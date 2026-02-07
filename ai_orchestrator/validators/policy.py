# ai_orchestrator/validators/policy.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..repo import Repo
from .types import ValidatorResult


@dataclass(frozen=True)
class ForbiddenImportRule:
    """
    Enforce: files under src_prefix cannot import from any forbidden_import_prefixes.

    Example:
      src_prefix="ui/"
      forbidden_import_prefixes=["backend", "server", "db"]
    """
    src_prefix: str
    forbidden_import_prefixes: List[str]


@dataclass(frozen=True)
class DependencyBoundaryRule:
    """
    Enforce: files under src_prefix may only import from allowed_import_prefixes (optional).
    If allowed_import_prefixes is empty, rule is ignored.
    """
    src_prefix: str
    allowed_import_prefixes: List[str]


_SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("private_key", re.compile(r"-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws_secret_key", re.compile(r"\b[0-9a-zA-Z/+]{40}\b")),
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("generic_token_assign", re.compile(r"\b(API_KEY|SECRET|TOKEN|PASSWORD)\b\s*[:=]\s*['\"][^'\"\n]{8,}['\"]", re.IGNORECASE)),
]


def run_policy_validator(
    *,
    name: str,
    repo: Repo,
    params: Dict[str, Any],
) -> ValidatorResult:
    """
    Dispatch policy validators by name.
    """
    if name == "forbidden_imports":
        return forbidden_imports(repo=repo, params=params)
    if name == "dependency_boundaries":
        return dependency_boundaries(repo=repo, params=params)
    if name == "secrets_scan":
        return secrets_scan(repo=repo, params=params)
    if name == "assert_text_contract":
        return assert_text_contract(repo=repo, params=params)

    return ValidatorResult(
        name=name,
        ok=False,
        exit_code=2,
        stdout="",
        stderr=f"Unknown policy validator: {name}",
        duration_s=0.0,
    )

def assert_text_contract(*, repo: Repo, params: Dict[str, Any]) -> ValidatorResult:
    """
    Generic text-based contract enforcement.

    Params:
      - contracts: list of objects:
          - path: repo-relative file path
          - require_all_regex: list[str] (all must match)
          - require_any_regex: list[str] (at least one must match)
          - message: str (optional failure hint)
    """
    import re
    import time

    t0 = time.time()
    contracts = params.get("contracts") or []
    if not isinstance(contracts, list) or not contracts:
        return ValidatorResult(
            name="assert_text_contract",
            ok=False,
            exit_code=2,
            stdout="",
            stderr="assert_text_contract: missing or invalid 'contracts' list in params",
            duration_s=time.time() - t0,
        )

    failures = []
    for c in contracts:
        if not isinstance(c, dict):
            failures.append("Invalid contract entry (not an object).")
            continue

        rp = str(c.get("path") or "").replace("\\", "/").strip()
        msg = str(c.get("message") or "").strip()
        all_rx = c.get("require_all_regex") or []
        any_rx = c.get("require_any_regex") or []

        if not rp:
            failures.append("Contract missing 'path'.")
            continue

        p = repo.root / rp
        if not p.exists():
            failures.append(f"{rp}: file does not exist. {msg}".strip())
            continue

        txt = p.read_text(encoding="utf-8", errors="ignore")

        # All-of
        if all_rx:
            if not isinstance(all_rx, list) or not all(isinstance(x, str) for x in all_rx):
                failures.append(f"{rp}: require_all_regex must be list[str].")
            else:
                for rx in all_rx:
                    if not re.search(rx, txt, flags=re.MULTILINE):
                        failures.append(f"{rp}: missing required pattern: {rx}. {msg}".strip())

        # Any-of
        if any_rx:
            if not isinstance(any_rx, list) or not all(isinstance(x, str) for x in any_rx):
                failures.append(f"{rp}: require_any_regex must be list[str].")
            else:
                if not any(re.search(rx, txt, flags=re.MULTILINE) for rx in any_rx):
                    failures.append(f"{rp}: none of require_any_regex matched. {msg}".strip())

    ok = len(failures) == 0
    return ValidatorResult(
        name="assert_text_contract",
        ok=ok,
        exit_code=0 if ok else 1,
        stdout="All text contracts satisfied." if ok else "",
        stderr="\n".join(failures) if not ok else "",
        duration_s=time.time() - t0,
    )

def forbidden_imports(*, repo: Repo, params: Dict[str, Any]) -> ValidatorResult:
    rules = []
    for r in params.get("rules", []) or []:
        rules.append(
            ForbiddenImportRule(
                src_prefix=str(r.get("src_prefix", "")),
                forbidden_import_prefixes=list(r.get("forbidden_import_prefixes", []) or []),
            )
        )

    if not rules:
        return ValidatorResult(
            name="forbidden_imports",
            ok=True,
            exit_code=0,
            stdout="No forbidden import rules configured; skipping.",
            stderr="",
            duration_s=0.0,
        )

    violations: List[str] = []

    # Simple import detection across TS/JS/PY
    re_js = re.compile(r"""from\s+['"]([^'"]+)['"]""")
    re_py = re.compile(r"^\s*(import\s+([A-Za-z0-9_\.]+)|from\s+([A-Za-z0-9_\.]+)\s+import\s+)", re.MULTILINE)

    all_files = repo.list_files(include_globs=["**/*.*"], exclude_globs=[])
    for p in all_files:
        rel = p.relative_to(repo.root).as_posix()
        for rule in rules:
            if not rel.startswith(rule.src_prefix):
                continue

            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # JS/TS imports
            for imp in re_js.findall(txt):
                for bad in rule.forbidden_import_prefixes:
                    if imp == bad or imp.startswith(bad + "/") or imp.startswith(bad + "."):
                        violations.append(f"{rel}: forbidden import '{imp}' (rule src_prefix={rule.src_prefix})")

            # Python imports
            for m in re_py.findall(txt):
                mod = m[1] or m[2]  # import x | from x import
                if not mod:
                    continue
                for bad in rule.forbidden_import_prefixes:
                    if mod == bad or mod.startswith(bad + "."):
                        violations.append(f"{rel}: forbidden import '{mod}' (rule src_prefix={rule.src_prefix})")

    if violations:
        return ValidatorResult(
            name="forbidden_imports",
            ok=False,
            exit_code=1,
            stdout="Forbidden import violations detected:\n" + "\n".join(violations[:200]),
            stderr="Fix the above imports or adjust architecture.forbidden_imports rules in config.",
            duration_s=0.0,
        )

    return ValidatorResult(
        name="forbidden_imports",
        ok=True,
        exit_code=0,
        stdout="No forbidden import violations detected.",
        stderr="",
        duration_s=0.0,
    )


def dependency_boundaries(*, repo: Repo, params: Dict[str, Any]) -> ValidatorResult:
    rules = []
    for r in params.get("rules", []) or []:
        rules.append(
            DependencyBoundaryRule(
                src_prefix=str(r.get("src_prefix", "")),
                allowed_import_prefixes=list(r.get("allowed_import_prefixes", []) or []),
            )
        )

    active = [r for r in rules if r.src_prefix and r.allowed_import_prefixes]
    if not active:
        return ValidatorResult(
            name="dependency_boundaries",
            ok=True,
            exit_code=0,
            stdout="No dependency boundary rules configured; skipping.",
            stderr="",
            duration_s=0.0,
        )

    violations: List[str] = []
    re_js = re.compile(r"""from\s+['"]([^'"]+)['"]""")
    re_py = re.compile(r"^\s*(import\s+([A-Za-z0-9_\.]+)|from\s+([A-Za-z0-9_\.]+)\s+import\s+)", re.MULTILINE)

    all_files = repo.list_files(include_globs=["**/*.*"], exclude_globs=[])
    for p in all_files:
        rel = p.relative_to(repo.root).as_posix()
        for rule in active:
            if not rel.startswith(rule.src_prefix):
                continue

            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            imports = []
            for imp in re_js.findall(txt):
                imports.append(imp)
            for m in re_py.findall(txt):
                mod = m[1] or m[2]
                if mod:
                    imports.append(mod)

            for imp in imports:
                # If it looks like a local prefix (no http(s), not built-in), enforce allowlist by prefix
                if imp.startswith(("http://", "https://")):
                    continue
                if imp.startswith((".", "..")):
                    continue  # relative import: allow
                if not any(imp == ok or imp.startswith(ok + "/") or imp.startswith(ok + ".") for ok in rule.allowed_import_prefixes):
                    violations.append(
                        f"{rel}: import '{imp}' not allowed under boundary (src_prefix={rule.src_prefix})"
                    )

    if violations:
        return ValidatorResult(
            name="dependency_boundaries",
            ok=False,
            exit_code=1,
            stdout="Dependency boundary violations detected:\n" + "\n".join(violations[:200]),
            stderr="Adjust imports or update architecture.dependency_boundaries rules in config.",
            duration_s=0.0,
        )

    return ValidatorResult(
        name="dependency_boundaries",
        ok=True,
        exit_code=0,
        stdout="No dependency boundary violations detected.",
        stderr="",
        duration_s=0.0,
    )


def secrets_scan(*, repo: Repo, params: Dict[str, Any]) -> ValidatorResult:
    """
    Deterministic scan of git diff (preferred) and/or changed files for secrets.

    Params:
      - use_git_diff: bool (default True)
      - scan_paths: optional list[str] relative paths to scan (if provided, scans these)
    """
    use_git_diff = bool(params.get("use_git_diff", True))
    scan_paths = list(params.get("scan_paths", []) or [])

    findings: List[str] = []
    scanned_bytes = 0

    def scan_text(label: str, txt: str) -> None:
        nonlocal scanned_bytes
        scanned_bytes += len(txt.encode("utf-8", errors="ignore"))
        for pname, pat in _SECRET_PATTERNS:
            if pat.search(txt):
                findings.append(f"{label}: matched {pname}")

    if use_git_diff:
        diff = repo.run_command("git diff").stdout or ""
        if diff.strip():
            scan_text("git diff", diff)

    # If specific paths provided, scan them; else scan files changed in diff.
    if scan_paths:
        targets = scan_paths
    else:
        names = repo.run_command("git diff --name-only").stdout.splitlines()
        targets = [n.strip() for n in names if n.strip()]

    for rp in targets[:200]:
        p = repo.root / rp
        if not p.exists() or not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        scan_text(rp, txt[:200_000])  # cap per file

    if findings:
        return ValidatorResult(
            name="secrets_scan",
            ok=False,
            exit_code=1,
            stdout="Potential secrets detected:\n" + "\n".join(findings[:200]),
            stderr="Remove secrets from code/diff and use environment variables or secret managers.",
            duration_s=0.0,
        )

    return ValidatorResult(
        name="secrets_scan",
        ok=True,
        exit_code=0,
        stdout=f"Secrets scan OK. Scanned ~{scanned_bytes} bytes.",
        stderr="",
        duration_s=0.0,
    )
