# ai_orchestrator/graph/context.py
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class ContextPackLimits:
    """
    Hard bounds to keep context packs small and predictable.
    """
    max_total_chars: int = 30_000
    max_file_chars: int = 6_000
    max_constraints_chars: int = 10_000
    max_module_summary_chars: int = 8_000
    max_files_in_pack: int = 12
    max_modules_in_pack: int = 6


class ContextPackBuilder:
    """
    Deterministic context builder for large repos.

    Produces a bounded context pack containing:
      - constraints docs excerpts
      - cached module summaries keyed by hash
      - targeted file excerpts

    Caching:
      - module summaries stored under .ai-orchestrator/cache/module_summaries/<module>.json
      - each summary includes a "content_hash" so we can skip regenerating

    Notes:
      - This implementation uses deterministic heuristics for summaries (no LLM),
        prioritizing reproducibility and bounded size.
      - You can later swap summary generation to LLM-assisted if desired.
    """

    def __init__(self, repo_root: Path, limits: Optional[ContextPackLimits] = None):
        self.repo_root = repo_root
        self.limits = limits or ContextPackLimits()

        self._ao_dir = self.repo_root / ".ai-orchestrator"
        self._cache_dir = self._ao_dir / "cache" / "module_summaries"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Public API ----------------

    def build(
        self,
        *,
        selected_files: List[Path],
        node_id: str,
        phase_name: str,
        run_dir: Path,
        node_dir: Path,
        failure_hints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build and persist a context pack. Returns the dict.

        failure_hints can include:
          - "failed_validator": name
          - "validator_stderr": string
          - "changed_paths": list[str]
        """
        selected_rel = [p.relative_to(self.repo_root).as_posix() for p in selected_files]

        constraints = self._load_constraints_docs()
        modules = self._infer_modules(selected_rel)
        modules = modules[: self.limits.max_modules_in_pack]

        module_summaries = [self._get_or_build_module_summary(m) for m in modules]

        # Target files: selected files first, then heuristically related
        target_rel_paths = self._target_files(selected_rel, failure_hints=failure_hints)
        target_rel_paths = target_rel_paths[: self.limits.max_files_in_pack]

        files_payload = []
        for rp in target_rel_paths:
            files_payload.append(
                {
                    "path": rp,
                    "excerpt": self._read_file_excerpt(rp, self.limits.max_file_chars),
                    "sha256": self._sha256_of_file(rp),
                }
            )

        pack = {
            "meta": {
                "phase": phase_name,
                "node_id": node_id,
                "limits": {
                    "max_total_chars": self.limits.max_total_chars,
                    "max_file_chars": self.limits.max_file_chars,
                    "max_constraints_chars": self.limits.max_constraints_chars,
                    "max_module_summary_chars": self.limits.max_module_summary_chars,
                    "max_files_in_pack": self.limits.max_files_in_pack,
                    "max_modules_in_pack": self.limits.max_modules_in_pack,
                },
                "selected_files": selected_rel,
                "target_files": target_rel_paths,
                "modules": modules,
                "failure_hints": failure_hints or {},
            },
            "constraints": constraints,
            "module_summaries": module_summaries,
            "files": files_payload,
        }

        pack = self._enforce_total_size(pack)
        try:
            encoded_len = len(json.dumps(pack, ensure_ascii=False))
        except Exception:
            encoded_len = -1

        log.info(
            "Context pack persisted",
            extra={"fields": {
                "node_id": node_id,
                "phase": phase_name,
                "selected_files": len(selected_files),
                "target_files": len(pack.get("meta", {}).get("target_files", []) or []),
                "modules": len(pack.get("meta", {}).get("modules", []) or []),
                "size_chars": encoded_len,
                "node_dir": str(node_dir.relative_to(self.repo_root).as_posix()),
            }},
        )

        # Persist: per-node + run index + latest pointer
        (node_dir / "context_pack.json").write_text(
            json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Run-level index (append/update)
        self._update_run_index(run_dir=run_dir, node_id=node_id, node_dir=node_dir)

        # Latest pointer
        latest = self._ao_dir / "context_pack.json"
        latest.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")

        return pack

    # ---------------- Constraints docs ----------------

    def _load_constraints_docs(self) -> Dict[str, str]:
        """
        Reads a small set of conventional docs if present:
          - docs/constraints.md
          - docs/architecture.md
          - docs/decisions/*.md (bounded)
        """
        docs = {}

        def read_if_exists(rel: str, limit: int) -> str:
            p = self.repo_root / rel
            if not p.exists() or not p.is_file():
                return ""
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""
            return txt[:limit]

        constraints_md = read_if_exists("docs/constraints.md", self.limits.max_constraints_chars)
        arch_md = read_if_exists("docs/architecture.md", self.limits.max_constraints_chars)

        decisions_dir = self.repo_root / "docs" / "decisions"
        decisions_texts: List[str] = []
        if decisions_dir.exists() and decisions_dir.is_dir():
            for p in sorted(decisions_dir.glob("*.md"))[:10]:
                try:
                    decisions_texts.append(p.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    continue
        decisions_blob = "\n\n".join(decisions_texts)[: self.limits.max_constraints_chars]

        if constraints_md:
            docs["docs/constraints.md"] = constraints_md
        if arch_md:
            docs["docs/architecture.md"] = arch_md
        if decisions_blob:
            docs["docs/decisions/*.md"] = decisions_blob

        return docs

    # ---------------- Module summaries ----------------

    def _infer_modules(self, rel_paths: List[str]) -> List[str]:
        """
        Basic module inference: first path component (e.g. "src", "ui", "backend").
        Falls back to directory name if file is at root.
        """
        modules: List[str] = []
        for rp in rel_paths:
            parts = rp.split("/")
            mod = parts[0] if len(parts) > 1 else "(root)"
            if mod not in modules:
                modules.append(mod)
        return modules

    def _get_or_build_module_summary(self, module_name: str) -> Dict[str, Any]:
        """
        Build deterministic summary for module:
          - list top-level imports in files
          - list classes and function defs (language-agnostic-ish heuristic)
          - include file list
        Cached by a module content hash.
        """
        module_files = self._module_files(module_name)
        content_hash = self._hash_paths(module_files)

        cache_path = self._cache_dir / f"{self._safe_name(module_name)}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("content_hash") == content_hash:
                    return cached
            except Exception:
                pass

        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("content_hash") == content_hash:
                    log.debug("Module summary cache hit", extra={"fields": {"module": module_name}})
                    return cached
            except Exception:
                pass
        log.debug("Module summary cache miss", extra={"fields": {"module": module_name}})

        summary_text = self._summarize_module_deterministic(module_files)

        # Bound summary size
        summary_text = summary_text[: self.limits.max_module_summary_chars]

        payload = {
            "module": module_name,
            "content_hash": content_hash,
            "files": [p for p in module_files],
            "summary": summary_text,
        }
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def _module_files(self, module_name: str) -> List[str]:
        """
        Returns rel paths belonging to the module, bounded by a file cap for perf.
        """
        if module_name == "(root)":
            candidates = [p for p in self.repo_root.glob("*") if p.is_file()]
        else:
            candidates = list((self.repo_root / module_name).rglob("*"))

        rels: List[str] = []
        for p in candidates:
            if not p.is_file():
                continue
            rp = p.relative_to(self.repo_root).as_posix()
            if rp.startswith(".git/") or rp.startswith(".ai-orchestrator/"):
                continue
            # avoid huge binary-like files
            if self._is_probably_binary(p):
                continue
            rels.append(rp)

        return rels[:200]  # hard cap

    def _summarize_module_deterministic(self, rel_paths: List[str]) -> str:
        imports: List[str] = []
        symbols: List[str] = []

        import_re = re.compile(r"^\s*(import\s+[^\s;]+|from\s+[^\s;]+\s+import\s+.+)", re.MULTILINE)
        class_re = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
        def_re = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
        js_fn_re = re.compile(r"^\s*(export\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
        ts_type_re = re.compile(r"^\s*(export\s+)?(type|interface)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

        for rp in rel_paths[:80]:
            txt = self._read_file_excerpt(rp, 8000)
            if not txt:
                continue

            for m in import_re.findall(txt):
                if m not in imports:
                    imports.append(m)

            for m in class_re.findall(txt):
                sym = f"class {m}"
                if sym not in symbols:
                    symbols.append(sym)

            for m in def_re.findall(txt):
                sym = f"def {m}()"
                if sym not in symbols:
                    symbols.append(sym)

            for m in js_fn_re.findall(txt):
                name = m[1]
                sym = f"function {name}()"
                if sym not in symbols:
                    symbols.append(sym)

            for m in ts_type_re.findall(txt):
                name = m[2]
                sym = f"{m[1]} {name}"
                if sym not in symbols:
                    symbols.append(sym)

        lines = []
        lines.append("Files:")
        for rp in rel_paths[:40]:
            lines.append(f"  - {rp}")
        if len(rel_paths) > 40:
            lines.append(f"  ... ({len(rel_paths) - 40} more)")

        lines.append("\nKey imports (sample):")
        for im in imports[:25]:
            lines.append(f"  - {im}")
        if len(imports) > 25:
            lines.append(f"  ... ({len(imports) - 25} more)")

        lines.append("\nKey symbols (sample):")
        for s in symbols[:35]:
            lines.append(f"  - {s}")
        if len(symbols) > 35:
            lines.append(f"  ... ({len(symbols) - 35} more)")

        return "\n".join(lines)

    # ---------------- Target file selection ----------------

    def _target_files(self, selected_rel: List[str], failure_hints: Optional[Dict[str, Any]]) -> List[str]:
        """
        Targeting heuristics (deterministic, bounded):
          1) include selected files (in order)
          2) if failure_hints includes changed_paths, include those
          3) include import-neighbors of selected files (same module or directly imported local paths)
        """
        out: List[str] = []
        seen = set()

        def add(rp: str) -> None:
            if rp in seen:
                return
            seen.add(rp)
            out.append(rp)

        for rp in selected_rel:
            add(rp)

        if failure_hints:
            for rp in failure_hints.get("changed_paths", [])[:20]:
                add(rp)

        # import neighbors
        for rp in selected_rel[:8]:
            neighbors = self._import_neighbors(rp)
            for nb in neighbors[:6]:
                add(nb)

        return out

    def _import_neighbors(self, rel_path: str) -> List[str]:
        """
        Best-effort inference of local import neighbors for TS/JS/PY:
          - from './x' or '../x'
          - import ... from './x'
          - python: from package import ... (maps only within same top module)
        """
        p = self.repo_root / rel_path
        if not p.exists():
            return []

        txt = self._read_file_excerpt(rel_path, 12000)
        if not txt:
            return []

        neighbors: List[str] = []

        # JS/TS relative imports
        re_js = re.compile(r"""from\s+['"](\.{1,2}/[^'"]+)['"]""")
        for m in re_js.findall(txt):
            resolved = self._resolve_relative_import(rel_path, m)
            if resolved:
                neighbors.append(resolved)

        # Python relative-ish: from <top>.<...> import ...
        re_py = re.compile(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+", re.MULTILINE)
        top = rel_path.split("/")[0] if "/" in rel_path else ""
        for mod in re_py.findall(txt):
            if top and mod.startswith(top + "."):
                guess = (mod.replace(".", "/") + ".py")
                if (self.repo_root / guess).exists():
                    neighbors.append(guess)

        # keep unique order
        seen = set()
        out = []
        for n in neighbors:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _resolve_relative_import(self, rel_path: str, import_path: str) -> Optional[str]:
        """
        Resolve ./foo or ../foo relative to rel_path directory.
        Attempts common extensions.
        """
        base_dir = (self.repo_root / rel_path).parent
        candidate = (base_dir / import_path).resolve()

        # Ensure within repo
        try:
            candidate.relative_to(self.repo_root)
        except Exception:
            return None

        # Try direct file, then with extensions, then index files.
        exts = ["", ".ts", ".tsx", ".js", ".jsx", ".py"]
        for ext in exts:
            p = Path(str(candidate) + ext)
            if p.exists() and p.is_file():
                return p.relative_to(self.repo_root).as_posix()

        if candidate.exists() and candidate.is_dir():
            for idx in ["index.ts", "index.tsx", "index.js", "index.jsx"]:
                p = candidate / idx
                if p.exists() and p.is_file():
                    return p.relative_to(self.repo_root).as_posix()

        return None

    # ---------------- Utilities ----------------

    def _read_file_excerpt(self, rel_path: str, limit: int) -> str:
        p = self.repo_root / rel_path
        if not p.exists() or not p.is_file():
            return ""
        if self._is_probably_binary(p):
            return ""
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        return txt[:limit]

    def _sha256_of_file(self, rel_path: str) -> str:
        p = self.repo_root / rel_path
        if not p.exists() or not p.is_file():
            return ""
        try:
            data = p.read_bytes()
        except Exception:
            return ""
        return hashlib.sha256(data).hexdigest()

    def _hash_paths(self, rel_paths: List[str]) -> str:
        h = hashlib.sha256()
        for rp in rel_paths:
            h.update(rp.encode("utf-8"))
            h.update(b"\0")
            # include content hash
            h.update(self._sha256_of_file(rp).encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _safe_name(self, module_name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", module_name)

    def _is_probably_binary(self, path: Path) -> bool:
        # quick heuristic: null bytes early in file
        try:
            chunk = path.read_bytes()[:2048]
        except Exception:
            return True
        return b"\x00" in chunk

    def _enforce_total_size(self, pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure pack is bounded by truncating excerpts/summaries further if needed.
        """
        encoded = json.dumps(pack, ensure_ascii=False)
        if len(encoded) <= self.limits.max_total_chars:
            return pack

        # Reduce file excerpts
        for f in pack.get("files", []):
            if "excerpt" in f and isinstance(f["excerpt"], str):
                f["excerpt"] = f["excerpt"][: max(500, self.limits.max_file_chars // 2)]

        # Reduce module summaries
        for ms in pack.get("module_summaries", []):
            if "summary" in ms and isinstance(ms["summary"], str):
                ms["summary"] = ms["summary"][: max(800, self.limits.max_module_summary_chars // 2)]

        # Reduce constraints
        for k, v in list(pack.get("constraints", {}).items()):
            if isinstance(v, str):
                pack["constraints"][k] = v[: max(800, self.limits.max_constraints_chars // 2)]

        # Final hard trim if still too large: drop non-selected files
        encoded = json.dumps(pack, ensure_ascii=False)
        if len(encoded) > self.limits.max_total_chars:
            selected = set(pack.get("meta", {}).get("selected_files", []))
            pack["files"] = [f for f in pack.get("files", []) if f.get("path") in selected]

        return pack

    def _update_run_index(self, run_dir: Path, node_id: str, node_dir: Path) -> None:
        index_path = run_dir / "context_pack_index.json"
        index = {}
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                index = {}
        index[node_id] = str(node_dir.relative_to(run_dir).as_posix()) + "/context_pack.json"
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
