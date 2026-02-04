# ai_orchestrator/repo.py
from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .patching import FileContentPatch, Patch, UnifiedDiffPatch
import logging
log = logging.getLogger(__name__)

@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Repo:
    def __init__(self, root: Path):
        self.root = root

    # ---------- Files ----------
    def list_files(
        self,
        include_globs: List[str],
        exclude_globs: List[str],
    ) -> List[Path]:
        """
        Lists repository files matching include/exclude globs.

        Safety invariants:
          - Always excludes .git/** and .ai-orchestrator/** regardless of config.
        """
        files: List[Path] = []

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            rel = path.relative_to(self.root)
            rel_str = rel.as_posix()

            # Always ignore repository metadata and orchestrator artifacts.
            if rel_str == ".git" or rel_str.startswith(".git/"):
                continue
            if rel_str == ".ai-orchestrator" or rel_str.startswith(".ai-orchestrator/"):
                continue

            if exclude_globs and any(fnmatch.fnmatch(rel_str, pat) for pat in exclude_globs):
                continue

            # IMPORTANT: Path.match handles ** semantics better than fnmatch for includes.
            if include_globs and not any(rel.match(pat) for pat in include_globs):
                continue

            files.append(path)

        return files

    # ---------- Commands ----------
    def run_command(
        self,
        cmd: str,
        env: Optional[dict] = None,
    ) -> CommandResult:
        log.debug("Running command", extra={"fields": {"cmd": cmd}})
        proc = subprocess.run(
            cmd,
            cwd=self.root,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
        )
        res = CommandResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if not res.ok:
            log.warning("Command failed", extra={"fields": {"cmd": cmd, "returncode": res.returncode}})
        return res
    

    def run_tests(self, test_command: Optional[str]) -> CommandResult:
        if not test_command:
            return CommandResult(0, "No test_command configured; skipping.", "")
        return self.run_command(test_command)

    def run_build(self, build_command: Optional[str]) -> CommandResult:
        if not build_command:
            return CommandResult(0, "No build_command configured; skipping.", "")
        return self.run_command(build_command)

    # ---------- Git helpers ----------
    def git(self, args: Iterable[str]) -> CommandResult:
        cmd = "git " + " ".join(args)
        return self.run_command(cmd)

    def ensure_branch(self, branch_name: str) -> None:
        branches = self.git(["branch", "--list"]).stdout
        if branch_name not in branches:
            self.git(["checkout", "-b", branch_name])
        else:
            self.git(["checkout", branch_name])

    def commit_all(self, message: str) -> None:
        self.git(["add", "."])
        self.git(["commit", "-m", message])

    # ---------- Unified diff hardening ----------

    _HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")

    @staticmethod
    def _diff_sanity_check(diff_text: str) -> Tuple[bool, str]:
        """
        Fast local validation to prevent common 'No valid patches in input' failures.

        Accept:
          - modern header blocks: 'diff --git a/... b/...'
          - plus-file headers: '--- ...' and '+++ ...' where each can be:
              - 'a/<path>' / 'b/<path>'
              - '/dev/null' (new/deleted file)
        Require:
          - at least one valid range hunk header:
              @@ -l,s +l,s @@
          - hard reject any invalid hunk header (e.g. naked '@@').

        NOTE: We intentionally DO NOT require both 'diff --git' and legacy headers simultaneously.
        Many valid diffs contain both, but some tools emit only one style.
        """
        if not isinstance(diff_text, str) or not diff_text.strip():
            return False, "Diff is empty."

        txt = diff_text.strip("\n")

        lines = txt.splitlines()

        has_diff_git = any(line.startswith("diff --git ") for line in lines)
        has_minus = any(line.startswith("--- ") for line in lines)
        has_plus = any(line.startswith("+++ ") for line in lines)

        if not (has_diff_git or (has_minus and has_plus)):
            return False, (
                "Diff must include either a 'diff --git a/... b/...' header or a '--- ...'/'+++ ...' header pair."
            )

        # Validate that ---/+++ paths look git-ish if present
        if has_minus or has_plus:
            if not (has_minus and has_plus):
                return False, "Diff contains only one of '---' or '+++' headers; both are required."
            # Allow: a/<path>, b/<path>, /dev/null
            bad_headers: List[str] = []
            for line in lines:
                if line.startswith("--- "):
                    p = line[4:].strip()
                    if p != "/dev/null" and not (p.startswith("a/") or p.startswith("b/")):
                        bad_headers.append(line)
                if line.startswith("+++ "):
                    p = line[4:].strip()
                    if p != "/dev/null" and not (p.startswith("a/") or p.startswith("b/")):
                        bad_headers.append(line)
            if bad_headers:
                return False, f"Diff contains non-git file header(s): {bad_headers[:3]!r}"

        # Hard reject invalid hunks; require at least one valid hunk header.
        saw_hunk = False
        for line in lines:
            if line.startswith("@@"):
                saw_hunk = True
                if Repo._HUNK_RE.match(line) is None:
                    return False, (
                        "Diff contains an invalid hunk header. Expected '@@ -l,s +l,s @@' "
                        f"but got: {line!r}"
                    )

        if not saw_hunk:
            return False, "Diff contains no hunk headers ('@@ -l,s +l,s @@')."

        return True, "ok"

    def check_unified_diff(self, diff_text: str) -> CommandResult:
        """
        Runs `git apply --check` to produce diagnostics for failures without applying.
        """
        # (2) Windows-hardening: ensure trailing newline so stdin piping can't produce "corrupt patch" due to EOF edge cases.
        if isinstance(diff_text, str) and diff_text and not diff_text.endswith("\n"):
            diff_text = diff_text + "\n"

        ok, msg = self._diff_sanity_check(diff_text)
        if not ok:
            log.warning("Diff sanity check failed", extra={"fields": {"reason": msg}})
            return CommandResult(returncode=2, stdout="", stderr=f"Invalid unified diff: {msg}")

        args = ["git", "apply", "--check", "--whitespace=nowarn", "--recount"]
        proc = subprocess.run(
            args,
            cwd=self.root,
            input=diff_text,
            text=True,
            capture_output=True,
        )
        stderr = proc.stderr or ""
        if proc.returncode != 0 and "No valid patches in input" in stderr:
            stderr = (
                "git apply --check failed: No valid patches in input.\n"
                "This usually means the diff is malformed (missing file headers or range hunk headers).\n"
                f"Raw git stderr:\n{proc.stderr}"
            )
        return CommandResult(returncode=proc.returncode, stdout=proc.stdout, stderr=stderr)


    def apply_unified_diff(self, diff_text: str, *, update_index: bool = False) -> CommandResult:
        """
        Apply a unified diff using `git apply`, with a fail-fast preflight.

        Flow:
          1) local sanity check
          2) git apply --check
          3) git apply [--index]

        Uses:
          git apply --whitespace=nowarn --recount [--index]
        """
        # (2) Windows-hardening: ensure trailing newline so stdin piping can't produce "corrupt patch" due to EOF edge cases.
        if isinstance(diff_text, str) and diff_text and not diff_text.endswith("\n"):
            diff_text = diff_text + "\n"

        ok, msg = self._diff_sanity_check(diff_text)
        if not ok:
            log.warning("Diff sanity check failed", extra={"fields": {"reason": msg}})
            return CommandResult(returncode=2, stdout="", stderr=f"Invalid unified diff: {msg}")

        check_res = self.check_unified_diff(diff_text)
        if not check_res.ok:
            return check_res

        args = ["git", "apply", "--whitespace=nowarn", "--recount"]
        if update_index:
            args.append("--index")

        proc = subprocess.run(
            args,
            cwd=self.root,
            input=diff_text,
            text=True,
            capture_output=True,
        )
        stderr = proc.stderr or ""
        if proc.returncode != 0 and "No valid patches in input" in stderr:
            stderr = (
                "git apply failed: No valid patches in input.\n"
                "The diff is not git-applyable. Ensure it includes headers and range hunks.\n"
                f"Raw git stderr:\n{proc.stderr}"
            )
        return CommandResult(returncode=proc.returncode, stdout=proc.stdout, stderr=stderr)


    def apply_patches(self, patches: List[Patch]) -> CommandResult:
        """
        Apply a list of patches. Stops on first failure and returns that failure result.
        """
        for p in patches:
            if isinstance(p, FileContentPatch):
                p.path.parent.mkdir(parents=True, exist_ok=True)
                p.path.write_text(p.new_content, encoding="utf-8")
                continue

            if isinstance(p, UnifiedDiffPatch):
                res = self.apply_unified_diff(p.diff_text, update_index=False)
                if not res.ok:
                    return res
                continue

            raise TypeError(f"Unknown patch type: {type(p)}")

        return CommandResult(0, "All patches applied.", "")
