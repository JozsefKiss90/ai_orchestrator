# ai_orchestrator/repo.py
from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .patching import FileContentPatch, Patch, UnifiedDiffPatch


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
        proc = subprocess.run(
            cmd,
            cwd=self.root,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return CommandResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

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

    @staticmethod
    def _diff_sanity_check(diff_text: str) -> Tuple[bool, str]:
        """
        Fast local validation to prevent common 'No valid patches in input' failures.

        We accept either:
          - modern header: "diff --git a/... b/..."
          OR
          - legacy header pair: "--- a/..." and "+++ b/..."

        And we require at least one range hunk header:
          @@ -l,s +l,s @@

        This is intentionally simple string matching (no regex) to avoid false negatives.
        """
        if not isinstance(diff_text, str) or not diff_text.strip():
            return False, "Diff is empty."

        txt = diff_text.strip()

        has_diff_git = "diff --git " in txt
        has_legacy_headers = ("--- a/" in txt) and ("+++ b/" in txt)
        if not (has_diff_git and has_legacy_headers):
            return False, (
                "Diff must include both 'diff --git a/... b/...' and '--- a/...'/ '+++ b/...'."
            )

        # Require at least one proper range hunk header
        # (the earlier failure mode was '@@' without ranges).
        for line in txt.splitlines():
            if line.startswith("@@"):
                # must look like: @@ -1,6 +1,11 @@
                if "@@ -" not in line or " +" not in line or " @@" not in line:
                    return False, (
                        "Diff contains an invalid hunk header. Expected '@@ -l,s +l,s @@' "
                        f"but got: {line!r}"
                    )

        return True, "ok"

    def _normalize_unified_diff(self, diff_text: str) -> str:
        """
        Normalize diff text for git apply:
        - normalize line endings to LF
        - ensure final trailing newline (git can treat missing final newline as corrupt patch)
        """
        txt = (diff_text or "").replace("\r\n", "\n").replace("\r", "\n")
        if txt and not txt.endswith("\n"):
            txt += "\n"
        return txt


    def check_unified_diff(self, diff_text: str) -> CommandResult:
        """
        Runs `git apply --check` to produce diagnostics for failures without applying.
        """
        diff_text = self._normalize_unified_diff(diff_text)
        ok, msg = self._diff_sanity_check(diff_text)

        if not ok:
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
        # Normalize a common opaque failure into something actionable
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
        diff_text = self._normalize_unified_diff(diff_text)
        ok, msg = self._diff_sanity_check(diff_text)

        if not ok:
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
