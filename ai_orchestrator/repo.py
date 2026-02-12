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
    def run_command(self, cmd: str | List[str]) -> CommandResult:
        """
        Run a command in the repo root.

        Accepts either:
        - a string (legacy) -> executed via shell
        - a list of argv tokens (preferred) -> executed without shell (safe)
        """
        try:
            if isinstance(cmd, list):
                p = subprocess.run(
                    cmd,
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    shell=False,
                )
                res = CommandResult(p.returncode, p.stdout or "", p.stderr or "")
            else:
                p = subprocess.run(
                    cmd,
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    shell=True,
                )
                res = CommandResult(p.returncode, p.stdout or "", p.stderr or "")

            if not res.ok:
                log.warning("Command failed", extra={"fields": {"cmd": cmd, "returncode": res.returncode}})
            return res
        except Exception as e:
            log.warning("Command failed", extra={"fields": {"cmd": cmd, "error": str(e)}})
            return CommandResult(1, "", str(e))

    def run_tests(self, test_command: Optional[str]) -> CommandResult:
        if not test_command:
            return CommandResult(0, "No test_command configured; skipping.", "")
        return self.run_command(test_command)

    def run_build(self, build_command: Optional[str]) -> CommandResult:
        if not build_command:
            return CommandResult(0, "No build_command configured; skipping.", "")
        return self.run_command(build_command)

    # ---------- Git helpers ----------
    def git(self, args: List[str]) -> CommandResult:
        return self.run_command(["git", *args])

    def ensure_branch(self, branch_name: str) -> None:
        branches = self.git(["branch", "--list"]).stdout
        if branch_name not in branches:
            self.git(["checkout", "-b", branch_name])
        else:
            self.git(["checkout", branch_name])

    @staticmethod
    def _is_nothing_to_commit(res: CommandResult) -> bool:
        """
        Git returns non-zero for 'nothing to commit' in some cases.
        This is a benign no-op and must NOT be treated as pipeline failure.
        """
        txt = (res.stdout or "") + "\n" + (res.stderr or "")
        txt_l = txt.lower()
        needles = [
            "nothing to commit",
            "working tree clean",
            "no changes added to commit",
            "nothing added to commit",
        ]
        return any(n in txt_l for n in needles)

    def commit_all(self, message: str) -> CommandResult:
        """
        Stage all changes and commit.

        Returns:
        - If staging fails: staging result
        - Else: commit result
        - If commit is a benign no-op ('nothing to commit'): ok result (returncode=0)
        """
        add_res = self.git(["add", "."])
        if not add_res.ok:
            return add_res

        commit_res = self.git(["commit", "-m", message])
        if commit_res.ok:
            return commit_res

        if self._is_nothing_to_commit(commit_res):
            # Normalize benign no-op into success.
            return CommandResult(
                0,
                commit_res.stdout or "Nothing to commit; working tree clean.",
                commit_res.stderr or "",
            )

        return commit_res

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

        if has_minus or has_plus:
            if not (has_minus and has_plus):
                return False, "Diff contains only one of '---' or '+++' headers; both are required."
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
        """
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
