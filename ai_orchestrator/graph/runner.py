# ai_orchestrator/graph/runner.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import RepoConfig
from ..llm import LLMClient, SCHEMA_REPAIR_UNIFIED_DIFF_V1
from ..patching import FileContentPatch, Patch, UnifiedDiffPatch
from ..repo import Repo
from ..validators.pipeline import ValidatorPipeline
from ..validators.types import PipelineResult, ValidatorSpec
from ..phases.base import PhaseContext
from .context import ContextPackBuilder
from .types import DAG, Node, NodeResult


# Local schema: full-file fallback (new file creation without unified diffs).
SCHEMA_REPAIR_FILES_FULL_CONTENT_V1: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        "explanation": {"type": "string"},
    },
    "required": ["files", "explanation"],
    "additionalProperties": False,
}


class DagRunner:
    def __init__(self, cfg: RepoConfig, repo: Repo, llm: LLMClient):
        self.cfg = cfg
        self.repo = repo
        self.llm = llm

    # ---------------- Artifacts ----------------

    def _init_run_dir(self, run_name: str) -> Path:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.repo.root / ".ai-orchestrator" / "runs" / f"{run_id}-{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_json(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _write_latest_validator_results(self, obj: Any) -> None:
        """
        Optional Phase 2/4 convenience pointer (Acceptance D):
          .ai-orchestrator/validator_results.json
        """
        p = self.repo.root / ".ai-orchestrator" / "validator_results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(p, obj)

    # ---------------- Git snapshot (reproducibility) ----------------

    def _git_snapshot(self) -> Dict[str, Any]:
        head = (self.repo.git(["rev-parse", "HEAD"]).stdout or "").strip()
        branch = (self.repo.git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout or "").strip()
        porcelain = self.repo.git(["status", "--porcelain"]).stdout or ""
        return {
            "head_sha": head,
            "branch": branch,
            "dirty": bool(porcelain.strip()),
            "status_porcelain": porcelain,
        }

    # ---------------- Topological sort ----------------

    def _toposort(self, dag: DAG) -> List[Node]:
        nodes_by_id: Dict[str, Node] = {n.id: n for n in dag.nodes}

        in_deg: Dict[str, int] = {n.id: 0 for n in dag.nodes}
        adj: Dict[str, List[str]] = {n.id: [] for n in dag.nodes}

        for n in dag.nodes:
            for d in n.deps:
                if d not in nodes_by_id:
                    raise ValueError(f"Node {n.id} depends on unknown node {d}")
                adj[d].append(n.id)
                in_deg[n.id] += 1

        queue: List[str] = [nid for nid, deg in in_deg.items() if deg == 0]
        ordered: List[Node] = []

        while queue:
            nid = queue.pop(0)
            ordered.append(nodes_by_id[nid])
            for nxt in adj[nid]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)

        if len(ordered) != len(dag.nodes):
            raise ValueError("DAG contains a cycle; cannot execute.")
        return ordered

    # ---------------- Validators ----------------

    def _select_validator_specs(self, names: List[str]) -> List[ValidatorSpec]:
        if not names:
            return []

        specs_by_name: Dict[str, ValidatorSpec] = {v.name: v for v in self.cfg.validators}
        out: List[ValidatorSpec] = []
        for name in names:
            if name not in specs_by_name:
                raise ValueError(f"Unknown validator requested: {name}")
            out.append(specs_by_name[name])
        return out

    def _run_validators(self, validators: List[str]) -> PipelineResult:
        specs = self._select_validator_specs(validators)
        pipeline = ValidatorPipeline(validators=specs, stop_on_fail=True)
        return pipeline.run(self.repo)

    # ---------------- Unified diff repair helpers ----------------

    def _extract_paths_from_diff(self, diff_text: str) -> List[str]:
        """
        Best-effort path extraction. Handles:
          +++ b/path
          --- a/path
          /dev/null
        """
        paths: List[str] = []
        for line in diff_text.splitlines():
            if line.startswith("+++ "):
                token = line[4:].strip()
            elif line.startswith("--- "):
                token = line[4:].strip()
            else:
                continue

            token = token.replace("b/", "", 1) if token.startswith("b/") else token
            token = token.replace("a/", "", 1) if token.startswith("a/") else token
            if token != "/dev/null":
                paths.append(token)

        seen = set()
        out: List[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _minimal_file_snippets(self, rel_paths: List[str], max_chars: int = 4000) -> Dict[str, str]:
        snippets: Dict[str, str] = {}
        for rp in rel_paths:
            p = self.repo.root / rp
            if not p.exists() or not p.is_file():
                snippets[rp] = ""
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = ""
            snippets[rp] = txt[:max_chars]
        return snippets

    def _repair_unified_diff_json(
        self,
        *,
        failing_diff: str,
        apply_error: str,
        check_error: str,
        file_snippets: Dict[str, str],
        attempt: int,
        json_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Resilient wrapper: retries LLM JSON/schema parsing failures.
        """
        system_prompt = (
            "You are an expert engineer. You repair unified diffs so they apply cleanly with `git apply`.\n"
            "Rules:\n"
            "- Output a corrected unified diff.\n"
            "- Keep edits minimal.\n"
            "- Do not introduce unrelated changes.\n"
            "- Ensure headers and file paths are correct.\n"
            "- CRITICAL: Never output a naked '@@' line. Every hunk header must be like '@@ -l,s +l,s @@'.\n"
            "- Include 'diff --git', '--- ...', '+++ ...' for every changed file (use /dev/null for new/deleted files).\n"
        )
        user_prompt = (
            f"Attempt {attempt}: The following unified diff failed to apply.\n\n"
            "FAILURE DETAILS:\n"
            f"- git apply stderr:\n{apply_error}\n\n"
            f"- git apply --check stderr:\n{check_error}\n\n"
            "FAILING DIFF:\n"
            "```diff\n"
            f"{failing_diff}\n"
            "```\n\n"
            "CURRENT FILE SNIPPETS (may be empty if file missing):\n"
            + "\n\n".join(
                f"// FILE: {path}\n```code\n{snippet}\n```"
                for path, snippet in file_snippets.items()
            )
            + "\n\nReturn ONLY JSON: { diff: string, explanation: string }"
        )

        last_err: Optional[Exception] = None
        for r in range(1, json_retries + 1):
            try:
                return self.llm.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt + (f"\n\n(If your previous output was not valid JSON, fix it. Retry {r}/{json_retries}.)"),
                    schema=SCHEMA_REPAIR_UNIFIED_DIFF_V1,
                    schema_name="repair_unified_diff_v1",
                )
            except Exception as e:
                last_err = e

        raise RuntimeError(f"LLM returned malformed JSON for diff repair after {json_retries} retries: {last_err}")

    def _repair_files_full_content_json(
        self,
        *,
        allowed_paths: List[str],
        apply_error: str,
        check_error: str,
        failing_diff: str,
        file_snippets: Dict[str, str],
        json_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Fallback repair: return full file contents for the affected files.
        This supports NEW FILE CREATION without unified diffs by producing FileContentPatch.

        (1) Fix: strict-schema compatibility.
            Your JSON schema validator requires `required` to include every key in `properties`.
            Therefore we require BOTH "files" and "explanation".
        """
        system_prompt = (
            "You are an expert engineer. The unified diff workflow failed.\n"
            "Your job now is to provide FULL file contents for the files that should exist after the change.\n"
            "Rules:\n"
            "- You may ONLY return paths from the provided allowed_paths list.\n"
            "- Return complete file contents for each path.\n"
            "- Keep edits minimal and consistent with the intended refactor.\n"
            "- Do not include markdown fences inside the JSON.\n"
        )
        user_prompt = (
            "The patch could not be applied even after diff repair attempts.\n\n"
            "FAILURE DETAILS:\n"
            f"- git apply stderr:\n{apply_error}\n\n"
            f"- git apply --check stderr:\n{check_error}\n\n"
            "ORIGINAL / LAST DIFF (for intent):\n"
            "```diff\n"
            f"{failing_diff}\n"
            "```\n\n"
            "ALLOWED PATHS:\n"
            + "\n".join(f"- {p}" for p in allowed_paths)
            + "\n\n"
            "CURRENT FILE SNIPPETS (may be empty if file missing):\n"
            + "\n\n".join(
                f"// FILE: {path}\n```code\n{snippet}\n```"
                for path, snippet in file_snippets.items()
            )
            + "\n\nReturn ONLY JSON: { files: [{path: string, content: string}, ...], explanation: string }"
        )

        # (1) Strict-schema fix: require BOTH keys.
        schema = dict(SCHEMA_REPAIR_FILES_FULL_CONTENT_V1)
        schema["required"] = ["files", "explanation"]

        last_err: Optional[Exception] = None
        for r in range(1, json_retries + 1):
            try:
                return self.llm.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                    + (f"\n\n(If your previous output was not valid JSON, fix it. Retry {r}/{json_retries}.)"),
                    schema=schema,
                    schema_name="repair_files_full_content_v1",
                )
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"LLM returned malformed JSON for file-content repair after {json_retries} retries: {last_err}"
        )


    def _safe_relpath(self, rp: str) -> Optional[Path]:
        """
        Validate and normalize a repo-relative path string.
        Reject absolute paths, drive letters, and parent traversal.
        """
        try:
            p = Path(rp)
        except Exception:
            return None
        if p.is_absolute():
            return None
        # Block Windows drive-letter like "C:..."
        if len(p.parts) > 0 and p.parts[0].endswith(":"):
            return None
        if ".." in p.parts:
            return None
        # Normalize
        return Path(p.as_posix())

    def _apply_patches_with_repair(
        self,
        patches: List[Patch],
        node_dir: Path,
    ) -> Tuple[bool, List[Patch], str]:
        """
        Apply patches. If unified diff fails, attempt repair.
        If unified diff still fails, fall back to full-file patches (FileContentPatch).

        UX semantics:
          - Only create patch_apply_error.txt if the FINAL outcome is failure.
          - If repaired, write patch_apply_error_initial.txt + patch_apply_report.json.
        Returns: (ok, possibly_updated_patches, message)

        (3) Improvement: include direct artifact paths in the failure message so users don't have to hunt.
        """

        def _artifact_hint() -> str:
            # Use paths relative to repo root if possible (much easier to paste/open)
            try:
                rel = node_dir.relative_to(self.repo.root).as_posix()
                base = rel
            except Exception:
                base = str(node_dir)
            return (
                f"Artifacts: {base}/patch_apply_report.json ; "
                f"{base}/patch_apply_error_initial.txt ; "
                f"{base}/patch_apply_check_error_initial.txt ; "
                f"{base}/patch_repair_context.json"
            )

        report: Dict[str, Any] = {
            "initial_ok": None,
            "repaired": False,
            "fallback_file_content_used": False,
            "attempts": 0,
            "initial_apply_error": "",
            "initial_check_error": "",
            "final_apply_error": "",
            "final_check_error": "",
            "message": "",
        }

        res = self.repo.apply_patches(patches)
        report["initial_ok"] = bool(res.ok)
        if res.ok:
            report["message"] = "Applied patches successfully."
            self._write_json(node_dir / "patch_apply_report.json", report)
            return True, patches, report["message"]

        unified = next((p for p in patches if isinstance(p, UnifiedDiffPatch)), None)
        if unified is None:
            report["final_apply_error"] = (res.stderr or res.stdout or "").strip()
            report["message"] = "Patch application failed (non-unified-diff)."
            self._write_json(node_dir / "patch_apply_report.json", report)
            self._write_text(node_dir / "patch_apply_error.txt", report["final_apply_error"])
            return False, patches, f'{report["message"]} {_artifact_hint()}'

        check_res = self.repo.check_unified_diff(unified.diff_text)
        report["initial_apply_error"] = (res.stderr or res.stdout or "").strip()
        report["initial_check_error"] = (check_res.stderr or check_res.stdout or "").strip()

        self._write_text(node_dir / "patch_apply_error_initial.txt", report["initial_apply_error"])
        self._write_text(node_dir / "patch_apply_check_error_initial.txt", report["initial_check_error"])

        rel_paths = self._extract_paths_from_diff(unified.diff_text)
        snippets = self._minimal_file_snippets(rel_paths)

        self._write_json(
            node_dir / "patch_repair_context.json",
            {
                "apply_error": report["initial_apply_error"],
                "check_error": report["initial_check_error"],
                "paths": rel_paths,
                "snippets": snippets,
                "original_diff": unified.diff_text,
            },
        )

        # -------- Stage 1: unified diff repair --------
        max_attempts = 2
        candidate = unified.diff_text
        last_apply_err = report["initial_apply_error"]
        last_check_err = report["initial_check_error"]

        for attempt in range(1, max_attempts + 1):
            report["attempts"] = attempt

            try:
                data = self._repair_unified_diff_json(
                    failing_diff=candidate,
                    apply_error=last_apply_err,
                    check_error=last_check_err,
                    file_snippets=snippets,
                    attempt=attempt,
                    json_retries=2,
                )
                candidate = (data.get("diff") or "").strip()
            except Exception as e:
                # Do NOT poison the attempt; record and continue to next attempt.
                self._write_text(node_dir / f"patch_repair_llm_error_{attempt}.txt", str(e))
                candidate = candidate.strip()

            self._write_json(node_dir / f"patch_repair_attempt_{attempt}.json", {"diff": candidate})

            res2 = self.repo.apply_unified_diff(candidate, update_index=False)
            if res2.ok:
                report["repaired"] = True
                report["message"] = f"Patch repair succeeded on attempt {attempt}."
                self._write_json(node_dir / "patch_apply_report.json", report)
                return True, [UnifiedDiffPatch(diff_text=candidate)], report["message"]

            check_res2 = self.repo.check_unified_diff(candidate)
            last_apply_err = (res2.stderr or res2.stdout or "").strip()
            last_check_err = (check_res2.stderr or check_res2.stdout or "").strip()
            report["final_apply_error"] = last_apply_err
            report["final_check_error"] = last_check_err

        # -------- Stage 2: file-content fallback (new file creation without diffs) --------
        allowed_paths = [p for p in rel_paths if self._safe_relpath(p) is not None]

        if allowed_paths:
            try:
                fallback_data = self._repair_files_full_content_json(
                    allowed_paths=allowed_paths,
                    apply_error=last_apply_err,
                    check_error=last_check_err,
                    failing_diff=candidate or unified.diff_text,
                    file_snippets=snippets,
                    json_retries=2,
                )
                files = fallback_data.get("files") or []
                file_patches: List[Patch] = []
                for item in files:
                    rp = (item.get("path") or "").strip()
                    content = item.get("content")
                    if content is None:
                        continue

                    rp_norm = self._safe_relpath(rp)
                    if rp_norm is None:
                        continue
                    rp_str = rp_norm.as_posix()
                    if rp_str not in allowed_paths:
                        continue

                    file_patches.append(
                        FileContentPatch(
                            path=(self.repo.root / rp_str),
                            new_content=str(content),
                        )
                    )

                if file_patches:
                    res3 = self.repo.apply_patches(file_patches)
                    if res3.ok:
                        report["fallback_file_content_used"] = True
                        report["message"] = "File-content fallback repair succeeded (wrote full file contents)."
                        self._write_json(node_dir / "patch_apply_report.json", report)
                        self._write_json(node_dir / "patch_repair_files_full_content.json", fallback_data)
                        return True, file_patches, report["message"]

                    report["final_apply_error"] = (res3.stderr or res3.stdout or "").strip()
                    report["final_check_error"] = ""
                    self._write_json(node_dir / "patch_repair_files_full_content.json", fallback_data)

            except Exception as e:
                self._write_text(node_dir / "patch_repair_files_full_content_error.txt", str(e))

        # FINAL failure: now write patch_apply_error.txt
        report["message"] = (
            "Unified diff failed to apply even after repair attempts (and file-content fallback failed)."
        )
        self._write_json(node_dir / "patch_apply_report.json", report)
        self._write_text(
            node_dir / "patch_apply_error.txt",
            report["final_apply_error"] or report["initial_apply_error"],
        )
        if report["final_check_error"] or report["initial_check_error"]:
            self._write_text(
                node_dir / "patch_apply_check_error.txt",
                report["final_check_error"] or report["initial_check_error"],
            )

        return False, patches, f'{report["message"]} {_artifact_hint()}'
    
    def _patch_to_log(self, patch: Patch) -> Dict[str, Any]:
        if isinstance(patch, UnifiedDiffPatch):
            return {"type": "unified_diff", "diff_text": patch.diff_text}
        elif isinstance(patch, FileContentPatch):
            return {
                "type": "file_content",
                "path": str(patch.path.relative_to(self.repo.root).as_posix()),
                "new_content_snippet": patch.new_content[:1000],
            }
        else:
            return {"type": "unknown", "repr": str(patch)}

    # ---------------- Execution ----------------

    def execute(
        self,
        dag: DAG,
        *,
        run_name: str,
        dry_run: bool = False,
        branch_name: Optional[str] = None,
    ) -> List[NodeResult]:
        run_dir = self._init_run_dir(run_name)
        nodes_dir = run_dir / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)

        if branch_name:
            self.repo.ensure_branch(branch_name)

        snapshot = self._git_snapshot()
        ctx_builder = ContextPackBuilder(repo_root=self.repo.root)

        run_meta = {
            "run_name": run_name,
            "dry_run": dry_run,
            "branch": branch_name,
            "commit_policy": dag.commit_policy,
            "default_validators": list(dag.default_validators),
            "git": snapshot,
            "nodes": [
                {
                    "id": n.id,
                    "phase": n.phase_name,
                    "deps": n.deps,
                    "validators": n.validators,
                    "commit": n.commit,
                }
                for n in dag.nodes
            ],
        }
        self._write_json(run_dir / "run.json", run_meta)

        ordered = self._toposort(dag)
        results: List[NodeResult] = []
        all_ok = True

        last_failure_hints: Optional[Dict[str, Any]] = None

        for node in ordered:
            node_dir = nodes_dir / node.id
            node_dir.mkdir(parents=True, exist_ok=True)

            node_validators = node.validators if node.validators else list(dag.default_validators)
            ctx = PhaseContext(phase_name=node.phase_name)

            all_files = self.repo.list_files(
                include_globs=self.cfg.include_globs,
                exclude_globs=self.cfg.exclude_globs,
            )
            all_files = sorted(all_files, key=lambda p: p.relative_to(self.repo.root).as_posix())

            selected = node.phase.select_files(
                repo=self.repo,
                files=all_files,
                llm=self.llm,
                ctx=ctx,
                max_files=self.cfg.max_files_per_run,
            )

            self._write_json(
                node_dir / "selected_files.json",
                {"files": [p.relative_to(self.repo.root).as_posix() for p in selected]},
            )

            if not selected:
                nr = NodeResult(
                    node_id=node.id,
                    ok=True,
                    message="No files selected; node skipped.",
                    applied_patches_count=0,
                    validators_ok=True,
                    stopped_early=False,
                    validator_names=node_validators,
                    artifacts_dir=str(node_dir),
                )
                results.append(nr)
                continue

            context_pack = ctx_builder.build(
                selected_files=selected,
                node_id=node.id,
                phase_name=node.phase_name,
                run_dir=run_dir,
                node_dir=node_dir,
                failure_hints=last_failure_hints,
            )
            ctx.state["context_pack"] = context_pack
            ctx.state["node_objective"] = getattr(node, "objective", "")

            patches: List[Patch] = node.phase.generate_patches(
                repo=self.repo,
                files=selected,
                llm=self.llm,
                ctx=ctx,
            )

            try:
                self._write_json(
                    node_dir / "patches.json",
                    {"patches": [self._patch_to_log(p) for p in patches]},
                )
            except Exception:
                self._write_text(node_dir / "patches.txt", str(patches))

            if dry_run:
                nr = NodeResult(
                    node_id=node.id,
                    ok=True,
                    message="dry-run: patches generated; no apply/validate/commit.",
                    applied_patches_count=len(patches),
                    validators_ok=True,
                    stopped_early=False,
                    validator_names=node_validators,
                    artifacts_dir=str(node_dir),
                )
                results.append(nr)
                continue

            ok_apply, patches, msg_apply = self._apply_patches_with_repair(patches, node_dir)
            if not ok_apply:
                nr = NodeResult(
                    node_id=node.id,
                    ok=False,
                    message=msg_apply,
                    applied_patches_count=0,
                    validators_ok=False,
                    stopped_early=True,
                    validator_names=node_validators,
                    artifacts_dir=str(node_dir),
                )
                results.append(nr)
                all_ok = False
                last_failure_hints = {
                    "failed_stage": "apply_patches",
                    "message": msg_apply,
                }
                break

            pipeline_result = self._run_validators(node_validators)
            validator_json = ValidatorPipeline.to_json_dict(pipeline_result)

            self._write_json(node_dir / "validator_results.json", validator_json)

            self._write_latest_validator_results(
                {
                    "run": run_name,
                    "node_id": node.id,
                    "phase": node.phase_name,
                    "validators": node_validators,
                    **validator_json,
                }
            )

            if not pipeline_result.ok:
                nr = NodeResult(
                    node_id=node.id,
                    ok=False,
                    message="Validation failed.",
                    applied_patches_count=len(patches),
                    validators_ok=False,
                    stopped_early=pipeline_result.stopped_early,
                    validator_names=node_validators,
                    artifacts_dir=str(node_dir),
                )
                results.append(nr)
                all_ok = False

                failed = next((r for r in pipeline_result.results if not r.ok), None)
                last_failure_hints = {
                    "failed_stage": "validators",
                    "failed_validator": failed.name if failed else "",
                    "validator_stderr": failed.stderr if failed else "",
                    "validator_stdout": failed.stdout if failed else "",
                }
                break

            if dag.commit_policy == "per_node" and node.commit:
                self.repo.commit_all(f"[ai-orchestrator] {node.phase_name} (node {node.id})")

            nr = NodeResult(
                node_id=node.id,
                ok=True,
                message=msg_apply,
                applied_patches_count=len(patches),
                validators_ok=True,
                stopped_early=False,
                validator_names=node_validators,
                artifacts_dir=str(node_dir),
            )
            results.append(nr)
            last_failure_hints = None

        if not dry_run and dag.commit_policy == "end" and all_ok:
            self.repo.commit_all(f"[ai-orchestrator] DAG run {run_name}")

        return results
