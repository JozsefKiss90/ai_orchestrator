# ai_orchestrator/graph/runner.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import RepoConfig
from ..llm import LLMClient, SCHEMA_REPAIR_UNIFIED_DIFF_V1
from ..patching import Patch, UnifiedDiffPatch
from ..repo import Repo, CommandResult
from ..validators.pipeline import ValidatorPipeline
from ..validators.types import PipelineResult, ValidatorSpec
from ..phases.base import PhaseContext
from .context import ContextPackBuilder
from .types import DAG, Node, NodeResult


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
        """
        Capture the exact repo state used for file selection and patch application.
        This makes selection reproducible and makes "why did context differ?" debuggable.
        """
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
        """
        Filter configured validators by name, preserving the `names` order.
        Unknown names raise.
        """
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

    # ---------------- Unified diff repair ----------------

    def _extract_paths_from_diff(self, diff_text: str) -> List[str]:
        paths: List[str] = []
        for line in diff_text.splitlines():
            if line.startswith("+++ "):
                token = line[4:].strip()
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

    def _repair_unified_diff(
        self,
        *,
        failing_diff: str,
        apply_error: str,
        check_error: str,
        file_snippets: Dict[str, str],
        attempt: int,
    ) -> str:
        system_prompt = (
            "You are an expert engineer. You repair unified diffs so they apply cleanly with `git apply`.\n"
            "Rules:\n"
            "- Output a corrected unified diff.\n"
            "- Keep edits minimal.\n"
            "- Do not introduce unrelated changes.\n"
            "- Ensure headers and file paths are correct.\n"
            "- CRITICAL: Never output a naked '@@' line. Every hunk header must be like '@@ -l,s +l,s @@'.\n"
            "- Include 'diff --git', '--- a/...', '+++ b/...' for every changed file.\n"
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

        data = self.llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SCHEMA_REPAIR_UNIFIED_DIFF_V1,
            schema_name="repair_unified_diff_v1",
        )
        return (data.get("diff") or "").strip()

    def _apply_patches_with_repair(
        self,
        patches: List[Patch],
        node_dir: Path,
    ) -> tuple[bool, List[Patch], str]:
        """
        Apply patches. If unified diff fails, attempt repair up to 2 times.
        UX semantics:
          - Only create patch_apply_error.txt if the FINAL outcome is failure.
          - If repaired, write patch_apply_error_initial.txt + patch_apply_report.json.
        Returns: (ok, possibly_updated_patches, message)
        """
        report: Dict[str, Any] = {
            "initial_ok": None,
            "repaired": False,
            "attempts": 0,
            "initial_apply_error": "",
            "initial_check_error": "",
            "final_apply_error": "",
            "final_check_error": "",
            "message": "",
        }

        # Initial apply (may fail for format or context reasons)
        res = self.repo.apply_patches(patches)
        report["initial_ok"] = bool(res.ok)
        if res.ok:
            report["message"] = "Applied patches successfully."
            self._write_json(node_dir / "patch_apply_report.json", report)
            return True, patches, report["message"]

        # Capture initial failure detail (but do NOT write patch_apply_error.txt yet)
        unified = next((p for p in patches if isinstance(p, UnifiedDiffPatch)), None)
        if unified is None:
            report["final_apply_error"] = (res.stderr or res.stdout or "").strip()
            report["message"] = "Patch application failed (non-unified-diff)."
            self._write_json(node_dir / "patch_apply_report.json", report)
            self._write_text(node_dir / "patch_apply_error.txt", report["final_apply_error"])
            return False, patches, report["message"]

        check_res = self.repo.check_unified_diff(unified.diff_text)
        report["initial_apply_error"] = (res.stderr or res.stdout or "").strip()
        report["initial_check_error"] = (check_res.stderr or check_res.stdout or "").strip()

        # Persist initial-only artifacts (these are not "final failure" signals)
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

        max_attempts = 2
        candidate = unified.diff_text

        for attempt in range(1, max_attempts + 1):
            report["attempts"] = attempt
            try:
                candidate = self._repair_unified_diff(
                    failing_diff=candidate,
                    apply_error=report["initial_apply_error"],
                    check_error=report["initial_check_error"],
                    file_snippets=snippets,
                    attempt=attempt,
                )
            except Exception as e:
                self._write_text(node_dir / f"patch_repair_llm_error_{attempt}.txt", str(e))

            self._write_json(node_dir / f"patch_repair_attempt_{attempt}.json", {"diff": candidate})

            # Apply repaired diff directly
            res2 = self.repo.apply_unified_diff(candidate, update_index=False)
            if res2.ok:
                report["repaired"] = True
                report["message"] = f"Patch repair succeeded on attempt {attempt}."
                self._write_json(node_dir / "patch_apply_report.json", report)
                # Return repaired patch list (single unified diff)
                return True, [UnifiedDiffPatch(diff_text=candidate)], report["message"]

            # Keep updating "final" details for debugging; still not final until attempts exhausted
            check_res2 = self.repo.check_unified_diff(candidate)
            report["final_apply_error"] = (res2.stderr or res2.stdout or "").strip()
            report["final_check_error"] = (check_res2.stderr or check_res2.stdout or "").strip()

        # FINAL failure: now write patch_apply_error.txt
        report["message"] = "Unified diff failed to apply even after repair attempts."
        self._write_json(node_dir / "patch_apply_report.json", report)
        self._write_text(node_dir / "patch_apply_error.txt", report["final_apply_error"] or report["initial_apply_error"])
        self._write_text(
            node_dir / "patch_apply_check_error.txt", report["final_check_error"] or report["initial_check_error"]
        )
        return False, patches, report["message"]

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

        # If a branch was specified, ensure we're on it BEFORE selecting files.
        if branch_name:
            self.repo.ensure_branch(branch_name)

        # Capture snapshot for reproducibility and debugging.
        snapshot = self._git_snapshot()

        # Context pack builder (Commit 15): deterministic, bounded, cached summaries
        ctx_builder = ContextPackBuilder(repo_root=self.repo.root)

        # Run metadata
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

        # Failure hints from previous node (feeds ContextPackBuilder heuristics deterministically)
        last_failure_hints: Optional[Dict[str, Any]] = None

        for node in ordered:
            node_dir = nodes_dir / node.id
            node_dir.mkdir(parents=True, exist_ok=True)

            node_validators = node.validators if node.validators else list(dag.default_validators)

            # Always pass a real PhaseContext
            ctx = PhaseContext(phase_name=node.phase_name)

            # File list is sorted deterministically for reproducible selection
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

            # Build bounded context pack and attach to ctx for the phase to consume
            context_pack = ctx_builder.build(
                selected_files=selected,
                node_id=node.id,
                phase_name=node.phase_name,
                run_dir=run_dir,
                node_dir=node_dir,
                failure_hints=last_failure_hints,
            )
            ctx.state["context_pack"] = context_pack

            # Generate patches
            patches: List[Patch] = node.phase.generate_patches(
                repo=self.repo,
                files=selected,
                llm=self.llm,
                ctx=ctx,
            )

            # Log patches (best-effort)
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

            # Apply patches (with repair)
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

            # Validate
            pipeline_result = self._run_validators(node_validators)
            validator_json = ValidatorPipeline.to_json_dict(pipeline_result)

            self._write_json(node_dir / "validator_results.json", validator_json)

            # Latest pointer for validator results
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

            # Commit policy
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

        # Commit-at-end
        if not dry_run and dag.commit_policy == "end" and all_ok:
            self.repo.commit_all(f"[ai-orchestrator] DAG run {run_name}")

        # Write summary
        self._write_json(
            run_dir / "summary.json",
            {
                "ok": all(r.ok for r in results),
                "results": [asdict(r) for r in results],
            },
        )

        return results

    def _patch_to_log(self, p: Patch) -> Dict[str, Any]:
        if isinstance(p, UnifiedDiffPatch):
            return {"kind": "unified_diff", "diff_preview": p.diff_text[:2000]}
        if hasattr(p, "path") and hasattr(p, "new_content"):
            rel = getattr(p, "path").relative_to(self.repo.root).as_posix()
            return {
                "kind": "file_content",
                "path": rel,
                "new_content_preview": getattr(p, "new_content")[:2000],
            }
        return {"kind": "unknown", "repr": repr(p)}
