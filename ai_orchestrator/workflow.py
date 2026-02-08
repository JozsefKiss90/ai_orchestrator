# ai_orchestrator/workflow.py
from __future__ import annotations

from typing import Dict, Type, List

from .config import RepoConfig
from .llm import LLMClient
from .repo import Repo
from .phases.base import Phase
from .phases.oop_refactor import OopRefactorPhase
from .phases.plan import PlanPhase
from .graph.types import DAG, Node
from .graph.runner import DagRunner
from .graph.planner import DagPlanner, PlannedDAGSpec, PlannedNodeSpec
from .phases.scaffold import ScaffoldPhase
import logging

log = logging.getLogger(__name__)
PHASE_REGISTRY: Dict[str, Type[Phase]] = {
    ScaffoldPhase.name: ScaffoldPhase,
    OopRefactorPhase.name: OopRefactorPhase,
    PlanPhase.name: PlanPhase,  # invoked only with --use-plan
}



class WorkflowRunner:
    def __init__(self, cfg: RepoConfig):
        self.cfg = cfg
        self.repo = Repo(cfg.root)
        self.llm = LLMClient(cfg.llm)

    def _ensure_branch(self, branch_name: str, dry_run: bool) -> None:
        if dry_run:
            return
        self.repo.ensure_branch(branch_name)

    def _single_node_dag(self, phase_name: str) -> DAG:
        phase_cls = PHASE_REGISTRY[phase_name]
        phase_instance: Phase = phase_cls()  # type: ignore[call-arg]

        # Default validators: tests if configured; else run the DAG defaults anyway
        default_validators = ["tests"]
        
        return DAG(
            nodes=[
                Node(
                    id="node-1",
                    phase_name=phase_name,
                    phase=phase_instance,
                    deps=[],
                    validators=default_validators,  # preserves existing semantics: tests gate commit
                    commit=True,
                )
            ],
            commit_policy="per_node",
            default_validators=default_validators,
        )

    def _planned_dag(self, requested_phase: str) -> DAG:
        # Planner phase produces a structured spec
        plan_phase = PlanPhase()

        available_phases = [k for k in PHASE_REGISTRY.keys() if k != "plan"]
        available_validators = [v.name for v in self.cfg.validators]

        plan_json = plan_phase.generate_plan(
            llm=self.llm,
            ctx=None,  # PlanPhase doesn't depend on ctx for now
            available_phases=available_phases,
            requested_phase=requested_phase,
            available_validators=available_validators,
        )

        # --------------------------
        # HARDEN: sanitize + enforce
        # --------------------------
        known = set(available_validators)

        def _only_known(xs: object) -> List[str]:
            if not isinstance(xs, list):
                return []
            out: List[str] = []
            seen = set()
            for x in xs:
                if not isinstance(x, str):
                    continue
                if x not in known:
                    continue
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        # Naming convention: repo-provided goal contracts
        contract_validators = sorted([v for v in available_validators if isinstance(v, str) and v.startswith("goal_contract")])

        # Baseline: always run tests if present
        tests_only: List[str] = ["tests"] if "tests" in known else []
        refactor_validators: List[str] = tests_only + contract_validators

        commit_policy = plan_json.get("commit_policy", "per_node")
        nodes_json = plan_json.get("nodes", [])

        # Default validators: always refactor set (tests + contracts) for safety
        default_validators = _only_known(plan_json.get("default_validators", [])) or list(refactor_validators)

        spec_nodes: List[PlannedNodeSpec] = []
        for n in nodes_json:
            node_id = str(n.get("id", "") or "")
            phase = str(n.get("phase", "") or "")
            deps = list(n.get("deps", []) or [])
            obj = str(n.get("objective", "") or "")

            # Start with planner-provided validators, but sanitize.
            node_validators = _only_known(n.get("validators", []))

            # Enforce deterministic gating:
            # - scaffold node(s): tests only
            # - oop_refactor / verify nodes: tests + goal_contract*
            if phase == "scaffold":
                node_validators = list(tests_only)
            elif phase == "oop_refactor":
                node_validators = list(refactor_validators)
            else:
                # Any other phase: if planner omitted validators, fall back to default.
                if not node_validators:
                    node_validators = list(default_validators)

            spec_nodes.append(
                PlannedNodeSpec(
                    id=node_id,
                    phase=phase,
                    deps=deps,
                    validators=node_validators,
                    objective=obj,
                )
            )

        spec = PlannedDAGSpec(
            nodes=spec_nodes,
            commit_policy=commit_policy,
            default_validators=list(default_validators),
        )

        planner = DagPlanner(phase_registry=PHASE_REGISTRY)
        return planner.build(spec)


    def run_phase(self, phase_name: str, dry_run: bool = False, use_plan: bool = False) -> None:
        if use_plan:
            # Plan-first DAG
            log.info("Building DAG", extra={"fields": {"use_plan": use_plan, "phase": phase_name}})

            dag = self._planned_dag(requested_phase=phase_name)
            run_name = f"plan-{phase_name}"
        else:
            # Single-node DAG wrapping the requested phase
            if phase_name not in PHASE_REGISTRY:
                raise ValueError(f"Unknown phase: {phase_name}")
            dag = self._single_node_dag(phase_name)
            run_name = phase_name

        # Branch for safety (skip in dry-run)
        branch_name = f"ai/{run_name}"
        log.info("Ensuring branch", extra={"fields": {"branch": branch_name, "dry_run": dry_run}})

        self._ensure_branch(branch_name, dry_run=dry_run)

        runner = DagRunner(cfg=self.cfg, repo=self.repo, llm=self.llm)
        log.info("Executing DAG", extra={"fields": {"run_name": run_name, "commit_policy": dag.commit_policy, "dry_run": dry_run}})

        results = runner.execute(
            dag,
            run_name=run_name,
            dry_run=dry_run,
            branch_name=None if dry_run else branch_name,
        )

        # Preserve prior behavior: print a summary and exit
        ok = all(r.ok for r in results)
        log.info("DAG finished", extra={"fields": {"ok": ok, "run_name": run_name, "nodes": len(results)}})

        if ok:
            print("DAG run completed successfully.")
        else:
            print("DAG run failed. See per-node artifacts in .ai-orchestrator/runs/*/nodes/*")
