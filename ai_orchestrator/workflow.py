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

        # Default validators: tests if configured; else empty.
        default_validators = ["tests"] if any(v.name == "tests" for v in self.cfg.validators) else []

        return DAG(
            nodes=[
                Node(
                    id="node-1",
                    phase_name=phase_name,
                    phase=phase_instance,
                    deps=[],
                    validators=default_validators,
                    commit=True,
                )
            ],
            commit_policy="per_node",
            default_validators=default_validators,
        )

    def _planned_dag(self, requested_phase: str) -> DAG:
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

        contract_validators = sorted(
            [v for v in available_validators if isinstance(v, str) and v.startswith("goal_contract")]
        )

        tests_only: List[str] = ["tests"] if "tests" in known else []
        refactor_validators: List[str] = tests_only + contract_validators

        commit_policy = str(plan_json.get("commit_policy", "per_node") or "per_node")
        nodes_json = plan_json.get("nodes", [])

        # IMPORTANT: default validators must NOT be injected into explicitly planned nodes,
        # otherwise validator gating leaks into phase-2 (oop_refactor_run).
        # We keep defaults minimal and only apply to phases not explicitly handled below.
        default_validators = _only_known(plan_json.get("default_validators", [])) or list(tests_only)

        spec_nodes: List[PlannedNodeSpec] = []
        for n in nodes_json:
            node_id = str(n.get("id", "") or "")
            phase = str(n.get("phase", "") or "")
            deps = list(n.get("deps", []) or [])
            obj = str(n.get("objective", "") or "")
            obj_l = obj.lower()

            # Start with planner-provided validators, but sanitize.
            node_validators = _only_known(n.get("validators", []))

            # Enforce intended gating:
            # - scaffold: tests only
            # - oop_refactor: NO validators unless explicitly "validators only" node
            if phase == "scaffold":
                node_validators = list(tests_only)

            elif phase == "oop_refactor":
                if "validators only" in obj_l or "run validators only" in obj_l or "tests-only" in obj_l:
                    node_validators = list(refactor_validators)
                else:
                    node_validators = []  # phase-2 commit happens without validators

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
            log.info("Building DAG", extra={"fields": {"use_plan": use_plan, "phase": phase_name}})
            dag = self._planned_dag(requested_phase=phase_name)
            run_name = f"plan-{phase_name}"
        else:
            if phase_name not in PHASE_REGISTRY:
                raise ValueError(f"Unknown phase: {phase_name}")
            dag = self._single_node_dag(phase_name)
            run_name = phase_name

        branch_name = f"ai/{run_name}"
        log.info("Ensuring branch", extra={"fields": {"branch": branch_name, "dry_run": dry_run}})
        self._ensure_branch(branch_name, dry_run=dry_run)

        runner = DagRunner(cfg=self.cfg, repo=self.repo, llm=self.llm)
        log.info(
            "Executing DAG",
            extra={"fields": {"run_name": run_name, "commit_policy": dag.commit_policy, "dry_run": dry_run}},
        )

        results = runner.execute(
            dag,
            run_name=run_name,
            dry_run=dry_run,
            branch_name=None if dry_run else branch_name,
        )

        ok = all(r.ok for r in results)
        log.info("DAG finished", extra={"fields": {"ok": ok, "run_name": run_name, "nodes": len(results)}})

        if ok:
            print("DAG run completed successfully.")
        else:
            print("DAG run failed. See per-node artifacts in .ai-orchestrator/runs/*/nodes/*")
