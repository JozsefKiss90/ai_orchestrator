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

        commit_policy = plan_json.get("commit_policy", "per_node")
        default_validators = plan_json.get("default_validators", ["tests"])
        nodes_json = plan_json.get("nodes", [])

        spec_nodes: List[PlannedNodeSpec] = []
        for n in nodes_json:
            spec_nodes.append(
                PlannedNodeSpec(
                    id=n["id"],
                    phase=n["phase"],
                    deps=list(n.get("deps", [])),
                    validators=list(n.get("validators", [])),
                    objective=str(n.get("objective", "") or ""),
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
        self._ensure_branch(branch_name, dry_run=dry_run)

        runner = DagRunner(cfg=self.cfg, repo=self.repo, llm=self.llm)
        results = runner.execute(
            dag,
            run_name=run_name,
            dry_run=dry_run,
            branch_name=None if dry_run else branch_name,
        )

        # Preserve prior behavior: print a summary and exit
        ok = all(r.ok for r in results)
        if ok:
            print("DAG run completed successfully.")
        else:
            print("DAG run failed. See per-node artifacts in .ai-orchestrator/runs/*/nodes/*")
