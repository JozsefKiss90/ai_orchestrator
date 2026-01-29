# ai_orchestrator/graph/planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .types import DAG, Node
from ..phases.base import Phase


@dataclass(frozen=True)
class PlannedNodeSpec:
    id: str
    phase: str
    deps: List[str]
    validators: List[str]


@dataclass(frozen=True)
class PlannedDAGSpec:
    nodes: List[PlannedNodeSpec]
    commit_policy: str
    default_validators: List[str]


class DagPlanner:
    """
    Turns a structured DAG plan (JSON spec) into an executable DAG by mapping phase names
    to actual Phase instances.
    """

    def __init__(self, phase_registry: Dict[str, type[Phase]]):
        self.phase_registry = phase_registry

    def build(self, spec: PlannedDAGSpec) -> DAG:
        nodes: List[Node] = []
        for n in spec.nodes:
            if n.phase not in self.phase_registry:
                raise ValueError(f"Planner returned unknown phase: {n.phase}")

            phase_instance: Phase = self.phase_registry[n.phase]()  # type: ignore[call-arg]
            nodes.append(
                Node(
                    id=n.id,
                    phase_name=n.phase,
                    phase=phase_instance,
                    deps=list(n.deps or []),
                    validators=list(n.validators or []),
                    commit=True,
                )
            )

        return DAG(
            nodes=nodes,
            commit_policy=spec.commit_policy or "per_node",
            default_validators=list(spec.default_validators or ["tests"]),
        )
