# ai_orchestrator/graph/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..phases.base import Phase


@dataclass(frozen=True)
class Node:
    """
    A single unit of work in a DAG.

    - id: stable identifier in the DAG
    - phase_name: used for logging/metadata; runner uses `phase` for execution
    - phase: a Phase instance implementing select_files + generate_patches
    - deps: node IDs that must complete before this node runs
    - validators: list of validator names to run for this node (e.g., ["lint","tests"])
      If empty, runner uses DAG.default_validators.
    - commit: whether to commit after this node if validation passes (when commit_policy="per_node")
    """
    id: str
    phase_name: str
    phase: Phase
    deps: List[str] = field(default_factory=list)
    validators: List[str] = field(default_factory=list)
    commit: bool = True


@dataclass(frozen=True)
class DAG:
    """
    A directed acyclic graph of Nodes.

    - nodes: list of nodes
    - commit_policy:
        - "per_node": commit after each successful node (default; recommended for refactors)
        - "end": commit once at the end if all nodes succeed
    - default_validators: used when a node.validators is empty
    """
    nodes: List[Node]
    commit_policy: str = "per_node"
    default_validators: List[str] = field(default_factory=lambda: ["tests"])


@dataclass
class NodeResult:
    node_id: str
    ok: bool
    message: str
    applied_patches_count: int = 0
    validators_ok: bool = True
    stopped_early: bool = False
    validator_names: List[str] = field(default_factory=list)
    artifacts_dir: Optional[str] = None
