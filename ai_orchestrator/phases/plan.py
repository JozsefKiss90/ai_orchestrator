# ai_orchestrator/phases/plan.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import Phase, PhaseContext
from ..llm import LLMClient


SCHEMA_DAG_PLAN_V1: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "commit_policy": {"type": "string", "enum": ["per_node", "end"]},
        "default_validators": {"type": "array", "items": {"type": "string"}},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "phase": {"type": "string"},
                    "deps": {"type": "array", "items": {"type": "string"}},
                    "validators": {"type": "array", "items": {"type": "string"}},
                    "objective": {"type": "string"},
                },
                "required": ["id", "phase", "deps", "validators", "objective"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["commit_policy", "default_validators", "nodes"],
    "additionalProperties": False,
}


class PlanPhase(Phase):
    """
    Planner-only phase. It does not apply patches.

    It returns a JSON spec that graph/planner.py will map to actual Phase instances.
    """
    name = "plan"

    def generate_plan(
        self,
        *,
        llm: LLMClient,
        ctx: PhaseContext,
        available_phases: List[str],
        requested_phase: str,
        available_validators: List[str],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a software delivery planner. You output a small DAG plan of steps.\n"
            "You must only use available phase names and available validators.\n"
            "Prefer small steps and deterministic validation.\n"
        )

        user_prompt = (
            f"Create a DAG plan for running the orchestrator.\n\n"
            f"Requested target phase: {requested_phase}\n"
            f"Available phases: {available_phases}\n"
            f"Available validators: {available_validators}\n\n"
            "Hard requirements:\n"
            "- Output EXACTLY 3 nodes.\n"
            "- commit_policy MUST be 'per_node'.\n"
            "- Node 1 MUST be phase 'scaffold'.\n"
            "- Node 2 MUST be phase 'oop_refactor' and depend on Node 1.\n"
            "- Node 3 MUST be phase 'oop_refactor' OR 'scaffold' is NOT allowed; it MUST be any phase that makes sense "
            "for verification, but since phases are limited, use phase 'oop_refactor' with no patches and validators only.\n"
            "- Every node MUST include validators ['tests'] if 'tests' is available.\n"
            "- Deps must refer only to earlier node ids.\n\n"
            "Return ONLY JSON per schema."
        )

        return llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SCHEMA_DAG_PLAN_V1,
            schema_name="dag_plan_v1",
        )

    # Not used for patch generation
    def generate_patches(self, repo, files, llm, ctx):
        return []
