# ai_orchestrator/phases/plan.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

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


@dataclass(frozen=True)
class GoalEdits:
    creates: List[Tuple[str, str]]  # (path, hint)
    updates: List[Tuple[str, str]]  # (path, instruction)


class PlanPhase(Phase):
    """
    Planner-only phase. It does not apply patches.

    It returns a JSON spec that graph/planner.py will map to actual Phase instances.

    Patch behavior:
      - Read docs/ORCHESTRATOR_GOAL.md (repo-local) and extract "Create <file>" + "Update <file> so ..."
      - Override LLM-proposed node objectives so they ALWAYS mention the correct repo-relative paths.
      - This ensures downstream select_files includes those paths (your select_files prioritizes paths in objective).
    """
    name = "plan"

    @staticmethod
    def _read_goal_text() -> str:
        goal_path = Path("docs") / "ORCHESTRATOR_GOAL.md"
        if not goal_path.exists():
            return ""
        return goal_path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _normalize_repo_relpath(p: str) -> str:
        p = (p or "").strip().strip("\"'`")
        p = p.replace("\\", "/")
        if p.startswith("./"):
            p = p[2:]
        return p

    @classmethod
    def _extract_goal_edits(cls, goal_text: str) -> GoalEdits:
        """
        Extract directives from ORCHESTRATOR_GOAL.md.

        Supported patterns (intentionally conservative):
          - Create ... `path/to/file.py`
          - **`path/to/file.py`** under "Create the following new files"
          - Update `path/to/file.py` so <instruction>
          - * Update `path/to/file.py` so <instruction>
        """
        creates: List[Tuple[str, str]] = []
        updates: List[Tuple[str, str]] = []

        if not goal_text.strip():
            return GoalEdits(creates=creates, updates=updates)

        lines = goal_text.splitlines()

        # A) "Update `file` so ..." bullet lines
        upd_re = re.compile(
            r"^\s*(?:[-*]\s*)?Update\s+`([^`]+)`\s+so\s+(.*)\s*$",
            re.IGNORECASE,
        )
        for ln in lines:
            m = upd_re.match(ln)
            if not m:
                continue
            path = cls._normalize_repo_relpath(m.group(1))
            instr = m.group(2).strip()
            if path:
                updates.append((path, instr))

        # B) "Create ..." lines mentioning backticked paths
        #    We extract any backticked path on a line containing "Create"
        create_line_re = re.compile(r"^\s*(?:[-*]\s*)?Create\b", re.IGNORECASE)
        backtick_path_re = re.compile(r"`([^`]+)`")
        for ln in lines:
            if not create_line_re.search(ln):
                continue
            for bt in backtick_path_re.findall(ln):
                path = cls._normalize_repo_relpath(bt)
                if path:
                    creates.append((path, "create"))

        # C) Under "Create the following new files:" sections, you often have enumerated items with **`path`**
        strong_backtick_re = re.compile(r"\*\*`([^`]+)`\*\*")
        for ln in lines:
            for bt in strong_backtick_re.findall(ln):
                path = cls._normalize_repo_relpath(bt)
                if path:
                    creates.append((path, "create"))

        # Dedup while preserving order
        def _dedup(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
            seen = set()
            out: List[Tuple[str, str]] = []
            for p, h in items:
                if p in seen:
                    continue
                seen.add(p)
                out.append((p, h))
            return out

        return GoalEdits(creates=_dedup(creates), updates=_dedup(updates))

    @staticmethod
    def _ensure_paths_in_objective(obj: str, paths: List[str]) -> str:
        obj = (obj or "").strip()
        for p in paths:
            if p and p not in obj:
                obj += ("\n" if obj else "") + f"- {p}"
        return obj

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
            "Use default validators unless additional safeguards are explicitly requested.\n"
            "Node objectives MUST include repo-relative file paths for any files to be created/edited.\n"
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
            "- Node 3 MUST be verification-only (validators only) and depend on Node 2.\n"
            "- Node 3 objective MUST include the phrase 'validators only'.\n"
            "- Every node objective MUST mention explicit repo-relative file paths to create/edit.\n"
            "- Deps must refer only to earlier node ids.\n\n"
            "Validation guidance:\n"
            "- Use default validators (e.g. tests) unless otherwise required.\n"
            "- Do NOT invent validators.\n"
            "- If unsure, include only tests.\n\n"
            "Return ONLY JSON per schema."
        )

        spec = llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SCHEMA_DAG_PLAN_V1,
            schema_name="dag_plan_v1",
        )

        # ---- Post-process objectives from the goal doc (project-agnostic) ----
        goal_text = self._read_goal_text()
        edits = self._extract_goal_edits(goal_text)

        create_paths = [p for p, _ in edits.creates]
        update_paths = [p for p, _ in edits.updates]
        all_paths = sorted({*create_paths, *update_paths})

        # Build deterministic objective strings if goal has directives.
        scaffold_obj = ""
        if create_paths:
            scaffold_obj = (
                "Create the following files as specified in docs/ORCHESTRATOR_GOAL.md:\n"
                + "\n".join(f"- {p}" for p in create_paths)
            )

        oop_obj = ""
        if edits.updates:
            lines = ["Implement the following goal-directed edits:"]
            for p, instr in edits.updates:
                if instr:
                    lines.append(f"- Edit {p} to {instr}")
                else:
                    lines.append(f"- Edit {p} according to docs/ORCHESTRATOR_GOAL.md")
            oop_obj = "\n".join(lines)

        verify_obj = ""
        if all_paths:
            verify_obj = (
                "validators only\n"
                "Validate goal contracts and behavior for these paths:\n"
                + "\n".join(f"- {p}" for p in all_paths)
            )
        else:
            verify_obj = "validators only"

        # Rewrite objectives per phase (do NOT alter deps/validators here).
        # If goal had no directives, keep LLM objectives.
        if goal_text.strip() and (create_paths or update_paths):
            nodes = spec.get("nodes") or []
            for n in nodes:
                phase = str(n.get("phase") or "")
                if phase == "scaffold" and scaffold_obj:
                    n["objective"] = scaffold_obj
                elif phase == "oop_refactor" and oop_obj and "validators only" not in str(n.get("objective") or "").lower():
                    n["objective"] = oop_obj
                elif "validators only" in str(n.get("objective") or "").lower():
                    n["objective"] = verify_obj

                # Safety: ensure referenced paths appear even if LLM output survives.
                # This forces downstream file selection to include the goal paths.
                if phase == "scaffold" and create_paths:
                    n["objective"] = self._ensure_paths_in_objective(str(n.get("objective") or ""), create_paths)
                if phase == "oop_refactor" and update_paths:
                    n["objective"] = self._ensure_paths_in_objective(str(n.get("objective") or ""), update_paths)

        return spec
