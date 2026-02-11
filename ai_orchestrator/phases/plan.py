# ai_orchestrator/phases/plan.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
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

    IMPORTANT:
    - The DAG "phase" values must be valid registered phase names.
    - Do NOT allow the model to invent phases like "noop".
    - We can still have a verification-only node by using an existing phase
      (e.g. oop_refactor) with objective containing "validators only",
      because oop_refactor.generate_patches() returns [] in that case.

    Responsibilities:
      - Read docs/ORCHESTRATOR_GOAL.md and extract Create/Update directives
      - Read docs/**/*.puml and extract PlantUML @file mappings
      - Build a deterministic 3-node plan:
          1) scaffold
          2) oop_refactor
          3) oop_refactor (validators only)
      - Ensure node objectives always mention explicit repo-relative paths
        so downstream file selection is forced to include those paths.
    """

    name = "plan"

    @staticmethod
    def _read_goal_text() -> str:
        goal_path = Path("docs") / "ORCHESTRATOR_GOAL.md"
        if not goal_path.exists():
            return ""
        return goal_path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _read_puml_texts() -> List[str]:
        docs_dir = Path("docs")
        if not docs_dir.exists():
            return []
        out: List[str] = []
        for p in docs_dir.rglob("*.puml"):
            if not p.is_file():
                continue
            out.append(p.read_text(encoding="utf-8", errors="ignore"))
        return out

    @staticmethod
    def _normalize_repo_relpath(p: str) -> str:
        p = (p or "").strip().strip("\"'`")
        p = p.replace("\\", "/")
        if p.startswith("./"):
            p = p[2:]
        return p

    @classmethod
    def _extract_plantuml_blocks(cls, markdown: str) -> List[str]:
        if not markdown:
            return []
        blocks: List[str] = []
        fence_re = re.compile(r"```plantuml\s*(.*?)```", re.IGNORECASE | re.DOTALL)
        for m in fence_re.finditer(markdown):
            blocks.append(m.group(1).strip())
        return blocks

    @classmethod
    def _extract_uml_file_mappings(cls, texts: List[str]) -> List[str]:
        """
        Extract @file repo-relative mappings from PlantUML text.

        Convention:
            @file <repo-relative-path>

        Returns normalized POSIX repo-relative paths (deduped, order-preserving).
        """
        file_re = re.compile(r"^\s*@file\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
        seen: Set[str] = set()
        out: List[str] = []
        for t in texts:
            if not t:
                continue
            for raw in file_re.findall(t):
                p = cls._normalize_repo_relpath(raw)
                if not p:
                    continue
                pp = Path(p)
                # reject absolute / traversal / drive-letter
                if pp.is_absolute():
                    continue
                if pp.parts and pp.parts[0].endswith(":"):
                    continue
                if ".." in pp.parts:
                    continue
                rp = pp.as_posix()
                if rp not in seen:
                    seen.add(rp)
                    out.append(rp)
        return out

    @classmethod
    def _extract_goal_edits(cls, goal_text: str) -> GoalEdits:
        creates: List[Tuple[str, str]] = []
        updates: List[Tuple[str, str]] = []

        if not goal_text.strip():
            return GoalEdits(creates=creates, updates=updates)

        lines = goal_text.splitlines()

        # Update `file` so ...
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
            if path and path.lower() != "@file":
                updates.append((path, instr))

        # Create ... with backticked paths
        create_line_re = re.compile(r"^\s*(?:[-*]\s*)?Create\b", re.IGNORECASE)
        backtick_path_re = re.compile(r"`([^`]+)`")
        for ln in lines:
            if not create_line_re.search(ln):
                continue
            for bt in backtick_path_re.findall(ln):
                path = cls._normalize_repo_relpath(bt)
                if not path or path.lower() == "@file":
                    continue
                if "<" in path or ">" in path:
                    continue
                if ("/" not in path and "\\" not in path) and not path.endswith((".py", ".puml", ".md", ".json")):
                    continue
                creates.append((path, "create"))

        # **`path`** pattern
        strong_backtick_re = re.compile(r"\*\*`([^`]+)`\*\*")
        for ln in lines:
            for bt in strong_backtick_re.findall(ln):
                path = cls._normalize_repo_relpath(bt)
                if not path or path.lower() == "@file":
                    continue
                if "<" in path or ">" in path:
                    continue
                if ("/" not in path and "\\" not in path) and not path.endswith((".py", ".puml", ".md", ".json")):
                    continue
                creates.append((path, "create"))

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
        # ---- Parse goal + UML ----
        goal_text = self._read_goal_text()
        edits = self._extract_goal_edits(goal_text)

        puml_texts = self._read_puml_texts()
        puml_texts.extend(self._extract_plantuml_blocks(goal_text))
        uml_paths = self._extract_uml_file_mappings(puml_texts)

        create_paths = [p for p, _ in edits.creates]
        update_paths = [p for p, _ in edits.updates]

        all_paths = sorted({*create_paths, *update_paths, *uml_paths})

        # ---- Build deterministic objectives ----
        scaffold_lines: List[str] = ["Create the following files:"]
        if create_paths:
            scaffold_lines.append("From docs/ORCHESTRATOR_GOAL.md:")
            scaffold_lines.extend([f"- {p}" for p in create_paths])
        if uml_paths:
            scaffold_lines.append("From PlantUML @file mappings (architecture):")
            scaffold_lines.extend([f"- {p}" for p in uml_paths])
        scaffold_obj = "\n".join(scaffold_lines)

        oop_lines: List[str] = ["Implement the following goal-directed edits:"]
        if edits.updates:
            for p, instr in edits.updates:
                if instr:
                    oop_lines.append(f"- Edit {p} to {instr}")
                else:
                    oop_lines.append(f"- Edit {p} according to docs/ORCHESTRATOR_GOAL.md")
        if uml_paths:
            oop_lines.append("Implement the PlantUML @file targets (architecture):")
            oop_lines.extend([f"- {p}" for p in uml_paths])
        oop_obj = "\n".join(oop_lines)

        verify_obj = "validators only"
        if all_paths:
            verify_obj += (
                "\nValidate goal contracts and behavior for these paths:\n"
                + "\n".join(f"- {p}" for p in all_paths)
            )

        # ---- Validators ----
        default_validators = ["tests"] if "tests" in (available_validators or []) else list(available_validators or [])
        if not default_validators:
            default_validators = ["tests"]

        # ---- Phases: must be registered ----
        # We *require* scaffold + oop_refactor to exist if you’re calling this.
        # Verification node uses oop_refactor with "validators only" objective so it generates no patches.
        nodes = [
            {
                "id": "scaffold_intro",
                "phase": "scaffold",
                "deps": [],
                "validators": [],
                "objective": scaffold_obj,
            },
            {
                "id": "oop_refactor_run",
                "phase": "oop_refactor",
                "deps": ["scaffold_intro"],
                "validators": [],
                "objective": oop_obj,
            },
            {
                "id": "verification_step",
                "phase": "oop_refactor",
                "deps": ["oop_refactor_run"],
                "validators": [],
                "objective": verify_obj,
            },
        ]

        # Hard safety: also force paths into objectives (selection relies on this).
        if create_paths:
            nodes[0]["objective"] = self._ensure_paths_in_objective(nodes[0]["objective"], create_paths)
        if uml_paths:
            nodes[0]["objective"] = self._ensure_paths_in_objective(nodes[0]["objective"], uml_paths)
            nodes[1]["objective"] = self._ensure_paths_in_objective(nodes[1]["objective"], uml_paths)
        if update_paths:
            nodes[1]["objective"] = self._ensure_paths_in_objective(nodes[1]["objective"], update_paths)
        if all_paths:
            nodes[2]["objective"] = self._ensure_paths_in_objective(nodes[2]["objective"], all_paths)

        spec = {
            "commit_policy": "per_node",
            "default_validators": default_validators,
            "nodes": nodes,
        }

        # Keep output shape compatible with the existing planner schema.
        return spec
