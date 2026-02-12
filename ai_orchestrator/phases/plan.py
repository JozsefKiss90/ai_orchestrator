from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
import json
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

    Responsibilities:
      - Read docs/ORCHESTRATOR_GOAL.md and extract Create/Update directives
      - Read docs/**/*.puml and extract PlantUML @file mappings
      - Read .ai-orchestrator.json policy contracts and inject them into the OOP objective
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
            if p.is_file():
                out.append(p.read_text(encoding="utf-8", errors="ignore"))
        return out

    @staticmethod
    def _normalize_repo_relpath(p: str) -> str:
        p = (p or "").strip().strip("\"'`")
        p = p.replace("\\", "/")
        p = p.rstrip(").,;:")
        if p.startswith("./"):
            p = p[2:]
        return p

    @staticmethod
    def _is_safe_repo_relpath(p: str) -> bool:
        if not p:
            return False
        if "://" in p:
            return False
        pp = Path(p)
        if pp.is_absolute():
            return False
        if pp.parts and pp.parts[0].endswith(":"):
            return False
        if ".." in pp.parts or "." in pp.parts:
            return False
        return True

    @staticmethod
    def _looks_like_file_path(p: str) -> bool:
        """
        Prevent garbage tokens like '@file' or plain identifiers from being treated as paths.

        Heuristic:
          - must contain at least one slash (repo-rel) OR be under docs/ explicitly
          - must end with a plausible extension ('.py', '.md', '.json', '.puml', etc.)
        """
        if not p or not isinstance(p, str):
            return False
        p = p.replace("\\", "/").strip()
        if p.lower() == "@file":
            return False
        if "/" not in p:
            return False
        # Require an extension; this keeps us from selecting tokens like 'demo-hello'
        if "." not in Path(p).name:
            return False
        return True

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
                if not cls._is_safe_repo_relpath(p):
                    continue
                rp = Path(p).as_posix()
                if not cls._looks_like_file_path(rp):
                    continue
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
            if path and cls._is_safe_repo_relpath(path):
                rp = Path(path).as_posix()
                if cls._looks_like_file_path(rp):
                    updates.append((rp, instr))

        # Create ... with backticked paths (goal may list created files)
        create_line_re = re.compile(r"^\s*(?:[-*]\s*)?Create\b", re.IGNORECASE)
        backtick_path_re = re.compile(r"`([^`]+)`")
        for ln in lines:
            if not create_line_re.search(ln):
                continue
            for bt in backtick_path_re.findall(ln):
                path = cls._normalize_repo_relpath(bt)
                if not path:
                    continue
                if "<" in path or ">" in path:
                    continue
                if not cls._is_safe_repo_relpath(path):
                    continue
                rp = Path(path).as_posix()
                if not cls._looks_like_file_path(rp):
                    continue
                creates.append((rp, "create"))

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
    def _read_policy_contracts() -> Dict[str, Dict[str, List[str]]]:
        """
        Load assert_text_contract policy contracts from .ai-orchestrator.json.

        Returns mapping:
          path -> { "require_all_regex": [...], "require_any_regex": [...], "forbid_any_regex": [...] }
        """
        cfg_path = Path(".ai-orchestrator.json")
        if not cfg_path.exists():
            return {}

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8", errors="ignore") or "{}")
        except Exception:
            return {}

        validators = cfg.get("validators")
        if not isinstance(validators, list):
            return {}

        out: Dict[str, Dict[str, List[str]]] = {}
        for v in validators:
            if not isinstance(v, dict):
                continue
            if v.get("kind") != "policy":
                continue
            if v.get("policy") != "assert_text_contract":
                continue
            params = v.get("params") or {}
            if not isinstance(params, dict):
                continue
            contracts = params.get("contracts")
            if not isinstance(contracts, list):
                continue

            for c in contracts:
                if not isinstance(c, dict):
                    continue
                path = c.get("path")
                if not isinstance(path, str) or not path.strip():
                    continue

                def _lst(key: str) -> List[str]:
                    val = c.get(key)
                    if not isinstance(val, list):
                        return []
                    return [x for x in val if isinstance(x, str) and x.strip()]

                rp = path.replace("\\", "/").strip()
                out[rp] = {
                    "require_all_regex": _lst("require_all_regex"),
                    "require_any_regex": _lst("require_any_regex"),
                    "forbid_any_regex": _lst("forbid_any_regex"),
                }

        return out

    @staticmethod
    def _ensure_paths_in_objective(obj: str, paths: List[str]) -> str:
        obj = (obj or "").strip()
        for p in paths:
            if p and p not in obj:
                obj += ("\n" if obj else "") + f"- {p}"
        return obj

    @staticmethod
    def _contract_checklist_human(
        contract_map: Dict[str, Dict[str, List[str]]],
        *,
        only_paths: List[str] | None = None,
    ) -> str:
        """
        Convert regex contracts into an explicit, human checklist.

        Keep it literal: name drift is the main failure mode.
        """
        if not contract_map:
            return ""

        only = set(only_paths or [])
        lines: List[str] = []
        lines.append("REQUIRED PUBLIC API (MUST MATCH EXACT NAMES; validator-enforced):")

        def add(path: str, item: str) -> None:
            lines.append(f"- {path}: {item}")

        for path in sorted(contract_map.keys()):
            if only and path not in only:
                continue
            hints = contract_map.get(path) or {}
            pats = list(hints.get("require_all_regex") or [])

            have_any = False
            for pat in pats:
                if r"class\s+" in pat:
                    m = re.search(r"class\\s\+([A-Za-z_][A-Za-z0-9_]*)", pat)
                    if m:
                        add(path, f"class `{m.group(1)}`")
                        have_any = True
                if r"\bdef\s+" in pat or "def\\s+" in pat:
                    m = re.search(r"def\\s\+([A-Za-z_][A-Za-z0-9_]*)", pat)
                    if m:
                        add(path, f"method `def {m.group(1)}(...)`")
                        have_any = True
                m = re.fullmatch(r"\\b([A-Za-z_][A-Za-z0-9_]*)\\b", pat.strip())
                if m:
                    add(path, f"must reference token `{m.group(1)}` (import/use/field)")
                    have_any = True
                if "selectedFiles" in pat or "selected_files" in pat:
                    add(path, "field/attribute `selectedFiles` OR `selected_files`")
                    have_any = True

            if have_any:
                continue

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def generate_plan(
        self,
        *,
        llm: LLMClient,
        ctx: PhaseContext,
        available_phases: List[str],
        requested_phase: str,
        available_validators: List[str],
    ) -> Dict[str, Any]:
        goal_text = self._read_goal_text()
        edits = self._extract_goal_edits(goal_text)

        puml_texts = self._read_puml_texts()
        puml_texts.extend(self._extract_plantuml_blocks(goal_text))
        uml_paths = self._extract_uml_file_mappings(puml_texts)

        create_paths = [p for p, _ in edits.creates]
        update_paths = [p for p, _ in edits.updates]

        all_paths = sorted({*create_paths, *update_paths, *uml_paths})

        contract_map = self._read_policy_contracts()
        checklist = self._contract_checklist_human(contract_map, only_paths=all_paths or None)

        scaffold_lines: List[str] = ["Create the following files (skeletons OK, must be valid Python):"]
        if create_paths:
            scaffold_lines.append("From docs/ORCHESTRATOR_GOAL.md:")
            scaffold_lines.extend([f"- {p}" for p in create_paths])
        if uml_paths:
            scaffold_lines.append("From PlantUML @file mappings (architecture):")
            scaffold_lines.extend([f"- {p}" for p in uml_paths])
        scaffold_obj = "\n".join(scaffold_lines)

        oop_lines: List[str] = [
            "Implement the UML-defined classes with MINIMAL coherent behavior.",
            "CRITICAL: DO NOT rename required public methods/fields. Use the exact names below.",
        ]
        if checklist:
            oop_lines.append("")
            oop_lines.append(checklist)

        if edits.updates:
            oop_lines.append("")
            oop_lines.append("Goal-directed edits:")
            for p, instr in edits.updates:
                oop_lines.append(f"- Edit {p}: {instr or 'follow docs/ORCHESTRATOR_GOAL.md'}")

        if uml_paths:
            oop_lines.append("")
            oop_lines.append("Implement the PlantUML @file targets (architecture):")
            oop_lines.extend([f"- {p}" for p in uml_paths])

        oop_obj = "\n".join(oop_lines)

        verify_lines = ["validators only", "Re-run the same contract validators to confirm deterministic compliance."]
        if all_paths:
            verify_lines.append("Validate these paths:")
            verify_lines.extend([f"- {p}" for p in all_paths])
        if checklist:
            verify_lines.append("")
            verify_lines.append(checklist)
        verify_obj = "\n".join(verify_lines)

        default_validators = list(available_validators or [])
        if not default_validators:
            default_validators = ["tests"]

        nodes = [
            {"id": "scaffold_intro", "phase": "scaffold", "deps": [], "validators": [], "objective": scaffold_obj},
            {"id": "oop_refactor_run", "phase": "oop_refactor", "deps": ["scaffold_intro"], "validators": [], "objective": oop_obj},
            {"id": "verification_step", "phase": "oop_refactor", "deps": ["oop_refactor_run"], "validators": [], "objective": verify_obj},
        ]

        if create_paths:
            nodes[0]["objective"] = self._ensure_paths_in_objective(nodes[0]["objective"], create_paths)
        if uml_paths:
            nodes[0]["objective"] = self._ensure_paths_in_objective(nodes[0]["objective"], uml_paths)
            nodes[1]["objective"] = self._ensure_paths_in_objective(nodes[1]["objective"], uml_paths)
        if update_paths:
            nodes[1]["objective"] = self._ensure_paths_in_objective(nodes[1]["objective"], update_paths)
        if all_paths:
            nodes[2]["objective"] = self._ensure_paths_in_objective(nodes[2]["objective"], all_paths)

        return {
            "commit_policy": "per_node",
            "default_validators": default_validators,
            "nodes": nodes,
        }
