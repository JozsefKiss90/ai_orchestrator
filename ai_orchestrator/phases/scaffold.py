# ai_orchestrator/phases/scaffold.py
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

from .base import Phase, PhaseContext
from ..llm import LLMClient
from ..patching import Patch, FileContentPatch

SCHEMA_SCAFFOLD_V1: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["files", "notes"],
    "additionalProperties": False,
}


class ScaffoldPhase(Phase):
    """
    Pure scaffolding: create new files and minimal supporting edits if needed.

    Project-agnostic enforcement:
      - Reads repo-local `.ai-orchestrator.json` (the orchestrated repo contract).
      - Enforces `phase_rules.scaffold.create_only` and `phase_rules.scaffold.modify_allow` if present.
      - Treats PlantUML '@file <path>' mappings as normative file targets.
    """

    name = "scaffold"

    def _load_phase_rules(self, repo) -> Tuple[Set[str], Set[str]]:
        cfg_path = repo.root / ".ai-orchestrator.json"
        if not cfg_path.exists():
            return set(), set()

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8", errors="ignore") or "{}")
        except Exception:
            return set(), set()

        pr = cfg.get("phase_rules") or {}
        if not isinstance(pr, dict):
            return set(), set()

        sc = pr.get("scaffold") or {}
        if not isinstance(sc, dict):
            return set(), set()

        def _norm_list(v: Any) -> Set[str]:
            if not isinstance(v, list):
                return set()
            out: Set[str] = set()
            for x in v:
                if not isinstance(x, str):
                    continue
                rp = self._normalize_relpath(x)
                if rp and self._safe_repo_relpath(rp) is not None:
                    out.add(rp)
            return out

        create_only = _norm_list(sc.get("create_only"))
        modify_allow = _norm_list(sc.get("modify_allow"))
        return create_only, modify_allow

    @staticmethod
    def _normalize_relpath(s: str) -> str:
        s = (s or "").strip().strip("\"'`")
        s = s.replace("\\", "/")
        s = s.rstrip(").,;:")
        if s.startswith("./"):
            s = s[2:]
        return s

    @staticmethod
    def _safe_repo_relpath(s: str) -> Optional[str]:
        if not s:
            return None
        if "://" in s:
            return None
        p = Path(s)
        if p.is_absolute():
            return None
        if len(p.parts) > 0 and p.parts[0].endswith(":"):
            return None
        if ".." in p.parts:
            return None
        return Path(p.as_posix()).as_posix()

    @staticmethod
    def _parse_puml_file_map(puml_text: str) -> Dict[str, List[str]]:
        """
        Extract mapping: repo_rel_path -> list of class/interface names.
        Convention:
          class Foo { ... }
          note top of Foo
            @file path/to/file.py
          end note
        """
        mapping: DefaultDict[str, List[str]] = defaultdict(list)
        in_note_for_symbol: Optional[str] = None

        for raw in (puml_text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("'"):
                continue

            m_note = re.match(r"^note\s+top\s+of\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if m_note:
                in_note_for_symbol = m_note.group(1)
                continue

            if line.lower() == "end note":
                in_note_for_symbol = None
                continue

            if in_note_for_symbol:
                m_file = re.search(r"@file\s+(.+)$", line)
                if m_file:
                    rp = ScaffoldPhase._normalize_relpath(m_file.group(1))
                    rp = ScaffoldPhase._safe_repo_relpath(rp) or ""
                    if rp:
                        mapping[rp].append(in_note_for_symbol)

        return dict(mapping)

    @staticmethod
    def _skeleton_for(path_rel: str, symbols: List[str]) -> str:
        header = [
            '"""',
            f"Auto-generated scaffold for: {path_rel}",
            "Generated from PlantUML @file mappings.",
            '"""',
            "",
            "from __future__ import annotations",
            "",
        ]
        body: List[str] = []
        for sym in symbols:
            body.extend(
                [
                    f"class {sym}:",
                    "    def __init__(self, *args, **kwargs):",
                    "        pass",
                    "",
                ]
            )
        if not body:
            body = ["# TODO: implement", ""]
        return "\n".join(header + body)

    def generate_patches(
        self,
        repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
    ) -> List[Patch]:
        objective = str(ctx.state.get("node_objective") or "").strip()

        goal_rel = "docs/ORCHESTRATOR_GOAL.md"
        goal_path = repo.root / goal_rel
        goal_excerpt = ""
        if goal_path.exists():
            goal_excerpt = goal_path.read_text(encoding="utf-8", errors="ignore")[:6000]

        def extract_paths(text: str) -> List[str]:
            if not text:
                return []
            candidates = re.findall(r"(?<!\w)([A-Za-z0-9_.\-]+(?:[\\/][A-Za-z0-9_.\-]+)+)(?!\w)", text)
            out: List[str] = []
            seen = set()
            for c in candidates:
                rp = self._normalize_relpath(c)
                rp = self._safe_repo_relpath(rp) or ""
                if not rp:
                    continue
                if rp not in seen:
                    seen.add(rp)
                    out.append(rp)
            return out

        # 1) parse UML @file mappings from selected .puml files
        uml_map: Dict[str, List[str]] = {}
        for p in files:
            if p.suffix.lower() != ".puml":
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for rp, syms in self._parse_puml_file_map(txt).items():
                uml_map.setdefault(rp, [])
                for s in syms:
                    if s not in uml_map[rp]:
                        uml_map[rp].append(s)

        uml_paths = set(uml_map.keys())

        # 2) Mentioned paths (objective + goal excerpt + UML @file targets)
        mentioned = set(extract_paths(objective) + extract_paths(goal_excerpt))
        mentioned |= uml_paths
        mentioned_list = sorted(mentioned)

        # 3) Phase allowlists from orchestrated repo config
        create_only, modify_allow = self._load_phase_rules(repo)

        create_gate = set(mentioned)
        if create_only:
            create_gate &= create_only

        modify_gate = set(mentioned)
        if modify_allow:
            modify_gate &= modify_allow

        # 4) Deterministic pre-pass: create missing files from UML @file
        deterministic_patches: List[Patch] = []
        for rp in sorted(uml_paths):
            if rp not in create_gate:
                continue
            abs_path = repo.root / rp
            if abs_path.exists():
                continue
            content = self._skeleton_for(rp, uml_map.get(rp, []))
            deterministic_patches.append(FileContentPatch(path=abs_path, new_content=content))

        # If we only need to create files, skip LLM entirely.
        if deterministic_patches and not modify_gate:
            return deterministic_patches

        # 5) Optional LLM scaffolding for allowed modifications (and any additional creations that are explicitly mentioned)
        system_prompt = (
            "You are a scaffolding tool.\n"
            "Return ONLY JSON.\n"
            "Do not invent files or directories.\n"
        )

        def _fmt_list(items: List[str]) -> str:
            return "\n".join(f"- {p}" for p in items) if items else "(none)"

        user_prompt = (
            "Create/modify files required by the repo goal and the node objective.\n\n"
            "Hard rules:\n"
            "- Output ONLY paths in MENTIONED PATHS (includes UML @file targets).\n"
            "- Additionally, obey the PHASE ALLOWLISTS below.\n"
            "- Provide complete file contents for each file you output.\n"
            "- Keep content minimal.\n\n"
            f"NODE OBJECTIVE:\n{objective}\n\n"
            f"GOAL EXCERPT ({goal_rel}):\n{goal_excerpt}\n\n"
            "MENTIONED PATHS:\n" + _fmt_list(mentioned_list) + "\n\n"
            "UML @file TARGETS:\n" + _fmt_list(sorted(uml_paths)) + "\n\n"
            "PHASE ALLOWLISTS (from .ai-orchestrator.json):\n"
            f"- scaffold.create_only (if present):\n{_fmt_list(sorted(create_only))}\n\n"
            f"- scaffold.modify_allow (if present):\n{_fmt_list(sorted(modify_allow))}\n\n"
            "EFFECTIVE GATES THIS RUN:\n"
            f"- allowed creations:\n{_fmt_list(sorted(create_gate))}\n\n"
            f"- allowed modifications:\n{_fmt_list(sorted(modify_gate))}\n\n"
            "Return JSON with:\n"
            "{ files: [ {path, content} ], notes: string }\n"
        )

        data = llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SCHEMA_SCAFFOLD_V1,
            schema_name="scaffold_v1",
        )

        patches: List[Patch] = list(deterministic_patches)

        for item in data.get("files", []):
            rp_raw = (item.get("path") or "").strip()
            rp = self._normalize_relpath(rp_raw)
            rp = self._safe_repo_relpath(rp) or ""
            content = item.get("content")

            if not rp or content is None:
                continue
            if rp not in mentioned:
                continue

            p = repo.root / rp
            exists = p.exists()

            if exists:
                if rp not in modify_gate:
                    continue
                patches.append(FileContentPatch(path=p, new_content=str(content)))
            else:
                if rp not in create_gate:
                    continue
                patches.append(FileContentPatch(path=p, new_content=str(content)))

        return patches
