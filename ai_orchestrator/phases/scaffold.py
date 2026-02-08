# ai_orchestrator/phases/scaffold.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
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
    """
    name = "scaffold"

    def _load_phase_rules(self, repo) -> Tuple[Set[str], Set[str]]:
        """
        Returns (create_only, modify_allow) as normalized repo-relative POSIX paths.
        Missing keys => empty sets (meaning "no allowlist specified").
        """
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
        """
        Validate and normalize a repo-relative path string.
        Reject absolute paths, drive letters, and parent traversal.
        """
        if not s:
            return None
        if "://" in s:
            return None
        p = Path(s)
        if p.is_absolute():
            return None
        # Windows drive-letter like "C:..."
        if len(p.parts) > 0 and p.parts[0].endswith(":"):
            return None
        if ".." in p.parts:
            return None
        return Path(p.as_posix()).as_posix()

    def generate_patches(
        self,
        repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
    ) -> List[Patch]:
        """
        Goal-driven scaffolding.

        - No hard-coded file names in pipeline.
        - Create/modify only files explicitly required by NODE OBJECTIVE and/or repo goal doc.
        - Enforce repo-declared allowlists for this phase via `.ai-orchestrator.json`:
            phase_rules.scaffold.create_only
            phase_rules.scaffold.modify_allow
        """
        import re

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

        # Mentioned paths (objective + goal). We never allow creation outside this set.
        mentioned = sorted(set(extract_paths(objective) + extract_paths(goal_excerpt)))

        # Phase allowlists from orchestrated repo config
        create_only, modify_allow = self._load_phase_rules(repo)

        # Effective allowlists:
        # - If create_only is non-empty: it becomes a hard gate for creations.
        # - If modify_allow is non-empty: it becomes a hard gate for modifications.
        create_gate = set(mentioned)
        if create_only:
            create_gate = create_gate & create_only

        modify_gate = set(mentioned)
        if modify_allow:
            modify_gate = modify_gate & modify_allow

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
            "- Output ONLY paths that are explicitly mentioned by path in NODE OBJECTIVE or GOAL EXCERPT.\n"
            "- Additionally, obey the PHASE ALLOWLISTS below.\n"
            "- Provide complete file contents for each file you output.\n"
            "- Keep content minimal.\n\n"
            f"NODE OBJECTIVE:\n{objective}\n\n"
            f"GOAL EXCERPT ({goal_rel}):\n{goal_excerpt}\n\n"
            "MENTIONED PATHS:\n" + _fmt_list(mentioned) + "\n\n"
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

        patches: List[Patch] = []
        for item in data.get("files", []):
            rp_raw = (item.get("path") or "").strip()
            rp = self._normalize_relpath(rp_raw)
            rp = self._safe_repo_relpath(rp) or ""
            content = item.get("content")

            if not rp or content is None:
                continue

            # Must be mentioned, always.
            if rp not in mentioned:
                continue

            p = repo.root / rp
            exists = p.exists()

            if exists:
                # Modification: allowed only if modify_gate allows it.
                if rp not in modify_gate:
                    continue
                patches.append(FileContentPatch(path=p, new_content=str(content)))
            else:
                # Creation: allowed only if create_gate allows it.
                if rp not in create_gate:
                    continue
                patches.append(FileContentPatch(path=p, new_content=str(content)))

        return patches
