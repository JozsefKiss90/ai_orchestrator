# ai_orchestrator/phases/oop_refactor.py
from __future__ import annotations

from pathlib import Path
from typing import List, Set

from .base import Phase, PhaseContext
from ..llm import LLMClient, SCHEMA_FILE_CONTENT_V1
from ..patching import Patch, FileContentPatch


class OopRefactorPhase(Phase):
    name = "oop_refactor"

    def generate_patches(
        self,
        repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
    ) -> List[Patch]:
        """
        Robust mode: return full file contents only.

        HARD SAFETY:
          - Only emit patches for:
              (a) repo-relative paths that are in SELECTED FILES, OR
              (b) repo-relative paths explicitly mentioned in docs/ORCHESTRATOR_GOAL.md
          - This prevents planner/objective hallucinations from creating new files.
        """
        system_prompt = (
            "You are a senior software engineer.\n"
            "Return ONLY JSON matching the schema.\n"
            "Do not include markdown fences.\n"
        )

        objective_raw = str(ctx.state.get("node_objective") or "")
        objective = objective_raw.strip().lower()

        # Verification-only node: no patches
        if "validators only" in objective or "run validators only" in objective or "tests-only" in objective:
            return []

        # ---- Allowlist computation ----
        selected_rel_paths: List[str] = [p.relative_to(repo.root).as_posix() for p in files]
        selected_set: Set[str] = set(selected_rel_paths)

        goal_allow: Set[str] = set()
        goal_rel = "docs/ORCHESTRATOR_GOAL.md"
        goal_path = repo.root / goal_rel
        if goal_path.exists():
            import re

            txt = goal_path.read_text(encoding="utf-8", errors="ignore")
            # conservative path extractor: tokens containing at least one "/" or "\".
            candidates = re.findall(r"(?<!\w)([A-Za-z0-9_.\-]+(?:[\\/][A-Za-z0-9_.\-]+)+)(?!\w)", txt)
            for c in candidates:
                rp = c.replace("\\", "/").strip().strip("\"'`").rstrip(").,;:")
                if rp.startswith("./"):
                    rp = rp[2:]
                # reject traversal/absolute
                p = Path(rp)
                if p.is_absolute() or ".." in p.parts or (p.parts and p.parts[0].endswith(":")):
                    continue
                goal_allow.add(p.as_posix())

        allowed_paths: Set[str] = set(selected_set) | set(goal_allow)

        # ---- Build context blobs (as before) ----
        context_pack = ctx.state.get("context_pack")
        constraints_blob = ""
        module_summaries_blob = ""

        if isinstance(context_pack, dict):
            constraints = context_pack.get("constraints", {}) or {}
            if constraints:
                constraints_blob = "CONSTRAINTS / ARCHITECTURE DOCS (excerpts):\n" + "\n\n".join(
                    f"## {k}\n{v}" for k, v in constraints.items() if isinstance(v, str) and v.strip()
                )
            ms = context_pack.get("module_summaries", []) or []
            if ms:
                module_summaries_blob = "MODULE SUMMARIES:\n" + "\n\n".join(
                    f"## Module: {m.get('module')}\n{m.get('summary','')}"
                    for m in ms
                    if isinstance(m, dict)
                )

        file_blobs = []
        for path in files:
            rel = path.relative_to(repo.root).as_posix()
            content = path.read_text(encoding="utf-8", errors="ignore")
            file_blobs.append(f"// FILE: {rel}\n```code\n{content}\n```")

        user_prompt = (
            "TASK:\n"
            "- Apply OOP refactoring if required by the goal.\n"
            "- Implement the current NODE OBJECTIVE precisely.\n\n"
            "OUTPUT FORMAT:\n"
            'Return JSON: { "files": [ {"path": string, "new_content": string}, ... ] }\n\n'
            "Rules:\n"
            "- You may edit ONLY files listed in SELECTED FILES.\n"
            "- You may create/modify additional files ONLY if they are explicitly mentioned in docs/ORCHESTRATOR_GOAL.md.\n"
            "- For any file you edit, return its COMPLETE new content.\n"
            "- Paths must be repo-relative (use forward slashes).\n\n"
            "HARD ALLOWLIST (do not output paths outside this set):\n"
            + "\n".join(f"- {p}" for p in sorted(allowed_paths))
            + "\n\n"
        )

        if objective_raw.strip():
            user_prompt += f"NODE OBJECTIVE:\n{objective_raw}\n\n"
        if constraints_blob:
            user_prompt += constraints_blob + "\n\n"
        if module_summaries_blob:
            user_prompt += module_summaries_blob + "\n\n"

        user_prompt += "SELECTED FILES:\n" + "\n".join(f"- {p}" for p in selected_rel_paths) + "\n\n"
        user_prompt += "FILE CONTENTS:\n" + "\n\n".join(file_blobs)

        data = llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SCHEMA_FILE_CONTENT_V1,
            schema_name="file_content_v1",
        )

        patches: List[Patch] = []
        for item in data.get("files", []):
            rp = (item.get("path") or "").strip().replace("\\", "/")
            new_content = item.get("new_content")
            if not rp or new_content is None:
                continue

            # HARD BLOCK hallucinations
            if rp not in allowed_paths:
                continue

            patches.append(FileContentPatch(path=repo.root / rp, new_content=str(new_content)))

        return patches

    def select_files(
        self,
        repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
        max_files: int,
    ) -> List[Path]:
        """
        Keep your existing dynamic selection logic unchanged.
        (This file is a full replacement, but selection remains identical to your prior version.)
        """
        # Import the existing implementation from your current file if you want,
        # but for "full replacement" we inline it by reusing the exact prior code.
        import re

        max_files = max(1, int(max_files))

        def _normalize_relpath(s: str) -> str:
            s = s.strip().strip("\"'`")
            s = s.replace("\\", "/")
            s = s.rstrip(").,;:")
            if s.startswith("./"):
                s = s[2:]
            return s

        def _safe_repo_relpath(s: str) -> str | None:
            s = _normalize_relpath(s)
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

        def _extract_paths_from_objective(text: str) -> List[str]:
            if not text:
                return []
            candidates = re.findall(r"(?<!\w)([A-Za-z0-9_.\-]+(?:[\\/][A-Za-z0-9_.\-]+)+)(?!\w)", text)
            out: List[str] = []
            seen = set()
            for c in candidates:
                rp = _safe_repo_relpath(c)
                if not rp:
                    continue
                if rp not in seen:
                    seen.add(rp)
                    out.append(rp)
            return out

        def _score_candidate(p: Path, *, obj_tokens: set[str], obj_paths: set[str], touched: set[str]) -> int:
            rel = p.relative_to(repo.root).as_posix()
            name = p.name.lower()
            parts = {seg.lower() for seg in Path(rel).parts}
            score = 0
            if rel in obj_paths:
                score += 1000
            if rel in touched:
                score += 800
            if p.suffix == ".py":
                score += 50
            elif p.suffix in {".md", ".json", ".yaml", ".yml"}:
                score += 10
            score += 5 * len((parts | {name}) & obj_tokens)
            for op in obj_paths:
                op_parts = {seg.lower() for seg in Path(op).parts[:-1]}
                if op_parts and (op_parts & parts):
                    score += 15
                    break
            if len(Path(rel).parts) == 1 and p.suffix == ".py":
                score += 10
            return score

        objective = str(ctx.state.get("node_objective") or "")
        obj_paths_list = _extract_paths_from_objective(objective)

        touched_list = ctx.state.get("touched_paths") or []
        if not isinstance(touched_list, list):
            touched_list = []

        touched_norm: List[str] = []
        seen = set()
        for t in touched_list:
            if not isinstance(t, str):
                continue
            rp = _safe_repo_relpath(t)
            if not rp:
                continue
            if rp not in seen:
                seen.add(rp)
                touched_norm.append(rp)

        obj_paths = set(obj_paths_list)
        touched = set(touched_norm)
        obj_tokens = {tok.lower() for tok in re.findall(r"[A-Za-z0-9_.\-]+", objective) if tok}

        selected: List[Path] = []
        selected_rel: set[str] = set()

        def _try_add_rel(rp: str) -> None:
            if len(selected) >= max_files:
                return
            p = repo.root / rp
            if not p.exists() or not p.is_file():
                return
            if rp in selected_rel:
                return
            selected.append(p)
            selected_rel.add(rp)

        for rp in obj_paths_list:
            _try_add_rel(rp)

        for rp in touched_norm:
            _try_add_rel(rp)

        if len(selected) >= max_files:
            return selected[:max_files]

        scored = []
        for p in files:
            if not p.is_file():
                continue
            rel = p.relative_to(repo.root).as_posix()
            if rel in selected_rel:
                continue
            scored.append((_score_candidate(p, obj_tokens=obj_tokens, obj_paths=obj_paths, touched=touched), rel, p))

        scored.sort(key=lambda x: (-x[0], x[1]))
        for _, rel, p in scored:
            if len(selected) >= max_files:
                break
            selected.append(p)
            selected_rel.add(rel)

        return selected[:max_files]
