# ai_orchestrator/validators/pipeline.py
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional, List

from ..repo import Repo, CommandResult
from .types import PipelineResult, ValidatorResult, ValidatorSpec
from .policy import run_policy_validator

import logging
log = logging.getLogger(__name__)

class ValidatorPipeline:
    """
    Deterministic validation pipeline.

    Supports:
      - shell validators (Repo.run_command)
      - policy validators (in-process)
    """

    def __init__(self, validators: list[ValidatorSpec], stop_on_fail: bool = True):
        self.validators = validators
        self.stop_on_fail = stop_on_fail

    def run(self, repo: Repo) -> PipelineResult:
        results: list[ValidatorResult] = []
        stopped_early = False

        for spec in self.validators:
            start = time.time()

            if spec.kind == "policy":
                params = spec.params or {}
                log.info("Validator start", extra={"fields": {"validator": spec.name, "kind": spec.kind}})
                vr = run_policy_validator(name=spec.policy or spec.name, repo=repo, params=params)
                vr.duration_s = float(time.time() - start)  # type: ignore[misc]
                results.append(vr)
                if self.stop_on_fail and not vr.ok:
                    stopped_early = True
                    break
                continue

            # shell validator (default)
            if not spec.command:
                results.append(
                    ValidatorResult(
                        name=spec.name,
                        ok=True,
                        exit_code=0,
                        stdout=f"{spec.name}: skipped (no command configured).",
                        stderr="",
                        duration_s=float(time.time() - start),
                    )
                )
                continue

            cmd_res: CommandResult = repo.run_command(spec.command)
            vr = ValidatorResult(
                name=spec.name,
                ok=cmd_res.ok,
                exit_code=cmd_res.returncode,
                stdout=cmd_res.stdout,
                stderr=cmd_res.stderr,
                duration_s=float(time.time() - start),
            )
            results.append(vr)
            log.info(
                "Validator end",
                extra={"fields": {"validator": vr.name, "ok": vr.ok, "exit_code": vr.exit_code, "duration_s": round(vr.duration_s, 3)}},
            )

            if self.stop_on_fail and not vr.ok:
                log.warning("Stopping early due to validator failure", extra={"fields": {"failed_validator": vr.name}})
                stopped_early = True
                break

        return PipelineResult(results=results, stopped_early=stopped_early)

    @staticmethod
    def to_json_dict(pipeline_result: PipelineResult) -> dict:
        return {
            "ok": pipeline_result.ok,
            "stopped_early": pipeline_result.stopped_early,
            "results": [asdict(r) for r in pipeline_result.results],
        }

    @staticmethod
    def extract_tests_result(pipeline_result: PipelineResult) -> Optional[ValidatorResult]:
        return pipeline_result.get("tests")

    @staticmethod
    def filter_specs(all_specs: List[ValidatorSpec], names: List[str]) -> List[ValidatorSpec]:
        specs_by_name = {s.name: s for s in all_specs}
        out: List[ValidatorSpec] = []
        for n in names:
            if n not in specs_by_name:
                raise ValueError(f"Unknown validator: {n}")
            out.append(specs_by_name[n])
        return out
