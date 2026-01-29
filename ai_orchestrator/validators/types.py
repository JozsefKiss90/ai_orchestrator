# ai_orchestrator/validators/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class ValidatorSpec:
    """
    Declarative validator definition loaded from repo config.

    Backward compatible fields:
      - name
      - command
      - shell

    New fields:
      - kind: "shell" (default) or "policy"
      - policy: for kind="policy", the policy validator name
      - params: for kind="policy", a JSON object with validator parameters
    """
    name: str
    command: str = ""
    shell: bool = True
    kind: str = "shell"
    policy: str = ""
    params: Optional[Dict[str, Any]] = None


@dataclass
class ValidatorResult:
    name: str
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def summary(self) -> str:
        status = "OK" if self.ok else "FAIL"
        return f"{self.name}: {status} (exit={self.exit_code}, {self.duration_s:.2f}s)"


@dataclass
class PipelineResult:
    results: list[ValidatorResult]
    stopped_early: bool = False

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    def get(self, name: str) -> Optional[ValidatorResult]:
        for r in self.results:
            if r.name == name:
                return r
        return None
