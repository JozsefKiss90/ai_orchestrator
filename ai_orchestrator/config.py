# ai_orchestrator/config.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from .llm import LLMConfig
from .validators.types import ValidatorSpec
from .validators.builtins import default_validators


CONFIG_FILENAME = ".ai-orchestrator.json"


@dataclass
class RepoConfig:
    root: Path
    llm: LLMConfig
    test_command: Optional[str]
    build_command: Optional[str]
    max_files_per_run: int
    include_globs: List[str]
    exclude_globs: List[str]
    validators: List[ValidatorSpec]
    architecture: Dict[str, Any]


def _parse_architecture(raw: dict) -> Dict[str, Any]:
    """
    Optional top-level config section:

    "architecture": {
      "layering": {
        "layers": {...},
        "rules": [...]
      },
      "secrets": {
        "deny_patterns": [...],
        "scan_paths": [...]
      }
    }
    """
    arch = raw.get("architecture", {})
    if isinstance(arch, dict):
        return arch
    return {}


def _parse_validators(raw: dict, test_command: Optional[str], architecture: Dict[str, Any]) -> List[ValidatorSpec]:
    """
    Validators config supports two kinds:

    1) Shell validators:
      {"name": "lint", "command": "npm run lint", "shell": true}

    2) Policy validators:
      {"name": "layering", "kind": "policy", "policy": "layering", "params": {...}}

    Auto-wiring behavior:
      - if kind == "policy" and params is omitted -> params = architecture.get(policy, {})
    """
    validators_raw = raw.get("validators")
    if not validators_raw:
        return default_validators(test_command=test_command)

    if not isinstance(validators_raw, list):
        raise ValueError("'validators' must be a list if provided.")

    specs: List[ValidatorSpec] = []

    for v in validators_raw:
        if not isinstance(v, dict):
            raise ValueError("Each validator entry must be an object/dict.")

        name = str(v.get("name", "")).strip()
        if not name:
            raise ValueError("Validator entry missing required field: 'name'")

        kind = str(v.get("kind", "shell")).strip() or "shell"
        shell = bool(v.get("shell", True))

        if kind == "policy":
            policy = str(v.get("policy", "")).strip() or name

            params = v.get("params", None)
            if params is None:
                # auto-wire architecture config into policy validators if available
                params = architecture.get(policy, {}) if isinstance(architecture, dict) else {}

            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise ValueError(f"Validator '{name}' policy params must be an object/dict.")

            specs.append(
                ValidatorSpec(
                    name=name,
                    kind="policy",
                    policy=policy,
                    params=params,
                    command="",
                    shell=shell,
                )
            )
            continue

        # default: shell validator
        command = str(v.get("command", "")).strip()
        if not command:
            # allow empty command (interpreted as skip), consistent with your pipeline behavior
            command = ""

        specs.append(
            ValidatorSpec(
                name=name,
                kind="shell",
                policy="",
                params={},
                command=command,
                shell=shell,
            )
        )

    return specs


def load_repo_config(repo_root: Path) -> RepoConfig:
    cfg_path = repo_root / CONFIG_FILENAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file {CONFIG_FILENAME} not found in {repo_root}")

    raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    llm_cfg = LLMConfig(
        model=raw.get("model", "gpt-5.1-codex-mini"),
        temperature=raw.get("temperature", 0.2),
        max_output_tokens=raw.get("max_output_tokens", 4096),
    )

    test_command = raw.get("test_command")
    build_command = raw.get("build_command")

    architecture = _parse_architecture(raw)
    validators = _parse_validators(raw=raw, test_command=test_command, architecture=architecture)

    return RepoConfig(
        root=repo_root,
        llm=llm_cfg,
        test_command=test_command,
        build_command=build_command,
        max_files_per_run=int(raw.get("max_files_per_run", 5)),
        include_globs=raw.get("include_globs", ["**/*.py"]),
        exclude_globs=raw.get("exclude_globs", []),
        validators=validators,
        architecture=architecture,
    )
