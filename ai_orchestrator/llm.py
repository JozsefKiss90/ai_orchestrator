# ai_orchestrator/llm.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from collections.abc import Mapping

from openai import OpenAI


# ---- Schema registry ----
# NOTE: For OpenAI Structured Outputs with strict=True, the server requires:
# - schema.required must exist
# - required must include ALL keys present in properties

SCHEMA_FILE_CONTENT_V1: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "new_content": {"type": "string"},
                },
                "required": ["path", "new_content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["files"],
    "additionalProperties": False,
}

SCHEMA_UNIFIED_DIFF_V1: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "diff": {"type": "string", "minLength": 1},
        "notes": {"type": "string"},
    },
    "required": ["diff", "notes"],
    "additionalProperties": False,
}

SCHEMA_REPAIR_UNIFIED_DIFF_V1: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "diff": {"type": "string", "minLength": 1},
        "explanation": {"type": "string"},
    },
    "required": ["diff", "explanation"],
    "additionalProperties": False,
}


@dataclass
class LLMConfig:
    model: str = "gpt-5.1-codex-mini"
    temperature: float = 0.2
    max_output_tokens: int = 4096


class LLMClient:
    def __init__(self, config: LLMConfig):
        self._client = OpenAI()
        self._config = config

    def _supports_temperature(self) -> bool:
        m = (self._config.model or "").lower()
        if "codex" in m:
            return False
        if m.startswith("o"):
            return False
        return True

    def _responses_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"max_output_tokens": self._config.max_output_tokens}
        if self._supports_temperature():
            kwargs["temperature"] = self._config.temperature
        return kwargs

    @staticmethod
    def _preflight_strict_schema(schema: Dict[str, Any]) -> None:
        props = schema.get("properties")
        req = schema.get("required")
        if not isinstance(props, dict) or not props:
            raise ValueError("Invalid schema: missing/empty 'properties'.")
        if not isinstance(req, list):
            raise ValueError("Invalid schema: missing 'required' list (strict mode requires it).")
        missing = [k for k in props.keys() if k not in req]
        if missing:
            raise ValueError(
                "Invalid strict schema: 'required' must include every key in 'properties'. "
                f"Missing: {missing}"
            )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """
        Responses SDK can put text in multiple places. Try, in order:
          1) response.output_text (if present)
          2) concatenate response.output[*].content[*].text (or .output_text)
        """
        raw = getattr(response, "output_text", None)
        if isinstance(raw, str) and raw.strip():
            return raw

        # Fallback: dig into response.output structure
        out = getattr(response, "output", None)
        if isinstance(out, list):
            parts: list[str] = []
            for item in out:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for c in content:
                    # Common shapes: c.text (string), or c has dict-like fields
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
                        continue
                    # Some SDK shapes store as output_text
                    t2 = getattr(c, "output_text", None)
                    if isinstance(t2, str) and t2.strip():
                        parts.append(t2)
                        continue
            joined = "\n".join(parts).strip()
            if joined:
                return joined

        return ""

    @staticmethod
    def _coerce_parsed(parsed: Any) -> Optional[Dict[str, Any]]:
        """
        output_parsed can be:
          - dict
          - Mapping
          - Pydantic-like object with model_dump()
        """
        if parsed is None:
            return None
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, Mapping):
            return dict(parsed)
        md = getattr(parsed, "model_dump", None)
        if callable(md):
            dumped = md()
            if isinstance(dumped, dict):
                return dumped
        return None

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs = extra_kwargs or {}
        response = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **self._responses_kwargs(),
            **kwargs,
        )
        txt = self._extract_response_text(response)
        return txt

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        *,
        schema_name: str = "orchestrator_payload",
    ) -> Dict[str, Any]:
        self._preflight_strict_schema(schema)

        # Preferred path: Responses API + Structured Outputs via text.format
        try:
            response = self._client.responses.create(
                model=self._config.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
                **self._responses_kwargs(),
            )

            # Prefer SDK-provided parsed output (best case)
            parsed = self._coerce_parsed(getattr(response, "output_parsed", None))
            if isinstance(parsed, dict):
                return parsed

            # Fallback: parse text extracted from response structure
            raw = self._extract_response_text(response)
            if not raw.strip():
                # Codex models do NOT support chat completions.
                # Retry Responses once with a simplified instruction.
                retry_response = self._client.responses.create(
                    model=self._config.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": (
                                user_prompt
                                + "\n\nIMPORTANT: Return ONLY valid JSON matching the schema. "
                                "No prose, no markdown, no explanation."
                            ),
                        },
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "schema": schema,
                            "strict": True,
                        }
                    },
                    **self._responses_kwargs(),
                )

                parsed_retry = self._coerce_parsed(getattr(retry_response, "output_parsed", None))
                if isinstance(parsed_retry, dict):
                    return parsed_retry

                raw_retry = self._extract_response_text(retry_response)
                if raw_retry.strip():
                    try:
                        return json.loads(raw_retry)
                    except json.JSONDecodeError as e:
                        raise RuntimeError(
                            f"Retry Responses returned invalid JSON: {e}\nRaw:\n{raw_retry}"
                        )

                rid = getattr(response, "id", None)
                raise RuntimeError(
                    "Model returned empty output for structured JSON twice via Responses API. "
                    f"response.id={rid!r} model={self._config.model!r}"
                )

            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Model returned invalid JSON: {e}\nRaw output:\n{raw}")

        except TypeError:
            # Compatibility fallback: Chat Completions API
            chat = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                },
                temperature=self._config.temperature if self._supports_temperature() else 1.0,
            )
            content = chat.choices[0].message.content or ""
            if not content.strip():
                raise RuntimeError("Chat Completions returned empty content for JSON response.")
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Chat Completions returned invalid JSON: {e}\nRaw output:\n{content}")
