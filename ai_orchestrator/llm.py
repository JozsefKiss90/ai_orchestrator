# ai_orchestrator/llm.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from collections.abc import Mapping

from openai import OpenAI

import logging

log = logging.getLogger(__name__)

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
    # strict-mode rule: required must include every key in properties
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
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        """
        Get attribute or dict-key.
        """
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """
        Responses SDK can put text in multiple places, and can return objects or dict-like items.

        Try, in order:
          1) response.output_text
          2) response.output[*].content[*].text
          3) response.output[*].content[*].output_text
          4) response.output[*].content[*] dict shapes: {"type":"output_text","text":...}
        """
        raw = LLMClient._get(response, "output_text", None)
        if isinstance(raw, str) and raw.strip():
            return raw

        out = LLMClient._get(response, "output", None)
        if isinstance(out, list):
            parts: list[str] = []
            for item in out:
                content = LLMClient._get(item, "content", None)
                if not isinstance(content, list):
                    continue
                for c in content:
                    # object shapes
                    t = LLMClient._get(c, "text", None)
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
                        continue
                    t2 = LLMClient._get(c, "output_text", None)
                    if isinstance(t2, str) and t2.strip():
                        parts.append(t2)
                        continue
                    # dict shapes (common in some SDK versions)
                    if isinstance(c, Mapping):
                        # sometimes content item is {"type":"output_text","text":"..."}
                        tt = c.get("text")
                        if isinstance(tt, str) and tt.strip():
                            parts.append(tt)
                            continue
                        tt2 = c.get("output_text")
                        if isinstance(tt2, str) and tt2.strip():
                            parts.append(tt2)
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
        log.info("LLM complete_text start", extra={"fields": {"model": self._config.model}})
        log.debug(
            "LLM prompts",
            extra={"fields": {"system_chars": len(system_prompt or ""), "user_chars": len(user_prompt or "")}},
        )

        response = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **self._responses_kwargs(),
            **kwargs,
        )
        return self._extract_response_text(response)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        *,
        schema_name: str = "orchestrator_payload",
    ) -> Dict[str, Any]:
        """
        Robust JSON completion:
        1) Try Responses API Structured Outputs (strict json_schema).
        2) If structured output is empty/invalid, fall back to plain-text generation
           and extract/parse JSON locally.
        3) If that is ALSO empty twice, fall back to Chat Completions json_schema.
        """
        self._preflight_strict_schema(schema)

        def _json_retry_prompt(reason: str) -> str:
            return (
                user_prompt
                + "\n\nIMPORTANT:\n"
                + "- Return ONLY valid JSON matching the schema.\n"
                + "- Do NOT include markdown fences.\n"
                + "- Do NOT include comments.\n"
                + "- Ensure all string values escape newlines as \\n (no raw newlines).\n"
                + f"- Previous output was invalid/empty because: {reason}\n"
            )

        def _extract_first_json_object(txt: str) -> str:
            if not txt:
                return ""
            s = txt.strip()
            if s.startswith("{") and s.endswith("}"):
                return s

            start = s.find("{")
            if start < 0:
                return ""

            in_str = False
            esc = False
            depth = 0
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
            return ""

        def _plain_fallback(prompt: str, reason: str) -> Dict[str, Any]:
            """
            Non-structured fallback: ask for JSON-only, then locally extract/parse.
            If empty twice, do ONE smaller retry (short prompt) before giving up.
            """
            raw1 = (self.complete_text(system_prompt=system_prompt, user_prompt=prompt) or "").strip()
            js1 = _extract_first_json_object(raw1)
            if js1:
                try:
                    return json.loads(js1)
                except json.JSONDecodeError as e:
                    raw2 = (self.complete_text(system_prompt=system_prompt, user_prompt=_json_retry_prompt(f"{reason}; parse error: {e}")) or "").strip()
                    js2 = _extract_first_json_object(raw2)
                    if js2:
                        return json.loads(js2)
                    raise RuntimeError(
                        f"Model returned non-parseable JSON in fallback twice. Last error: {e}\nRaw:\n{raw2}"
                    )

            raw2 = (self.complete_text(system_prompt=system_prompt, user_prompt=_json_retry_prompt(f"{reason}; no JSON object")) or "").strip()
            js2 = _extract_first_json_object(raw2)
            if js2:
                return json.loads(js2)

            # One last attempt: shrink prompt hard (common fix for “silent empty” outputs)
            mini_user = (
                "Return ONLY JSON matching the schema. No prose.\n\n"
                "Schema keys:\n"
                + ", ".join(list(schema.get("properties", {}).keys()))
                + "\n\n"
                "Task summary (keep it minimal):\n"
                + (user_prompt[:800] if user_prompt else "")
            )
            raw3 = (self.complete_text(system_prompt=system_prompt, user_prompt=mini_user) or "").strip()
            js3 = _extract_first_json_object(raw3)
            if not js3:
                raise RuntimeError(
                    "Model returned empty/non-JSON output in fallback three times.\n"
                    f"Raw1:\n{raw1}\n\nRaw2:\n{raw2}\n\nRaw3:\n{raw3}"
                )
            return json.loads(js3)

        def _chat_completions_fallback(reason: str) -> Dict[str, Any]:
            """
            Final fallback: use Chat Completions response_format json_schema (often more stable for some models).
            """
            log.warning(
                "Falling back to chat.completions json_schema",
                extra={"fields": {"schema_name": schema_name, "reason": reason, "model": self._config.model}},
            )
            chat = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _json_retry_prompt(reason)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
                temperature=self._config.temperature if self._supports_temperature() else 1.0,
            )
            content = (chat.choices[0].message.content or "").strip()
            if not content:
                # last resort: plain parse from chat text
                chat2 = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": _json_retry_prompt(f"{reason}; chat empty")},
                    ],
                    temperature=self._config.temperature if self._supports_temperature() else 1.0,
                )
                content2 = (chat2.choices[0].message.content or "").strip()
                js = _extract_first_json_object(content2)
                if not js:
                    raise RuntimeError(f"Chat fallback returned empty twice. Raw:\n{content2}")
                return json.loads(js)

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                js = _extract_first_json_object(content)
                if not js:
                    raise
                return json.loads(js)

        # -------- Preferred path: Responses API + Structured Outputs --------
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

            parsed = self._coerce_parsed(self._get(response, "output_parsed", None))
            if isinstance(parsed, dict):
                return parsed

            raw = (self._extract_response_text(response) or "").strip()
            if raw:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as e:
                    log.warning(
                        "Structured JSON raw unparsable; retrying once then fallback",
                        extra={"fields": {"schema_name": schema_name, "error": str(e)}},
                    )
                    retry_response = self._client.responses.create(
                        model=self._config.model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": _json_retry_prompt(str(e))},
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
                    parsed_retry = self._coerce_parsed(self._get(retry_response, "output_parsed", None))
                    if isinstance(parsed_retry, dict):
                        return parsed_retry

                    raw_retry = (self._extract_response_text(retry_response) or "").strip()
                    if raw_retry:
                        try:
                            return json.loads(raw_retry)
                        except json.JSONDecodeError:
                            # fall back to plain extraction
                            return _plain_fallback(user_prompt, "structured invalid twice")

                    # retry empty -> fallback
                    return _plain_fallback(user_prompt, "structured retry empty")

            # raw empty: retry once structured
            log.warning("LLM structured output empty; retrying once", extra={"fields": {"schema_name": schema_name}})
            retry_response = self._client.responses.create(
                model=self._config.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _json_retry_prompt("empty output")},
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

            parsed_retry = self._coerce_parsed(self._get(retry_response, "output_parsed", None))
            if isinstance(parsed_retry, dict):
                return parsed_retry

            raw_retry = (self._extract_response_text(retry_response) or "").strip()
            if raw_retry:
                try:
                    return json.loads(raw_retry)
                except json.JSONDecodeError:
                    return _plain_fallback(user_prompt, "structured retry unparsable")

            # Empty twice -> plain fallback, if that fails -> chat fallback
            log.warning(
                "Structured JSON empty twice; falling back to plain JSON extraction",
                extra={"fields": {"schema_name": schema_name, "model": self._config.model}},
            )
            try:
                return _plain_fallback(user_prompt, "structured empty twice")
            except Exception as e:
                return _chat_completions_fallback(f"structured empty twice; plain fallback failed: {e}")

        except TypeError:
            # Compatibility path (older SDKs): use chat.completions json_schema
            return _chat_completions_fallback("responses structured TypeError")
