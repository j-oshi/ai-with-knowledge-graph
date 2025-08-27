import httpx
import json
import logging
from typing import Any, Optional

from graphiti_core.llm_client.client import LLMClient, DEFAULT_MAX_TOKENS
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _normalize_edges(data: dict) -> dict:
    """
    Ensure all required fields exist in edges for Pydantic validation.
    Injects 'fact': None if missing and replaces None IDs with 0.
    """
    if isinstance(data, dict) and "edges" in data:
        for edge in data["edges"]:
            if "fact" not in edge:
                edge["fact"] = None
            # 🔧 Fix: Ensure entity IDs are always integers
            if edge.get("source_entity_id") is None:
                edge["source_entity_id"] = 0
            if edge.get("target_entity_id") is None:
                edge["target_entity_id"] = 0

    if isinstance(data, dict) and "entity_resolutions" in data:
        for edge in data["entity_resolutions"]:
            if "duplicate_idx" in edge and "duplicates" not in edge:
                edge["duplicates"] = []

    # print("--------------normalized edges ----------")
    # print(data)
    # print("-----------------------------------------")
    return data


class OllamaClient(LLMClient):
    """
    Minimal Ollama client compatible with Graphiti's LLMClient interface.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(config, cache=cache)
        if self.config.base_url is None or self.config.base_url.strip() == "":
            self.config.base_url = DEFAULT_OLLAMA_BASE_URL

        self.max_tokens = max_tokens

    def _generation_url(self) -> str:
        base = (self.config.base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        if base.endswith("/api/generate"):
            return base
        if base.endswith("/api"):
            return f"{base}/generate"
        return f"{base}/api/generate"

    def _get_model_for_size(self, model_size: ModelSize) -> str:
        if model_size == ModelSize.small and self.small_model:
            return self.small_model
        return self.model

    def _flatten_messages(self, messages: list[Message]) -> str:
        parts: list[str] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            parts.append(f"{m.role}: {m.content}")
        return "\n".join(parts)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        prompt = self._flatten_messages(messages)

        if response_model is not None:
            schema = response_model.model_json_schema()

            # print("----------print schema-----------")
            # print(schema)
            # print("---------------------------------")
            prompt += (
                "\n\n"
                "You must respond ONLY with a valid JSON object that contains example data conforming to this schema:\n"
                f"{json.dumps(schema, indent=2)}\n\n"
                "⚠️ Do NOT return the schema itself. "
                "Instead, return a JSON object with real values that satisfy the schema. "
                "Always include valid integers for `source_entity_id` and `target_entity_id`. "
                "If you cannot determine an ID, return `0` instead of null. "
                "If you cannot find any data, return an empty `edges` array."
            )

        model_name = self._get_model_for_size(model_size)
        url = self._generation_url()

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",   # Ollama formats output as JSON
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
            "keep_alive": "5m",
        }

        async with httpx.AsyncClient(timeout=720) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw = data.get("response", "")
        # print("--------------raw ollama response ----------")
        # print(raw)
        # print("--------------------------------------------")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw[start : end + 1]
                parsed = json.loads(candidate)
            else:
                raise json.JSONDecodeError("Response was not valid JSON", raw, 0)

        # Normalize before returning to avoid Pydantic validation errors
        return _normalize_edges(parsed)
