import httpx
import json
import logging
import typing
from typing import ClassVar
from typing import Any, Optional

from graphiti_core.llm_client.client import MULTILINGUAL_EXTRACTION_RESPONSES, LLMClient, DEFAULT_MAX_TOKENS
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message

from pydantic import BaseModel

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

    return data

def _normalize_llm_output(parsed: dict, response_model: Optional[type[BaseModel]]) -> dict:
    """
    Normalize Ollama JSON response to ensure required keys exist
    and maintain consistency across ExtractedEntities / Edges / Resolutions.
    """
    if not isinstance(parsed, dict):
        return {}

    # --- Always enforce required keys by schema ---
    if response_model:
        if response_model.__name__ == "ExtractedEdges":
            parsed.setdefault("edges", [])
        elif response_model.__name__ == "ExtractedEntities":
            parsed.setdefault("extracted_entities", [])

    # --- Ensure entity_resolutions consistency ---
    resolutions = parsed.get("entity_resolutions", [])
    extracted_entities = parsed.get("extracted_entities", [])

    if resolutions and not extracted_entities:
        # Auto-create stub entities to match resolutions
        for res in resolutions:
            stub = {
                "name": res.get("name", f"Entity_{res.get('id', len(extracted_entities))}"),
                "entity_type_id": 0  # default fallback
            }
            extracted_entities.append(stub)
        parsed["extracted_entities"] = extracted_entities

    # Clamp invalid resolution IDs
    for res in resolutions:
        if isinstance(res.get("id"), int) and res["id"] >= len(parsed["extracted_entities"]):
            res["id"] = max(0, len(parsed["extracted_entities"]) - 1)

    return parsed

class OllamaClient(LLMClient):
    """
    Minimal Ollama client compatible with Graphiti's LLMClient interface.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

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
        if base.endswith("/api/chat"):
            return base
        if base.endswith("/api"):
            return f"{base}/chat"
        return f"{base}/api/chat"

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
        ollama_messages = []

        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                ollama_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':                
                if response_model is not None:
                    schema = response_model.model_json_schema()
                    m.content += (
                        "You must respond ONLY with a valid JSON object INSTANCE that conforms to this schema:\n\n"
                        f"{json.dumps(schema, indent=2)}\n\n"
                        "⚠️ IMPORTANT RULES:\n"
                        "1. Do NOT return the schema itself.\n"
                        "2. Do NOT include '$defs', 'properties', or 'title'.\n"
                        "3. Instead, return a JSON object populated with example values.\n"
                        "4. Always include the required fields:\n"
                        "   - For ExtractedEdges: return {\"edges\": [ ... ]}\n"
                        "   - For ExtractedEntities: return {\"extracted_entities\": [ ... ]}\n"
                        "5. If there are no values, return an empty array for the required key.\n"
                    )
                ollama_messages.append({'role': 'system', 'content': m.content})

        # prompt = self._flatten_messages(messages)

        model_name = self._get_model_for_size(model_size)
        url = self._generation_url()

        payload = {
            "model": model_name,
            "messages": ollama_messages,
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

        raw = data.get("message", {}).get("content", "")

        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw[start : end + 1]
                parsed = json.loads(candidate)
            else:
                raise json.JSONDecodeError("Response was not valid JSON", raw, 0)

        # Normalize for required keys
        parsed = _normalize_edges(parsed)
        # parsed = _normalize_llm_output(parsed, response_model)

        return parsed

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_error = None

        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        # Add multilingual extraction instructions
        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens=max_tokens, model_size=model_size
                )
                return response
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f'The previous response attempt was invalid. '
                    f'Error type: {e.__class__.__name__}. '
                    f'Error details: {str(e)}. '
                    f'Please try again with a valid response, ensuring the output matches '
                    f'the expected format and constraints.'
                )

                error_message = Message(role='user', content=error_context)
                messages.append(error_message)
                logger.warning(
                    f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                )

        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')
