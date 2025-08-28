import httpx
from collections.abc import Iterable
from pydantic import Field

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


class OllamaEmbedderConfig(EmbedderConfig):
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    base_url: str = "http://127.0.0.1:11434"  
    timeout: int = 240


class OllamaEmbedder(EmbedderClient):
    """
    Ollama Embedder Client \
    Calls Ollama's /api/embeddings endpoint to generate vector embeddings.
    """

    def __init__(self, config: OllamaEmbedderConfig | None = None):
        if config is None:
            config = OllamaEmbedderConfig()
        self.config = config

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embeddings for a single input.
        """
        if isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            input_str = " ".join(input_data)
        elif isinstance(input_data, str):
            input_str = input_data
        else:
            raise TypeError("Ollama embedder only supports str or list[str] inputs")

        url = f"{self.config.base_url}/api/embeddings"
        payload = {"model": self.config.embedding_model, "prompt": input_str}

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        embedding = data.get("embedding", [])
        return embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Ollama currently does not support true batch embedding.
        We emulate it by looping requests.
        """
        results: list[list[float]] = []
        for text in input_data_list:
            results.append(await self.create(text))
        return results