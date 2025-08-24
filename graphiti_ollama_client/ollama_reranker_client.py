import json
import logging
from typing import Any
import numpy as np
import ollama
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMConfig, RateLimitError
from graphiti_core.prompts import Message
from graphiti_core.cross_encoder.client import CrossEncoderClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3"


class OllamaRerankerClient(CrossEncoderClient):
    """
    Reranker client that uses the Ollama API.

    This reranker uses the Ollama API to run a simple relevance classifier prompt
    concurrently for each passage. It prompts the model to return a JSON object
    containing a relevance score from 0.0 to 1.0. This score is then used to rank
    the passages.

    Note: The original OpenAI implementation used token log-probabilities to
    determine the score. As Ollama does not expose this feature, this
    implementation relies on the model's ability to provide a structured JSON
    response with a score, which is a common and effective workaround.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: ollama.AsyncClient | None = None,
    ):
        """
        Initialize the OllamaRerankerClient with the provided configuration and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including
                model, and other settings.
            client (ollama.AsyncClient | None): An optional async client instance to use.
                If not provided, a new ollama.AsyncClient is created.
        """
        if config is None:
            config = LLMConfig()

        self.config = config
        if client is None:
            self.client = ollama.AsyncClient()
        else:
            self.client = client

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Ranks a list of passages based on their relevance to a query.

        Args:
            query (str): The query string.
            passages (list[str]): The list of passages to be ranked.

        Returns:
            list[tuple[str, float]]: A sorted list of (passage, score) tuples,
                from most relevant to least relevant.
        """
        ollama_messages_list: Any = [
            [
                Message(
                    role="system",
                    content="""You are an expert tasked with determining whether a passage is relevant to a query.
                    Your response must be a single JSON object with the following schema:
                    {
                        "relevance_score": number // A float from 0.0 to 1.0 indicating relevance.
                    }""",
                ),
                Message(
                    role="user",
                    content=f"""
                    QUERY: {query}
                    PASSAGE: {passage}
                    
                    Respond with a relevance score from 0.0 (not relevant at all) to 1.0 (highly relevant).
                    """,
                ),
            ]
            for passage in passages
        ]

        try:
            responses = await semaphore_gather(
                *[
                    self.client.chat(
                        model=self.config.model or DEFAULT_MODEL,
                        messages=[m.dict() for m in ollama_messages],
                        format="json",
                        options={"temperature": 0.0},
                    )
                    for ollama_messages in ollama_messages_list
                ]
            )

            scores: list[float] = []
            for response in responses:
                try:
                    content = response["message"]["content"]
                    data = json.loads(content)
                    score = float(data.get("relevance_score", 0.0))
                    scores.append(score)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Failed to parse JSON response: {content}. Error: {e}")
                    scores.append(0.0) 

            results = [(passage, score) for passage, score in zip(passages, scores, strict=True)]
            results.sort(reverse=True, key=lambda x: x[1])
            return results
        except ollama.ResponseError as e:
            if "rate limit" in str(e).lower():
                raise RateLimitError from e
            else:
                logger.error(f"Ollama API error: {e}")
                raise
        except Exception as e:
            logger.error(f"Error in generating LLM response: {e}")
            raise