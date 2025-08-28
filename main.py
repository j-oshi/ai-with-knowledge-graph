from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_ollama_client.ollama_client import OllamaClient
from graphiti_ollama_client.ollama_embedder import OllamaEmbedder, OllamaEmbedderConfig
from graphiti_ollama_client.ollama_reranker_client import OllamaRerankerClient

from ollama import chat
import os
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from helper import check_if_model_exist

load_dotenv()

AI_MODEL = "qwen2.5vl:7b" # Set up from ollama.com
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

# Neo4j connection details
NEO4j_URI = 'neo4j://127.0.0.1:7687'
NEO4j_USER = 'neo4j'
NEO4j_PASSWORD = os.getenv('NEO4j_PASSWORD')

# Configure Ollama LLM client
llm_config = LLMConfig(
    api_key="abc",  
    model=AI_MODEL,
    small_model=AI_MODEL,
    base_url=OLLAMA_BASE_URL,
)

llm_client = OllamaClient(config=llm_config)

# Ollama model
AI_MODEL = "qwen2.5vl:7b"
print(check_if_model_exist(AI_MODEL))

# Initialize Graphiti with Neo4j connection
graphiti = Graphiti(
    NEO4j_URI,
    NEO4j_USER,
    NEO4j_PASSWORD,
    llm_client=llm_client,
    embedder=OllamaEmbedder(
        config=OllamaEmbedderConfig(
            api_key="abc",
            embedding_model=EMBEDDING_MODEL,
            embedding_dim=768,
            base_url=OLLAMA_BASE_URL,
        )
    ),
    cross_encoder=OllamaRerankerClient(client=llm_client, config=llm_config),
)

# ---------------- Search result wrapper ----------------
@dataclass
class GraphitiSearchResult:
    uuid: str
    fact: str
    source_node_uuid: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None

# ---------------- Graphiti search tool ----------------
async def search_graphiti(query: str) -> List[GraphitiSearchResult]:
    """Search Graphiti knowledge graph for relevant facts."""
    try:
        results = await graphiti.search(query)
        formatted_results = []
        for result in results:
            formatted_results.append(
                GraphitiSearchResult(
                    uuid=result.uuid,
                    fact=result.fact,
                    source_node_uuid=getattr(result, 'source_node_uuid', None),
                    valid_at=str(result.valid_at) if getattr(result, 'valid_at', None) else None,
                    invalid_at=str(result.invalid_at) if getattr(result, 'invalid_at', None) else None,
                )
            )
        print('test test')
        print(formatted_results)
        return formatted_results
    except Exception as e:
        print(f"Error searching Graphiti: {e}")
        raise

# ---------------- System prompt ----------------
system_prompt = """You have access to a knowledge graph containing information about large language models (LLMs), including temporal data.
When answering user questions, call the `search_graphiti` tool if you need factual information.
If no relevant information is found, say you don’t know rather than inventing an answer."""

# ---------------- Ollama chat wrapper ----------------
async def ollama_chat(question: str):
    search_results = await search_graphiti(question)

    # Convert results into context for Ollama
    context_facts = "\n".join([f"- {r.fact} (valid: {r.valid_at}, invalid: {r.invalid_at})"
                               for r in search_results]) or "No results found in Graphiti."

    # Ollama’s chat is sync → run in thread executor to avoid blocking
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: chat(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"Graphiti search results:\n{context_facts}"}
            ]
        )
    )
    return response["message"]["content"]

async def main():
    print("Graphiti-Ollama Agent. Type 'exit' to quit.")
    while True:
        user_query = input("\n[You] ")
        if user_query.lower() in ["exit", "quit"]:
            break

        try:
            answer = await ollama_chat(user_query)
            print(f"\n[Assistant] {answer}")
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    asyncio.run(main())

