import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from ollama import chat
from colorama import Fore, Style
from utils.ollama_utils import check_if_model_exist
from utils.decorators import timer_decorator
from db_connector import get_top_k_similar_docs, check_db_connection
from ingestion.vector import get_embedding_ollama

load_dotenv()

AI_MODEL = os.getenv('AI_MODEL')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
EMBEDDING_MODEL = "nomic-embed-text:latest"

def get_completion_from_messages(messages: List[Dict[str, Any]], model: str = AI_MODEL, temperature: float = 0, max_tokens: int = 1000) -> str:
    """
    Calls the Ollama chat model to get a response and prints token usage.
    """
    try:
        response = chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        
        # Extract and print token counts
        prompt_tokens = response.get('prompt_eval_count')
        completion_tokens = response.get('eval_count')

        if prompt_tokens is not None:
            print(f"Number of input (prompt) tokens: {prompt_tokens}")
        if completion_tokens is not None:
            print(f"Number of output (completion) tokens: {completion_tokens}")

        # The 'total_duration' and 'eval_duration' can also be useful
        total_duration = response.get('total_duration')
        eval_duration = response.get('eval_duration')
        
        if total_duration is not None:
            print(f"Total duration: {total_duration / 1e9:.2f} seconds")
        if eval_duration is not None and completion_tokens is not None and eval_duration > 0:
            tokens_per_second = (completion_tokens / (eval_duration / 1e9))
            print(f"Response speed: {tokens_per_second:.2f} tokens/second")

        return response.get('message', {}).get('content', "Sorry, I couldn't get a response from the model.")
    
    except Exception as e:
        print(f"Error getting completion from Ollama: {e}")
        return "An error occurred while getting the model response."

@timer_decorator
async def process_input_with_retrieval(user_input: str) -> str:
    """
    Processes the user's input by retrieving relevant documents and generating a response.
    """
    # Step 1: Get documents related to the user input from the database
    query_embedding = get_embedding_ollama(user_input)
    related_docs = get_top_k_similar_docs(query_embedding)

    # Step 2: Format messages to pass to the model for RAG
    delimiter = "```"
    context = "\n".join(related_docs) if related_docs else "No relevant documents found."
    
    # We combine the system prompt, context, and user input into a single user message.
    # This is an effective pattern for RAG with Ollama.
    system_message = f"""
    You are a friendly chatbot. 
    You can answer questions about timescaledb, its features and its use cases. 
    You respond in a concise, technically credible tone.
    """
    
    user_message = f"""
    {system_message}
    
    Relevant Timescale case studies information:
    {context}
    
    User query: {delimiter}{user_input}{delimiter}
    """

    messages = [
        {"role": "user", "content": user_message},
    ]

    final_response = get_completion_from_messages(messages)
    return final_response

async def main():
    if not check_if_model_exist(AI_MODEL):
        return
    
    if check_db_connection() == False:
        return
    
    print("Ollama Agent. Type 'exit' or 'quit' to terminate running task.")
    while True:
        try:
            user_query = input("\n[You] ")
            if user_query.lower() in ["exit", "quit"]:
                break

            if len(user_query) > 0:
                answer = await process_input_with_retrieval(user_query)
                print(Fore.BLUE + f"\n[Assistant] {answer}")
                print(Style.RESET_ALL)

        except (KeyboardInterrupt, EOFError):
            print("\nTask terminated by user.")
            break
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted from the main thread.")