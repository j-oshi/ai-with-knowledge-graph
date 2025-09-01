import ollama
from ollama import chat
from dotenv import load_dotenv
import asyncio
import os
import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from colorama import Fore, Style
from utils.ollama_utils import check_if_model_exist

load_dotenv()
AI_MODEL = "qwen2.5vl:7b" 
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
EMBEDDING_MODEL = "nomic-embed-text:latest"

def get_embedding_ollama(text: str, model: str = EMBEDDING_MODEL):
    """
    Generates a vector embedding for the given text using Ollama.
    """
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error getting embedding from Ollama: {e}")
        return None

def get_top_k_similar_docs(query_embedding: list, k: int = 3) -> list:
    """
    Connects to the database and retrieves the top-k most similar documents.
    """
    if not query_embedding:
        return []

    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=POSTGRES_PORT,
            dbname="postgres",
            user="postgres",
            password=POSTGRES_PASSWORD
        )
        # Register pgvector extension once per connection
        register_vector(conn)
        cur = conn.cursor()

        embedding_array = np.array(query_embedding)

        # Get the top k most similar documents using the KNN <=> operator
        cur.execute("SELECT text_column FROM embeddings_table ORDER BY embedding_column <=> %s LIMIT %s", (embedding_array, k))
        top_docs = cur.fetchall()
        
        return [doc[0] for doc in top_docs]
    except psycopg2.Error as e:
        print(f"Database error occurred: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_completion_from_messages(messages: list, model: str = AI_MODEL, temperature: float = 0, max_tokens: int = 1000):
    """
    Calls the Ollama chat model to get a response.
    """
    try:
        response = chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response.get('message', {}).get('content', "Sorry, I couldn't get a response from the model.")
    except Exception as e:
        print(f"Error getting completion from Ollama: {e}")
        return "An error occurred while getting the model response."


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
    
    print("Ollama Agent. Type 'exit' or 'quit' to terminate running task.")
    while True:
        try:
            user_query = input("\n[You] ")
            if user_query.lower() in ["exit", "quit"]:
                break
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