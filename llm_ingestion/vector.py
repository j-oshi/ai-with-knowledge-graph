import ollama

def chunk_text(text, doc_name, chunk_size=500, overlap=100):
    """
    Splits a given text into smaller chunks for embedding, with metadata.

    Args:
        text (str): The input text to be chunked.
        doc_name (str): The name or title of the document the text belongs to.
        chunk_size (int, optional): The maximum size of each text chunk in words. Defaults to 500.
        overlap (int, optional): The number of words to overlap between consecutive chunks. Defaults to 100.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a chunk.
                    Each dictionary contains the chunked text and its associated metadata.
                    The format is: [{'text': str, 'metadata_': {'doc': str, 'index': int}}]
    """
    if text == "":
        print("No text for embedding.")
        return []
    if doc_name == "":
        print("No document title for embedding.")
        return []
    
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append({
            'text': chunk,
            'metadata_': {'doc': doc_name, 'index': i}
        })
    return chunks

def get_embedding_ollama(text, model="nomic-embed-text:latest"):
    """
    Generates a vector embedding for a given text using an Ollama-hosted model.

    This function sends the text to a running Ollama server to get its embedding vector.
    It requires the 'ollama' library and a local Ollama server to be running.

    Args:
        text (str): The text to be embedded.
        model (str, optional): The name of the Ollama model to use for embedding. 
                               Defaults to "nomic-embed-text:latest".

    Returns:
        list[float]: A list of floats representing the embedding vector for the text.
    
    Raises:
        requests.exceptions.RequestException: If there is an issue connecting to the Ollama server.
        KeyError: If the 'embedding' key is not present in the Ollama response.
    """
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]