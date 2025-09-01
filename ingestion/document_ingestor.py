from pathlib import Path
from ingestion.extractor.document_extractor import extract_to
from ingestion.vector import get_embedding_ollama, chunk_text
from utils.document_utils import get_document_filenames


def doc_to_vector():
    """
    Processes all documents in the designated data folder, chunks their content,
    and generates vector embeddings for each chunk.

    This function iterates through all supported document files within the 'data/documents'
    directory. For each file, it extracts the text content, breaks the content down
    into smaller, manageable chunks, and then uses a pre-configured embedding model
    (e.g., from Ollama) to generate a vector representation for each chunk.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a chunk
                    of document content with its corresponding vector embedding.
                    The format for each dictionary is expected to be:
                    [{'text': str, 'metadata_': {'doc': str, 'index': int}, 'embedding': list[float]}]
    """
    doc_list = get_document_filenames()
    chunks = []
    data_folder = Path("data/documents")
    for doc in doc_list:
        data = extract_to(f"{data_folder}/{doc}")
        file_name = data[0]
        file_content = data[1]
        chunks = chunk_text(file_content, file_name)
        for chunki in chunks:
            embedding = get_embedding_ollama(chunki['text'])
            chunki['embedding'] = embedding
    return chunks