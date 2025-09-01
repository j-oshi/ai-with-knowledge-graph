from pathlib import Path
from llm_ingestion.document_extractor import extract_to
from llm_ingestion.vector import get_embedding_ollama, chunk_text
from db_connector import insert_embeddings_to_db
from utils.document_utils import get_document_filenames


if __name__ == "__main__":
    doc_list = get_document_filenames()
    input_doc_paths = []
    chunks = []
    data_folder = Path("documents")
    for doc in doc_list:
        data = extract_to(f"{data_folder}/{doc}")
        file_name = data[0]
        file_content = data[1]
        chunks = chunk_text(file_content, file_name)
        for chunki in chunks:
            embedding = get_embedding_ollama(chunki['text'])
            chunki['embedding'] = embedding

    insert_embeddings_to_db(chunks)