from llm_ingestion.html_extractor import get_site_content, scrap_site_content
from llm_ingestion.vector import get_embedding_ollama, chunk_text
# from db_connector import insert_embeddings_to_db
from utils.document_utils import get_word_length

# import tiktoken
import ast
import pgvector
import math


SITE_NAME = "https://akilibookkeeping.com/"
if __name__ == "__main__":
    chunks = []
    site_content = get_site_content(SITE_NAME)
    # site_content = scrap_site_content("https://example.com/")
    # site_content = scrap_site_content(SITE_NAME)

    chunks = chunk_text(site_content, SITE_NAME)
    for chunki in chunks:
        embedding = get_embedding_ollama(chunki['text'])
        chunki['embedding'] = embedding
    print(chunks)
    # if len(chunks) > 0:
    #     insert_embeddings_to_db(chunks)