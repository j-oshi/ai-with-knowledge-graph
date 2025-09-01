import asyncio
from ingestion.document_ingestor import doc_to_vector
from db_connector import insert_embeddings_to_db
from utils.decorators import timer_decorator



@timer_decorator
async def execute_conversion():  
    print('start conversion') 
    embedded_text = doc_to_vector()
    print(embedded_text)
    insert_embeddings_to_db(embedded_text)

if __name__ == "__main__":
    asyncio.run(execute_conversion())