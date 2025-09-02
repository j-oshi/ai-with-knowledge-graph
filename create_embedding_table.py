import asyncio
from db_connector import create_embedding_table, install_vector_extension, create_index

async def install_estension_and_generate_table():  
    # install_vector_extension()
    # create_embedding_table()
    create_index()

if __name__ == "__main__":
    asyncio.run(install_estension_and_generate_table())