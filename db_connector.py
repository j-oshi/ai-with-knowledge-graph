import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from psycopg2 import OperationalError
from psycopg2.extras import Json
import numpy as np
from colorama import Fore, Style


load_dotenv()

POSTGRES_HOST = "localhost"
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DBNAME = os.getenv('POSTGRES_DBNAME')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')

def connect_pg():
    connect = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DBNAME,
        user=POSTGRES_USERNAME,
        password=POSTGRES_PASSWORD
    )
    return connect

def check_db_connection():
    """
    Attempts to establish a connection to a PostgreSQL database.

    Args:
        None

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    conn = None
    try:
        conn = connect_pg()
        print("Connection to the PostgreSQL database successful!")
        return True
    except OperationalError as e:
        print(Fore.RED+ f"The connection to the database failed.\nError: {e}")
        print(Style.RESET_ALL)
        return False
    finally:
        if conn:
            conn.close()


def insert_embeddings_to_db(data, table_name="embeddings_table"):
    try:
        conn = connect_pg()
        cursor = conn.cursor()

        sql = f"""
        INSERT INTO {table_name} (text_column, doc_name_column, embedding_column)
        VALUES %s
        """
        
        values = [(row['text'], row['metadata_']['doc'], row['embedding']) for row in data]
        
        # Use execute_values for bulk insertion
        execute_values(cursor, sql, values)
        
        conn.commit()
        print(f"Successfully inserted {len(values)} rows into {table_name}.")

    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print(f"Database error occurred: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("Database connection closed.")

def get_top_k_similar_docs(query_embedding: list, k: int = 3) -> list:
    """
    Connects to the database and retrieves the top-k most similar documents.
    """
    if not query_embedding:
        return []

    conn = None
    try:
        conn = connect_pg()
        # Register pgvector extension once per connection
        register_vector()
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
