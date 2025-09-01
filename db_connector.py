import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

POSTGRES_HOST = "localhost"
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DBNAME = os.getenv('POSTGRES_DBNAME')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')


def insert_embeddings_to_db(data, table_name="embeddings_table"):
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DBNAME,
            user=POSTGRES_USERNAME,
            password=POSTGRES_PASSWORD
        )
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