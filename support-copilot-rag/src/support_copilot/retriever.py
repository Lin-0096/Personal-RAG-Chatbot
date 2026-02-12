from chromadb import Client
from chromadb.config import Settings
from typing import List, Tuple

class Retriever:
    def __init__(self, db_path: str):
        self.client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=db_path))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        collection = self.client.get_collection("knowledge_base")
        results = collection.query(query=query, n_results=top_k)
        return [(result['document'], result['score']) for result in results['documents']]