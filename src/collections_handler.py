from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction


class CollectionsHandler:
    def __init__(self):
        self.collection: Collection = None  # type: ignore

    def init_collection(
        self,
        chroma_client: ClientAPI,
        collection_name: str,
        embeddings_function: EmbeddingFunction = None,  # type: ignore
    ):
        # i need to structure it this way as embedding_function has a default parameter
        if not embeddings_function:
            self.collection = chroma_client.get_or_create_collection(
                name=collection_name
            )
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=embeddings_function
        )

    def query(self, query, n_results, where_clause: dict = None):  # type: ignore
        return self.collection.query(
            query_texts=query, n_results=n_results, where=where_clause
        )

    def upsert(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)  # type: ignore

    def delete(self, where_clause: dict = None):  # type: ignore
        if not where_clause:
            raise ValueError(
                "Please provide a where condition to avoid deleting the whole collections"
            )
        self.collection.delete(where=where_clause)
