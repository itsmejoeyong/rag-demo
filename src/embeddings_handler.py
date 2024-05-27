from typing import Protocol
import chromadb.utils.embedding_functions as embedding_functions

from src.custom_embeddings_function.voyageai_embeddings_function import (
    VoyageAIEmbeddingFunction,
)


class EmbeddingsFunction(Protocol):
    def init_embedding_function(self): ...


# Implementing custom voyage ai logic as adapter as of 2024-05-27
class VoyageAIEmbeddingsAdapter:
    def __init__(self, api_key: str, model_name: str, input_type: str = "document"):
        self.api_key = api_key
        self.model_name = model_name
        self.input_type = input_type

    def init_embedding_function(self):
        voyageai_ef = VoyageAIEmbeddingFunction(
            api_key=self.api_key,
            model_name=self.model_name,
            input_type=self.input_type,
        )
        return voyageai_ef


# Implementing built in module as adapter
class OpenAIEmbeddingsAdapter:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_base: str = None,  # type: ignore
        api_type: str = None,  # type: ignore
        api_version: str = None,  # type: ignore
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        self.api_type = api_type
        self.api_version = api_version

    def init_embeddings_function(self):
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name=self.model_name,
            api_base=self.api_base,
            api_type=self.api_type,
            api_version=self.api_version,
        )
        return openai_ef


class EmbeddingsHandler:
    def __init__(self, embeddings_function: EmbeddingsFunction):
        self.embeddings_function = embeddings_function

    def init_embedding_function(self):
        embedding_function = self.embeddings_function.init_embedding_function()
        return embedding_function


def embeddings_factory(embeddings_function: str):
    if embeddings_function == "openai":
        return OpenAIEmbeddingsAdapter
    elif embedding_functions == "voyageai":
        return VoyageAIEmbeddingsAdapter
    else:
        raise ValueError(f"Unknown embeddings function: {embeddings_function}")
