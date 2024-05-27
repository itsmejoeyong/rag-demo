from typing import cast

import voyageai
from chromadb import Documents, EmbeddingFunction, Embeddings


# Created a custom VoyageAIEmbeddingFunction as of 2024-05-27
# voyageai docs: https://docs.voyageai.com/docs/embeddings
# chromadb OpenAIEmbeddingFunction source code line 112: https://github.com/chroma-core/chroma/blob/main/chromadb/utils/embedding_functions.py
class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, api_key: str, model_name: str, input_type: str):
        self.api_key = api_key
        self.model_name = model_name
        self.input_type = input_type
        self._client = voyageai.Client(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        input = [text.replace("\n", " ") for text in input]
        result = self._client.embed(
            texts=input, model=self.model_name, input_type=self.input_type
        )
        return cast(Embeddings, result.embeddings)
