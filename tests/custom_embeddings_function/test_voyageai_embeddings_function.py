from unittest.mock import MagicMock, patch
from src.custom_embeddings_function.voyageai_embeddings_function import (
    VoyageAIEmbeddingFunction,
)

import pytest


@pytest.fixture
def mock_voyage_client():
    mock_client = MagicMock()
    mock_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2], [0.3, 0.4]])
    with patch(
        "src.custom_embeddings_function.voyageai_embeddings_function.voyageai.Client",
        return_value=mock_client,
    ):
        yield mock_client


def test_voyageai_embedding_function_init(mock_voyage_client):
    embedding_function = VoyageAIEmbeddingFunction(
        api_key="test", model="test_model", input_type="test"
    )
    assert embedding_function.api_key == "test"
    assert embedding_function.model == "test_model"
    assert embedding_function.input_type == "test"
    assert embedding_function._client == mock_voyage_client


def test_voyageai_embedding_function_call(mock_voyage_client):
    embedding_function = VoyageAIEmbeddingFunction(
        api_key="test", model="test_model", input_type="test"
    )
    docs = ["Hello", "world!"]
    embeddings = embedding_function(docs)
    assert isinstance(embeddings, list)
    assert all(isinstance(embedding, list) for embedding in embeddings)
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    mock_voyage_client.embed.assert_called_once_with(
        texts=docs, model="test_model", input_type="test"
    )
