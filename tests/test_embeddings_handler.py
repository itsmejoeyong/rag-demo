from unittest.mock import MagicMock, patch
import pytest

from src.custom_embeddings_function.voyageai_embeddings_function import (
    VoyageAIEmbeddingFunction,
)
from src.embeddings_handler import (
    EmbeddingsHandler,
    OpenAIEmbeddingsAdapter,
    VoyageAIEmbeddingsAdapter,
    embeddings_factory,
)


@pytest.fixture
def voyageai_embeddings_adapter():
    with patch(
        "src.embeddings_handler.VoyageAIEmbeddingFunction"
    ) as MockVoyageAIEmbeddingFunction:
        yield MockVoyageAIEmbeddingFunction


@pytest.fixture
def openai_embeddings_adapter():
    with patch(
        "src.embeddings_handler.embedding_functions.OpenAIEmbeddingFunction"
    ) as MockOpenAIEmbeddingFunction:
        yield MockOpenAIEmbeddingFunction


@pytest.fixture
def embeddings_handler_voyageai(voyageai_embeddings_adapter):
    return EmbeddingsHandler(voyageai_embeddings_adapter)


@pytest.fixture
def embeddings_handler_openai(openai_embeddings_adapter):
    return EmbeddingsHandler(openai_embeddings_adapter)


def test_voyageai_init_embedding_function(voyageai_embeddings_adapter):
    adapter = VoyageAIEmbeddingsAdapter(
        api_key="test_api_key", model_name="test_model", input_type="test_input_type"
    )
    voyageai_ef = adapter.init_embedding_function()

    assert isinstance(voyageai_ef, MagicMock)
    voyageai_embeddings_adapter.assert_called_once_with(
        api_key="test_api_key", model_name="test_model", input_type="test_input_type"
    )


def test_openai_init_embeddings_function(openai_embeddings_adapter):
    adapter = OpenAIEmbeddingsAdapter(
        api_key="test_api_key", model_name="test_model_name"
    )

    openai_ef = adapter.init_embeddings_function()

    assert isinstance(openai_ef, MagicMock)
    openai_embeddings_adapter.assert_called_once_with(
        api_key="test_api_key",
        model_name="test_model_name",
        api_base=None,
        api_type=None,
        api_version=None,
    )


def test_embeddings_handler_voyageai(embeddings_handler_voyageai):
    ef = embeddings_handler_voyageai.init_embedding_function()
    assert isinstance(ef, MagicMock)
    assert isinstance(embeddings_handler_voyageai.embeddings_function, MagicMock)


def test_embeddings_handler_openai(embeddings_handler_openai):
    ef = embeddings_handler_openai.init_embedding_function()
    assert isinstance(ef, MagicMock)
    assert isinstance(embeddings_handler_openai.embeddings_function, MagicMock)


def test_embeddings_factory():
    ef_mapping = {
        "openai": OpenAIEmbeddingsAdapter,
        "voyageai": VoyageAIEmbeddingsAdapter,
    }
    for key, value in ef_mapping.items():
        assert embeddings_factory(key) == value


def test_embeddings_factory_error():
    with pytest.raises(Exception) as e:
        embeddings_factory("not implemented")
        assert str(e.value) in "Unknown embeddings function"
