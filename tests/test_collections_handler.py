from unittest.mock import MagicMock, patch
import pytest

from src.collections_handler import CollectionsHandler


@pytest.fixture
def mock_client_api_method():
    with patch(
        "src.collections_handler.ClientAPI.get_or_create_collection"
    ) as MockClientAPIMethod:
        MockClientAPIMethod.return_value = MagicMock()
        yield MockClientAPIMethod


@pytest.fixture
def mock_embeddings_function():
    MockEmbeddingsFunction = MagicMock()
    return MockEmbeddingsFunction


@pytest.fixture
def mock_query():
    with patch("src.collections_handler.Collection.query") as MockQuery:
        MockQuery.return_value = {"data": "your result"}
        yield MockQuery


@pytest.fixture
def mock_upsert():
    with patch("src.collections_handler.Collection.upsert") as MockUpsert:
        yield MockUpsert


@pytest.fixture
def mock_delete():
    with patch("src.collections_handler.Collection.delete") as MockDelete:
        yield MockDelete


def test_collections_handler_init_collection(
    mock_client_api_method, mock_embeddings_function
):
    MockClient = MagicMock()
    MockClient.get_or_create_collection = mock_client_api_method

    handler = CollectionsHandler()
    handler.init_collection(
        chroma_client=MockClient,
        collection_name="test",
        embeddings_function=mock_embeddings_function,
    )

    assert isinstance(handler.collection, MagicMock)
    mock_client_api_method.assert_called_once_with(
        name="test",
        embedding_function=mock_embeddings_function,
    )


def test_collections_handler_query(mock_query):
    handler = CollectionsHandler()
    handler.collection = MagicMock()
    handler.collection.query = mock_query
    result = handler.query(query=["test"], n_results=1)

    assert result == {"data": "your result"}
    handler.collection.query.assert_called_once_with(
        query_texts=["test"], n_results=1, where=None
    )


def test_collections_handler_upsert(mock_upsert):
    handler = CollectionsHandler()
    handler.collection = MagicMock()
    handler.collection.upsert = mock_upsert
    # by right there is not result, just testing implementation
    handler.upsert(
        documents=["doc1", "doc2"],
        metadatas=[{"type": "mock"}, {"type": "test"}],
        ids=["id1", "id2"],
    )

    handler.collection.upsert.assert_called_once_with(
        documents=["doc1", "doc2"],
        metadatas=[{"type": "mock"}, {"type": "test"}],
        ids=["id1", "id2"],
    )


def test_collections_handler_delete(mock_delete):
    handler = CollectionsHandler()
    handler.collection = MagicMock()
    handler.collection.delete = mock_delete

    handler.delete(where_clause={"type": "not mock"})

    handler.collection.delete.assert_called_once_with(where={"type": "not mock"})
    with pytest.raises(Exception) as e:
        handler.delete()
        assert str(e.value) in "Please provide a where condition"
