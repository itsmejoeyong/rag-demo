from unittest.mock import MagicMock, patch
import pytest

from src.database_handler import DatabaseHandler


@pytest.fixture
def database_handler_client():
    with patch("src.database_handler.Client") as MockClient:
        MockClient.return_value = MagicMock()
        yield MockClient


@pytest.fixture
def database_handler_persistent_client():
    with patch("src.database_handler.PersistentClient") as MockClient:
        MockClient.return_value = MagicMock()
        yield MockClient


def test_database_handler_client(database_handler_client):
    handler = DatabaseHandler()

    assert isinstance(handler.chroma_client, MagicMock)
    assert handler.is_persistent is False
    assert handler.database_path == "databases/"
    database_handler_client.assert_called_once_with()


def test_database_handler_persistent_client(database_handler_persistent_client):
    handler = DatabaseHandler(is_persistent=True, database_path="test/")

    assert isinstance(handler.chroma_client, MagicMock)
    assert handler.is_persistent is True
    assert handler.database_path == "test/"
    database_handler_persistent_client.assert_called_once_with(path="test/")
