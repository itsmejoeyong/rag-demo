import os
from unittest.mock import MagicMock, patch
from src.text_handlers import (
    SemanticSplitterAdapter,
    SentenceSplitterAdapter,
    TextHandlers,
    text_splitter_factory,
)
from langchain.docstore.document import Document

import pytest

DUMMY_DATA = dummy_data = (
    "dignissim cras tincidunt lobortis feugiat vivamus at augue eget arcu dictum varius duis at consectetur lorem donec massa sapien faucibus et molestie ac feugiat sed lectus vestibulum mattis ullamcorper velit sed ullamcorper morbi tincidunt ornare massa eget egestas purus viverra accumsan in nisl nisi scelerisque eu ultrices vitae auctor eu augue ut lectus arcu bibendum at varius vel pharetra vel turpis nunc eget lorem dolor sed viverra ipsum nunc aliquet bibendum enim facilisis gravida neque convallis a cras semper auctor neque vitae tempus quam pellentesque nec nam aliquam sem et tortor consequat id porta nibh venenatis cras sed felis eget"
)


@pytest.fixture
def sentence_splitter_adapter():
    return SentenceSplitterAdapter(DUMMY_DATA, 400, 0)


@pytest.fixture
def semantic_splitter_adapter():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-api-key"}):
        mock_embeddings = MagicMock()
        with patch("src.text_handlers.OpenAIEmbeddings", return_value=mock_embeddings):
            mock_splitter = MagicMock()
            mock_splitter.create_documents.return_value = [
                Document(page_content="chunk1"),
                Document(page_content="chunk2"),
            ]
            with patch("src.text_handlers.SemanticChunker", return_value=mock_splitter):
                yield SemanticSplitterAdapter(DUMMY_DATA, "dummy-model")


@pytest.fixture
def text_handler_sentence_splitter(sentence_splitter_adapter):
    return TextHandlers(sentence_splitter_adapter)


@pytest.fixture
def text_handler_semantic_splitter(semantic_splitter_adapter):
    return TextHandlers(semantic_splitter_adapter)


def test_sentence_splitter_split_text(sentence_splitter_adapter):
    page_contents = sentence_splitter_adapter.split_text()
    assert len(page_contents) == 2


def test_semantic_splitter_split_text(semantic_splitter_adapter):
    page_contents = semantic_splitter_adapter.split_text()
    assert len(page_contents) == 2
    assert page_contents == ["chunk1", "chunk2"]


def test_text_handler_sentence_splitter(text_handler_sentence_splitter):
    page_contents = text_handler_sentence_splitter.split_text()
    assert len(page_contents) == 2


def test_text_handler_semantic_splitter(text_handler_semantic_splitter):
    page_contents = text_handler_semantic_splitter.split_text()
    assert len(page_contents) == 2
    assert page_contents == ["chunk1", "chunk2"]


def test_text_splitter_factory_return_sentence_splitter():
    splitter = text_splitter_factory("sentence")
    assert splitter == SentenceSplitterAdapter


def test_text_splitter_factory_return_semantic_splitter():
    splitter = text_splitter_factory("semantic")
    assert splitter == SemanticSplitterAdapter


def test_text_splitter_factory_return_error():
    with pytest.raises(Exception) as e:
        text_splitter_factory("does not exist")
        assert "Unknown splitter" in str(e.value)
