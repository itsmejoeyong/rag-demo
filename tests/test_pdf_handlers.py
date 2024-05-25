from unittest.mock import MagicMock, patch
import pytest

from src.pdf_handlers import (
    PyMuPDFLoaderAdapter,
    PyPDFLoaderAdapter,
    PdfHandler,
    pdf_factory,
)

DUMMY_PAGE = ["page1", "page2"]
DUMMY_PATH = "path/to_pdf.pdf"


@pytest.fixture
def pymupdf_loader_adapter():
    dummy_path = DUMMY_PATH
    mock_loader = MagicMock()
    mock_loader.load.return_value = DUMMY_PAGE

    with patch("src.pdf_handlers.PyMuPDFLoader", return_value=mock_loader):
        yield PyMuPDFLoaderAdapter(dummy_path)


@pytest.fixture
def pypdf_loader_adapter():
    dummy_path = DUMMY_PATH
    mock_loader = MagicMock()
    mock_loader.load_and_split.return_value = DUMMY_PAGE

    with patch("src.pdf_handlers.PyPDFLoader", return_value=mock_loader):
        yield PyPDFLoaderAdapter(dummy_path)


@pytest.fixture
def pdf_handler_pymupdf_loader(pymupdf_loader_adapter):
    return PdfHandler(pymupdf_loader_adapter)


@pytest.fixture
def pdf_handler_pypdf_loader(pypdf_loader_adapter):
    return PdfHandler(pypdf_loader_adapter)


def test_pymupdf_load_pdf_data(pymupdf_loader_adapter):
    data = pymupdf_loader_adapter.load_pdf_data()
    assert len(data) == 2
    assert data == DUMMY_PAGE


def test_pypdf_load_pdf_data(pypdf_loader_adapter):
    data = pypdf_loader_adapter.load_pdf_data()
    assert len(data) == 2
    assert data == DUMMY_PAGE


def test_pdf_handler_pymupdf(pdf_handler_pymupdf_loader):
    data = pdf_handler_pymupdf_loader.load_pdf_data()
    assert len(data) == 2
    assert data == DUMMY_PAGE


def test_pdf_handler_pypdf(pdf_handler_pypdf_loader):
    data = pdf_handler_pypdf_loader.load_pdf_data()
    assert len(data) == 2
    assert data == ["page1", "page2"]


def test_pdf_factory_pymupdf():
    with patch(
        "src.pdf_handlers.PyMuPDFLoaderAdapter", return_value=PyMuPDFLoaderAdapter
    ):
        loader = pdf_factory("pymupdf", DUMMY_PATH)
        assert loader == PyMuPDFLoaderAdapter


def test_pdf_factory_pypdf():
    with patch("src.pdf_handlers.PyPDFLoaderAdapter", return_value=PyPDFLoaderAdapter):
        loader = pdf_factory("pypdf", DUMMY_PATH)
        assert loader == PyPDFLoaderAdapter


def test_pdf_factory_error():
    with pytest.raises(ValueError) as e:
        pdf_factory("doesn't exist", DUMMY_PATH)
        assert "Unknown pdf loader" in str(e.value)
