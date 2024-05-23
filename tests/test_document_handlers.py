from unittest.mock import MagicMock, patch
from src.document_handlers import PdfHandler

import pytest


@pytest.fixture
def pdf_handler():
    return PdfHandler()


def test_load_pdf_data(pdf_handler):
    mock_file_path = "dummy/path.pdf"

    pymupdf_mock = MagicMock()
    pymupdf_mock.load.return_value = ["page1", "page2"]

    with patch("src.document_handlers.PyMuPDFLoader", return_value=pymupdf_mock):
        data = pdf_handler.load_pdf_data(mock_file_path)

        assert data == ["page1", "page2"]
        pymupdf_mock.load.assert_called_once()
        pymupdf_mock.load.assert_called_with()
