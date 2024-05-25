from typing import Protocol
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader


class PdfLoader(Protocol):
    def load_pdf_data(self) -> list: ...


class PyMuPDFLoaderAdapter:
    def __init__(self, path_to_pdf: str):
        self.loader = PyMuPDFLoader(path_to_pdf)

    def load_pdf_data(self) -> list:
        data = self.loader.load()
        return data


class PyPDFLoaderAdapter:
    def __init__(self, path_to_pdf: str):
        self.loader = PyPDFLoader(path_to_pdf)

    def load_pdf_data(self) -> list:
        data = self.loader.load_and_split()
        return data


class PdfHandler:
    def __init__(self, pdf_loader: PdfLoader):
        self.pdf_loader = pdf_loader

    def load_pdf_data(self):
        data = self.pdf_loader.load_pdf_data()
        return data


def pdf_factory(pdf_loader: str, path_to_pdf: str):
    if pdf_loader == "pymupdf":
        return PyMuPDFLoaderAdapter(path_to_pdf)
    elif pdf_loader == "pypdf":
        return PyPDFLoaderAdapter(path_to_pdf)
    else:
        raise ValueError(f"Unknown pdf loader: {pdf_loader}")
