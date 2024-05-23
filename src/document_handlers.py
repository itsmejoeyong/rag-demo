from langchain_community.document_loaders import PyMuPDFLoader


class PdfHandler:
    def load_pdf_data(self, path_to_pdf: str) -> list:
        loader = PyMuPDFLoader(path_to_pdf)
        data = loader.load()
        return data
