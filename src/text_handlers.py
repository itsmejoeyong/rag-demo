from typing import Protocol
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


class TextSplitters(Protocol):
    def split_text(self) -> list: ...


class SentenceSplitterAdapter:
    def __init__(
        self,
        data: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str] = None,  # type: ignore
        keep_separator: bool = True,
        length_function=len,
    ):
        self.data = data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.keep_separator = keep_separator
        self.length_function = length_function

    def split_text(self) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=self.keep_separator,
            length_function=self.length_function,
            is_separator_regex=False,
        )

        texts = text_splitter.create_documents([self.data])
        page_contents = [content.page_content for content in texts]
        return page_contents


class SemanticSplitterAdapter:
    def __init__(self, data: str, model: str):
        self.model = model
        self.data = data

    def split_text(self) -> list[str]:
        # please set os.environ["OPENAI_API_KEY"]
        text_splitter = SemanticChunker(OpenAIEmbeddings(model=self.model))
        texts = text_splitter.create_documents([self.data])
        page_content = [content.page_content for content in texts]
        return page_content


class TextHandlers:
    def __init__(self, spliter: TextSplitters):
        self.spliter = spliter

    def split_text(self):
        page_content = self.spliter.split_text()
        return page_content


def text_splitter_factory(splitter: str):
    if splitter == "sentence":
        return SentenceSplitterAdapter
    elif splitter == "semantic":
        return SemanticSplitterAdapter
    else:
        raise ValueError(f"Unknown splitter: {splitter}")
