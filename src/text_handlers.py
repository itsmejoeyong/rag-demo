from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextHandler:
    def split_text_recursively(
        self,
        chunk_size: int,
        chunk_overlap: int,
        data: str,
        separators: list[str] = None,  # type: ignore
        keep_separator: bool = True,
        length_function=len,
    ) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            length_function=length_function,
            is_separator_regex=False,
        )

        texts = text_splitter.create_documents([data])
        page_contents = [content.page_content for content in texts]
        return page_contents
