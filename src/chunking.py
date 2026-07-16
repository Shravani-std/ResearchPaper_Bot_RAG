from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,

):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        separators=[
            "\n\n",   # Paragraph breaks
            "\n",     # Line breaks
            " ",      # Spaces
            ""        # Character level
            ],

        length_function=len,
        is_separator_regex=False
    )

    chunks = splitter.split_text(text)
    return chunks

# if __name__ == "__main__":
#     sample_text = (
#         "This is a sample text that will be split into chunks. "
#         "The text splitter will use the specified chunk size and overlap to create manageable pieces of text. "
#         "This is useful for processing large documents or texts in smaller segments."
#     )

#     chunks = chunk_text(sample_text)
#     print(f"Number of chunks: {len(chunks)}")
#     for i, chunk in enumerate(chunks):
#         print(f"Chunk {i + 1}: {chunk}")

# print("chunking.py is running")
