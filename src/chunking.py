from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 200,
    chunk_overlap: int = 100,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",   # Paragraph breaks
            "\n",     # Line breaks
            ". ",     # Sentence boundaries
            " ",      # Spaces
            ""        # Character level
        ],
        length_function=lambda text: len(text.split()),
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    return chunks



# from ingestion.data_loader import load_documents

# if __name__ == "__main__":

#     DATA_PATH = r"D:\AI\RAG\Projects\data"

#     documents = load_documents(DATA_PATH)

#     chunked_docs = chunk_documents(
#         documents,
#         chunk_size=200,
#         chunk_overlap=50
#     )

#     print(f"Loaded {len(documents)} pages")
#     print(f"Created {len(chunked_docs)} chunks\n")

#     for i, doc in enumerate(chunked_docs[:5]):
#         print(f"Chunk {i+1}")
#         print("Metadata:", doc.metadata)
#         print(doc.page_content[:300])
#         print("-" * 50)