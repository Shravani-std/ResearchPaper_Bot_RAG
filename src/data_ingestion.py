from src.chunking import chunk_text
from src.embedding import JinaEmbedding
from utils.helpers import batch_chunks
from ingestion.text_extraction import extract_text_from_pdf


def ingest_document(
    pdf_path: str,
    batch_size: int = 50,
):
    """
    Complete ingestion pipeline.

    Steps:
    1. Extract text
    2. Chunk text
    3. Generate embeddings
    4. Return chunks and embeddings
    """

    # Step 1: Extract text
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk text
    chunks = chunk_text(text)

    # Step 3: Initialize embedder
    embedder = JinaEmbedding()

    all_vectors = []

    # Step 4: Generate embeddings in batches
    for batch in batch_chunks(chunks, batch_size=batch_size):
        vectors = embedder.embed(batch)
        all_vectors.extend(vectors)

    return chunks, all_vectors


if __name__ == "__main__":
    pdf_path = r"D:\AI\RAG\Projects\data\2601.10034v2.pdf"
    chunks, embeddings = ingest_document(pdf_path)
    print(f"[DEBUG] Total chunks: {len(chunks)}")
    print(f"[DEBUG] Total embeddings: {embeddings}")