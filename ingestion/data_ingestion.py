from ingestion.data_loader import load_documents
from src.chunking import chunk_documents
from src.embedding import JinaEmbedding
from src.vectorstore import QdrantStore
from core.logger import get_logger, log_step
import uuid

logger = get_logger("ingestion")

DATA_PATH = r"D:\AI\RAG\Projects\data"


def run_ingestion(pdf_path: str = None):

    path_to_load = pdf_path if pdf_path else DATA_PATH

    # ==========================================================
    # Step 1 : Load Documents
    # ==========================================================
    log_step("=" * 60)
    log_step("Step 1 : Loading Documents...")
    log_step("=" * 60)

    documents = load_documents(path_to_load)

    log_step(f"Loaded {len(documents)} pages/documents.")

    # ==========================================================
    # Step 2 : Chunk Documents
    # ==========================================================
    log_step("=" * 60)
    log_step("Step 2 : Chunking Documents...")
    log_step("=" * 60)

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        chunk_overlap=50,
    )

    log_step(f"Created {len(chunks)} chunks.")

    # ==========================================================
    # Step 3 : Generate Embeddings
    # ==========================================================
    log_step("=" * 60)
    log_step("Step 3 : Generating Embeddings...")
    log_step("=" * 60)

    embedder = JinaEmbedding()

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedder.embed(texts)

    log_step(f"Generated {len(embeddings)} embeddings.")
    log_step(f"Embedding Dimension : {len(embeddings[0])}")

    # ==========================================================
    # Step 4 : Prepare Documents for Qdrant
    # ==========================================================
    log_step("=" * 60)
    log_step("Step 4 : Preparing Documents...")
    log_step("=" * 60)

    documents_to_store = []

    # ----------------------------------------------------------
    # Group chunks by source document
    # ----------------------------------------------------------
    grouped_documents = {}

    for chunk, embedding in zip(chunks, embeddings):

        source = chunk.metadata.get("source", "")

        if source not in grouped_documents:
            grouped_documents[source] = []

        grouped_documents[source].append((chunk, embedding))












    for document_id, (source, chunk_list) in enumerate(grouped_documents.items()):

        total_chunks = len(chunk_list)

        # Generate a unique id for every chunk in this document up front,
        # so previous/next can reference them before we loop
        chunk_ids = [str(uuid.uuid4()) for _ in range(total_chunks)]

        for chunk_index, (chunk, embedding) in enumerate(chunk_list):

            current_chunk_id = chunk_ids[chunk_index]

            previous_chunk = (
                chunk_ids[chunk_index - 1]
                if chunk_index > 0
                else None
            )

            next_chunk = (
                chunk_ids[chunk_index + 1]
                if chunk_index < total_chunks - 1
                else None
            )

            documents_to_store.append(
                {
                    "chunk_id": current_chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "previous_chunk": previous_chunk,
                    "next_chunk": next_chunk,
                    "total_chunks": total_chunks,
                    "text": chunk.page_content,
                    "embedding": embedding,
                    "source": source,
                    "page": chunk.metadata.get("page", 0),
                    "section": chunk.metadata.get("section", "Unknown"),
                }
            )


















    log_step(f"Prepared {len(documents_to_store)} vectors.")

    # ==========================================================
    # Step 5 : Upload to Qdrant
    # ==========================================================
    log_step("=" * 60)
    log_step("Step 5 : Uploading to Qdrant...")
    log_step("=" * 60)

    store = QdrantStore()

    store.create_collection(
        vector_size=len(embeddings[0])
    )

    store.upload_documents(
        documents_to_store
    )

    log_step("Ingestion Completed Successfully.")
    log_step("=" * 60)
    log_step(f"Total Pages Loaded   : {len(documents)}")
    log_step(f"Total Chunks Created : {len(chunks)}")
    log_step(f"Total Embeddings     : {len(embeddings)}")
    log_step(f"Stored in Collection : {store.collection_name}")
    log_step("=" * 60)


if __name__ == "__main__":
    run_ingestion(DATA_PATH)