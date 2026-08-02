from ingestion.data_loader import load_documents
from src.chunking import chunk_documents
from src.embedding import JinaEmbedding
from src.vectorstore import QdrantStore

DATA_PATH = r"D:\AI\RAG\Projects\data"


def run_ingestion():

    # ==========================================================
    # Step 1 : Load Documents
    # ==========================================================
    print("=" * 60)
    print("Step 1 : Loading Documents...")
    print("=" * 60)

    documents = load_documents(DATA_PATH)

    print(f"Loaded {len(documents)} pages/documents.\n")

    # ==========================================================
    # Step 2 : Chunk Documents
    # ==========================================================
    print("=" * 60)
    print("Step 2 : Chunking Documents...")
    print("=" * 60)

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        chunk_overlap=50,
    )

    print(f"Created {len(chunks)} chunks.\n")

    # ==========================================================
    # Step 3 : Generate Embeddings
    # ==========================================================
    print("=" * 60)
    print("Step 3 : Generating Embeddings...")
    print("=" * 60)

    embedder = JinaEmbedding()

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedder.embed(texts)

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Embedding Dimension : {len(embeddings[0])}\n")

    # ==========================================================
    # Step 4 : Prepare Documents for Qdrant
    # ==========================================================
    print("=" * 60)
    print("Step 4 : Preparing Documents...")
    print("=" * 60)

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

    global_chunk_id = 0

    # ----------------------------------------------------------
    # Process each document separately
    # ----------------------------------------------------------
    for document_id, (source, chunk_list) in enumerate(grouped_documents.items()):

        total_chunks = len(chunk_list)

        for chunk_index, (chunk, embedding) in enumerate(chunk_list):

            previous_chunk = (
                global_chunk_id - 1
                if chunk_index > 0
                else None
            )

            next_chunk = (
                global_chunk_id + 1
                if chunk_index < total_chunks - 1
                else None
            )

            documents_to_store.append(
                {
                    "chunk_id": global_chunk_id,

                    "document_id": document_id,

                    "chunk_index": chunk_index,

                    "previous_chunk": previous_chunk,

                    "next_chunk": next_chunk,

                    "total_chunks": total_chunks,

                    "text": chunk.page_content,

                    "embedding": embedding,

                    "source": source,

                    "page": chunk.metadata.get("page", 0),

                    "section": chunk.metadata.get(
                        "section",
                        "Unknown",
                    ),
                }
            )

            global_chunk_id += 1

    print(f"Prepared {len(documents_to_store)} vectors.\n")

    # ==========================================================
    # Step 5 : Upload to Qdrant
    # ==========================================================
    print("=" * 60)
    print("Step 5 : Uploading to Qdrant...")
    print("=" * 60)

    store = QdrantStore()

    store.create_collection(
        vector_size=len(embeddings[0])
    )

    store.upload_documents(
        documents_to_store
    )

    print("\nIngestion Completed Successfully.\n")

    print("=" * 60)
    print(f"Total Pages Loaded   : {len(documents)}")
    print(f"Total Chunks Created : {len(chunks)}")
    print(f"Total Embeddings     : {len(embeddings)}")
    print(f"Stored in Collection : {store.collection_name}")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()