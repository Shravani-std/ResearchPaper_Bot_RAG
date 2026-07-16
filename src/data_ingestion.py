from pathlib import Path
from typing import List, Dict, Any

from ingestion.text_extraction import extract_text_from_pdf
from src.chunking import chunk_text
from src.embedding import JinaEmbedding
from src.vectorstore import QdrantStore
from utils.helpers import batch_chunks


def ingest_document(pdf_path: str) -> List[Dict[str, Any]]:
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return []

    chunks = chunk_text(text)
    embedder = JinaEmbedding()

    all_vectors = []
    for batch in batch_chunks(chunks):
        vectors = embedder.embed(batch)
        all_vectors.extend(vectors)

    source = Path(pdf_path).name
    documents = []

    for idx, (chunk, vector) in enumerate(zip(chunks, all_vectors)):
        documents.append(
            {
                "chunk_id": idx,
                "text": chunk,
                "embedding": vector,
                "source": source,
            }
        )

    store = QdrantStore()
    store.create_collection()
    store.upload_documents(documents)
    return documents