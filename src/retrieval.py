from __future__ import annotations

from typing import Any, List

from src.embedding import JinaEmbedding
from src.vectorstore import QdrantStore


class Retriever:
    def __init__(self) -> None:
        self.embedder = JinaEmbedding()
        self.vectorstore = QdrantStore()

    def retrieve(self, question: str, top_k: int = 5) -> List[Any]:
        query_vector = self.embedder.embed([question])[0]
        return self.vectorstore.search(query_vector=query_vector, top_k=top_k)