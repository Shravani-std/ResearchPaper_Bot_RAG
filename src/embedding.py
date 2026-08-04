from __future__ import annotations

import math
import re
import time
from typing import List

import requests

from core.config import settings
from core.logger import get_logger

logger = get_logger("embedding")


class JinaEmbedding:
    BASE_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        model: str = "jina-embeddings-v3",
        batch_size: int = 20,
        max_retries: int = 3,
    ) -> None:

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

        self.headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: List[str]) -> List[List[float]]:

        if not texts:
            return []

        texts = [text if text else "" for text in texts]

        if not settings.JINA_API_KEY:
            logger.warning("JINA_API_KEY not found. Using fallback embeddings.")
            return [self._fallback_embedding(text) for text in texts]

        embeddings = []

        for start in range(0, len(texts), self.batch_size):

            batch = texts[start:start + self.batch_size]

            payload = {
                "model": self.model,
                "input": batch
            }

            success = False

            for attempt in range(self.max_retries):

                try:

                    response = requests.post(
                        self.BASE_URL,
                        headers=self.headers,
                        json=payload,
                        timeout=60,
                    )

                    response.raise_for_status()

                    data = response.json()

                    batch_embeddings = [
                        item["embedding"]
                        for item in data["data"]
                    ]

                    embeddings.extend(batch_embeddings)

                    success = True

                    # Small delay to avoid rate limit
                    time.sleep(1)

                    break

                except requests.RequestException as e:

                    logger.warning(
                        "Embedding batch %s attempt %s/%s failed",
                        start // self.batch_size + 1,
                        attempt + 1,
                        self.max_retries,
                    )
                    logger.warning(str(e))

                    if (
                        hasattr(e, "response")
                        and e.response is not None
                        and e.response.status_code == 429
                    ):

                        retry_after = e.response.headers.get("Retry-After")
                        wait_time = int(retry_after) if retry_after else 30

                        logger.warning(
                            "Rate limit reached while embedding. Waiting %s seconds.",
                            wait_time,
                        )
                        time.sleep(wait_time)

                    else:

                        if attempt < self.max_retries - 1:
                            time.sleep(2)

            if not success:
                raise RuntimeError(
                    f"Failed to generate embeddings for batch {start // self.batch_size + 1}"
                )

        return embeddings

    @staticmethod
    def _fallback_embedding(text: str) -> List[float]:

        tokens = re.findall(r"\w+", text.lower())

        vector = [0.0] * 1024

        for token in tokens:
            idx = sum(ord(c) for c in token) % 1024
            vector[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))

        if norm == 0:
            return vector

        return [v / norm for v in vector]


# if __name__ == "__main__":

    # from ingestion.data_loader import load_documents
    # from src.chunking import chunk_documents

    # DATA_PATH = r"D:\AI\RAG\Projects\data"

    # print("Loading documents...")
    # documents = load_documents(DATA_PATH)

    # print("Chunking documents...")
    # chunks = chunk_documents(documents)

    # # Test only first 5 chunks
    # chunks = chunks[:5]

    # texts = [doc.page_content for doc in chunks]

    # print("Generating embeddings...")

    # embedder = JinaEmbedding()

    # embeddings = embedder.embed(texts)

    # print("=" * 60)
    # print(f"Chunks: {len(chunks)}")
    # print(f"Embeddings: {len(embeddings)}")

    # if embeddings:
    #     print(f"Embedding Dimension: {len(embeddings[0])}")

    # print("=" * 60)

    # for i, chunk in enumerate(chunks):

    #     print(f"\nChunk {i + 1}")
    #     print(f"Source : {chunk.metadata.get('source')}")
    #     print(f"Page   : {chunk.metadata.get('page')}")
    #     print(f"Words  : {len(chunk.page_content.split())}")
    #     print(f"Vector Dimension : {len(embeddings[i])}")

    #     print("-" * 60)