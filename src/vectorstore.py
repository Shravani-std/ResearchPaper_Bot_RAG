from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Any
from core.config import settings
from core.logger import get_logger

logger = get_logger("vectorstore")



class QdrantStore:
    def __init__(self):

        self.collection_name = settings.COLLECTION_NAME or "AIResearch_Bot"

        # self.client = QdrantClient(
        #     url=settings.QDRANT_URL,
        #     api_key=settings.QDRANT_API_KEY,
        # )
        if settings.QDRANT_API_KEY:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
            )
        # print(self.client.get_collections())

    def create_collection(self, vector_size: int):

        collections = self.client.get_collections().collections

        names = [c.name for c in collections]

        if self.collection_name in names:
            logger.info("Collection '%s' already exists.", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        logger.info("Collection '%s' created.", self.collection_name)

    def upload_documents(
    self,
    documents: List[dict[str, Any]],
    batch_size: int = 100,
):

        total = len(documents)

        for start in range(0, total, batch_size):

            batch = documents[start:start + batch_size]

            points = []

            for doc in batch:

                points.append(
                    PointStruct(
                        id=doc["chunk_id"],
                        vector=doc["embedding"],
                        payload={
                            "text": doc["text"],
                            "source": doc["source"],
                            "page": doc["page"],
                            "section": doc["section"],
                            "document_id":doc["document_id"],
                            "chunk_id": doc["chunk_id"],
                            "chunk_index": doc["chunk_index"],
                            "previous_chunk": doc["previous_chunk"],
                            "next_chunk": doc["next_chunk"],
                            "total_chunks": doc["total_chunks"],
                        },
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(
                "Uploaded %s/%s vectors to collection '%s'",
                min(start + batch_size, total),
                total,
                self.collection_name,
            )

        logger.info("All vectors uploaded successfully.")

    def search(self, query_vector: List[float], top_k: int = 5):


        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        return results.points

  

# if __name__ == "__main__":

#     from src.embedding import JinaEmbedding

#     # Create embedder
#     embedder = JinaEmbedding()

#     # Sample text
#     text = "Transformers use self-attention to model relationships between words."

#     # Generate embedding
#     embedding = embedder.embed([text])[0]

#     print(f"Embedding Dimension: {len(embedding)}")

#     # Connect to Qdrant
#     store = QdrantStore()

#     # Create collection (only once)
#     store.create_collection(len(embedding))

#     # Prepare document
#     documents = [
#         {
#             "chunk_id": 1,
#             "text": text,
#             "source": "sample.pdf",
#             "page": 1,
#             "section": "Introduction",
#             "embedding": embedding,
#         }
#     ]

#     # Upload
#     store.upload_documents(documents)

#     print("\nDocument uploaded successfully.")

#     # Search using the same embedding
#     results = store.search(embedding)

#     print("\nSearch Results\n")

#     for result in results:
#         print(f"Score : {result.score:.4f}")
#         print(f"Source: {result.payload['source']}")
#         print(f"Page  : {result.payload['page']}")
#         print(f"Text  : {result.payload['text']}")
#         print("-" * 60)