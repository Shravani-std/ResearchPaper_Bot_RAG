from typing import List, Dict, Any

from retrieval._3_hybrid_search import HybridRetriever
from src.vectorstore import QdrantStore
from core.logger import get_logger, log_step

logger = get_logger("contextual_retrieval")


class ContextualRetriever:
    def __init__(self):
        # self.hybrid = HybridRetriever()
        self.vector_store = QdrantStore()


    def retrieve(
            self,
            query: str,
            top_k: int = 5,

    ) -> List[Dict[str, Any]]:

        # Step -1 : Hybrid Retrieval

        # hybrid_results = self.hybrid.retrieve(
        #     query=query,
        #     top_k=top_k,
        # )

        contextual_results = []
        visited = set()

        # step - 2 : Expand Neighbors

        # for doc in hybrid_results:
        #     ids = [
        #         doc["previous_chunk"],
        #         doc["chunk_id"],
        #         doc["next_chunk"],
        #     ]


        #     for chunk_id in ids:
        #         if chunk_id is None:
        #             continue
        #         if chunk_id in visited:
        #             continue

        #         visited.add(chunk_id)

        #         neighbour  = self.get_chunk(chunk_id)
        #         if neighbour :
        #             contextual_results.append(neighbour )


        return contextual_results

    def get_chunk(
        self,
        chunk_id: int,
    ) -> Dict[str, Any] | None:

        """
        Fetch a chunk by chunk_id.
        """

        results = self.vector_store.client.scroll(
            collection_name=self.vector_store.collection_name,
            scroll_filter={
                "must": [
                    {
                        "key": "chunk_id",
                        "match": {
                            "value": chunk_id
                        }
                    }
                ]
            },
            limit=1,
        )

        points = results[0]

        if not points:
            return None

        point = points[0]

        payload = point.payload

        return {
            "chunk_id": payload["chunk_id"],
            "document_id": payload["document_id"],
            "chunk_index": payload["chunk_index"],
            "text": payload["text"],
            "source": payload["source"],
            "page": payload["page"],
            "section": payload["section"],
        }
    def expand(
    self,
    hybrid_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

        contextual_results = []
        visited = set()

        log_step("Expanding hybrid results with neighboring chunks.")

        for doc in hybrid_results:

            ids = [
                doc["previous_chunk"],
                doc["chunk_id"],
                doc["next_chunk"],
            ]

            for chunk_id in ids:

                if chunk_id is None:
                    continue

                if chunk_id in visited:
                    continue

                visited.add(chunk_id)

                neighbour = self.get_chunk(chunk_id)

                if neighbour:
                    contextual_results.append(neighbour)

        return contextual_results

# if __name__ == "__main__":

#     retriever = ContextualRetriever()

#     query = "Why was XGBoost chosen as the surrogate model instead of linear regression?"

#     docs = retriever.retrieve(
#         query=query,
#         top_k=3,
#     )

#     print("\nContextual Retrieval Results\n")

#     for i, doc in enumerate(docs):

#         print("=" * 70)

#         print(f"Chunk ID : {doc['chunk_id']}")

#         print(f"Page     : {doc['page']}")

#         print(f"Source   : {doc['source']}")

#         print()

#         print(doc["text"][:400])


