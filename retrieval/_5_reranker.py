from typing import List, Dict

from sentence_transformers import CrossEncoder
from core.logger import get_logger, log_step

logger = get_logger("reranker")


class Reranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
    ):

        log_step("=" * 60)
        log_step("Loading Cross Encoder...")
        log_step("=" * 60)

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

        log_step(f"Loaded {model_name}")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:

        if not documents:
            log_step("No documents available for reranking.")
            return []

        # ------------------------------------
        # Build Query-Document pairs
        # ------------------------------------

        pairs = [
            (query, doc["text"])
            for doc in documents
        ]

        # ------------------------------------
        # Predict relevance scores
        # ------------------------------------

        log_step(f"Reranking {len(documents)} documents for query.")
        scores = self.model.predict(pairs)

        # ------------------------------------
        # Attach scores
        # ------------------------------------

        for doc, score in zip(documents, scores):

            doc["rerank_score"] = float(score)

        # ------------------------------------
        # Sort
        # ------------------------------------

        documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        log_step(f"Reranking completed. Returning top {min(top_k, len(documents))} documents.")
        return documents[:top_k]

# if __name__ == "__main__":

#     from retrieval._3_hybrid_search import HybridRetriever

#     retriever = HybridRetriever()

#     query = "Why was XGBoost chosen as the surrogate model instead of linear regression?"

#     documents = retriever.retrieve(
#         query=query,
#         top_k=10,
#     )

#     reranker = Reranker()

#     results = reranker.rerank(
#         query=query,
#         documents=documents,
#         top_k=5,
#     )

#     print("\nFinal Results\n")

#     for i, doc in enumerate(results):

#         print("=" * 70)

#         print(f"Rank : {i+1}")

#         print(f"Rerank Score : {doc['rerank_score']:.4f}")

#         print(f"Source : {doc['source']}")

#         print(f"Page : {doc['page']}")

#         print()

#         print(doc["text"][:500])