from typing import List, Dict

from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
    ):

        print("=" * 60)
        print("Loading Cross Encoder...")
        print("=" * 60)

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

        print(f"Loaded {model_name}")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:

        if not documents:
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

        return documents[:top_k]

# if __name__ == "__main__":

#     from retrieval._3_hybrid_search import HybridRetriever

#     retriever = HybridRetriever()

#     query = "What is Retrieval Augmented Generation?"

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