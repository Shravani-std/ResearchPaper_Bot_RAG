from typing import List, Dict, Any

from retrieval._1_semantic_search import SemanticRetriever
from retrieval._2_bm25_search import BM25Retriever
from retrieval._6_hyde import HyDE


class HybridRetriever:

    def __init__(self):

        self.semantic = SemanticRetriever(top_k=20)
        self.bm25 = BM25Retriever()
        self.hyde = HyDE()

        self.rrf_k = 60

    def reciprocal_rank_fusion(
        self,
        semantic_docs: List[Dict[str, Any]],
        bm25_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        scores = {}
        documents = {}

        # Semantic Results
        for rank, doc in enumerate(semantic_docs):

            chunk_id = doc["chunk_id"]

            documents[chunk_id] = doc

            scores.setdefault(chunk_id, 0.0)

            scores[chunk_id] += 1 / (self.rrf_k + rank + 1)

        # BM25 Results
        for rank, doc in enumerate(bm25_docs):

            chunk_id = doc["chunk_id"]

            if chunk_id not in documents:
                documents[chunk_id] = doc

            scores.setdefault(chunk_id, 0.0)

            scores[chunk_id] += 1 / (self.rrf_k + rank + 1)

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        fused_results = []

        for chunk_id, score in ranked:

            doc = documents[chunk_id].copy()

            doc["hybrid_score"] = score

            fused_results.append(doc)

        return fused_results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
    ) -> List[Dict[str, Any]]:

        print("=" * 60)
        print("Hybrid Retrieval")
        print("=" * 60)

        # -------------------------
        # HyDE
        # -------------------------

        if use_hyde:

            print("\nGenerating HyDE document...\n")

            hypothetical_document, query_embedding = (
                self.hyde.transform_query(query)
            )

            print("=" * 60)
            print("Hypothetical Document")
            print("=" * 60)
            print(hypothetical_document)
            print()

        else:

            query_embedding = None

        # -------------------------
        # Semantic Search
        # -------------------------

        semantic_docs = self.semantic.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=20,
        )

        # -------------------------
        # BM25 Search
        # -------------------------

        bm25_docs = self.bm25.retrieve(
            query=query,
            top_k=20,
        )

        # -------------------------
        # RRF Fusion
        # -------------------------

        fused_results = self.reciprocal_rank_fusion(
            semantic_docs,
            bm25_docs,
        )

        return fused_results[:top_k]


if __name__ == "__main__":

    retriever = HybridRetriever()

    query = "What is Retrieval Augmented Generation?"

    docs = retriever.retrieve(
        query=query,
        top_k=10,
        use_hyde=True,
    )

    print("\nHybrid Search Results\n")

    for i, doc in enumerate(docs):

        print("=" * 70)

        print(f"Rank          : {i + 1}")
        print(f"Hybrid Score  : {doc['hybrid_score']:.5f}")
        print(f"Source        : {doc['source']}")
        print(f"Page          : {doc['page']}")
        print(f"Chunk ID      : {doc['chunk_id']}")

        print()
        print(doc["text"][:500])
        print()