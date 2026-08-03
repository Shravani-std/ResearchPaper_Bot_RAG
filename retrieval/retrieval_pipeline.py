from typing import List, Dict, Any

from retrieval._6_hyde import HyDE
from retrieval._3_hybrid_search import HybridRetriever
from retrieval._4_contextual_retrieval import ContextualRetriever
from retrieval._5_reranker import Reranker


class RetrievalPipeline:

    def __init__(self):

        self.hyde = HyDE()

        self.hybrid = HybridRetriever()

        self.contextual = ContextualRetriever()

        self.reranker = Reranker()

    def retrieve(
        self,
        query: str,
        use_hyde: bool = True,
        top_k: int = 5,
    ) -> List[Dict]:

        print("=" * 70)
        print("Retrieval Pipeline")
        print("=" * 70)

        print(f"Query : {query}")

        # ---------------------------------------
        # Step 1 : HyDE
        # ---------------------------------------

        if use_hyde:

            print("\nStep 1 : HyDE")

            hypothetical_document = (
                self.hyde.generate_hypothetical_document(query)
            )

            print(hypothetical_document)

            search_query = hypothetical_document

        else:

            search_query = query

        # ---------------------------------------
        # Step 2 : Hybrid Search
        # ---------------------------------------

        print("\nStep 2 : Hybrid Retrieval")

        hybrid_docs = self.hybrid.retrieve(
            query=search_query,
            top_k=20,
        )

        print(f"Retrieved {len(hybrid_docs)} chunks.")

        # ---------------------------------------
        # Step 3 : Contextual Retrieval
        # ---------------------------------------

        print("\nStep 3 : Contextual Retrieval")

        contextual_docs = self.contextual.expand(
            hybrid_docs
        )

        print(f"Expanded to {len(contextual_docs)} chunks.")

        # ---------------------------------------
        # Step 4 : Reranking
        # ---------------------------------------

        print("\nStep 4 : Cross Encoder")

        final_docs = self.reranker.rerank(
            query=query,
            documents=contextual_docs,
            top_k=top_k,
        )

        print(f"Final {len(final_docs)} chunks.")

        return final_docs
    
if __name__ == "__main__":

    pipeline = RetrievalPipeline()

    query = "What is Retrieval Augmented Generation?"

    results = pipeline.retrieve(
        query=query,
        use_hyde=True,
        top_k=5,
    )

    print("\n" + "=" * 80)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 80)

    for i, doc in enumerate(results):

        print(f"\nRank {i+1}")
        print(f"Source       : {doc['source']}")
        print(f"Page         : {doc['page']}")
        print(f"Chunk ID     : {doc['chunk_id']}")

        if "hybrid_score" in doc:
            print(f"Hybrid Score : {doc['hybrid_score']:.4f}")

        if "rerank_score" in doc:
            print(f"Rerank Score : {doc['rerank_score']:.4f}")

        print("-" * 60)
        print(doc["text"][:500])