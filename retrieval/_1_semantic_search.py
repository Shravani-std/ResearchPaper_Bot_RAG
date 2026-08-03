from typing import List, Dict, Any
from src.embedding import JinaEmbedding
from src.vectorstore import QdrantStore

class SemanticRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.embedding_model = JinaEmbedding()
        self.vector_store = QdrantStore()

    def retrieve(
    self,
    query: str,
    query_embedding: List[float] = None,
    top_k: int = None,
) -> List[Dict[str, Any]]:

        print("=" * 60)
        print("Semantic Retrieval")
        print("=" * 60)

        print(f"Query : {query}")

        if top_k is None:
            top_k = self.top_k

        # Generate embedding only if HyDE didn't provide one
        if query_embedding is None:
            query_embedding = self.embedding_model.embed([query])[0]

        print(f"Generated Query Embedding : {len(query_embedding)} dimensions")

        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
        )

        retrieved_docs = []

        for result in results:
            retrieved_docs.append(
                {
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "source": result.payload.get("source", ""),
                    "page": result.payload.get("page", ""),
                    "section": result.payload.get("section", ""),
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "chunk_index": result.payload.get("chunk_index", ""),
                    "previous_chunk": result.payload.get("previous_chunk", ""),
                    "next_chunk": result.payload.get("next_chunk", ""),
                    "total_chunks": result.payload.get("total_chunks", ""),
                    "document_id": result.payload.get("document_id", ""),
                }
            )

        return retrieved_docs

# if __name__ == "__main__":

    # retriever = SemanticRetriever(top_k=5)

    # query = "What is Retrieval Augmented Generation?"

    # results = retriever.retrieve(query)

    # print("\nRetrieved Documents\n")

    # for i, doc in enumerate(results):

    #     print("=" * 70)

    #     print(f"Rank      : {i+1}")

    #     print(f"Score     : {doc['score']:.4f}")

    #     print(f"Source    : {doc['source']}")

    #     print(f"Page      : {doc['page']}")

    #     print(f"Section   : {doc['section']}")

    #     print(f"Chunk ID  : {doc['chunk_id']}")

    #     print()

    #     print(doc["text"][:500])

    #     print()