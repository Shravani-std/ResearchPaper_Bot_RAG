from typing import List, Dict, Any
from rank_bm25 import BM25Okapi 
import re

from ingestion.data_ingestion import load_documents
from src.chunking import chunk_documents

STOPWORDS = {
    "the",
    "is",
    "of",
    "a",
    "an",
    "and",
    "to",
    "in",
    "for",
    "on",
}
def tokenize(text: str):

    tokens = re.findall(r"\b\w+\b", text.lower())
    return [token for token in tokens if token not in STOPWORDS]
class BM25Retriever:
    def __init__(self):
        print("=" * 60)
        print("Building BM25 Index...")
        print("=" * 60)

        # Load All Documents 
        documents = load_documents(r"D:\AI\RAG\Projects\data")

        # Chunk Documents
        self.documents = chunk_documents(documents)

        #Tokenize
        self.corpus = [
            tokenize(doc.page_content)
            for doc in self.documents
        ]

        # Build BM25 Index
        self.bm25 = BM25Okapi(self.corpus)
        print(f"Indexed {len(self.documents)} chunks.\n")
    
    def retrieve(self, query: str, top_k: int = 5, ) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        score = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(score),
            key=lambda x: x[1],
            reverse=True,
        )
        results = []

        for rank, (idx, score) in enumerate(ranked[:top_k]):
            chunk = self.documents[idx]

            results.append(
                {
                    "rank": rank + 1,
                    "score": float(score),
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                    "document_id": chunk.metadata.get("document_id", ""),
                    "page": chunk.metadata.get("page", 0),
                    "section": chunk.metadata.get("section", "Unknown"),

                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "chunk_index": chunk.metadata.get("chunk_index"),

                    "previous_chunk": chunk.metadata.get("previous_chunk"),
                    "next_chunk": chunk.metadata.get("next_chunk"),

                    "total_chunks": chunk.metadata.get("total_chunks"),
                }
            )
        return results



# if __name__ == "__main__":

#     retriever = BM25Retriever()

#     query = "What is Retrieval Augmented Generation?"

#     results = retriever.retrieve(
#         query=query,
#         top_k=5,
#     )

#     print("\nRetrieved Documents\n")

#     for i, doc in enumerate(results):

#         print("=" * 70)

#         print(f"Rank      : {doc['rank']}")
#         print(f"Score     : {doc['score']:.4f}")
#         print(f"Source    : {doc['source']}")
#         print(f"Page      : {doc['page']}")
#         print(f"Section   : {doc['section']}")
#         print(f"Chunk ID  : {doc['chunk_id']}")

#         print()
#         print(doc["text"][:500])
#         print()      