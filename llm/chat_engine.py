from typing import Dict

from retrieval.retrieval_pipeline import RetrievalPipeline
from llm.prompt_builder import PromptBuilder
from llm.openrouter_llm import OpenRouterLLM


class ChatEngine:

    def __init__(self):

        print("=" * 70)
        print("Initializing RAG Chat Engine...")
        print("=" * 70)

        self.retriever = RetrievalPipeline()

        self.prompt_builder = PromptBuilder()

        self.llm = OpenRouterLLM()

        print("Chat Engine Ready.\n")

    def chat(
        self,
        query: str,
        use_hyde: bool = True,
    ) -> Dict:

        print("=" * 70)
        print("User Query")
        print("=" * 70)

        print(query)

        # ------------------------------------
        # Retrieval
        # ------------------------------------

        documents = self.retriever.retrieve(
            query=query,
            use_hyde=use_hyde,
            top_k=5,
        )

        # ------------------------------------
        # Build Prompt
        # ------------------------------------

        prompt = self.prompt_builder.build(
            query=query,
            documents=documents,
        )

        # ------------------------------------
        # Generate Answer
        # ------------------------------------

        answer = self.llm.generate(
            prompt=prompt,
        )

        return {

            "query": query,

            "answer": answer,

            "documents": documents,

            "prompt": prompt,

        }
if __name__ == "__main__":

    engine = ChatEngine()

    query = "What is Retrieval Augmented Generation?"

    result = engine.chat(
        query=query,
        use_hyde=True,
    )

    print("\n")
    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n")
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for i, doc in enumerate(result["documents"]):

        print(f"\nDocument {i+1}")

        print(f"Source : {doc['source']}")

        print(f"Page   : {doc['page']}")

        print(f"Score  : {doc['rerank_score']:.4f}")

        print("-" * 50)

        print(doc["text"][:300])
