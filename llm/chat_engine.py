from typing import Dict

from retrieval.retrieval_pipeline import RetrievalPipeline
from llm.prompt_builder import PromptBuilder
from llm.openrouter_llm import OpenRouterLLM
from core.logger import get_logger, log_step

logger = get_logger("chat_engine")


class ChatEngine:

    def __init__(self):

        log_step("=" * 70)
        log_step("Initializing RAG Chat Engine...")
        log_step("=" * 70)

        self.retriever = RetrievalPipeline()

        self.prompt_builder = PromptBuilder()

        self.llm = OpenRouterLLM()

        log_step("Chat Engine Ready.")

    def chat(
        self,
        query: str,
        use_hyde: bool = True,
    ) -> Dict:

        log_step("=" * 70)
        log_step("User Query")
        log_step("=" * 70)

        log_step(query)

        # ------------------------------------
        # Retrieval
        # ------------------------------------

        log_step("Starting retrieval for user query.")
        documents = self.retriever.retrieve(
            query=query,
            use_hyde=use_hyde,
            top_k=5,
        )
        log_step(f"Retrieved {len(documents)} candidate documents.")

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

        log_step("Generating answer from retrieved context.")
        answer = self.llm.generate(
            prompt=prompt,
        )
        log_step("Answer generation completed.")

        return {

            "query": query,

            "answer": answer,

            "documents": documents,

            "prompt": prompt,

        }
    
# if __name__ == "__main__":

#     engine = ChatEngine()

#     query = "Why was XGBoost chosen as the surrogate model instead of linear regression?"

#     result = engine.chat(
#         query=query,
#         use_hyde=True,
#     )

#     print("\n")
#     print("=" * 80)
#     print("FINAL ANSWER")
#     print("=" * 80)

#     print(result["answer"])

#     print("\n")
#     print("=" * 80)
#     print("SOURCES")
#     print("=" * 80)

#     for i, doc in enumerate(result["documents"]):

#         print(f"\nDocument {i+1}")

#         print(f"Source : {doc['source']}")

#         print(f"Page   : {doc['page']}")

#         print(f"Score  : {doc['rerank_score']:.4f}")

#         print("-" * 50)

#         print(doc["text"][:300])
