from __future__ import annotations
print("1. Starting main")

import argparse
from typing import List

from src.retrieval import Retriever
print("2. Retriever imported")
from src.llm import GeminiLLM
print("3. LLM imported")
from src.data_ingestion import ingest_document
print("4. Ingestion of files")

def build_context(results: List[object]) -> str:
    context_parts = []
    for result in results:
        payload = getattr(result, "payload", {}) or {}
        text = payload.get("text") or ""
        if text:
            context_parts.append(text)
    return "\n\n".join(context_parts)


def run_query(question: str) -> str:
    retriever = Retriever()
    llm = GeminiLLM()
    results = retriever.retrieve(question)
    context = build_context(results)
    return llm.generate_answer(context, question)


def run_ingestion(pdf_path: str) -> None:
    documents = ingest_document(pdf_path)
    print(f"Prepared {len(documents)} chunks from {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG project CLI")
    parser.add_argument("--query", help="Ask a question to the RAG pipeline")
    parser.add_argument("--ingest", help="Path to a PDF document to ingest")
    args = parser.parse_args()

    if args.ingest:
        run_ingestion(args.ingest)
        return

    if args.query:
        print(run_query(args.query))
        return

    question = input("Ask: ")
    print(run_query(question))


if __name__ == "__main__":
    main()