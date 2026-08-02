from typing import Optional

import requests

from core.config import settings
from src.embedding import JinaEmbedding


class HyDE:

    def __init__(self):

        self.embedder = JinaEmbedding()

        self.url = f"{settings.OPENROUTER_BASE_URL}/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

    def generate_hypothetical_document(
        self,
        query: str,
    ) -> str:

        prompt = f"""
Write a short factual paragraph that answers the question.

Do not say "I don't know."

Question:
{query}

Hypothetical Answer:
"""

        payload = {
            "model": settings.HYDE_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.3,
            "max_tokens": 250,
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    def transform_query(self, query: str):

        hypothetical_document = self.generate_hypothetical_document(query)

        embedding = self.embedder.embed(
            [hypothetical_document]
        )[0]

        return hypothetical_document, embedding

if __name__ == "__main__":

    hyde = HyDE()

    query = "Our users can't log in, what could be wrong?"

    document, embedding = hyde.transform_query(query)

    print("=" * 60)

    print("Original Query\n")

    print(query)

    print()

    print("=" * 60)

    print("Hypothetical Document\n")

    print(document)

    print()

    print("=" * 60)

    print(f"Embedding Dimension : {len(embedding)}")