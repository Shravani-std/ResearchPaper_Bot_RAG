from typing import Optional

import requests

from core.config import settings
from src.embedding import JinaEmbedding
from core.logger import get_logger, log_step

logger = get_logger("hyde")


class HyDE:

    def __init__(self):

        self.embedder = JinaEmbedding()

        self.url = f"{settings.OPENROUTER_BASE_URL}/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {settings.OPEN_ROUTER_API}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _resolve_model(model_name: Optional[str]) -> str:
        if not model_name:
            return "meta-llama/llama-3.2-3b-instruct"

        if model_name.endswith(":free"):
            return model_name.rsplit(":", 1)[0]

        return model_name

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

        model_name = self._resolve_model(settings.HYDE_MODEL)

        payload = {
            "model": model_name,
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

        log_step(f"HyDE response status: {response.status_code}")
        log_step(f"HyDE response body: {response.text}")

        if response.status_code == 404:
            logger.warning("OpenRouter rejected the requested model. Retrying with the base model slug.")
            payload["model"] = "meta-llama/llama-3.2-3b-instruct"
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            log_step(f"HyDE retry status: {response.status_code}")
            log_step(f"HyDE retry response: {response.text}")

        response.raise_for_status()

        data = response.json()
        log_step("HyDE document generated successfully.")

        return data["choices"][0]["message"]["content"].strip()

    def transform_query(self, query: str):

        hypothetical_document = self.generate_hypothetical_document(query)

        embedding = self.embedder.embed(
            [hypothetical_document]
        )[0]

        return hypothetical_document, embedding

# if __name__ == "__main__":

#     hyde = HyDE()

#     query = "Our users can't log in, what could be wrong?"

#     document, embedding = hyde.transform_query(query)

#     print("=" * 60)

#     print("Original Query\n")

#     print(query)

#     print()

#     print("=" * 60)

#     print("Hypothetical Document\n")

#     print(document)

#     print()

#     print("=" * 60)

#     print(f"Embedding Dimension : {len(embedding)}")