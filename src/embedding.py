from __future__ import annotations

import math
import re
import time
from typing import List

import requests

from core.config import settings


class JinaEmbedding:
    BASE_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self) -> None:
        self.headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if not settings.JINA_API_KEY:
            return [self._fallback_embedding(text) for text in texts]

        payload = {"model": "jina-embeddings-v3", "input": texts}

        for attempt in range(3):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data.get("data", [])]
            except requests.RequestException:
                if attempt == 2:
                    break
                time.sleep(2)

        return [self._fallback_embedding(text) for text in texts]

    @staticmethod
    def _fallback_embedding(text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vector = [0.0] * 64
        for token in tokens:
            index = sum(ord(char) for char in token) % 64
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]