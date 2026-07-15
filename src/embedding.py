import time
import requests

from core.config import settings


class JinaEmbedding:

    BASE_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self):

        self.headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json"
        }

    def embed(self, texts):

        payload = {
            "model": "jina-embeddings-v3",
            "input": texts
        }

        for attempt in range(3):

            try:

                response = requests.post(
                    self.BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )

                response.raise_for_status()

                data = response.json()

                return [
                    item["embedding"]
                    for item in data["data"]
                ]

            except requests.RequestException:

                if attempt == 2:
                    raise

                time.sleep(2)