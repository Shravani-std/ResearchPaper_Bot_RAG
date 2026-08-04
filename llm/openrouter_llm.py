from __future__ import annotations

import time
from typing import List, Dict

import requests

from core.config import settings
from core.logger import get_logger, log_step

logger = get_logger("openrouter_llm")


class OpenRouterLLM:

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ):

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self.url = (
            f"{settings.OPENROUTER_BASE_URL}/chat/completions"
        )

        self.headers = {
            "Authorization": f"Bearer {settings.OPEN_ROUTER_API}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:

        messages: List[Dict] = []

        if system_prompt:

            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        for attempt in range(self.max_retries):

            try:

                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                return (
                    data["choices"][0]
                    ["message"]
                    ["content"]
                    .strip()
                )

            except requests.RequestException as e:

                logger.warning(
                    "OpenRouter attempt %s/%s failed",
                    attempt + 1,
                    self.max_retries,
                )
                logger.warning(str(e))

                if (
                    hasattr(e, "response")
                    and e.response is not None
                ):

                    if e.response.status_code == 429:

                        retry = e.response.headers.get(
                            "Retry-After",
                            20,
                        )

                        logger.warning(
                            "Rate limit reached. Waiting %ss before retrying.",
                            retry,
                        )

                        time.sleep(int(retry))

                        continue

                if attempt < self.max_retries - 1:

                    time.sleep(2)

                else:

                    raise RuntimeError(
                        "OpenRouter request failed."
                    ) from e
# if __name__ == "__main__":

#     llm = OpenRouterLLM()

#     answer = llm.generate(
#         prompt="Explain Retrieval Augmented Generation."
#     )

#     print(answer)