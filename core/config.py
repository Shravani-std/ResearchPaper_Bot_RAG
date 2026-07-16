import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    JINA_API_KEY = os.getenv("JINA_API_EMBEDDINGS")
    GEMINI_API_KEY = os.getenv("GEMINI_LLM_API")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "AIResearch_Bot")


settings = Settings()