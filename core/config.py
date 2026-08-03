import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    JINA_API_KEY = os.getenv("JINA_API_EMBEDDINGS")
    OPEN_ROUTER_API = os.getenv("OPEN_ROUTER_API")
    HYDE_MODEL = os.getenv("HYDE_MODEL")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "AIResearch_Bot")


settings = Settings()