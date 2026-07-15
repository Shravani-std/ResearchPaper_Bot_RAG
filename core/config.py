from dotenv import load_dotenv
import os

load_dotenv()

class Settings:

    JINA_API_KEY = os.getenv("JINA_API_EMBEDDINGS")

    GEMINI_API_KEY = os.getenv("GEMINI_LLM_API")

    # QDRANT_URL = os.getenv("QDRANT_URL")

    # QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    # COLLECTION_NAME = os.getenv("COLLECTION_NAME")


settings = Settings()