# RAG Project

## Setup

1. Create and activate the virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Ensure your `.env` file contains the API keys for Jina, Gemini, and Qdrant.

## Usage

- Ingest a PDF:
  - `python main.py --ingest path/to/file.pdf`
- Query the indexed content:
  - `python main.py --query "your question"`
