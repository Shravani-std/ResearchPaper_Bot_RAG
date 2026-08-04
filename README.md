# AI Research Assistant — RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for querying a corpus of research papers through natural-language chat. Built with a hybrid retrieval architecture (semantic + lexical search), HyDE query expansion, contextual chunk expansion, and cross-encoder reranking, served through a Streamlit chat interface.

## What This Project Does

Upload PDF research papers, ingest them into a searchable vector index, and ask natural-language questions. The system retrieves the most relevant passages using multiple retrieval strategies, reranks them for relevance, and generates a grounded answer with source citations (document, page, and confidence score).

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  1. HyDE (Hypothetical Document Embeddings)  │
│     LLM drafts a hypothetical answer to      │
│     improve the semantic search query        │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  2. Hybrid Retrieval                         │
│     • Semantic search (Qdrant + Jina embed)  │
│     • BM25 lexical search                    │
│     • Reciprocal Rank Fusion (RRF)           │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  3. Contextual Expansion                     │
│     Pulls in previous/next neighboring       │
│     chunks for better local context          │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  4. Cross-Encoder Reranking                  │
│     BAAI/bge-reranker-base scores query-doc  │
│     pairs for true relevance, top-K selected │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  5. Prompt Construction + LLM Generation     │
│     Retrieved chunks → prompt → OpenRouter   │
│     LLM → final cited answer                 │
└─────────────────────────────────────────────┘
    │
    ▼
Answer + Sources (Streamlit chat UI)
```

## Ingestion Pipeline

```
PDF/TXT files
    │
    ▼
Load & extract text (PyMuPDF / fitz + LangChain TextLoader)
    │
    ▼
Chunk (RecursiveCharacterTextSplitter, ~200 words, 50-100 overlap)
    │
    ▼
Embed (Jina Embeddings v3, 1024-dim)
    │
    ▼
Store (Qdrant vector database, with chunk_id, source, page,
        previous_chunk/next_chunk links for contextual expansion)
```

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF parsing | PyMuPDF (fitz) |
| Chunking | LangChain (`RecursiveCharacterTextSplitter`) |
| Embeddings | Jina Embeddings v3 (via API) |
| Vector store | Qdrant |
| Lexical search | BM25 (`rank_bm25`) |
| Query expansion | HyDE (via OpenRouter LLM) |
| Reranking | Cross-encoder — `BAAI/bge-reranker-base` |
| Generation | LLM via OpenRouter API |

## Project Structure

```
Projects/
├── app.py                          # Streamlit chat UI
├── ingestion/
│   ├── data_loader.py              # PDF/TXT loading + text cleaning
│   └── data_ingestion.py           # Full ingestion orchestration
├── src/
│   ├── chunking.py                 # Document chunking
│   ├── embedding.py                # Jina embedding client
│   └── vectorstore.py              # Qdrant client wrapper
├── retrieval/
│   ├── _1_semantic_search.py       # Vector similarity search
│   ├── _2_bm25_search.py           # Lexical (BM25) search
│   ├── _3_hybrid_search.py         # RRF fusion of semantic + BM25
│   ├── _4_contextual_retrieval.py  # Neighbor chunk expansion
│   ├── _5_reranker.py              # Cross-encoder reranking
│   ├── _6_hyde.py                  # HyDE query expansion
│   └── retrieval_pipeline.py       # Orchestrates steps 1-5
├── llm/
│   ├── chat_engine.py              # Top-level chat orchestration
│   ├── prompt_builder.py           # Builds LLM prompt from retrieved docs
│   └── openrouter_llm.py           # LLM generation client
└── core/
    └── config.py                   # API keys & settings
```

## How to Run

```bash
# From the project root
python -m streamlit run app.py
```

1. Upload a PDF in the sidebar and click **"Ingest PDF"** — this chunks, embeds, and stores it in Qdrant.
2. Ask a question in the chat box.
3. The system retrieves, reranks, and generates a cited answer.

## Key Design Notes

- **Hybrid retrieval**: combining semantic (meaning-based) and BM25 (keyword-based) search catches both paraphrased and exact-term queries.
- **HyDE**: rather than embedding the raw question, an LLM first drafts a hypothetical answer — this tends to match the phrasing/style of source documents better than a short question does.
- **Contextual expansion**: retrieved chunks pull in their immediate neighbors, so the LLM sees more surrounding context than a single ~200-word chunk.
- **Reranking**: a cross-encoder re-scores the expanded candidate set for true relevance, since embedding similarity and BM25 scores alone are noisy signals.
- **Consistent chunk IDs**: ingestion and BM25 indexing both read from the same Qdrant collection, so `chunk_id`/`previous_chunk`/`next_chunk` stay consistent across retrieval methods.

## Known Limitations / Next Steps

- No conversation memory — each question is answered independently of prior turns.
- No deduplication check on re-ingesting the same document.
- Citation formatting in generated answers can be inconsistent — worth tightening the prompt template in `prompt_builder.py`.
- Single flat Qdrant collection — no per-document filtering exposed in the UI yet.
