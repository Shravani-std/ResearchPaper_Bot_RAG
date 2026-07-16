from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import streamlit as st

from src.data_ingestion import ingest_document
from src.retrieval import Retriever
from src.llm import GeminiLLM

st.set_page_config(page_title="Jina RAG Assistant", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .block-container { padding-top: 2rem; }
    .chat-user { background: #1F2937; padding: 15px; border-radius: 12px; margin-bottom: 10px; }
    .chat-bot { background: #111827; padding: 15px; border-radius: 12px; margin-bottom: 10px; border-left: 5px solid #4F46E5; }
    .stButton > button { width: 100%; background: #4F46E5; color: white; border-radius: 10px; height: 45px; border: none; }
    .stTextInput > div > div > input { background: #1F2937; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("⚙ Settings")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if st.sidebar.button("Ingest PDF"):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            ingest_document(temp_path)
            st.sidebar.success("PDF indexed successfully")
        except Exception as exc:
            st.sidebar.error(f"Upload failed: {exc}")
        finally:
            Path(temp_path).unlink(missing_ok=True)

st.title("🤖 AI Research RAG Assistant")
st.caption("Ask questions from your uploaded PDFs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div class='chat-user'>👤 {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>🤖 {message['content']}</div>", unsafe_allow_html=True)

question = st.chat_input("Ask anything...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    retriever = Retriever()
    llm = GeminiLLM()
    results = retriever.retrieve(question)
    context_parts = []
    for result in results:
        payload = getattr(result, "payload", {}) or {}
        text = payload.get("text") or ""
        if text:
            context_parts.append(text)
    context = "\n\n".join(context_parts)
    if context.strip():
        answer = llm.generate_answer(context, question)
    else:
        answer = "No indexed content is available yet. Please upload and ingest a PDF first."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()