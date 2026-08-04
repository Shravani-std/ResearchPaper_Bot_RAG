import tempfile
from pathlib import Path

import streamlit as st

from core.config import settings
from ingestion.data_ingestion import run_ingestion
from llm.chat_engine import ChatEngine

st.set_page_config(
    page_title="AI Research RAG",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp{ background:#0E1117; }
.chat-user{ background:#1F2937; padding:15px; border-radius:12px; margin-bottom:10px; }
.chat-bot{ background:#111827; padding:15px; border-radius:12px; margin-bottom:10px; border-left:5px solid #4F46E5; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar — Bring Your Own Keys
# -----------------------------

st.sidebar.title("🔑 API Configuration")

with st.sidebar.form("config_form"):

    jina_key = st.text_input("Jina API Key", type="password")
    openrouter_key = st.text_input("OpenRouter API Key", type="password")
    openrouter_url = st.text_input(
        "OpenRouter Base URL",
        value="https://openrouter.ai/api/v1",
    )
    qdrant_url = st.text_input("Qdrant URL")
    qdrant_key = st.text_input("Qdrant API Key", type="password")
    collection_name = st.text_input(
        "Collection Name",
        value="AIResearch_Bot",
    )

    save_config = st.form_submit_button("Save Configuration")

    if save_config:

        settings.JINA_API_KEY = jina_key
        settings.OPEN_ROUTER_API = openrouter_key
        settings.OPENROUTER_BASE_URL = openrouter_url
        settings.QDRANT_URL = qdrant_url
        settings.QDRANT_API_KEY = qdrant_key
        settings.COLLECTION_NAME = collection_name

        st.session_state.config_ready = True

        # Force re-creation of the engine with the new keys
        if "engine" in st.session_state:
            del st.session_state["engine"]

        st.sidebar.success("Configuration saved.")

if not st.session_state.get("config_ready"):
    st.info("👈 Enter your API keys in the sidebar to get started.")
    st.stop()

st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if st.sidebar.button("Ingest PDF"):
    if uploaded_file:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name
        try:
            run_ingestion(pdf_path)
            st.sidebar.success("Document Indexed Successfully!")
        except Exception as e:
            st.sidebar.error(str(e))
        finally:
            Path(pdf_path).unlink(missing_ok=True)

# -----------------------------
# Chat Engine
# -----------------------------

if "engine" not in st.session_state:

    st.session_state.engine = ChatEngine()

engine = st.session_state.engine

# -----------------------------
# Chat History
# -----------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

st.title("🤖 AI Research Assistant")

st.caption(
    "Powered by Hybrid Search + HyDE + Contextual Retrieval + Cross Encoder"
)

# -----------------------------
# Process Section
# -----------------------------

with st.expander("🔍 Process — What happens after you ask a question?"):

    st.markdown(
        """
**Step 1 — HyDE (Hypothetical Document Embeddings)**
The LLM drafts a short hypothetical answer to your question first.
This hypothetical text is embedded instead of the raw question,
because it tends to match the phrasing and style of source documents
more closely than a short question does.

**Step 2 — Hybrid Retrieval**
Two searches run in parallel over the ingested documents:
- *Semantic search* — vector similarity in Qdrant using Jina embeddings
- *BM25 search* — classic keyword/lexical matching

Both result sets are merged using **Reciprocal Rank Fusion (RRF)**,
so a chunk that ranks well in either search gets weighted appropriately.

**Step 3 — Contextual Expansion**
Each retrieved chunk pulls in its immediate neighboring chunks
(the text right before and after it in the original document),
so the LLM sees fuller local context instead of an isolated ~200-word snippet.

**Step 4 — Cross-Encoder Reranking**
A cross-encoder model (`BAAI/bge-reranker-base`) scores every
query–chunk pair for true relevance and keeps only the top-K
most relevant chunks — this catches cases where embedding
similarity alone is misleading.

**Step 5 — Answer Generation**
The final set of chunks is assembled into a prompt and sent to
the LLM (via OpenRouter), which generates a grounded answer
citing the specific source documents and pages it used.
        """
    )

# -----------------------------
# Display Messages
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# -----------------------------
# User Input
# -----------------------------

question = st.chat_input(
    "Ask a research question..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = engine.chat(
                query=question,
                use_hyde=True,
            )

            answer = response["answer"]

            st.markdown(answer)

#             with st.expander("📄 Retrieved Sources"):

#                 for i, doc in enumerate(response["documents"]):

#                     st.markdown(
#                         f"""
# **{i+1}. {Path(doc['source']).name}** — Page {doc['page']}
# Rerank Score: `{doc.get('rerank_score', 0):.4f}`

# > {doc['text'][:400]}...
# """
#                     )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
