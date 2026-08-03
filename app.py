import tempfile
from pathlib import Path

import streamlit as st

from ingestion.data_ingestion import run_ingestion
from llm.chat_engine import ChatEngine

# -----------------------------
# Streamlit Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Research RAG",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# CSS
# -----------------------------

st.markdown(
    """
<style>

.stApp{
    background:#0E1117;
}

.chat-user{
    background:#1F2937;
    padding:15px;
    border-radius:12px;
    margin-bottom:10px;
}

.chat-bot{
    background:#111827;
    padding:15px;
    border-radius:12px;
    margin-bottom:10px;
    border-left:5px solid #4F46E5;
}

</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

if st.sidebar.button("Ingest PDF"):

    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as tmp:

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
            "role":"user",
            "content":question,
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

            with st.expander("Retrieved Sources"):

                for i, doc in enumerate(response["documents"]):

                    st.markdown(
                        f"""
### {i+1}

**Source:** {doc['source']}

**Page:** {doc['page']}

**Score:** {doc['rerank_score']:.4f}

{doc['text'][:400]}...
"""
                    )

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer,
        }
    )