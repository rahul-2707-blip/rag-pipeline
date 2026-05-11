"""Streamlit dashboard for the RAG pipeline.

Run with: streamlit run app.py
"""
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))
load_dotenv()

from rag.ask import ask as ask_question
from rag.retrieve import RetrievalConfig
from rag.store import list_documents


st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title("RAG Pipeline · Hybrid Retrieval Demo")


@st.cache_resource(show_spinner="Warming up models (first run can take a few minutes on CPU)…")
def _warmup():
    from rag.embed import embed_text
    from rag.retrieve import _reranker

    embed_text("warmup")
    _reranker().predict([("warmup query", "warmup passage")])
    return True


_warmup()
st.caption(
    "Ask questions of the FastAPI documentation. Retrieval combines dense embeddings "
    "(BGE-small) with BM25 keyword search via Reciprocal Rank Fusion, then reranks "
    "with a cross-encoder. Every claim is citation-verified."
)


with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox("Chunking strategy", ["recursive", "fixed", "semantic"], index=0)
    mode = st.radio("Retrieval mode", ["hybrid", "dense", "sparse"], index=0,
                    help="Toggle to compare hybrid vs. dense-only vs. sparse-only.")
    use_rerank = st.checkbox("Use cross-encoder reranker", value=True)
    verify = st.checkbox("Verify citations (slower)", value=True,
                         help="Disable for ~3x faster responses; loses the citation audit display.")
    final_k = st.slider("Final chunks (k)", min_value=1, max_value=10, value=5)
    st.divider()
    if st.button("Refresh document list"):
        st.session_state.documents = list_documents()


if "documents" not in st.session_state:
    try:
        st.session_state.documents = list_documents()
    except Exception as e:
        st.session_state.documents = []
        st.error(f"DB connection failed: {e}")


with st.expander(f"Indexed documents ({len(st.session_state.documents)})"):
    for d in st.session_state.documents[:50]:
        st.write(f"· **{d['title']}** — `{d['source']}` ({d['chunk_count']} chunks)")
    if len(st.session_state.documents) > 50:
        st.caption(f"…and {len(st.session_state.documents) - 50} more")


question = st.text_input("Ask a question:", placeholder="e.g. How do I declare a path parameter?")
ask_btn = st.button("Ask", type="primary", disabled=not question)


if ask_btn and question:
    config = RetrievalConfig(
        strategy=strategy,
        mode=mode,
        use_rerank=use_rerank,
        final_k=final_k,
    )
    with st.spinner("Retrieving and generating…"):
        bundle = ask_question(question, config=config, verify=verify)

    # Confidence chips
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite confidence", f"{bundle.confidence.composite:.2f}")
    c2.metric("Retrieval", f"{bundle.confidence.retrieval:.2f}")
    c3.metric("Citation coverage", f"{bundle.confidence.citation_coverage:.2f}")
    c4.metric("Completeness", f"{bundle.confidence.completeness:.2f}")

    if bundle.refused:
        st.warning(bundle.answer)
    else:
        st.subheader("Answer")
        st.write(bundle.answer)

        if bundle.citation_verdicts:
            st.subheader("Citation verification")
            for v in bundle.citation_verdicts:
                icon = "✅" if v.supported else "⚠️"
                st.markdown(f"{icon} **[{v.citation_num}]** _{v.sentence}_  \n→ {v.reason}")

    st.subheader("Retrieved chunks")
    tab_labels = [f"[{i+1}] {c.source}" for i, c in enumerate(bundle.chunks)]
    if tab_labels:
        tabs = st.tabs(tab_labels)
        for i, (tab, chunk) in enumerate(zip(tabs, bundle.chunks)):
            with tab:
                st.caption(
                    f"section: {chunk.section or '—'} · chunk #{chunk.chunk_index} · "
                    f"dense={chunk.dense_score:.3f} · sparse={chunk.sparse_score:.3f} · "
                    f"rrf={chunk.rrf_score:.3f} · rerank={chunk.rerank_score:.3f}"
                )
                st.code(chunk.text, language="markdown")
