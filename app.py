"""Orbitra Assist - Streamlit chat UI for the RAG system.

Run:  python -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from orbitra_rag.config import get_settings
from orbitra_rag.providers import ConfigurationError
from orbitra_rag.rag import RAGPipeline

PRODUCTS = {
    "All products": None,
    "RoboVac R7 (robot vacuum)": "robovac-r7",
    "BrewMate C3 (coffee machine)": "brewmate-c3",
    "AirPure A5 (air purifier)": "airpure-a5",
    "ThermoSmart T2 (thermostat)": "thermosmart-t2",
}
EXAMPLES = [
    "My vacuum shows E06, what should I do?",
    "How do I descale the coffee machine and how long does it take?",
    "Can I use the thermostat with electric baseboard heaters?",
    "My RoboVac battery died after 18 months. Is it covered by warranty?",
]
BASELINE_TITLE = "Answer of the same LLM without RAG (for comparison)"
BASELINE_WARNING = "No context from the knowledge base was provided to the model for this answer."

st.set_page_config(page_title="Orbitra Assist", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def render_sources(hits) -> None:
    with st.expander(f"Sources: {len(hits)} retrieved chunks", expanded=False):
        for i, hit in enumerate(hits, start=1):
            p = hit.payload
            st.markdown(f"**[{i}]** `{p.get('doc_id')}` › {p.get('section')} — similarity **{hit.score:.3f}**")
            st.caption(f"product: {p.get('product_id')} · type: {p.get('doc_type')} · file: {p.get('source')}")
            st.code(p.get("text", ""), language=None, wrap_lines=True)


def render_baseline(text: str) -> None:
    with st.expander(BASELINE_TITLE):
        st.warning(BASELINE_WARNING)
        st.markdown(text)


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_sources(msg["hits"])
        if msg.get("baseline"):
            render_baseline(msg["baseline"])
        if msg.get("meta"):
            st.caption(msg["meta"])


# ---------------------------------------------------------------------------- sidebar

settings = get_settings()
with st.sidebar:
    st.header("⚙️ Settings")
    product_label = st.selectbox("Product filter", list(PRODUCTS))
    top_k = st.slider("Number of retrieved chunks (top-k)", 1, 10, 5)
    show_sources = st.toggle("Show retrieved sources", value=True)
    compare = st.toggle("Compare with LLM without RAG", value=False)
    if st.button("🗑️ Clear chat", width="stretch"):
        st.session_state.messages = []

    st.divider()
    st.subheader("System")
    try:
        pipeline = get_pipeline()
        points = pipeline.store.count()
        st.success(f"Vector DB: {points} chunks in `{pipeline.store.collection}`")
    except ConfigurationError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Vector DB is not reachable ({settings.qdrant_url}). Run `docker compose up -d`.\n\n{exc}")
        st.stop()
    st.caption(
        f"Provider: **{settings.provider}**  \n"
        f"Embeddings: `{settings.embedding_model}` ({settings.embedding_dim} dims)  \n"
        f"LLM: `{settings.chat_model}`  \n"
        f"Vector DB: Qdrant @ {settings.qdrant_url}"
    )

# ---------------------------------------------------------------------------- main area

st.title("🤖 Orbitra Assist")
st.caption("Support assistant for Orbitra Home products. Answers are based on the official documentation (RAG).")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try one of these questions:**")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, width="stretch"):
            st.session_state.pending = example
            st.rerun()

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask about your Orbitra device...") or st.session_state.pop("pending", None)

if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            # 1-3. question -> embedding -> vector search in Qdrant
            with st.spinner("Searching the knowledge base..."):
                hits, timings, stream = pipeline.stream_answer(
                    question, product_id=PRODUCTS[product_label], top_k=top_k
                )
            # 4. question + retrieved chunks in one prompt -> LLM, answer streamed to the UI
            answer = st.write_stream(stream)

            baseline = None
            if compare:
                with st.spinner("Asking the LLM without RAG..."):
                    baseline = pipeline.answer_without_rag(question).text

            meta = (
                f"embedding {timings['embedding_s']:.2f}s · vector search {timings['search_s'] * 1000:.0f} ms · "
                f"filter: {PRODUCTS[product_label] or 'none'} · top-k: {top_k}"
            )
            # The streamed text is already on screen; render only the extras here.
            if show_sources:
                render_sources(hits)
            if baseline:
                render_baseline(baseline)
            st.caption(meta)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "hits": hits if show_sources else None,
                    "baseline": baseline,
                    "meta": meta,
                }
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {exc}")
