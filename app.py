"""
AI Plant Care Assistant
-----------------------
Streamlit + Ollama (100% local, free)

Features
1. Upload plant-care documents (.txt / .pdf) -> used as the knowledge base (RAG)
2. Upload a plant photo -> vision model identifies the plant / spots problems
3. Ask questions about watering, sunlight, soil, fertilizer, pests, general care
4. Answers are grounded in your documents, and sources are shown
"""

import io

import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader

# ----------------------------- CONFIG ---------------------------------------
CHAT_MODEL = "llama3.2"          # text model
VISION_MODEL = "llava"           # image model
EMBED_MODEL = "nomic-embed-text" # embedding model
CHUNK_SIZE = 600                 # characters per chunk
CHUNK_OVERLAP = 100
TOP_K = 3                        # chunks retrieved per question

SYSTEM_PROMPT = (
    "You are a friendly, expert plant care assistant. "
    "Answer questions about watering, sunlight, soil, fertilizer, pests, "
    "repotting and general maintenance. "
    "Use the CONTEXT from the user's documents first. If the context does not "
    "contain the answer, say so briefly, then give general best-practice advice. "
    "Keep answers clear and practical, using short bullet points."
)


# ----------------------------- HELPERS --------------------------------------
def read_file(uploaded) -> str:
    """Extract text from an uploaded .txt or .pdf file."""
    if uploaded.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded.read()))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def embed(texts):
    """Return a numpy array of embeddings for a list of texts."""
    res = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.array(res["embeddings"], dtype=np.float32)


def build_index(files):
    """Read files, chunk them and embed every chunk."""
    chunks, sources = [], []
    for f in files:
        for c in chunk_text(read_file(f)):
            chunks.append(c)
            sources.append(f.name)
    if not chunks:
        return None
    vectors = embed(chunks)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    return {"chunks": chunks, "sources": sources, "vectors": vectors}


def retrieve(query: str, index, k: int = TOP_K):
    """Find the k most similar chunks to the query (cosine similarity)."""
    q = embed([query])[0]
    q /= np.linalg.norm(q) + 1e-10
    scores = index["vectors"] @ q
    top = np.argsort(scores)[::-1][:k]
    return [(index["chunks"][i], index["sources"][i], float(scores[i])) for i in top]


def identify_plant(image_bytes: bytes) -> str:
    """Use the vision model to identify the plant and check its health."""
    prompt = (
        "You are a plant expert. Look at this photo and reply in this format:\n"
        "Plant name: <common name and scientific name if known>\n"
        "Health: <Healthy / Needs attention>\n"
        "Visible issues: <yellow leaves, brown tips, pests, wilting, etc. or 'None'>\n"
        "Quick tip: <one short care tip>"
    )
    res = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
    )
    return res["message"]["content"]


def stream_answer(question: str, context: str, plant_info: str, history):
    """Stream the answer from the chat model."""
    user_msg = (
        f"PLANT FROM PHOTO:\n{plant_info or 'No photo uploaded.'}\n\n"
        f"CONTEXT FROM DOCUMENTS:\n{context or 'No documents uploaded.'}\n\n"
        f"QUESTION: {question}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]  # short memory of the conversation
    messages.append({"role": "user", "content": user_msg})
    for chunk in ollama.chat(model=CHAT_MODEL, messages=messages, stream=True):
        yield chunk["message"]["content"]


# ----------------------------- UI -------------------------------------------
st.set_page_config(page_title="AI Plant Care Assistant", page_icon="🌿", layout="wide")
st.title("🌿 AI Plant Care Assistant")
st.caption("Upload plant-care notes and a plant photo, then ask about watering, sunlight and maintenance.")

# session state
st.session_state.setdefault("index", None)
st.session_state.setdefault("plant_info", "")
st.session_state.setdefault("messages", [])

# ---- Sidebar: inputs ----
with st.sidebar:
    st.header("1. Knowledge base")
    docs = st.file_uploader("Upload plant-care files (.txt, .pdf)",
                            type=["txt", "pdf"], accept_multiple_files=True)
    if st.button("Process documents", use_container_width=True):
        if docs:
            with st.spinner("Reading and indexing documents..."):
                st.session_state.index = build_index(docs)
            if st.session_state.index:
                n = len(st.session_state.index["chunks"])
                st.success(f"Indexed {len(docs)} file(s) into {n} chunks.")
            else:
                st.warning("No readable text found.")
        else:
            st.warning("Please upload at least one file.")

    st.header("2. Plant photo (optional)")
    img = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])
    if img:
        st.image(img, caption="Your plant", use_container_width=True)
        if st.button("Analyze plant", use_container_width=True):
            with st.spinner("Analyzing image with LLaVA..."):
                st.session_state.plant_info = identify_plant(img.getvalue())

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.plant_info = ""
        st.rerun()

    st.divider()
    st.caption(f"Models: {CHAT_MODEL} · {VISION_MODEL} · {EMBED_MODEL}")

# ---- Main area ----
if st.session_state.plant_info:
    st.subheader("📷 Plant analysis")
    st.info(st.session_state.plant_info)

st.subheader("💬 Ask a question")
cols = st.columns(4)
quick = {
    "💧 Watering": "How often should I water this plant?",
    "☀️ Sunlight": "How much sunlight does this plant need?",
    "🌱 Maintenance": "What general maintenance does this plant need?",
    "🐛 Problems": "What are common problems and how do I fix them?",
}
clicked = None
for col, (label, q) in zip(cols, quick.items()):
    if col.button(label, use_container_width=True):
        clicked = q

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📚 Sources used"):
                for text, src, score in m["sources"]:
                    st.markdown(f"**{src}** (similarity {score:.2f})\n\n> {text}")

question = st.chat_input("e.g. How often should I water my snake plant?") or clicked

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    sources, context = [], ""
    if st.session_state.index:
        sources = retrieve(question, st.session_state.index)
        context = "\n\n".join(f"[{s}] {t}" for t, s, _ in sources)

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_answer(question, context, st.session_state.plant_info, history)
            )
            if sources:
                with st.expander("📚 Sources used"):
                    for text, src, score in sources:
                        st.markdown(f"**{src}** (similarity {score:.2f})\n\n> {text}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Could not reach Ollama. Is it running and are the models pulled?\n\n{e}")
