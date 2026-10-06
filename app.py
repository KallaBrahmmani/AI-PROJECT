"""
🌿 AI Plant Care Assistant  (Streamlit + Ollama, 100% local)
-------------------------------------------------------------
Features
1. Upload plant-care documents (.txt / .pdf) -> knowledge base (RAG)
2. Upload a plant photo -> vision model identifies plant + checks health
3. Ask about watering, sunlight, soil, fertilizer, pests, maintenance
4. NEW: Plant Uses -> medicinal, culinary, household, environmental uses
        + safety / toxicity information
5. Sources shown for every answer, modern green UI
"""

import io
import re

import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader

# ============================== CONFIG ======================================
CHAT_MODEL = "llama3.2"           # text model
VISION_MODEL = "llava"            # image model
EMBED_MODEL = "nomic-embed-text"  # embedding model
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3

MEDICAL_KEYWORDS = (
    "medic", "cure", "treat", "remedy", "health", "heal", "disease",
    "ayurved", "herbal", "benefit", "use of", "uses of", "uses", "edible",
    "eat", "tea", "toxic", "poison", "safe",
)

DISCLAIMER = (
    "⚠️ **Note:** Medicinal uses mentioned here are *traditional or commonly "
    "reported* uses, not medical advice. Always consult a doctor or qualified "
    "practitioner before using any plant for health purposes, and check "
    "toxicity for children and pets."
)

SYSTEM_PROMPT = (
    "You are a friendly, expert plant care and plant knowledge assistant. "
    "You help with watering, sunlight, soil, fertilizer, pests, repotting and "
    "general maintenance. You also explain the uses of plants: medicinal "
    "(traditional/commonly reported), culinary, household, ornamental and "
    "environmental uses, plus safety and toxicity. "
    "Use the CONTEXT from the user's documents first. If the context does not "
    "contain the answer, say so briefly and then give general, well-known "
    "information. Never claim a plant cures or treats a disease for certain; "
    "describe medicinal uses as traditional or reported. "
    "Keep answers clear and practical with short bullet points."
)

USES_PROMPT = """Create a clear guide on the uses of this plant: {plant}

Use the CONTEXT first, then general knowledge. Use exactly these sections with
short bullet points:

### 💊 Medicinal / Traditional Uses
(describe as traditional or commonly reported, not as proven cures)
### 🍽️ Culinary Uses
### 🏠 Household & Ornamental Uses
### 🌍 Environmental Benefits
### ⚠️ Safety & Toxicity
(pets, children, allergies, side effects, who should avoid it)

CONTEXT:
{context}
"""

# ============================== PAGE SETUP ==================================
st.set_page_config(page_title="AI Plant Care Assistant", page_icon="🌿", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 45%, #66bb6a 100%);
    padding: 2rem 2.2rem; border-radius: 22px; color: white;
    box-shadow: 0 10px 30px rgba(46,125,50,.35); margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
}
.hero::after {
    content: "🌿"; position: absolute; right: 25px; top: -10px;
    font-size: 8rem; opacity: .15;
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: white; }
.hero p  { margin: .4rem 0 0 0; font-size: 1.02rem; opacity: .93; }
.badge {
    display: inline-block; background: rgba(255,255,255,.2);
    padding: .22rem .8rem; border-radius: 999px; font-size: .78rem;
    margin: .8rem .4rem 0 0; backdrop-filter: blur(4px);
}

/* Cards */
.card {
    background: rgba(76,175,80,.08); border: 1px solid rgba(76,175,80,.35);
    border-left: 6px solid #43a047; padding: 1rem 1.2rem;
    border-radius: 14px; margin-bottom: 1rem;
}
.card h4 { margin: 0 0 .4rem 0; color: #43a047; }
.stat {
    background: rgba(76,175,80,.10); border-radius: 14px; padding: .7rem;
    text-align: center; border: 1px solid rgba(76,175,80,.3);
}
.stat b { font-size: 1.4rem; color: #43a047; display: block; }
.stat span { font-size: .75rem; opacity: .8; }

/* Buttons */
.stButton > button {
    border-radius: 12px; border: 1px solid rgba(76,175,80,.5);
    font-weight: 500; transition: all .2s ease;
}
.stButton > button:hover {
    background: #43a047; color: white; border-color: #43a047;
    transform: translateY(-2px); box-shadow: 0 6px 14px rgba(67,160,71,.35);
}

/* Horizontal radio as nav pills */
div[role="radiogroup"] { gap: .5rem; }
div[role="radiogroup"] > label {
    background: rgba(76,175,80,.10); border: 1px solid rgba(76,175,80,.35);
    padding: .4rem 1.1rem; border-radius: 999px; cursor: pointer;
}
div[role="radiogroup"] > label:has(input:checked) {
    background: #43a047; color: white;
}
div[role="radiogroup"] > label > div:first-child { display: none; }

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 16px; padding: .8rem 1rem; margin-bottom: .6rem;
    border: 1px solid rgba(76,175,80,.25);
}
section[data-testid="stSidebar"] { border-right: 1px solid rgba(76,175,80,.3); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================== HELPERS =====================================
def read_file(uploaded) -> str:
    """Extract text from an uploaded .txt or .pdf file."""
    if uploaded.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded.read()))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded.read().decode("utf-8", errors="ignore")


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def embed(texts):
    res = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.array(res["embeddings"], dtype=np.float32)


def build_index(files):
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


def retrieve(query, index, k=TOP_K):
    q = embed([query])[0]
    q /= np.linalg.norm(q) + 1e-10
    scores = index["vectors"] @ q
    top = np.argsort(scores)[::-1][:k]
    return [(index["chunks"][i], index["sources"][i], float(scores[i])) for i in top]


def identify_plant(image_bytes):
    prompt = (
        "You are a plant expert. Look at this photo and reply in EXACTLY this format:\n"
        "Plant name: <common name (scientific name if known)>\n"
        "Health: <Healthy / Needs attention>\n"
        "Visible issues: <yellow leaves, brown tips, pests, wilting etc. or None>\n"
        "Quick tip: <one short care tip>"
    )
    res = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
    )
    return res["message"]["content"]


def extract_plant_name(info):
    m = re.search(r"Plant name:\s*(.+)", info or "", re.IGNORECASE)
    if not m:
        return ""
    name = m.group(1).strip().strip("*")
    return re.sub(r"\s*\(.*?\)", "", name).strip()  # drop scientific name in brackets


def stream_chat(messages):
    for chunk in ollama.chat(model=CHAT_MODEL, messages=messages, stream=True):
        yield chunk["message"]["content"]


def stream_answer(question, context, plant_info, history):
    user_msg = (
        f"PLANT FROM PHOTO:\n{plant_info or 'No photo uploaded.'}\n\n"
        f"CONTEXT FROM DOCUMENTS:\n{context or 'No documents uploaded.'}\n\n"
        f"QUESTION: {question}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]
    messages.append({"role": "user", "content": user_msg})
    return stream_chat(messages)


def needs_disclaimer(text):
    t = text.lower()
    return any(k in t for k in MEDICAL_KEYWORDS)


def show_sources(sources):
    if sources:
        with st.expander("📚 Sources used"):
            for text, src, score in sources:
                st.markdown(f"**{src}** · similarity `{score:.2f}`\n\n> {text}")


# ============================== SESSION STATE ===============================
st.session_state.setdefault("index", None)
st.session_state.setdefault("plant_info", "")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uses_result", None)   # (plant, text, sources)
st.session_state.setdefault("detected_plant", "")
st.session_state.setdefault("applied_plant", "")

# ============================== SIDEBAR =====================================
with st.sidebar:
    st.markdown("## 🌱 Setup")

    st.markdown("**1️⃣ Knowledge base**")
    docs = st.file_uploader("Plant-care files (.txt, .pdf)", type=["txt", "pdf"],
                            accept_multiple_files=True, label_visibility="collapsed")
    if st.button("⚙️ Process documents", use_container_width=True):
        if docs:
            with st.spinner("Reading and indexing..."):
                st.session_state.index = build_index(docs)
            if st.session_state.index:
                st.success("Documents indexed!")
            else:
                st.warning("No readable text found.")
        else:
            st.warning("Upload at least one file.")

    st.markdown("**2️⃣ Plant photo** *(optional)*")
    img = st.file_uploader("Plant image", type=["jpg", "jpeg", "png"],
                           label_visibility="collapsed")
    if img:
        st.image(img, use_container_width=True)
        if st.button("🔍 Analyze plant", use_container_width=True):
            with st.spinner("Analyzing image with LLaVA..."):
                info = identify_plant(img.getvalue())
                st.session_state.plant_info = info
                name = extract_plant_name(info)
                if name:
                    st.session_state.detected_plant = name  # auto-fills Uses tab

    st.markdown("---")
    n_chunks = len(st.session_state.index["chunks"]) if st.session_state.index else 0
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="stat"><b>{n_chunks}</b><span>Chunks</span></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="stat"><b>{len(st.session_state.messages)//2}</b>'
                f'<span>Questions</span></div>', unsafe_allow_html=True)
    st.write("")
    if st.button("🗑️ Clear everything", use_container_width=True):
        st.session_state.messages = []
        st.session_state.plant_info = ""
        st.session_state.uses_result = None
        st.session_state.detected_plant = ""
        st.session_state.applied_plant = ""
        st.session_state.pop("plant_name_input", None)
        st.rerun()
    st.caption(f"🤖 {CHAT_MODEL} · 👁️ {VISION_MODEL} · 🧬 {EMBED_MODEL}")

# ============================== HERO ========================================
st.markdown(
    """
    <div class="hero">
        <h1>🌿 AI Plant Care Assistant</h1>
        <p>Your personal gardening expert: watering, sunlight, maintenance,
        and the medicinal & everyday uses of your plants.</p>
        <span class="badge">💧 Watering</span><span class="badge">☀️ Sunlight</span>
        <span class="badge">🛠️ Maintenance</span><span class="badge">💊 Medicinal Uses</span>
        <span class="badge">📷 Image Analysis</span><span class="badge">🔒 100% Local</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.plant_info:
    st.markdown(
        f'<div class="card"><h4>📷 Plant analysis</h4>'
        f'{st.session_state.plant_info.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

mode = st.radio("Mode", ["💬 Ask the Assistant", "🌼 Plant Uses & Benefits", "ℹ️ About"],
                horizontal=True, label_visibility="collapsed")
st.write("")

# ============================== MODE 1: CHAT ================================
if mode == "💬 Ask the Assistant":
    quick = {
        "💧 Watering": "How often should I water this plant?",
        "☀️ Sunlight": "How much sunlight does this plant need?",
        "🛠️ Maintenance": "What general maintenance does this plant need?",
        "🐛 Problems": "What are common problems and how do I fix them?",
        "💊 Uses": "What are the uses and medicinal benefits of this plant?",
    }
    cols = st.columns(len(quick))
    clicked = None
    for col, (label, q) in zip(cols, quick.items()):
        if col.button(label, use_container_width=True):
            clicked = q

    if not st.session_state.messages:
        st.markdown(
            '<div class="card"><h4>👋 Welcome!</h4>Upload your plant-care notes and/or a '
            'plant photo in the sidebar, then ask a question below. Try: '
            '<i>"What are the medicinal uses of tulsi?"</i> or '
            '<i>"How often should I water aloe vera?"</i></div>',
            unsafe_allow_html=True,
        )

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑‍🌾" if m["role"] == "user" else "🌿"):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_sources(m.get("sources"))

    question = st.chat_input("Ask about watering, sunlight, uses, medicinal benefits...") or clicked

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🌾"):
            st.markdown(question)

        sources, context = [], ""
        if st.session_state.index:
            plant = extract_plant_name(st.session_state.plant_info)
            sources = retrieve(f"{plant} {question}".strip(), st.session_state.index)
            context = "\n\n".join(f"[{s}] {t}" for t, s, _ in sources)

        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]]

        with st.chat_message("assistant", avatar="🌿"):
            try:
                answer = st.write_stream(
                    stream_answer(question, context, st.session_state.plant_info, history)
                )
                if needs_disclaimer(question):
                    st.warning(DISCLAIMER)
                    answer += "\n\n" + DISCLAIMER
                show_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Could not reach Ollama. Is it running and are the models pulled?\n\n{e}")

# ============================== MODE 2: USES ================================
elif mode == "🌼 Plant Uses & Benefits":
    st.markdown(
        '<div class="card"><h4>🌼 Discover what a plant is useful for</h4>'
        'Enter a plant name (or analyze a photo, the name is filled automatically) '
        'to get its medicinal, culinary, household and environmental uses, '
        'with safety information.</div>',
        unsafe_allow_html=True,
    )
    det = st.session_state.detected_plant
    if det and st.session_state.applied_plant != det:
        st.session_state.plant_name_input = det
        st.session_state.applied_plant = det
    c1, c2 = st.columns([3, 1])
    plant_name = c1.text_input("Plant name", key="plant_name_input",
                               placeholder="e.g. Tulsi, Aloe Vera, Neem, Snake Plant")
    c2.write("")
    c2.write("")
    go = c2.button("✨ Get uses", use_container_width=True, type="primary")

    if go:
        if not plant_name.strip():
            st.warning("Please enter a plant name first.")
        else:
            sources, context = [], ""
            if st.session_state.index:
                sources = retrieve(f"{plant_name} uses medicinal benefits",
                                   st.session_state.index)
                context = "\n\n".join(f"[{s}] {t}" for t, s, _ in sources)
            prompt = USES_PROMPT.format(plant=plant_name.strip(),
                                        context=context or "No documents uploaded.")
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}]
            try:
                with st.chat_message("assistant", avatar="🌿"):
                    text = st.write_stream(stream_chat(messages))
                st.session_state.uses_result = (plant_name.strip(), text, sources)
                st.rerun()
            except Exception as e:
                st.error(f"Could not reach Ollama. Is it running?\n\n{e}")

    if st.session_state.uses_result:
        name, text, srcs = st.session_state.uses_result
        st.markdown(f"## 🌿 {name.title()}")
        st.markdown(text)
        st.warning(DISCLAIMER)
        show_sources(srcs)

# ============================== MODE 3: ABOUT ===============================
else:
    st.markdown("""
<div class="card"><h4>ℹ️ How it works</h4>

**1. Documents → Knowledge base:** your files are split into chunks and converted to
vectors with `nomic-embed-text`.<br>
**2. Question → Retrieval:** the most similar chunks are found with cosine similarity.<br>
**3. Photo → Plant identity:** `llava` identifies the plant and visible health issues.<br>
**4. Answer:** `llama3.2` combines the question, the photo analysis and the retrieved
notes, and streams a grounded answer with sources.<br>
**5. Plant Uses:** a dedicated mode that produces medicinal, culinary, household,
environmental and safety information for any plant.
</div>

<div class="card"><h4>✅ Why it's useful</h4>

- Works offline and is free, so no data leaves your computer<br>
- Answers are grounded in your own documents, which reduces made-up advice<br>
- Helps beginners identify plants from a photo<br>
- Explains everyday and traditional uses, with safety warnings for pets and children
</div>

<div class="card"><h4>⚠️ Limitations</h4>

- Image identification may be inaccurate, and small local models can make mistakes<br>
- Medicinal information is educational only, not medical advice
</div>
""", unsafe_allow_html=True)
