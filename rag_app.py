import os
import streamlit as st
import tempfile
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ================= TOKEN =================
HF_TOKEN = os.getenv("HF_TOKEN")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }

.stApp { background: #f5f7ff; }
header[data-testid="stHeader"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #e8ecf8 !important;
    box-shadow: 4px 0 24px rgba(99,102,241,0.06) !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem !important; }

.sidebar-logo {
    display: flex; align-items: center; gap: 12px;
    padding: 0.4rem 0 1.6rem 0;
    border-bottom: 1.5px solid #f0f2ff;
    margin-bottom: 1.4rem;
}
.sidebar-logo-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3);
}
.sidebar-logo-text { font-size: 1.15rem; font-weight: 800; color: #1e1b4b; letter-spacing: -0.5px; }
.sidebar-logo-text span { color: #6366f1; }
.sidebar-logo-sub { font-family: 'Fira Code', monospace; font-size: 0.6rem; color: #a5b4fc; letter-spacing: 1px; }

.sidebar-section {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.62rem; font-weight: 500; color: #a0aec0;
    letter-spacing: 2.5px; text-transform: uppercase;
    margin: 1.3rem 0 0.7rem 0;
    display: flex; align-items: center; gap: 6px;
}

/* Sliders */
[data-testid="stSlider"] label { color: #64748b !important; font-size: 0.8rem !important; font-weight: 500 !important; }
div[data-baseweb="slider"] [role="slider"] { background: #6366f1 !important; border-color: #6366f1 !important; }
div[data-baseweb="slider"] div[data-testid="stSliderTrackFill"] { background: #6366f1 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #fafbff !important;
    border: 2px dashed #c7d2fe !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; }
[data-testid="stFileUploader"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important; width: 100% !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    transition: all 0.2s ease !important; letter-spacing: 0.2px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 12px !important; border: none !important; font-size: 0.82rem !important; }

/* ── Topbar ── */
.chat-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2.5rem;
    background: #ffffff;
    border-bottom: 1.5px solid #e8ecf8;
    box-shadow: 0 2px 12px rgba(99,102,241,0.06);
    position: sticky; top: 0; z-index: 100;
}
.topbar-left { display: flex; align-items: center; gap: 12px; }
.topbar-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #eef2ff, #ede9fe);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.topbar-title { font-size: 1rem; font-weight: 700; color: #1e1b4b; }
.topbar-doc { font-family: 'Fira Code', monospace; font-size: 0.7rem; color: #94a3b8; margin-top: 1px; }
.topbar-right { display: flex; align-items: center; gap: 8px; }
.status-pill {
    display: flex; align-items: center; gap: 6px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 20px; padding: 0.3rem 0.8rem;
    font-family: 'Fira Code', monospace; font-size: 0.68rem; color: #16a34a;
}
.status-pill.offline { background: #fff1f2; border-color: #fecdd3; color: #e11d48; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Messages ── */
.chat-area { padding: 2rem 3rem; }

.msg-row { display: flex; gap: 12px; max-width: 860px; margin-bottom: 1.4rem; }
.msg-row.user { flex-direction: row-reverse; margin-left: auto; }

.msg-avatar {
    width: 36px; height: 36px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.msg-avatar.bot { background: linear-gradient(135deg, #eef2ff, #ede9fe); border: 1.5px solid #c7d2fe; }
.msg-avatar.user { background: linear-gradient(135deg, #fdf2f8, #fce7f3); border: 1.5px solid #fbcfe8; }

.msg-bubble {
    padding: 1rem 1.3rem; border-radius: 18px;
    font-size: 0.9rem; line-height: 1.7; max-width: 700px;
}
.msg-bubble.bot {
    background: #ffffff;
    border: 1.5px solid #e8ecf8;
    color: #334155;
    border-top-left-radius: 4px;
    box-shadow: 0 2px 12px rgba(99,102,241,0.07);
}
.msg-bubble.user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    border-top-right-radius: 4px;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3);
}
.msg-meta {
    font-family: 'Fira Code', monospace; font-size: 0.62rem;
    color: #cbd5e1; margin-top: 5px; padding: 0 4px;
}
.msg-meta.user-meta { text-align: right; }

/* Sources */
.sources-label {
    font-family: 'Fira Code', monospace; font-size: 0.62rem;
    color: #94a3b8; letter-spacing: 1.5px; text-transform: uppercase;
    margin: 0.7rem 0 0.4rem 0.3rem;
}
.source-chip {
    background: #fafbff; border: 1.5px solid #e8ecf8;
    border-left: 3px solid #818cf8;
    border-radius: 10px; padding: 0.55rem 0.9rem;
    font-family: 'Fira Code', monospace; font-size: 0.7rem;
    color: #64748b; margin-bottom: 0.4rem; line-height: 1.6;
}

/* Empty state */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 52vh; gap: 1.2rem;
}
.empty-circle {
    width: 90px; height: 90px;
    background: linear-gradient(135deg, #eef2ff, #ede9fe);
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 2.5rem;
    box-shadow: 0 8px 24px rgba(99,102,241,0.15);
}
.empty-title { font-size: 1.1rem; font-weight: 700; color: #1e1b4b; }
.empty-text { font-family: 'Fira Code', monospace; font-size: 0.75rem; color: #94a3b8; text-align: center; line-height: 2; }

/* Input */
div[data-testid="stTextInput"] input {
    background: #ffffff !important;
    border: 2px solid #e8ecf8 !important;
    border-radius: 14px !important; color: #1e293b !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important; padding: 0.8rem 1.2rem !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.06) !important;
    transition: all 0.2s !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.12) !important;
}
div[data-testid="stTextInput"] input::placeholder { color: #cbd5e1 !important; }

/* Input area */
.input-area {
    padding: 1rem 2.5rem 1.8rem;
    background: #ffffff;
    border-top: 1.5px solid #e8ecf8;
}

/* Doc badge */
.doc-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: #fafbff; border: 1.5px solid #c7d2fe;
    border-radius: 10px; padding: 0.45rem 0.9rem;
    font-family: 'Fira Code', monospace; font-size: 0.72rem;
    color: #6366f1; margin-bottom: 0.9rem;
    box-shadow: 0 2px 8px rgba(99,102,241,0.08);
}

/* Stats */
.stats-row { display: flex; gap: 8px; margin-bottom: 1rem; }
.stat-box {
    flex: 1; background: linear-gradient(135deg, #fafbff, #f5f3ff);
    border: 1.5px solid #e0e7ff; border-radius: 12px;
    padding: 0.6rem 0.4rem; text-align: center;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stat-num { font-size: 1.15rem; font-weight: 800; color: #6366f1; display: block; }
.stat-label { font-size: 0.58rem; color: #a5b4fc; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

/* Send button override */
.send-btn > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
}

/* Model badge */
.model-badge {
    background: linear-gradient(135deg, #fafbff, #faf5ff);
    border: 1.5px solid #e0e7ff; border-radius: 12px;
    padding: 0.7rem 1rem; font-family: 'Fira Code', monospace;
    font-size: 0.72rem; color: #6366f1; line-height: 1.8;
}

/* Disabled input notice */
.disabled-notice {
    text-align: center; padding: 0.9rem;
    font-family: 'Fira Code', monospace; font-size: 0.75rem;
    color: #cbd5e1; background: #fafbff;
    border: 2px dashed #e8ecf8; border-radius: 14px;
    margin: 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ================= LLM =================
class HFChatLLM:
    def __init__(self):
        self.client = InferenceClient(token=HF_TOKEN)
        self.model  = "meta-llama/Meta-Llama-3-8B-Instruct"

    def generate(self, prompt):
        res = self.client.chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a RAG assistant. Answer ONLY from the provided context. If the answer is not found, say: I don't know from the document. Give clear structured answers."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=400,
            temperature=0.2,
        )
        return res.choices[0].message.content


@st.cache_resource
def get_llm():
    return HFChatLLM()


# ================= EMBEDDING =================
@st.cache_resource
def load_embed():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


# ================= BUILD INDEX =================
def build_index(file, chunk_size, overlap):
    suffix = file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(file.read())
        path = tmp.name
    docs     = TextLoader(path, encoding="utf-8").load() if suffix == "txt" else PyPDFLoader(path).load()
    chunks   = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_documents(docs)
    vectordb = FAISS.from_documents(chunks, load_embed())
    return vectordb, len(chunks), len(docs)


# ================= SESSION STATE =================
if "db"       not in st.session_state: st.session_state.db       = None
if "messages" not in st.session_state: st.session_state.messages = []
if "doc_info" not in st.session_state: st.session_state.doc_info = None


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🤖</div>
        <div>
            <div class="sidebar-logo-text">RAG<span>Bot</span></div>
            <div class="sidebar-logo-sub">POWERED BY LLAMA-3</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📁 &nbsp;Upload Document</div>', unsafe_allow_html=True)
    file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")

    if file:
        ext  = file.name.split(".")[-1].upper()
        size = round(file.size / 1024, 1)
        icon = "📕" if ext == "PDF" else "📄"
        st.markdown(f'<div class="doc-badge">{icon} &nbsp;{file.name}&nbsp;·&nbsp;{size} KB</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">⚙️ &nbsp;Settings</div>', unsafe_allow_html=True)
    chunk   = st.slider("Chunk Size", 300, 800, 500)
    overlap = st.slider("Overlap",    50,  200, 100)
    topk    = st.slider("Top K",      1,   5,   3)

    if file:
        if st.button("⚡  Process Document"):
            with st.spinner("Building index..."):
                db, total_chunks, total_docs = build_index(file, chunk, overlap)
                st.session_state.db       = db
                st.session_state.messages = []
                st.session_state.doc_info = {"name": file.name, "chunks": total_chunks, "pages": total_docs}
            st.rerun()

    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-box"><span class="stat-num">{info['pages']}</span><span class="stat-label">Pages</span></div>
            <div class="stat-box"><span class="stat-num">{info['chunks']}</span><span class="stat-label">Chunks</span></div>
            <div class="stat-box"><span class="stat-num">{topk}</span><span class="stat-label">Top K</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">🧠 &nbsp;Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-badge">
        ☁️ &nbsp;LLM · Llama-3-8B<br>
        🖥️ &nbsp;Embed · MiniLM-L6<br>
        🔍 &nbsp;Vector · FAISS
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️  Clear Conversation"):
            st.session_state.messages = []
            st.rerun()


# ================= TOPBAR =================
ready     = st.session_state.db is not None
doc_name  = st.session_state.doc_info["name"] if st.session_state.doc_info else "No document loaded"
pill_cls  = "status-pill" if ready else "status-pill offline"
pill_text = "● &nbsp;Ready" if ready else "● &nbsp;Offline"

st.markdown(f"""
<div class="chat-topbar">
    <div class="topbar-left">
        <div class="topbar-icon">💬</div>
        <div>
            <div class="topbar-title">Chat Assistant</div>
            <div class="topbar-doc">{doc_name}</div>
        </div>
    </div>
    <div class="topbar-right">
        <div class="{pill_cls}">{pill_text}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ================= CHAT MESSAGES =================
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-circle">🗂️</div>
        <div class="empty-title">Start a Conversation</div>
        <div class="empty-text">
            Upload a PDF or TXT file in the sidebar<br>
            click <b>Process Document</b><br>
            then ask anything about its contents
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div>
                    <div class="msg-bubble user">{msg["content"]}</div>
                    <div class="msg-meta user-meta">You</div>
                </div>
                <div class="msg-avatar user">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                chips = "".join(f'<div class="source-chip">{s}</div>' for s in msg["sources"])
                sources_html = f'<div class="sources-label">📎 &nbsp;Source Chunks</div>{chips}'
            st.markdown(f"""
            <div class="msg-row">
                <div class="msg-avatar bot">🤖</div>
                <div>
                    <div class="msg-bubble bot">{msg["content"]}</div>
                    <div class="msg-meta">RAGBot · Llama-3-8B</div>
                    {sources_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ================= INPUT =================
st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

if ready:
    c1, c2 = st.columns([5, 1])
    with c1:
        question = st.text_input(
            "q", placeholder="💬  Ask anything about your document...",
            label_visibility="collapsed", key="q"
        )
    with c2:
        send = st.button("Send →")

    if send and question.strip():
        retriever = st.session_state.db.as_retriever(search_kwargs={"k": topk})
        docs      = retriever.invoke(question)
        context   = "\n\n".join(d.page_content for d in docs)
        prompt    = f"Context:\n{context}\n\nQuestion:\n{question}"

        with st.spinner("Thinking..."):
            answer  = get_llm().generate(prompt)
            sources = [d.page_content[:200] + "..." for d in docs]

        st.session_state.messages.append({"role": "user",      "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()
else:
    st.markdown("""
    <div class="disabled-notice">
        ← Upload and process a document in the sidebar to start chatting
    </div>
    """, unsafe_allow_html=True)