"""
Admin Dashboard — Knowledge Base Management
"""

import os, json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from modules import store, knowledge_base as kb

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
CONFIG = json.loads((Path(__file__).parent.parent / "config.json").read_text()) \
    if (Path(__file__).parent.parent / "config.json").exists() else {}

st.set_page_config(page_title="Admin · Knowledge Base", page_icon="📋", layout="wide")
store.init_db()

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    background-color: #0D0E15 !important; color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] { background: #161927 !important; }
[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.stMetric label, .stMetric [data-testid="stMetricValue"] { color: #F1F5F9 !important; }

/* Drag-drop animation */
@keyframes dropBounce {
    0%   { transform: scale(1); }
    50%  { transform: scale(1.03); }
    100% { transform: scale(1); }
}
[data-testid="stFileUploader"]:hover {
    animation: dropBounce 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}
[data-testid="stFileUploader"] {
    background: #161927 !important;
    border: 2px dashed #2D3250 !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #B8A1EA !important;
}

.doc-card {
    background: #161927; border: 1px solid #1e2235; border-radius: 10px;
    padding: 16px 20px; margin: 8px 0;
}
.doc-card h4 { margin: 0 0 6px 0; color: #F1F5F9; }
.status-active { color: #86EFAC; }
.status-deleted { color: #FCA5A5; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.markdown("## 📋 Knowledge Base Management")
st.markdown("Upload business documents, view indexing status, and manage your knowledge base.")

# ── Upload ───────────────────────────────────────────────────────────
st.markdown("### Upload Documents")
uploaded = st.file_uploader(
    "Drag & drop files here (.pdf, .docx, .txt)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="kb_upload",
)

if uploaded:
    chunk_size = CONFIG.get("chunk_size", 400)
    overlap = CONFIG.get("chunk_overlap", 80)

    for f in uploaded:
        with st.status(f"Processing **{f.name}** …", expanded=True) as status:
            try:
                st.write("📄 Extracting text …")
                doc_id = kb.add_document(f, API_KEY, chunk_size, overlap)
                st.write(f"✅ Indexed successfully (doc #{doc_id})")
                status.update(label=f"{f.name} — Done ✓", state="complete")
            except Exception as e:
                st.error(f"Failed: {e}")
                status.update(label=f"{f.name} — Error", state="error")

    # Reload KB
    st.session_state.kb_loaded = False
    st.rerun()

# ── Document Table ───────────────────────────────────────────────────
st.markdown("### Indexed Documents")
docs = store.list_documents()

if not docs:
    st.info("No documents uploaded yet. Use the uploader above to add files.")
else:
    for doc in docs:
        status_class = "status-active" if doc["status"] == "active" else "status-deleted"
        st.markdown(f"""
        <div class="doc-card">
            <h4>📄 {doc['name']}</h4>
            <span class="{status_class}">{doc['status'].upper()}</span> ·
            <span style="color:#94A3B8">{doc['chunk_count']} chunks · {doc['file_type']} · {doc['uploaded_at']}</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 4])
        if doc["status"] == "active":
            if col1.button("🗑 Remove", key=f"del_{doc['id']}"):
                store.soft_delete_document(doc["id"])
                st.session_state.kb_loaded = False
                st.rerun()
        if col2.button("🔍 Chunks", key=f"chunks_{doc['id']}"):
            with st.expander(f"Chunks for {doc['name']}", expanded=True):
                chunks_data = store.load_active_chunks()
                doc_chunks = [c for c in chunks_data if c.get("source_file") == doc["name"]]
                for i, ch in enumerate(doc_chunks[:20]):
                    st.markdown(f"**Chunk {i+1}** (Page {ch.get('page_number', '?')})")
                    st.code(ch["text"][:300], language=None)
                if len(doc_chunks) > 20:
                    st.caption(f"… and {len(doc_chunks) - 20} more chunks")

# ── Stats ────────────────────────────────────────────────────────────
st.markdown("### Statistics")
active = [d for d in docs if d["status"] == "active"]
total_chunks = sum(d["chunk_count"] for d in active)
c1, c2, c3 = st.columns(3)
c1.metric("Active Documents", len(active))
c2.metric("Total Chunks", total_chunks)
c3.metric("Avg Chunks/Doc", round(total_chunks / max(len(active), 1), 1))
