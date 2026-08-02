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
    background-color: #F8FAFC !important; color: #0F172A !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] { background: #F1F5F9 !important; border-right: 1px solid #E2E8F0 !important; }
[data-testid="stSidebar"] * { color: #0F172A !important; }
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.stMetric label { color: #64748B !important; }
.stMetric [data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }

/* Drag-drop animation */
@keyframes dropBounce {
    0%   { transform: scale(1); }
    50%  { transform: scale(1.02); }
    100% { transform: scale(1); }
}
[data-testid="stFileUploader"]:hover {
    animation: dropBounce 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366F1 !important;
}

.doc-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 16px 20px; margin: 8px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}
.doc-card h4 { margin: 0 0 6px 0; color: #0F172A; }
.status-active { color: #059669; font-weight: 700; }
.status-deleted { color: #E11D48; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# Automatically clean up any past duplicate records on page load
store.deduplicate_documents()

if "processed_file_signatures" not in st.session_state:
    st.session_state.processed_file_signatures = set()

# ── Header ───────────────────────────────────────────────────────────
st.markdown("## 📋 Knowledge Base Management")
st.markdown("Upload business documents, view indexing status, and manage your ZeroBT knowledge base.")

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
    new_files_processed = False

    for f in uploaded:
        sig = f"{f.name}_{f.size}"
        if sig in st.session_state.processed_file_signatures:
            continue

        with st.status(f"Processing **{f.name}** …", expanded=True) as status:
            try:
                st.write("📄 Extracting text …")
                doc_id = kb.add_document(f, API_KEY, chunk_size, overlap)
                st.session_state.processed_file_signatures.add(sig)
                new_files_processed = True
                st.write(f"✅ Indexed successfully (doc #{doc_id})")
                status.update(label=f"{f.name} — Done ✓", state="complete")
            except Exception as e:
                st.error(f"Failed: {e}")
                status.update(label=f"{f.name} — Error", state="error")

    if new_files_processed:
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
            <span style="color:#64748B">{doc['chunk_count']} chunks · {doc['file_type']} · {doc['uploaded_at']}</span>
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
