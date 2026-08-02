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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.18) 0px, transparent 50%),
               radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.18) 0px, transparent 50%),
               linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 40%, #F5F3FF 100%) !important;
    background-attachment: fixed !important;
    color: #0F172A !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stMainBlockContainer"] {
    position: relative !important;
    border: 2px solid rgba(99, 102, 241, 0.28) !important;
    border-radius: 24px !important;
    padding: 32px 36px !important;
    margin-top: 14px !important;
    margin-bottom: 24px !important;
    background: linear-gradient(165deg, rgba(255, 255, 255, 0.65) 0%, rgba(248, 250, 252, 0.45) 100%) !important;
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    box-shadow: 0 20px 50px -10px rgba(99, 102, 241, 0.18),
                inset 0 1.5px 3px rgba(255, 255, 255, 0.95),
                inset 0 -2px 6px rgba(99, 102, 241, 0.08) !important;
}

[data-testid="stMainBlockContainer"]::before {
    content: '';
    position: absolute;
    top: -2px; left: 24px; right: 24px;
    height: 5px;
    border-radius: 6px 6px 8px 8px;
    background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 30%, #EC4899 65%, #06B6D4 100%);
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.5);
    z-index: 10;
}

[data-testid="stSidebar"] {
    background: linear-gradient(165deg, rgba(255, 255, 255, 0.78) 0%, rgba(241, 245, 249, 0.6) 100%) !important;
    backdrop-filter: blur(28px) saturate(220%) !important;
    border-right: 2.5px solid rgba(168, 85, 247, 0.35) !important;
    box-shadow: 12px 0 35px -10px rgba(99, 102, 241, 0.2) !important;
}

.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    border: 1.5px solid rgba(255, 255, 255, 0.8) !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(238, 242, 255, 0.65) 100%) !important;
    backdrop-filter: blur(16px) saturate(200%) !important;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.18), inset 0 1.5px 2px rgba(255, 255, 255, 0.95) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 18px 35px -8px rgba(99, 102, 241, 0.35) !important;
}

[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(241, 245, 249, 0.6) 100%) !important;
    backdrop-filter: blur(20px) !important;
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 18px !important;
    box-shadow: 0 14px 35px -10px rgba(99, 102, 241, 0.15), inset 0 1.5px 2px #FFFFFF !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #6366F1 !important;
    transform: scale(1.01) !important;
}

.doc-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(248, 250, 252, 0.65) 100%) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.85) !important;
    border-radius: 18px !important;
    padding: 18px 24px !important;
    margin: 12px 0 !important;
    box-shadow: 0 12px 30px -8px rgba(99, 102, 241, 0.15), inset 0 1.5px 2px rgba(255, 255, 255, 0.95) !important;
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
    "Drag & drop files here (All formats enabled: .txt, .pdf, .docx, .csv, .md, .json, etc.):",
    type=None,  # All file formats enabled
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
