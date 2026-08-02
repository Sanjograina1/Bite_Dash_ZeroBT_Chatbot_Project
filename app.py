"""
ZeroBT AI Support Chatbot — Main Chat Interface
==================================================
Pastel Matte Theme · Streaming RAG · Frustration Tracking · Gmail Escalation (8 Levels)
Run:  streamlit run app.py
"""

import json, os, re
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from modules import store, knowledge_base as kb, rag_engine, sentiment, escalation

# ── Env & Config ─────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_PASS = os.getenv("GMAIL_APP_PASSWORD", "")
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text()) if (Path(__file__).parent / "config.json").exists() else {}

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZeroBT · AI Customer Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════
# CSS — Pastel Bright & Matte Light Styling (High Contrast & Visible Fonts)
# ═════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg-main:    #F8FAFC;
    --bg-card:    #FFFFFF;
    --bg-user:    #EEF2FF;
    --bg-bot:     #FAF5FF;
    --border-user:#C7D2FE;
    --border-bot: #E9D5FF;
    --accent-bot: #7C3AED;
    --accent-usr: #4338CA;
    --txt-main:   #0F172A;
    --txt-muted:  #475569;
    --lvl1:       #10B981;
    --lvl2:       #059669;
    --lvl3:       #D97706;
    --lvl4:       #E11D48;
    --lvl5:       #EA580C;
    --lvl6:       #C026D3;
    --lvl7:       #E11D48;
    --lvl8:       #991B1B;
}

/* ── Global Page ── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--txt-main) !important;
    font-family: 'Inter', sans-serif !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #F1F5F9 !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--txt-main) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border-radius: 14px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
    animation: msgSpring 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    margin-bottom: 12px !important;
    padding: 14px 18px !important;
}
@keyframes msgSpring {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User Message Styling */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--bg-user) !important;
    border: 1px solid var(--border-user) !important;
}

/* Assistant Message Styling */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--bg-bot) !important;
    border: 1px solid var(--border-bot) !important;
}

/* Chat Text Elements */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: var(--txt-main) !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

/* ── Input Box ── */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
}
[data-testid="stChatInputTextArea"] {
    color: var(--txt-main) !important;
    font-size: 15px !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid #CBD5E1 !important;
    background: #FFFFFF !important;
    color: #334155 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #6366F1 !important;
    color: #4338CA !important;
    background: #EEF2FF !important;
    transform: translateY(-1px);
}
button[data-testid="stBaseButton-primary"] {
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    border: none !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background: #4338CA !important;
    box-shadow: 0 4px 12px rgba(79,70,229,0.25) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #475569 !important;
    font-weight: 600 !important;
}

/* ── Metrics ── */
.stMetric label { color: #64748B !important; font-size: 13px !important; }
.stMetric [data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }

/* ── Frustration Gauge ── */
.gauge-container { text-align: center; margin: 10px 0; }
.gauge-label {
    font-size: 12px; color: #64748B;
    text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
}
.gauge-score {
    font-size: 34px; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Typing Dots ── */
@keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}
.typing-dots span {
    display: inline-block;
    width: 8px; height: 8px;
    background: #7C3AED;
    border-radius: 50%;
    margin: 0 3px;
    animation: dotPulse 1.4s ease-in-out infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

/* ── Escalation Banner ── */
.esc-banner {
    background: #FEF2F2;
    border: 1.5px solid #FCA5A5;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 14px 0;
    color: #991B1B;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.08);
}
.esc-banner strong { font-size: 16px; color: #7F1D1D; }

.section-label {
    font-size: 11px; color: #64748B;
    text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700;
    margin: 16px 0 6px 0;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════

def _init():
    store.init_db()
    defaults = {
        "messages": [],
        "session_id": store.new_session_id(),
        "tracker": sentiment.FrustrationTracker(),
        "escalated": False,
        "escalation_info": None,
        "rated": False,
        "kb_loaded": False,
        "chunks": [],
        "vectors": None,
        "bm25": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ═════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE LOADING
# ═════════════════════════════════════════════════════════════════════

def _load_kb():
    if not st.session_state.kb_loaded:
        chunks, vectors, bm25 = kb.load_index()
        st.session_state.chunks = chunks
        st.session_state.vectors = vectors
        st.session_state.bm25 = bm25
        st.session_state.kb_loaded = True

_load_kb()

# ═════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 ZeroBT Support")
    st.markdown(f'<p class="section-label">Session ID: {st.session_state.session_id}</p>',
                unsafe_allow_html=True)
    st.divider()

    # ── Frustration Gauge ──
    score = st.session_state.tracker.score
    if score < 35:
        g_color = "#10B981"
    elif score < 65:
        g_color = "#D97706"
    else:
        g_color = "#DC2626"

    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-label">Frustration Meter</div>
        <div class="gauge-score" style="color:{g_color}">{int(score)}</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(min(score / 100.0, 1.0))
    st.divider()

    # ── KB Stats ──
    doc_count = len([d for d in store.list_documents() if d["status"] == "active"])
    chunk_count = len(st.session_state.chunks)
    st.markdown(f'<p class="section-label">Knowledge Base</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Docs", doc_count)
    c2.metric("Chunks", chunk_count)

    if doc_count == 0:
        st.warning("No documents uploaded yet. Add files in **Admin Dashboard**.")
    st.divider()

    # ── Voice Input ──
    st.markdown(f'<p class="section-label">Voice Input</p>', unsafe_allow_html=True)
    audio = st.audio_input("🎤 Speak query", key="voice_input")

    # ── New Session ──
    st.divider()
    if st.button("🔄 Start New Session", use_container_width=True):
        if st.session_state.messages:
            store.save_conversation(
                st.session_state.session_id,
                st.session_state.messages,
                st.session_state.tracker.history,
                "closed",
            )
        st.session_state.messages = []
        st.session_state.session_id = store.new_session_id()
        st.session_state.tracker = sentiment.FrustrationTracker()
        st.session_state.escalated = False
        st.session_state.escalation_info = None
        st.session_state.rated = False
        st.rerun()

# ═════════════════════════════════════════════════════════════════════
# VOICE → TEXT (Whisper)
# ═════════════════════════════════════════════════════════════════════

voice_text = None
if audio and API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=("audio.wav", audio.read())
        )
        voice_text = transcript.text
    except Exception:
        voice_text = None

# ═════════════════════════════════════════════════════════════════════
# CHAT AREA
# ═════════════════════════════════════════════════════════════════════

st.markdown("### ZeroBT Support Portal")
st.markdown("Ask anything regarding our policies, products, or service guidelines.")

# ── Render history ──
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Referenced Sources"):
                for src in msg["sources"]:
                    st.markdown(f"• **{src['file']}** — Page {src['page']}")

# ── Escalation banner ──
if st.session_state.escalated and st.session_state.escalation_info:
    info = st.session_state.escalation_info
    contact = info.get("contact", {})
    st.markdown(f"""
    <div class="esc-banner">
        <strong>⚠ Case Escalated to {contact.get('name','Support Team')} ({contact.get('role','Executive')}) — Level {info.get('level',1)}</strong><br>
        <span style="font-size:14px;color:#7F1D1D;">
            Our support hierarchy has been alerted via email. A ticket (ID: <code>{st.session_state.session_id}</code>) has been dispatched for review.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── CSAT Rating ──
if st.session_state.escalated and not st.session_state.rated:
    st.markdown("---")
    st.markdown("**Rate your ZeroBT support interaction:**")
    cols = st.columns(5)
    for i in range(5):
        if cols[i].button("⭐" * (i + 1), key=f"star_{i}"):
            store.save_rating(st.session_state.session_id, i + 1)
            st.session_state.rated = True
            st.rerun()

# ═════════════════════════════════════════════════════════════════════
# PROCESS USER INPUT
# ═════════════════════════════════════════════════════════════════════

user_input = st.chat_input("Type your question here…")
if voice_text and not user_input:
    user_input = voice_text

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # ── Sentiment analysis ──
    analysis = sentiment.analyze_message(user_input, API_KEY)
    frust_score = st.session_state.tracker.update(analysis)

    auto_level = st.session_state.tracker.should_auto_escalate()
    wants_human = analysis.get("wants_human", False)

    if (auto_level or wants_human) and not st.session_state.escalated:
        assessed = sentiment.assess_seriousness(
            st.session_state.messages, frust_score, API_KEY
        )
        esc_level = max(auto_level or 1, assessed.get("level", 1))
        reason = assessed.get("reason", "Customer frustration / escalation signal")
        if wants_human:
            reason = "Customer requested human intervention. " + reason

        with st.spinner("Escalating query to support team…"):
            esc_result = escalation.escalate(
                st.session_state.session_id,
                st.session_state.messages,
                esc_level, reason,
                API_KEY, GMAIL_USER, GMAIL_PASS,
            )
        st.session_state.escalated = True
        st.session_state.escalation_info = esc_result

        bot_msg = (
            f"I hear you. I have escalated your issue to "
            f"**{esc_result['contact'].get('name', 'our team')}** "
            f"({esc_result['contact'].get('role', 'Support')}) via email for immediate assistance. "
            f"Reference Ticket ID: `{st.session_state.session_id}`."
        )
        st.session_state.messages.append({"role": "assistant", "content": bot_msg, "sources": []})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(bot_msg)

    else:
        # ── Knowledge Base Verification & RAG ──
        if not st.session_state.chunks:
            # Trigger escalation to Business Director (Level 7) and Founder (Level 8)
            reason = "Knowledge base empty. Query requires executive review."
            store.log_knowledge_gap(user_input)
            with st.spinner("Escalating to Business Director & Founder…"):
                esc_result = escalation.escalate(
                    st.session_state.session_id,
                    st.session_state.messages,
                    7, reason,
                    API_KEY, GMAIL_USER, GMAIL_PASS,
                )
            st.session_state.escalated = True
            st.session_state.escalation_info = esc_result

            bot_msg = (
                "The policy documents do not contain the necessary information for your request. "
                "I am escalating this query to our Business Director and Founder for further review."
            )
            st.session_state.messages.append({"role": "assistant", "content": bot_msg, "sources": []})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(bot_msg)
        else:
            candidates = rag_engine.hybrid_search(
                user_input, st.session_state.chunks, st.session_state.vectors,
                st.session_state.bm25, API_KEY,
                CONFIG.get("top_k", 5) * 2,
            )
            reranked = rag_engine.rerank(user_input, candidates, API_KEY, CONFIG.get("top_k", 5))

            with st.chat_message("assistant", avatar="🤖"):
                typing_ph = st.empty()
                typing_ph.markdown(
                    '<div class="typing-dots"><span></span><span></span><span></span></div>',
                    unsafe_allow_html=True,
                )

                placeholder = st.empty()
                full_response = ""
                for token in rag_engine.generate_answer_stream(
                    user_input, reranked, st.session_state.messages, API_KEY
                ):
                    full_response += token
                    typing_ph.empty()
                    placeholder.markdown(full_response + "▌")

                meta = rag_engine.parse_answer_metadata(full_response)
                clean_text = meta["clean_text"]
                confidence = meta["confidence"]
                sources = meta["sources"]

                # Missing policy info detection
                missing_info_phrase = "policy documents do not contain the necessary information"
                if missing_info_phrase.lower() in clean_text.lower() or confidence < CONFIG.get("confidence_threshold", 0.6):
                    clean_text = (
                        "The policy documents do not contain the necessary information for your request. "
                        "I am escalating this query to our Business Director and Founder for further review."
                    )
                    confidence = 0.0

                placeholder.markdown(clean_text)

                if sources and confidence >= CONFIG.get("confidence_threshold", 0.6):
                    with st.expander("📚 Referenced Sources"):
                        for src in sources:
                            st.markdown(f"• **{src.get('file', '?')}** — Page {src.get('page', '?')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_text,
                "sources": sources if confidence >= CONFIG.get("confidence_threshold", 0.6) else [],
            })

            # Check if escalation needed for missing info or low confidence
            if confidence < CONFIG.get("confidence_threshold", 0.6) and not st.session_state.escalated:
                store.log_knowledge_gap(user_input)
                reason = f"Knowledge gap / policy information missing for query: '{user_input[:60]}'"
                with st.spinner("Escalating to Business Director & Founder…"):
                    esc_result = escalation.escalate(
                        st.session_state.session_id,
                        st.session_state.messages,
                        7, # Level 7 (Business Director)
                        reason,
                        API_KEY, GMAIL_USER, GMAIL_PASS,
                    )
                st.session_state.escalated = True
                st.session_state.escalation_info = esc_result

    store.save_conversation(
        st.session_state.session_id,
        st.session_state.messages,
        st.session_state.tracker.history,
    )
    st.rerun()
