"""
AI Customer Support Chatbot — Main Chat Interface
==================================================
Streaming RAG answers · Frustration gauge · Voice I/O · Escalation · CSAT
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
    page_title="AI Support · Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════
# CSS — Dark Slate + Pastel Highlights
# ═════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-main:    #0D0E15;
    --bg-card:    #161927;
    --bg-user:    #2D3250;
    --accent-bot: #B8A1EA;
    --accent-usr: #7DD3FC;
    --lvl1:       #86EFAC;
    --lvl2:       #FDE047;
    --lvl34:      #FCA5A5;
    --txt:        #F1F5F9;
    --txt-muted:  #94A3B8;
}

/* ── Global ── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--txt) !important;
    font-family: 'Inter', sans-serif !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid #1e2235;
}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--txt) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border-radius: 14px !important;
    border: 1px solid #1e2235 !important;
    animation: msgSpring 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    margin-bottom: 8px !important;
}
@keyframes msgSpring {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: var(--txt) !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border-color: #2D3250 !important;
}
[data-testid="stChatInputTextArea"] {
    color: var(--txt) !important;
    background: var(--bg-card) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(184,161,234,0.2);
}
button[data-testid="stBaseButton-primary"] {
    background: var(--accent-bot) !important;
    color: var(--bg-main) !important;
}

/* ── Expander (source citations) ── */
[data-testid="stExpander"] {
    background: #1a1d2e !important;
    border: 1px solid #2D3250 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p,
[data-testid="stExpander"] span {
    color: var(--txt-muted) !important;
}

/* ── Metrics / badges ── */
.stMetric label, .stMetric [data-testid="stMetricValue"] {
    color: var(--txt) !important;
}

/* ── Frustration Gauge ── */
.gauge-container { text-align: center; margin: 12px 0; }
.gauge-label {
    font-size: 11px; color: var(--txt-muted);
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;
}
.gauge-score {
    font-size: 32px; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Typing Indicator ── */
@keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}
.typing-dots span {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent-bot);
    border-radius: 50%;
    margin: 0 3px;
    animation: dotPulse 1.4s ease-in-out infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

/* ── AI Glow ── */
@keyframes aiGlow {
    0%, 100% { box-shadow: 0 0 12px rgba(184,161,234,0.2); }
    50%      { box-shadow: 0 0 28px rgba(184,161,234,0.5); }
}
.ai-glow { animation: aiGlow 2s ease-in-out infinite; border-radius: 50%; }

/* ── Escalation Banner ── */
@keyframes escPulse {
    0%, 100% { border-color: var(--lvl34); }
    50%      { border-color: rgba(252,165,165,0.3); }
}
.esc-banner {
    background: linear-gradient(135deg, #2a1520, #1e1525);
    border: 2px solid var(--lvl34);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    animation: escPulse 2s ease infinite;
}

/* ── Rating Stars ── */
.star-btn { font-size: 28px; cursor: pointer; transition: transform 0.15s; }
.star-btn:hover { transform: scale(1.3); }

/* ── Misc ── */
.section-label {
    font-size: 11px; color: var(--txt-muted);
    text-transform: uppercase; letter-spacing: 1.2px;
    margin: 16px 0 6px 0;
}
hr { border-color: #1e2235 !important; }
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
    st.markdown("## 💬 AI Support")
    st.markdown(f'<p class="section-label">Session: {st.session_state.session_id}</p>',
                unsafe_allow_html=True)
    st.divider()

    # ── Frustration Gauge ──
    score = st.session_state.tracker.score
    if score < 40:
        g_color = "#86EFAC"
    elif score < 70:
        g_color = "#FDE047"
    else:
        g_color = "#FCA5A5"

    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-label">Frustration Level</div>
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
        st.warning("No documents uploaded. Go to **Admin Dashboard** to add files.")
    st.divider()

    # ── Voice Input ──
    st.markdown(f'<p class="section-label">Voice Input</p>', unsafe_allow_html=True)
    audio = st.audio_input("🎤 Tap to speak", key="voice_input")

    # ── New Session ──
    st.divider()
    if st.button("🔄 New Chat Session", use_container_width=True):
        # Save current conversation
        if st.session_state.messages:
            store.save_conversation(
                st.session_state.session_id,
                st.session_state.messages,
                st.session_state.tracker.history,
                "closed",
            )
        # Reset
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

st.markdown("### Customer Support Chat")

# ── Render history ──
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Source citations for bot messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['file']}** — Page {src['page']}")

# ── Escalation banner ──
if st.session_state.escalated and st.session_state.escalation_info:
    info = st.session_state.escalation_info
    contact = info.get("contact", {})
    st.markdown(f"""
    <div class="esc-banner">
        <strong>⚠ Escalated to {contact.get('name','Agent')}
        ({contact.get('role','Support')}) — Level {info.get('level',1)}</strong><br>
        <span style="color:var(--txt-muted);font-size:13px;">
            A support agent has been notified and will follow up shortly.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── CSAT Rating ──
if st.session_state.escalated and not st.session_state.rated:
    st.markdown("---")
    st.markdown("**How was your experience?**")
    cols = st.columns(5)
    for i in range(5):
        if cols[i].button("⭐" * (i + 1), key=f"star_{i}"):
            store.save_rating(st.session_state.session_id, i + 1)
            st.session_state.rated = True
            st.rerun()

# ═════════════════════════════════════════════════════════════════════
# TTS Helper
# ═════════════════════════════════════════════════════════════════════

def _speak_js(text: str):
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("'", "\\'")
    st.components.v1.html(f"""
    <script>
    const u = new SpeechSynthesisUtterance(`{safe}`);
    u.rate = 1.0; u.pitch = 1.0;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
    </script>
    """, height=0)

# ═════════════════════════════════════════════════════════════════════
# PROCESS USER INPUT
# ═════════════════════════════════════════════════════════════════════

user_input = st.chat_input("Type your message…")
# Use voice input if available and no typed input
if voice_text and not user_input:
    user_input = voice_text

if user_input:
    # ── Add user message ──
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # ── Sentiment analysis ──
    analysis = sentiment.analyze_message(user_input, API_KEY)
    frust_score = st.session_state.tracker.update(analysis)

    # ── Check for auto-escalation via sentiment ──
    auto_level = st.session_state.tracker.should_auto_escalate()
    wants_human = analysis.get("wants_human", False)

    if (auto_level or wants_human) and not st.session_state.escalated:
        # Assess seriousness
        assessed = sentiment.assess_seriousness(
            st.session_state.messages, frust_score, API_KEY
        )
        esc_level = max(auto_level or 1, assessed.get("level", 1))
        reason = assessed.get("reason", "Emotional signal detected")
        if wants_human:
            reason = "Customer explicitly requested a human agent. " + reason

        # Escalate
        with st.spinner("Escalating to a support agent…"):
            esc_result = escalation.escalate(
                st.session_state.session_id,
                st.session_state.messages,
                esc_level, reason,
                API_KEY, GMAIL_USER, GMAIL_PASS,
            )
        st.session_state.escalated = True
        st.session_state.escalation_info = esc_result

        bot_msg = (
            f"I understand your frustration. I've escalated your case to "
            f"**{esc_result['contact'].get('name', 'our team')}** "
            f"({esc_result['contact'].get('role', 'Support')}) who will follow up shortly. "
            f"Your reference ID is **{st.session_state.session_id}**."
        )
        st.session_state.messages.append({"role": "assistant", "content": bot_msg, "sources": []})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(bot_msg)

    else:
        # ── RAG Response ──
        if not st.session_state.chunks:
            bot_msg = ("I don't have any knowledge base documents loaded yet. "
                       "Please ask an admin to upload business documents so I can help you.")
            st.session_state.messages.append({"role": "assistant", "content": bot_msg, "sources": []})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(bot_msg)
        else:
            # Hybrid search + re-rank
            candidates = rag_engine.hybrid_search(
                user_input, st.session_state.chunks, st.session_state.vectors,
                st.session_state.bm25, API_KEY,
                CONFIG.get("top_k", 5) * 2,
            )
            reranked = rag_engine.rerank(user_input, candidates, API_KEY,
                                         CONFIG.get("top_k", 5))

            # Streaming response
            with st.chat_message("assistant", avatar="🤖"):
                # Typing indicator
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
                placeholder.markdown(full_response)

                # Parse metadata
                meta = rag_engine.parse_answer_metadata(full_response)
                clean_text = meta["clean_text"]
                confidence = meta["confidence"]
                sources = meta["sources"]

                placeholder.markdown(clean_text)

                # Source citations
                if sources:
                    with st.expander("📚 Sources"):
                        for src in sources:
                            st.markdown(f"**{src.get('file', '?')}** — Page {src.get('page', '?')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_text,
                "sources": sources,
            })

            # ── Confidence check → knowledge gap / escalation ──
            threshold = CONFIG.get("confidence_threshold", 0.6)
            if confidence < threshold and not st.session_state.escalated:
                store.log_knowledge_gap(user_input)

                assessed = sentiment.assess_seriousness(
                    st.session_state.messages, frust_score, API_KEY
                )
                if assessed.get("level", 1) >= 2:
                    reason = f"Low confidence ({confidence:.0%}). " + assessed.get("reason", "")
                    with st.spinner("Escalating…"):
                        esc_result = escalation.escalate(
                            st.session_state.session_id,
                            st.session_state.messages,
                            assessed["level"], reason,
                            API_KEY, GMAIL_USER, GMAIL_PASS,
                        )
                    st.session_state.escalated = True
                    st.session_state.escalation_info = esc_result

    # ── Persist conversation ──
    store.save_conversation(
        st.session_state.session_id,
        st.session_state.messages,
        st.session_state.tracker.history,
    )
    st.rerun()
