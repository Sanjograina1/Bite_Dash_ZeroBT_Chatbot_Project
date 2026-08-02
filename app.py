"""
ZeroBT AI Support Chatbot — Main Chat Interface
==================================================
Glassmorphism Theme · Streaming RAG · gTTS Voice Output · Frustration Tracking · Gmail Escalation (8 Levels)
Run:  streamlit run app.py
"""

import io, json, os, re
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS

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
# CSS — Glassmorphism & Fluidic Pastel Styling
# ═════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg-main:    #F8FAFC;
    --bg-user:    rgba(238, 242, 255, 0.75);
    --bg-bot:     rgba(250, 245, 255, 0.85);
    --border-user:rgba(199, 210, 254, 0.8);
    --border-bot: rgba(233, 213, 255, 0.8);
    --txt-main:   #0F172A;
    --txt-muted:  #475569;
}

/* ── Fluidic Glass Background ── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 40%, #F3E8FF 100%) !important;
    background-attachment: fixed !important;
    color: var(--txt-main) !important;
    font-family: 'Inter', sans-serif !important;
}

header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Glass Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(241, 245, 249, 0.65) !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
    border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--txt-main) !important;
}

/* ── Glass Chat Message Bubbles ── */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.05) !important;
    animation: msgSpring 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    margin-bottom: 12px !important;
    padding: 16px 20px !important;
}

@keyframes msgSpring {
    from { opacity: 0; transform: translateY(10px) scale(0.99); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--bg-user) !important;
    border: 1px solid var(--border-user) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--bg-bot) !important;
    border: 1px solid var(--border-bot) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: var(--txt-main) !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

/* ── Glass Input Box ── */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1.5px solid rgba(203, 213, 225, 0.8) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04) !important;
}
[data-testid="stChatInputTextArea"] {
    color: var(--txt-main) !important;
    font-size: 15px !important;
}

/* ── Glass Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(203, 213, 225, 0.8) !important;
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: #334155 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    border-color: #6366F1 !important;
    color: #4338CA !important;
    background: rgba(238, 242, 255, 0.9) !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15) !important;
}
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25) !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.35) !important;
}

/* ── Glass Audio Toolbar ── */
.audio-toolbar {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(226, 232, 240, 0.8) !important;
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(226, 232, 240, 0.8) !important;
    border-radius: 12px !important;
}

/* ── Escalation Glass Banner ── */
.esc-banner {
    background: rgba(254, 242, 242, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border: 1.5px solid #FCA5A5 !important;
    border-radius: 14px;
    padding: 16px 20px;
    margin: 14px 0;
    color: #991B1B;
    box-shadow: 0 8px 24px rgba(239, 68, 68, 0.08);
}
.esc-banner strong { font-size: 16px; color: #7F1D1D; }

/* ── Frustration Meter ── */
.gauge-container { text-align: center; margin: 10px 0; }
.gauge-label {
    font-size: 12px; color: #64748B;
    text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
}
.gauge-score {
    font-size: 34px; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}

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
        "last_audio_bytes": None,
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

    # ── Voice Recording Input ──
    st.markdown(f'<p class="section-label">Voice Recording Input</p>', unsafe_allow_html=True)
    audio = st.audio_input("🎤 Record Audio Query", key="voice_input")

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
        st.session_state.last_audio_bytes = None
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
# CHAT AREA & AUDIO TOOLBAR
# ═════════════════════════════════════════════════════════════════════

st.markdown("### ZeroBT Support Portal")
st.markdown("Ask questions using typed text or interactive audio controls.")

# ── Glass Audio Toolbar Buttons ──
ac1, ac2, ac3 = st.columns([1.5, 1.5, 3])

read_last = ac1.button("🔊 Read Answer Aloud", use_container_width=True)
quick_audio = ac2.button("🎙️ Quick Voice Sample", use_container_width=True)

if read_last:
    assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if assistant_msgs:
        last_txt = assistant_msgs[-1]["content"]
        clean_for_speech = re.sub(r"[#*`_\-]", "", last_txt)[:300]
        try:
            tts = gTTS(text=clean_for_speech, lang="en")
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            st.session_state.last_audio_bytes = fp.read()
            st.success("Playing audio response below...")
        except Exception as e:
            st.error(f"TTS generation error: {e}")
    else:
        st.info("No assistant answer to read aloud yet.")

if st.session_state.last_audio_bytes:
    st.audio(st.session_state.last_audio_bytes, format="audio/mp3", autoplay=True)

sample_query = None
if quick_audio:
    sample_query = "What is the policy process if my query needs escalation?"

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
            Our support hierarchy has been alerted via email. Reference Ticket ID: <code>{st.session_state.session_id}</code>.
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
if sample_query:
    user_input = sample_query
elif voice_text and not user_input:
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
                placeholder = st.empty()
                full_response = ""
                for token in rag_engine.generate_answer_stream(
                    user_input, reranked, st.session_state.messages, API_KEY
                ):
                    full_response += token
                    placeholder.markdown(full_response + "▌")

                meta = rag_engine.parse_answer_metadata(full_response)
                clean_text = meta["clean_text"]
                confidence = meta["confidence"]
                sources = meta["sources"]

                missing_info_phrase = "policy documents do not contain the necessary information"
                if missing_info_phrase.lower() in clean_text.lower() or confidence < CONFIG.get("confidence_threshold", 0.6):
                    clean_text = (
                        "The policy documents do not contain the necessary information for your request. "
                        "I am escalating this query to our Business Director and Founder for further review."
                    )
                    confidence = 0.0

                placeholder.markdown(clean_text)

                # Generate TTS audio for the new answer
                try:
                    clean_for_tts = re.sub(r"[#*`_\-]", "", clean_text)[:300]
                    tts = gTTS(text=clean_for_tts, lang="en")
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.session_state.last_audio_bytes = fp.read()
                except Exception:
                    pass

                if sources and confidence >= CONFIG.get("confidence_threshold", 0.6):
                    with st.expander("📚 Referenced Sources"):
                        for src in sources:
                            st.markdown(f"• **{src.get('file', '?')}** — Page {src.get('page', '?')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_text,
                "sources": sources if confidence >= CONFIG.get("confidence_threshold", 0.6) else [],
            })

            if confidence < CONFIG.get("confidence_threshold", 0.6) and not st.session_state.escalated:
                store.log_knowledge_gap(user_input)
                reason = f"Knowledge gap / policy information missing for query: '{user_input[:60]}'"
                with st.spinner("Escalating to Business Director & Founder…"):
                    esc_result = escalation.escalate(
                        st.session_state.session_id,
                        st.session_state.messages,
                        7,
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
