"""
ZeroBT AI Support Chatbot — Main Chat Interface
==================================================
Glassmorphism Theme · Streaming RAG · Interactive Consent Escalation · gTTS Voice Output
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
        "escalation_level_index": 0,
        "pending_escalation": None,
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

    # ── Test Escalation Controls ──
    st.markdown(f'<p class="section-label">🧪 Test Escalation Tiers</p>', unsafe_allow_html=True)
    test_level = st.selectbox("Select Tier to Test", options=list(range(1, 9)),
                             format_func=lambda l: f"Level {l}: {escalation.get_hierarchy_contact(l).get('role')}")

    if st.button(f"📧 Send Test Level {test_level} Email", use_container_width=True):
        dummy_messages = st.session_state.messages if st.session_state.messages else [
            {"role": "user", "content": f"Test message triggering Level {test_level} escalation."},
            {"role": "assistant", "content": "I am looking into this issue for you."}
        ]
        reason = f"Test bot escalation trigger for Level {test_level}"
        with st.spinner(f"Sending test email for Level {test_level}…"):
            esc_result = escalation.escalate(
                st.session_state.session_id,
                dummy_messages,
                test_level, reason,
                API_KEY, GMAIL_USER, GMAIL_PASS,
            )
        if esc_result.get("sent"):
            st.success(f"✅ Level {test_level} email sent to {esc_result['to']}!")
            st.session_state.escalated = True
            st.session_state.escalation_info = esc_result
        else:
            st.error("❌ Failed to send email. Check credentials.")

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
        st.session_state.escalation_level_index = 0
        st.session_state.pending_escalation = None
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
        if msg["role"] == "assistant":
            if msg.get("query_understanding"):
                qu = msg["query_understanding"]
                with st.expander("🧠 Real-Time Query Understanding & Intent Analysis"):
                    st.markdown(f"**⚡ Detected Intent:** `{qu.get('intent', 'N/A')}`")
                    st.markdown(f"**🔍 Rewritten Search Query:** `{qu.get('rewritten_query', 'N/A')}`")
                    entities = ", ".join([f"`{e}`" for e in qu.get("entities", [])]) or "None"
                    st.markdown(f"**🏷️ Extracted Entities:** {entities}")
                    st.markdown(f"**⏱️ Urgency:** `{qu.get('urgency', 'Medium')}` · **Topic:** *{qu.get('core_topic', 'N/A')}*")
            if msg.get("sources"):
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

    user_lower = user_input.strip().lower()
    is_agree = any(w in user_lower for w in ["yes", "yep", "yeah", "sure", "please", "ok", "escalate", "agree"])
    is_decline = any(w in user_lower for w in ["no", "nope", "don't", "cancel", "fine", "nevermind"])

    # ── Real-Time Query Understanding ──
    qu_data = rag_engine.understand_query(user_input, st.session_state.messages[:-1], API_KEY)

    # ── Handle response to a pending escalation consent offer ──
    if st.session_state.pending_escalation is not None:
        pending = st.session_state.pending_escalation
        st.session_state.pending_escalation = None

        if is_agree and not is_decline:
            esc_level = pending.get("level", 1)
            reason = pending.get("reason", "Customer agreed to escalation")
            with st.spinner(f"Escalating query to Level {esc_level} support…"):
                esc_result = escalation.escalate(
                    st.session_state.session_id,
                    st.session_state.messages,
                    esc_level, reason,
                    API_KEY, GMAIL_USER, GMAIL_PASS,
                )
            st.session_state.escalated = True
            st.session_state.escalation_info = esc_result

            bot_msg = (
                f"Thank you for confirming. I have escalated your ticket to "
                f"**{esc_result['contact'].get('name', 'our team')}** "
                f"({esc_result['contact'].get('role', 'Support')}) via email for further review. "
                f"Reference Ticket ID: `{st.session_state.session_id}`.\n\n"
                f"How else can I help you today?"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_msg,
                "query_understanding": qu_data,
                "sources": []
            })
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(bot_msg)
            store.save_conversation(st.session_state.session_id, st.session_state.messages, st.session_state.tracker.history)
            st.rerun()

        elif is_decline:
            bot_msg = "Understood! I will not escalate this ticket. How else can I help you today?"
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_msg,
                "query_understanding": qu_data,
                "sources": []
            })
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(bot_msg)
            store.save_conversation(st.session_state.session_id, st.session_state.messages, st.session_state.tracker.history)
            st.rerun()

    # ── Sentiment analysis ──
    analysis = sentiment.analyze_message(user_input, API_KEY)
    frust_score = st.session_state.tracker.update(analysis)
    auto_level = st.session_state.tracker.should_auto_escalate()
    wants_human = analysis.get("wants_human", False)

    # ── Check if customer requested human or frustration is high ──
    if wants_human or auto_level is not None:
        curr_lvl = st.session_state.escalation_level_index
        next_level = min(6, curr_lvl + 1)
        st.session_state.escalation_level_index = next_level

        assessed = sentiment.assess_seriousness(
            st.session_state.messages, frust_score, API_KEY
        )
        esc_level = max(next_level, min(6, assessed.get("level", 1)))
        contact = escalation.get_hierarchy_contact(esc_level)

        st.session_state.pending_escalation = {
            "level": esc_level,
            "reason": "Customer requested human intervention" if wants_human else "High frustration signal detected",
        }

        bot_msg = (
            f"I notice this issue might benefit from specialized attention from "
            f"**{contact.get('name', 'our support team')}** ({contact.get('role', 'Support Tier')}).\n\n"
            f"Would you like me to escalate this ticket to our support team for further review?\n"
            f"*(Reply **Yes** to confirm escalation, or **No** to continue chatting with me)*"
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_msg,
            "query_understanding": qu_data,
            "sources": []
        })
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(bot_msg)

    else:
        # ── RAG Answer Generation ──
        if not st.session_state.chunks:
            store.log_knowledge_gap(user_input)
            st.session_state.pending_escalation = {"level": 7, "reason": "Knowledge base empty"}
            bot_msg = (
                "The policy documents do not contain the necessary information for your request.\n\n"
                "Would you like me to escalate this query to our Business Director and Founder for further review?\n"
                "*(Reply **Yes** to confirm escalation, or **No** to continue chatting with me)*"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_msg,
                "query_understanding": qu_data,
                "sources": []
            })
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(bot_msg)
        else:
            rewritten_q = qu_data.get("rewritten_query", user_input)
            candidates = rag_engine.hybrid_search(
                user_input, st.session_state.chunks, st.session_state.vectors,
                st.session_state.bm25, API_KEY,
                CONFIG.get("top_k", 5) * 2,
                search_query=rewritten_q,
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
                    store.log_knowledge_gap(user_input)
                    st.session_state.pending_escalation = {"level": 7, "reason": "Missing policy document information"}
                    clean_text = (
                        "The policy documents do not contain the necessary information for your request.\n\n"
                        "Would you like me to escalate this query to our Business Director and Founder for further review?\n"
                        "*(Reply **Yes** to confirm escalation, or **No** to continue chatting with me)*"
                    )
                    confidence = 0.0

                placeholder.markdown(clean_text)

                try:
                    clean_for_tts = re.sub(r"[#*`_\-]", "", clean_text)[:300]
                    tts = gTTS(text=clean_for_tts, lang="en")
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.session_state.last_audio_bytes = fp.read()
                except Exception:
                    pass

                # Render Query Understanding expander
                with st.expander("🧠 Real-Time Query Understanding & Intent Analysis"):
                    st.markdown(f"**⚡ Detected Intent:** `{qu_data.get('intent', 'N/A')}`")
                    st.markdown(f"**🔍 Rewritten Search Query:** `{qu_data.get('rewritten_query', 'N/A')}`")
                    entities = ", ".join([f"`{e}`" for e in qu_data.get("entities", [])]) or "None"
                    st.markdown(f"**🏷️ Extracted Entities:** {entities}")
                    st.markdown(f"**⏱️ Urgency:** `{qu_data.get('urgency', 'Medium')}` · **Topic:** *{qu_data.get('core_topic', 'N/A')}*")

                if sources and confidence >= CONFIG.get("confidence_threshold", 0.6):
                    with st.expander("📚 Referenced Sources"):
                        for src in sources:
                            st.markdown(f"• **{src.get('file', '?')}** — Page {src.get('page', '?')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_text,
                "query_understanding": qu_data,
                "sources": sources if confidence >= CONFIG.get("confidence_threshold", 0.6) else [],
            })

    store.save_conversation(
        st.session_state.session_id,
        st.session_state.messages,
        st.session_state.tracker.history,
    )
    st.rerun()
