"""
ZeroBT AI Support Chatbot — Ask ZeroBT Main Chat Interface
===========================================================
Glassmorphism Theme · Streaming RAG · Interactive Consent Escalation · gTTS Voice Output
"""

import io, importlib, json, os, re
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS

from modules import store, knowledge_base as kb, rag_engine, sentiment, escalation
importlib.reload(kb)

# ── Env & Config ─────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_PASS = os.getenv("GMAIL_APP_PASSWORD", "")
CONFIG_FILE = Path(__file__).parent.parent / "config.json"
CONFIG = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ask ZeroBT · AI Customer Support",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════
# CSS — Glassmorphism & Vibrant 3D Liquid Glass Styling
# ═════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
    --bg-mesh: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.18) 0px, transparent 50%),
               radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.18) 0px, transparent 50%),
               radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.15) 0px, transparent 50%),
               radial-gradient(at 0% 100%, rgba(6, 182, 212, 0.18) 0px, transparent 50%),
               linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 40%, #F5F3FF 70%, #FDF4FF 100%);
    --bg-user-glass: linear-gradient(135deg, rgba(224, 231, 255, 0.85) 0%, rgba(238, 242, 255, 0.65) 100%);
    --bg-bot-glass:  linear-gradient(135deg, rgba(250, 245, 255, 0.9) 0%, rgba(243, 232, 255, 0.7) 100%);
    --txt-main: #0F172A;
    --txt-muted: #475569;
}

/* ── 3D Liquid Mesh Background ── */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg-mesh) !important;
    background-attachment: fixed !important;
    color: var(--txt-main) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── 3D Decorative Glass Page Border Frame ── */
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

/* ── Multi-color Liquid Gradient Top Border Accent ── */
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

/* ── 3D Glass Bottom Frame Ornament ── */
[data-testid="stMainBlockContainer"]::after {
    content: '';
    position: absolute;
    bottom: -2px; left: 35%; right: 35%;
    height: 4px;
    border-radius: 8px 8px 0 0;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.6), transparent);
    z-index: 10;
}

header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

/* ── 3D Liquid Glass Sidebar & Divider Border ── */
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, rgba(255, 255, 255, 0.78) 0%, rgba(241, 245, 249, 0.6) 100%) !important;
    backdrop-filter: blur(28px) saturate(220%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(220%) !important;
    border-right: 2.5px solid rgba(168, 85, 247, 0.35) !important;
    box-shadow: 12px 0 35px -10px rgba(99, 102, 241, 0.2) !important;
}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--txt-main) !important;
}

/* ── 3D Liquid Glass Buttons ── */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    border: 1.5px solid rgba(255, 255, 255, 0.8) !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(238, 242, 255, 0.65) 100%) !important;
    backdrop-filter: blur(16px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(200%) !important;
    color: #334155 !important;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.18),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.95),
                inset 0 -2px 4px rgba(99, 102, 241, 0.08) !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}

.stButton > button:hover {
    border-color: #6366F1 !important;
    color: #4338CA !important;
    background: linear-gradient(135deg, rgba(238, 242, 255, 0.95) 0%, rgba(224, 231, 255, 0.8) 100%) !important;
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 18px 35px -8px rgba(99, 102, 241, 0.35),
                inset 0 1.5px 2px #FFFFFF !important;
}

.stButton > button:active {
    transform: translateY(1px) scale(0.98) !important;
    box-shadow: 0 4px 12px -2px rgba(99, 102, 241, 0.3),
                inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

/* ── Primary 3D Vibrant Liquid Button ── */
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #D946EF 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    box-shadow: 0 12px 30px -4px rgba(139, 92, 246, 0.45),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.6),
                inset 0 -3px 6px rgba(0, 0, 0, 0.2) !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}

button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #C026D3 100%) !important;
    box-shadow: 0 20px 40px -6px rgba(139, 92, 246, 0.6),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.8) !important;
    transform: translateY(-3px) scale(1.02) !important;
}

/* ── 3D Liquid Chat Message Bubbles ── */
[data-testid="stChatMessage"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(248, 250, 252, 0.6) 100%) !important;
    backdrop-filter: blur(20px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(200%) !important;
    border-radius: 20px !important;
    border: 1.5px solid rgba(255, 255, 255, 0.8) !important;
    box-shadow: 0 15px 35px -10px rgba(99, 102, 241, 0.12),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.95),
                inset 0 -2px 4px rgba(0, 0, 0, 0.03) !important;
    animation: msgSpring 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    margin-bottom: 16px !important;
    padding: 18px 24px !important;
}

@keyframes msgSpring {
    from { opacity: 0; transform: translateY(12px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--bg-user-glass) !important;
    border: 1.5px solid rgba(199, 210, 254, 0.9) !important;
    border-left: 5px solid #6366F1 !important;
    box-shadow: 0 15px 35px -10px rgba(99, 102, 241, 0.2),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.9) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--bg-bot-glass) !important;
    border: 1.5px solid rgba(233, 213, 255, 0.9) !important;
    border-left: 5px solid #A855F7 !important;
    box-shadow: 0 15px 35px -10px rgba(168, 85, 247, 0.2),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.9) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: var(--txt-main) !important;
    font-size: 15.5px !important;
    line-height: 1.65 !important;
}

/* ── 3D Liquid Chat Input Box ── */
[data-testid="stChatInput"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%) !important;
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    border: 2px solid rgba(199, 210, 254, 0.9) !important;
    border-radius: 18px !important;
    box-shadow: 0 14px 40px -10px rgba(99, 102, 241, 0.2),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.95) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #6366F1 !important;
    box-shadow: 0 18px 45px -8px rgba(99, 102, 241, 0.35),
                inset 0 1.5px 2px rgba(255, 255, 255, 1) !important;
}

[data-testid="stChatInputTextArea"] {
    color: var(--txt-main) !important;
    font-size: 15.5px !important;
}

/* ── 3D Liquid Cards & Expanders ── */
[data-testid="stExpander"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(241, 245, 249, 0.6) 100%) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px -8px rgba(99, 102, 241, 0.1),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.9) !important;
}

.doc-card, .metric-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(248, 250, 252, 0.65) 100%) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.85) !important;
    border-radius: 16px !important;
    padding: 18px 24px !important;
    margin: 12px 0 !important;
    box-shadow: 0 12px 30px -8px rgba(99, 102, 241, 0.15),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.95) !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease !important;
}

.doc-card:hover, .metric-card:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 18px 40px -10px rgba(99, 102, 241, 0.25),
                inset 0 1.5px 2px rgba(255, 255, 255, 1) !important;
}

/* ── 3D Escalation Glass Banner ── */
.esc-banner {
    background: linear-gradient(135deg, rgba(254, 242, 242, 0.9) 0%, rgba(254, 226, 226, 0.75) 100%) !important;
    backdrop-filter: blur(18px) saturate(200%) !important;
    border: 1.5px solid #FCA5A5 !important;
    border-left: 6px solid #EF4444 !important;
    border-radius: 16px;
    padding: 18px 24px;
    margin: 16px 0;
    color: #991B1B;
    box-shadow: 0 14px 35px -8px rgba(239, 68, 68, 0.2),
                inset 0 1.5px 2px rgba(255, 255, 255, 0.9) !important;
}

.gauge-container { text-align: center; margin: 12px 0; }
.gauge-label {
    font-size: 11px; color: #64748B;
    text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700;
}
.gauge-score {
    font-size: 38px; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    text-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
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
    if not st.session_state.kb_loaded or not st.session_state.chunks:
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
    st.markdown("## 💬 Ask ZeroBT")
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

st.markdown("### 💬 Ask ZeroBT — AI Customer Support")
st.markdown("Ask questions using typed text or interactive audio controls.")

# ── Glass Audio Toolbar Buttons & Speed Selection ──
ac1, ac2, ac3 = st.columns([2, 2.5, 2.5])

speech_speed = ac1.selectbox("Narration Speed", ["Slow (0.5x)", "Normal (1.0x)", "Fast (1.25x)"], index=1)
read_last = ac2.button("🔊 Read Answer Aloud", use_container_width=True)
quick_audio = ac3.button("🎙️ Quick Voice Sample", use_container_width=True)

if read_last:
    assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if assistant_msgs:
        last_txt = assistant_msgs[-1]["content"]
        clean_for_speech = re.sub(r"[#*`_\-]", "", last_txt)[:350]
        try:
            is_slow = (speech_speed == "Slow (0.5x)")
            tts = gTTS(text=clean_for_speech, lang="en", slow=is_slow)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            st.session_state.last_audio_bytes = fp.read()
            st.session_state.current_speed = speech_speed
            st.success(f"Generated audio at {speech_speed}. Click play below to listen.")
        except Exception as e:
            st.error(f"TTS generation error: {e}")
    else:
        st.info("No assistant answer to read aloud yet.")

if st.session_state.last_audio_bytes:
    # Manual playback only — NO autoplay
    st.audio(st.session_state.last_audio_bytes, format="audio/mp3", autoplay=False)

sample_query = None
if quick_audio:
    sample_query = "What is the policy process if my query needs escalation?"

# ── Department Tier Mapping ──
DEPARTMENTS_HIERARCHY = {
    "Tier 1: General Customer Care (Order Inquiries, Account & FAQs)": 1,
    "Tier 2: Delivery & Logistics Support (Rain Delays, Driver Tracking & Address Changes)": 2,
    "Tier 3: Billing & Refunds Department (FoodiePass, Promo Cash & UPI Disputes)": 3,
    "Tier 4: Food Safety & Packaging Quality (Torn Seals, Spills & Hygiene Complaints)": 4,
    "Level 5: Senior Support Agent (Complex Complaints & Priority Escalations)": 5,
    "Level 6: Support Management (Operational Failures & Service Disputes)": 6,
    "Level 7: Business Operations (Executive Policy & Legal Queries)": 7,
    "Level 8: Founder's Office (Critical Escalations & Leadership Override)": 8,
}

# ── Render history ──
for idx, msg in enumerate(st.session_state.messages):
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Render interactive department selection dropdown if unanswered fallback line was shown
            fallback_line = "I am sorry for the inconvinience if you select what area your issue is I can escalate it to the appropriate department."
            if fallback_line in msg["content"] and not st.session_state.escalated:
                st.markdown("---")
                sel_dept = st.selectbox(
                    "📌 Select the area/department for your issue:",
                    options=list(DEPARTMENTS_HIERARCHY.keys()),
                    key=f"dept_sel_{idx}"
                )
                if st.button("📤 Confirm & Escalate to Selected Department", key=f"btn_dept_esc_{idx}", type="primary", use_container_width=True):
                    chosen_level = DEPARTMENTS_HIERARCHY[sel_dept]
                    with st.spinner(f"Escalating ticket to {sel_dept}…"):
                        esc_result = escalation.escalate(
                            st.session_state.session_id,
                            st.session_state.messages,
                            chosen_level,
                            f"Customer selected department: {sel_dept}",
                            API_KEY, GMAIL_USER, GMAIL_PASS
                        )
                        st.session_state.escalated = True
                        st.session_state.escalation_info = esc_result
                    st.success(f"✅ Your ticket has been escalated to {esc_result['contact'].get('name')} ({sel_dept})!")
                    st.rerun()

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
# PROCESS USER INPUT & ATTACHMENTS
# ═════════════════════════════════════════════════════════════════════

with st.expander("📎 Attach Files to Query (All formats accepted: TXT, PDF, DOCX, CSV, Excel, Images, Audio, Code, etc.)", expanded=bool(st.session_state.get("chat_file_attachments"))):
    uploaded_files = st.file_uploader(
        "Drop or select any file format (including .txt, .pdf, .docx, .csv, etc.):",
        accept_multiple_files=True,
        type=None,  # All file formats enabled
        key="chat_file_attachments"
    )
    btn_send_files = st.button("📥 Send & Analyze Attached Files", use_container_width=True)

user_input = st.chat_input("Type your question here…")
if sample_query:
    user_input = sample_query
elif voice_text and not user_input:
    user_input = voice_text

# Check if user submitted text, pressed send files, or attached files
if user_input or btn_send_files or (uploaded_files and not st.session_state.messages):
    raw_user_text = user_input if user_input else ""
    attachment_summary_lines = []
    attachment_texts = []

    if uploaded_files:
        with st.spinner("📎 Extracting & processing attached files…"):
            for f in uploaded_files:
                f_bytes = f.read()
                f_name = f.name
                extracted = kb.extract_any_file(f_bytes, f_name, API_KEY)
                attachment_summary_lines.append(f"`{f_name}`")
                attachment_texts.append(f"--- Attachment File: {f_name} ---\n{extracted}\n")

    if attachment_texts:
        attached_names_str = ", ".join(attachment_summary_lines)
        full_attachment_block = "\n".join(attachment_texts)
        if raw_user_text:
            final_user_content = f"{raw_user_text}\n\n📎 **Attached Files ({len(uploaded_files)}):** {attached_names_str}\n\n```text\n{full_attachment_block}\n```"
        else:
            final_user_content = f"📎 **Attached Files ({len(uploaded_files)}):** {attached_names_str}\n\n```text\n{full_attachment_block}\n```"
    else:
        final_user_content = raw_user_text

    st.session_state.messages.append({"role": "user", "content": final_user_content})

    # ── Real-Time Sentiment & Frustration Meter Analysis ──
    analysis = sentiment.analyze_message(final_user_content, API_KEY)
    frust_score = st.session_state.tracker.update(analysis)
    st.session_state.current_frustration = frust_score

    with st.chat_message("user", avatar="👤"):
        st.markdown(final_user_content)

    user_lower = final_user_content.strip().lower()
    is_agree = any(w in user_lower for w in ["yes", "yep", "yeah", "sure", "please", "ok", "escalate", "agree"])
    is_decline = any(w in user_lower for w in ["no", "nope", "don't", "cancel", "fine", "nevermind"])

    # ── Real-Time Query Understanding ──
    qu_data = rag_engine.understand_query(final_user_content, st.session_state.messages[:-1], API_KEY)

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

    # ── Check for Profanity: Auto-Escalate to Senior Agent (Level 5) with Sanitized Summary ──
    has_profanity = analysis.get("has_profanity", False) or "profanity" in analysis.get("escalation_signals", []) or "abusive_language" in analysis.get("escalation_signals", [])

    if has_profanity:
        sanitized_summary = analysis.get("sanitized_summary", "Customer query containing heightened concern summarized for Senior Agent review.")
        with st.spinner("Escalating query to Senior Agent…"):
            esc_result = escalation.escalate(
                st.session_state.session_id,
                [
                    {"role": "user", "content": f"[Sanitized Issue Summary]: {sanitized_summary}"},
                    {"role": "assistant", "content": "Query escalated to Senior Agent for priority handling."}
                ],
                5,  # Level 5: Senior Agent
                f"Profanity detected — Sanitized issue summary for Senior Agent: {sanitized_summary}",
                API_KEY, GMAIL_USER, GMAIL_PASS,
            )
        st.session_state.escalated = True
        st.session_state.escalation_info = esc_result

        bot_msg = (
            f"I have detected heightened concern in your query. To assist you as quickly and respectfully as possible, "
            f"I have summarized your concern without any unparliamentary language and escalated your ticket directly to our **Senior Agent** ({esc_result['contact'].get('name', 'Shivi Kanojia')}) for priority review.\n\n"
            f"**📋 Issue Summary for Senior Agent:**\n"
            f"> *{sanitized_summary}*\n\n"
            f"Reference Ticket ID: `{st.session_state.session_id}`. Our Senior Agent has been notified via email."
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

    # ── Check if customer requested human or frustration is high ──
    auto_level = st.session_state.tracker.should_auto_escalate()
    wants_human = analysis.get("wants_human", False)

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
        fallback_required_msg = "I am sorry for the inconvinience if you select what area your issue is I can escalate it to the appropriate department."

        if not st.session_state.chunks:
            store.log_knowledge_gap(user_input)
            bot_msg = fallback_required_msg
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_msg,
                "query_understanding": qu_data,
                "sources": []
            })
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(bot_msg)
        else:
            with st.spinner("🤖 Analyzing policy documents & generating detailed response…"):
                rewritten_q = qu_data.get("rewritten_query", user_input)
                query_vars = qu_data.get("query_variations", [])
                candidates = rag_engine.hybrid_search(
                    user_input, st.session_state.chunks, st.session_state.vectors,
                    st.session_state.bm25, API_KEY,
                    top_k=CONFIG.get("top_k", 20) * 2,
                    search_query=rewritten_q,
                    query_variations=query_vars,
                )
                reranked = rag_engine.rerank(user_input, candidates, API_KEY, top_n=CONFIG.get("top_k", 20))

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
