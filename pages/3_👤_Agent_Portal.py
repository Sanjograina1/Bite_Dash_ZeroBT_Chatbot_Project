"""
Agent Portal — Live chat monitoring and takeover
"""

import json, os
import streamlit as st
from dotenv import load_dotenv

from modules import store

load_dotenv()
st.set_page_config(page_title="Agent Portal", page_icon="👤", layout="wide")
store.init_db()

# ── CSS ──
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
.chat-card {
    background: #161927; border: 1px solid #1e2235; border-radius: 10px;
    padding: 16px 20px; margin: 8px 0; cursor: pointer;
    transition: border-color 0.3s;
}
.chat-card:hover { border-color: #B8A1EA; }
</style>
""", unsafe_allow_html=True)

# ── Handle URL query params (from email action buttons) ──
params = st.query_params
if params.get("action") == "resolve" and params.get("ticket"):
    ticket = params["ticket"]
    escs = store.get_escalations()
    for e in escs:
        if e["session_id"] == ticket:
            store.resolve_escalation(e["id"])
    st.success(f"Ticket {ticket} marked as resolved.")
    st.query_params.clear()

# ── Header ──
st.markdown("## 👤 Agent Portal")
st.caption("Monitor active customer chats and take over conversations when needed.")

# ── Agent ID ──
if "agent_name" not in st.session_state:
    st.session_state.agent_name = ""

with st.sidebar:
    st.markdown("### Agent Settings")
    st.session_state.agent_name = st.text_input("Your Name", value=st.session_state.agent_name)
    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# ── Active Conversations ──
st.markdown("### Active Conversations")
conversations = store.get_active_conversations()

if not conversations:
    st.info("No active conversations right now.")
else:
    for conv in conversations:
        messages = json.loads(conv["messages"]) if isinstance(conv["messages"], str) else conv["messages"]
        frustration = json.loads(conv["frustration"]) if isinstance(conv["frustration"], str) else conv["frustration"]
        msg_count = len(messages)
        last_msg = messages[-1]["content"][:80] if messages else "—"
        frust_score = frustration[-1] if frustration else 0

        # Frustration color
        if frust_score < 40:
            f_color = "#86EFAC"
        elif frust_score < 70:
            f_color = "#FDE047"
        else:
            f_color = "#FCA5A5"

        status_badge = ""
        if conv["status"] == "takeover":
            status_badge = f'<span style="color:#B8A1EA;font-weight:600;"> 🔴 LIVE (Agent: {conv.get("agent_id", "?")})</span>'

        st.markdown(f"""
        <div class="chat-card">
            <strong>Session {conv['id']}</strong> {status_badge}<br>
            <span style="color:#94A3B8">{msg_count} messages · Last: "{last_msg}…"</span><br>
            <span style="color:{f_color};font-weight:600;">Frustration: {int(frust_score)}</span>
            <span style="color:#94A3B8;float:right;">{conv['updated_at']}</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 3])

        # Takeover button
        if conv["status"] != "takeover":
            if col1.button("🎧 Take Over", key=f"take_{conv['id']}"):
                if st.session_state.agent_name:
                    store.flag_takeover(conv["id"], st.session_state.agent_name)
                    st.rerun()
                else:
                    st.warning("Enter your name first.")
        else:
            if col1.button("↩ Release", key=f"rel_{conv['id']}"):
                store.release_takeover(conv["id"])
                st.rerun()

        # View transcript
        if col2.button("📋 View Chat", key=f"view_{conv['id']}"):
            with st.expander(f"Transcript — {conv['id']}", expanded=True):
                for m in messages:
                    sender = "👤 Customer" if m.get("role") == "user" else "🤖 Bot"
                    st.markdown(f"**{sender}:** {m['content']}")

# ── Escalation Queue ──
st.divider()
st.markdown("### Escalation Queue")

escs = store.get_escalations(resolved=False)
if not escs:
    st.info("No pending escalations.")
else:
    for e in escs:
        level_color = {1: "#86EFAC", 2: "#FDE047", 3: "#FCA5A5", 4: "#FCA5A5"}.get(e["level"], "#94A3B8")
        st.markdown(f"""
        <div style="background:#161927;border:1px solid #1e2235;border-left:3px solid {level_color};
             border-radius:8px;padding:14px 18px;margin:6px 0;">
            <strong style="color:{level_color}">Level {e['level']}</strong> ·
            Session {e['session_id']} ·
            <span style="color:#94A3B8">{e['reason'][:60]}</span><br>
            <span style="color:#94A3B8;font-size:12px;">Sent to: {e['email_sent']} · {e['created_at']}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✅ Mark Resolved", key=f"resolve_{e['id']}"):
            store.resolve_escalation(e["id"])
            st.rerun()
