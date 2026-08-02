"""
Agent Portal — Live chat monitoring and takeover for ZeroBT
"""

import json, os
import streamlit as st
from dotenv import load_dotenv

from modules import store, escalation

load_dotenv()
st.set_page_config(page_title="ZeroBT · Agent Portal", page_icon="👤", layout="wide")
store.init_db()

# ── CSS ──
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

.chat-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(248, 250, 252, 0.65) 100%) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.85) !important;
    border-radius: 18px !important;
    padding: 18px 24px; margin: 12px 0; cursor: pointer;
    box-shadow: 0 12px 30px -8px rgba(99, 102, 241, 0.15), inset 0 1.5px 2px rgba(255, 255, 255, 0.95) !important;
    transition: all 0.25s ease !important;
}
.chat-card:hover {
    border-color: #6366F1 !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 18px 40px -10px rgba(99, 102, 241, 0.25), inset 0 1.5px 2px #FFFFFF !important;
}
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
st.markdown("## 👤 ZeroBT Agent Portal")
st.caption("Monitor active customer chats and manually take over conversations in real time.")

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
    st.info("No active customer conversations right now.")
else:
    for conv in conversations:
        messages = json.loads(conv["messages"]) if isinstance(conv["messages"], str) else conv["messages"]
        frustration = json.loads(conv["frustration"]) if isinstance(conv["frustration"], str) else conv["frustration"]
        msg_count = len(messages)
        last_msg = messages[-1]["content"][:80] if messages else "—"
        frust_score = frustration[-1] if frustration else 0

        if frust_score < 35:
            f_color = "#10B981"
        elif frust_score < 65:
            f_color = "#D97706"
        else:
            f_color = "#DC2626"

        status_badge = ""
        if conv["status"] == "takeover":
            status_badge = f'<span style="color:#7C3AED;font-weight:700;"> 🔴 LIVE TAKE-OVER (Agent: {conv.get("agent_id", "?")})</span>'

        st.markdown(f"""
        <div class="chat-card">
            <strong>Session ID: {conv['id']}</strong> {status_badge}<br>
            <span style="color:#475569">{msg_count} messages · Last: "{last_msg}…"</span><br>
            <span style="color:{f_color};font-weight:700;">Frustration Meter: {int(frust_score)}</span>
            <span style="color:#64748B;float:right;">{conv['updated_at']}</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 3])

        if conv["status"] != "takeover":
            if col1.button("🎧 Take Over", key=f"take_{conv['id']}"):
                if st.session_state.agent_name:
                    store.flag_takeover(conv["id"], st.session_state.agent_name)
                    st.rerun()
                else:
                    st.warning("Please enter your name in the sidebar first.")
        else:
            if col1.button("↩ Release", key=f"rel_{conv['id']}"):
                store.release_takeover(conv["id"])
                st.rerun()

        if col2.button("📋 View Chat", key=f"view_{conv['id']}"):
            with st.expander(f"Transcript — Session {conv['id']}", expanded=True):
                for m in messages:
                    sender = "👤 Customer" if m.get("role") == "user" else "🤖 ZeroBT"
                    st.markdown(f"**{sender}:** {m['content']}")

# ── Escalation Queue ──
st.divider()
st.markdown("### Escalation Queue (8 Tiers)")

_LEVEL_PALETTE = {
    1: "#10B981", 2: "#059669", 3: "#D97706", 4: "#E11D48",
    5: "#EA580C", 6: "#C026D3", 7: "#E11D48", 8: "#991B1B"
}

escs = store.get_escalations(resolved=False)
if not escs:
    st.info("No pending escalations in queue.")
else:
    for e in escs:
        level_color = _LEVEL_PALETTE.get(e["level"], "#64748B")
        contact = escalation.get_hierarchy_contact(e["level"])
        role_title = contact.get("role", f"Level {e['level']}")

        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid {level_color};
             border-radius:8px;padding:14px 18px;margin:6px 0;box-shadow:0 1px 4px rgba(0,0,0,0.02);">
            <strong style="color:{level_color}">Level {e['level']} ({role_title})</strong> ·
            Session {e['session_id']} ·
            <span style="color:#334155">{e['reason'][:70]}</span><br>
            <span style="color:#64748B;font-size:13px;">Notification sent to: <strong>{e['email_sent']}</strong> · {e['created_at']}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✅ Mark Resolved", key=f"resolve_{e['id']}"):
            store.resolve_escalation(e["id"])
            st.rerun()
