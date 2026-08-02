"""
Analytics Dashboard — Business metrics and insights
"""

import os
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from modules import store

load_dotenv()
st.set_page_config(page_title="ZeroBT · Analytics", page_icon="📊", layout="wide")
store.init_db()

# ── CSS ──
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
.metric-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 20px 24px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}
.metric-card .value { font-size: 36px; font-weight: 700; color: #0F172A; }
.metric-card .label { font-size: 12px; color: #64748B; text-transform: uppercase; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 ZeroBT Business Analytics")

# ── Top Metrics ──
c1, c2, c3, c4 = st.columns(4)

conv_count = store.get_conversation_count()
avg_csat = store.get_avg_rating()
esc_stats = store.get_escalation_stats()
total_esc = sum(esc_stats.values())

c1.metric("Total Conversations", conv_count)
c2.metric("Avg CSAT Rating", f"{'⭐' * round(avg_csat)} ({avg_csat})" if avg_csat else "—")
c3.metric("Total Escalations", total_esc)
c4.metric("Escalation Rate", f"{total_esc / max(conv_count, 1) * 100:.0f}%" if conv_count else "—")

st.divider()

# ── Escalation Distribution ──
col_left, col_right = st.columns(2)

_LEVEL_PALETTE = {
    1: "#10B981", 2: "#059669", 3: "#D97706", 4: "#E11D48",
    5: "#EA580C", 6: "#C026D3", 7: "#E11D48", 8: "#991B1B"
}

with col_left:
    st.markdown("### Escalation Distribution")
    if esc_stats:
        labels = [f"Level {k}" for k in sorted(esc_stats.keys())]
        values = [esc_stats[k] for k in sorted(esc_stats.keys())]
        colors = [_LEVEL_PALETTE.get(k, "#64748B") for k in sorted(esc_stats.keys())]

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors),
            textfont=dict(color="#FFFFFF", size=13),
            hole=0.4,
        )])
        fig.update_layout(
            paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            showlegend=True,
            legend=dict(font=dict(color="#0F172A")),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No escalations recorded yet.")

# ── Knowledge Gaps ──
with col_right:
    st.markdown("### Top Knowledge Gaps")
    st.caption("Queries where ZeroBT escalated due to missing policy documents.")
    gaps = store.get_knowledge_gaps(10)
    if gaps:
        for i, g in enumerate(gaps, 1):
            st.markdown(
                f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
                f'border-radius:8px;padding:10px 14px;margin:4px 0;box-shadow:0 1px 3px rgba(0,0,0,0.02);">'
                f'<span style="color:#D97706;font-weight:700;">#{i}</span> '
                f'{g["query"][:80]} '
                f'<span style="color:#64748B;float:right;font-weight:600;">{g["count"]}×</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No knowledge gaps recorded yet.")

st.divider()

# ── CSAT Ratings ──
st.markdown("### Customer Satisfaction Ratings")
ratings = store.get_all_ratings()
if ratings:
    stars_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for r in ratings:
        stars_dist[r["stars"]] = stars_dist.get(r["stars"], 0) + 1

    fig2 = go.Figure(data=[go.Bar(
        x=[f"{k} ⭐" for k in range(1, 6)],
        y=[stars_dist[k] for k in range(1, 6)],
        marker_color=["#E11D48", "#EA580C", "#D97706", "#059669", "#10B981"],
        text=[stars_dist[k] for k in range(1, 6)],
        textposition="outside",
        textfont=dict(color="#0F172A"),
    )])
    fig2.update_layout(
        paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A"),
        xaxis=dict(color="#475569"), yaxis=dict(color="#475569", showgrid=False),
        margin=dict(t=20, b=40, l=40, r=20),
        height=250,
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No ratings collected yet.")

# ── Recent Escalations ──
st.markdown("### Recent Escalations")
escs = store.get_escalations()
if escs:
    for e in escs[:10]:
        level_color = _LEVEL_PALETTE.get(e["level"], "#64748B")
        resolved_badge = "✅ Resolved" if e["resolved"] else "⏳ Pending"
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid {level_color};'
            f'border-radius:8px;padding:12px 16px;margin:6px 0;box-shadow:0 1px 4px rgba(0,0,0,0.02);">'
            f'<strong style="color:{level_color}">Level {e["level"]}</strong> · '
            f'{e["reason"][:70]} · '
            f'<span style="color:#64748B">{e["email_sent"]} · {e["created_at"]}</span> · '
            f'{resolved_badge}</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("No escalations yet.")
