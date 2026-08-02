"""
Analytics Dashboard — Business metrics and insights
"""

import os
import json
import importlib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from modules import store, report_generator
importlib.reload(store)
importlib.reload(report_generator)

load_dotenv()
st.set_page_config(page_title="ZeroBT · Analytics", page_icon="📊", layout="wide")
store.init_db()

API_KEY = os.getenv("OPENAI_API_KEY", "")

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

.stMetric {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(248, 250, 252, 0.65) 100%) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.85) !important;
    border-radius: 18px !important;
    padding: 16px 20px !important;
    box-shadow: 0 12px 30px -8px rgba(99, 102, 241, 0.15), inset 0 1.5px 2px rgba(255, 255, 255, 0.95) !important;
}

.export-card {
    background: linear-gradient(135deg, rgba(238, 242, 255, 0.9) 0%, rgba(245, 243, 255, 0.8) 100%) !important;
    backdrop-filter: blur(20px) saturate(200%) !important;
    border: 2px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 20px !important;
    padding: 24px 28px !important;
    margin: 20px 0 !important;
    box-shadow: 0 16px 40px -10px rgba(99, 102, 241, 0.22), inset 0 1.5px 2px #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 ZeroBT Business Analytics & AI Intelligence Center")

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

# ═════════════════════════════════════════════════════════════════════
# 📥 TOP EXPORT ACTION CENTER (OPENAI POWERED)
# ═════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="export-card">
    <h3 style="margin: 0 0 6px 0; color: #4338CA; font-weight: 800; font-size: 22px;">
        ✨ Export Analytics & OpenAI AI Intelligence Report
    </h3>
    <p style="margin: 0 0 16px 0; color: #334155; font-size: 14px; line-height: 1.5;">
        Generate and download comprehensive analytics documents powered by <strong>OpenAI GPT-4o API</strong>. 
        Analyzes all chat sessions across <strong>Low, Medium, and High frustration levels</strong>, details reasons for escalation, 
        tracks <strong>pending queue counts</strong>, and uncovers underlying customer needs.
    </p>
</div>
""", unsafe_allow_html=True)

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    st.download_button(
        label="📊 Download Excel Report (.xlsx)",
        data=report_generator.generate_excel_report(API_KEY),
        file_name="zerobt_openai_intelligence_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
        key="btn_dl_excel_top"
    )

with col_exp2:
    st.download_button(
        label="📑 Download Text Report (.txt)",
        data=report_generator.generate_txt_report(API_KEY),
        file_name="zerobt_openai_intelligence_report.txt",
        mime="text/plain",
        use_container_width=True,
        type="secondary",
        key="btn_dl_txt_top"
    )

with col_exp3:
    st.download_button(
        label="📄 Download PDF Report (.pdf)",
        data=report_generator.generate_pdf_report(API_KEY),
        file_name="zerobt_openai_intelligence_report.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
        key="btn_dl_pdf_top"
    )

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

st.markdown(" ")
st.markdown("#### 📂 Raw CSV Data Log Downloads")

# 1. Escalations Log DataFrame
escs_df = pd.DataFrame(escs) if escs else pd.DataFrame(columns=["id", "session_id", "level", "reason", "email_sent", "created_at", "resolved"])

# 2. Knowledge Gaps DataFrame
gaps_data = store.get_knowledge_gaps(100)
gaps_df = pd.DataFrame(gaps_data) if gaps_data else pd.DataFrame(columns=["id", "query", "count", "last_seen"])

# 3. CSAT Ratings DataFrame
ratings_data = store.get_all_ratings()
ratings_df = pd.DataFrame(ratings_data) if ratings_data else pd.DataFrame(columns=["id", "session_id", "stars", "created_at"])

# 4. Master Analytics JSON Payload
analytics_json_data = {
    "summary_metrics": {
        "total_conversations": conv_count,
        "avg_csat_rating": avg_csat,
        "total_escalations": total_esc,
        "escalation_rate": f"{total_esc / max(conv_count, 1) * 100:.1f}%" if conv_count else "0%",
    },
    "escalation_distribution": esc_stats,
    "knowledge_gaps": gaps_data,
    "csat_ratings": ratings_data,
    "escalation_logs": escs,
}
json_bytes = json.dumps(analytics_json_data, indent=2).encode("utf-8")

# Export Buttons Layout
exp1, exp2, exp3, exp4 = st.columns(4)

with exp1:
    st.download_button(
        label="📊 Escalations Log (.csv)",
        data=escs_df.to_csv(index=False).encode("utf-8"),
        file_name="zerobt_escalations_log.csv",
        mime="text/csv",
        use_container_width=True,
    )

with exp2:
    st.download_button(
        label="🔍 Knowledge Gaps (.csv)",
        data=gaps_df.to_csv(index=False).encode("utf-8"),
        file_name="zerobt_knowledge_gaps.csv",
        mime="text/csv",
        use_container_width=True,
    )

with exp3:
    st.download_button(
        label="⭐ CSAT Ratings (.csv)",
        data=ratings_df.to_csv(index=False).encode("utf-8"),
        file_name="zerobt_csat_ratings.csv",
        mime="text/csv",
        use_container_width=True,
    )

with exp4:
    st.download_button(
        label="📦 Master Payload (.json)",
        data=json_bytes,
        file_name="zerobt_master_analytics.json",
        mime="application/json",
        use_container_width=True,
    )
