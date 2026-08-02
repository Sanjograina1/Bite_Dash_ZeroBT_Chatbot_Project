"""
ZeroBT AI Support Chatbot — Router & Master Navigation
======================================================
Uses Streamlit st.navigation for clean multi-page routing without redundant 'app' tab.
"""

import streamlit as st

pages = [
    st.Page("pages/0_💬_Ask_ZeroBT.py", title="Ask ZeroBT", icon="💬", default=True),
    st.Page("pages/1_📋_Admin_Dashboard.py", title="Admin Dashboard", icon="📋"),
    st.Page("pages/2_📊_Analytics.py", title="Analytics", icon="📊"),
    st.Page("pages/3_👤_Agent_Portal.py", title="Agent Portal", icon="👤"),
    st.Page("pages/4_🧪_Evaluation.py", title="Evaluation", icon="🧪"),
]

pg = st.navigation(pages)
pg.run()
