"""
Evaluation — RAG pipeline quality metrics for ZeroBT
"""

import os, json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from modules import store, knowledge_base as kb, rag_engine

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
CONFIG = json.loads((Path(__file__).parent.parent / "config.json").read_text()) \
    if (Path(__file__).parent.parent / "config.json").exists() else {}

st.set_page_config(page_title="ZeroBT · RAG Evaluation", page_icon="🧪", layout="wide")
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

.stMetric {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(248, 250, 252, 0.65) 100%) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.85) !important;
    border-radius: 18px !important;
    padding: 16px 20px !important;
    box-shadow: 0 12px 30px -8px rgba(99, 102, 241, 0.15), inset 0 1.5px 2px rgba(255, 255, 255, 0.95) !important;
}

.score-pass { color: #059669; font-weight: 800; }
.score-fail { color: #E11D48; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧪 ZeroBT RAG Pipeline Evaluation")
st.caption(
    "Benchmark ZeroBT retrieval and generation accuracy. "
    "Measures Faithfulness, Answer Relevance, and Context Recall."
)

# ── Load KB ──
chunks, vectors, bm25 = kb.load_index()
if not chunks:
    st.warning("No documents in knowledge base. Upload documents in Admin Dashboard first.")
    st.stop()

# ── Test Cases ──
st.markdown("### Define Test Cases")
st.caption("Add questions and optional ground-truth expected answers.")

if "eval_cases" not in st.session_state:
    st.session_state.eval_cases = [
        {"question": "What is your refund policy?", "expected": ""},
        {"question": "How do I contact customer support?", "expected": ""},
    ]

for i, case in enumerate(st.session_state.eval_cases):
    c1, c2, c3 = st.columns([3, 3, 0.5])
    st.session_state.eval_cases[i]["question"] = c1.text_input(
        "Question", value=case["question"], key=f"q_{i}", label_visibility="collapsed"
    )
    st.session_state.eval_cases[i]["expected"] = c2.text_input(
        "Expected (optional)", value=case["expected"], key=f"e_{i}", label_visibility="collapsed"
    )
    if c3.button("🗑", key=f"del_case_{i}"):
        st.session_state.eval_cases.pop(i)
        st.rerun()

if st.button("➕ Add Test Case"):
    st.session_state.eval_cases.append({"question": "", "expected": ""})
    st.rerun()

st.divider()

# ── Run Evaluation ──
if st.button("🚀 Run Evaluation Pipeline", type="primary", use_container_width=True):
    cases = [c for c in st.session_state.eval_cases if c["question"].strip()]
    if not cases:
        st.warning("Add at least one valid question.")
        st.stop()

    results = []
    progress = st.progress(0)
    status = st.status("Benchmarking RAG pipeline …", expanded=True)

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    for idx, case in enumerate(cases):
        q = case["question"]
        status.write(f"📝 Evaluating question: '{q}'")

        result = rag_engine.query_rag(q, chunks, vectors, bm25, API_KEY, top_k=CONFIG.get("top_k", 5))

        answer = result["clean_text"]
        confidence = result["confidence"]
        context = result.get("context_chunks", [])
        ctx_text = " ".join(c["text"] for c in context)

        faith_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content":
                    "You are an expert enterprise RAG faithfulness evaluator. "
                    "Analyze the given answer against the full context document text. "
                    "Score 0-10 how faithful this answer is to the provided context. "
                    "10 = every single claim is strictly supported by context. 0 = completely fabricated. "
                    "Return ONLY the numeric score integer from 0 to 10."},
                {"role": "user", "content":
                    f"Context:\n{ctx_text}\n\nGenerated Answer:\n{answer}"},
            ],
            temperature=0, max_tokens=10,
        )
        try:
            faithfulness = int(re.search(r'\d+', faith_resp.choices[0].message.content.strip()).group()) / 10.0
        except Exception:
            faithfulness = 0.5

        rel_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content":
                    "You are an expert enterprise customer support answer relevance evaluator. "
                    "Score 0-10 how relevant and thorough this answer is to the user's question. "
                    "10 = perfectly and exhaustively answers the question. 0 = completely irrelevant. "
                    "Return ONLY the numeric score integer from 0 to 10."},
                {"role": "user", "content":
                    f"Question: {q}\n\nGenerated Answer:\n{answer}"},
            ],
            temperature=0, max_tokens=10,
        )
        try:
            relevance = int(re.search(r'\d+', rel_resp.choices[0].message.content.strip()).group()) / 10.0
        except Exception:
            relevance = 0.5

        recall = None
        if case["expected"].strip():
            rec_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content":
                        "You are an expert context recall evaluator. "
                        "Score 0-10 how much of the expected ground-truth information is present in the context. "
                        "10 = all information is covered in context. 0 = none is present. "
                        "Return ONLY the numeric score integer from 0 to 10."},
                    {"role": "user", "content":
                        f"Expected Ground Truth: {case['expected']}\n\nContext Documents:\n{ctx_text}"},
                ],
                temperature=0, max_tokens=10,
            )
            try:
                recall = int(re.search(r'\d+', rec_resp.choices[0].message.content.strip()).group()) / 10.0
            except Exception:
                recall = 0.5

        results.append({
            "question": q,
            "answer": answer[:150],
            "confidence": confidence,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "recall": recall,
        })
        progress.progress((idx + 1) / len(cases))

    status.update(label="Evaluation benchmarking complete ✓", state="complete")

    st.markdown("### Benchmark Results")

    for r in results:
        f_class = "score-pass" if r["faithfulness"] >= 0.7 else "score-fail"
        r_class = "score-pass" if r["relevance"] >= 0.7 else "score-fail"
        recall_str = f'{r["recall"]:.0%}' if r["recall"] is not None else "—"
        rc_class = "score-pass" if r["recall"] and r["recall"] >= 0.7 else "score-fail"

        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:16px 20px;margin:8px 0;box-shadow:0 1px 4px rgba(0,0,0,0.02);">
            <strong>{r['question']}</strong><br>
            <span style="color:#475569;font-size:14px;">{r['answer']}…</span><br><br>
            <span>Confidence: <strong>{r['confidence']:.0%}</strong></span> ·
            <span>Faithfulness: <span class="{f_class}">{r['faithfulness']:.0%}</span></span> ·
            <span>Relevance: <span class="{r_class}">{r['relevance']:.0%}</span></span> ·
            <span>Context Recall: <span class="{rc_class}">{recall_str}</span></span>
        </div>
        """, unsafe_allow_html=True)

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["relevance"] for r in results) / len(results)
    recall_vals = [r["recall"] for r in results if r["recall"] is not None]
    avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else None

    st.markdown("### Aggregate Benchmark Scores")
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("Avg Faithfulness", f"{avg_faith:.0%}")
    ac2.metric("Avg Relevance", f"{avg_rel:.0%}")
    ac3.metric("Avg Context Recall", f"{avg_recall:.0%}" if avg_recall else "—")

    import csv, io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    st.download_button("📥 Download Benchmark CSV", buf.getvalue(), "rag_evaluation_results.csv", "text/csv")
