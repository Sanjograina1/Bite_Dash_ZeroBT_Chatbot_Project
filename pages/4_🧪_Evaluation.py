"""
Evaluation — RAG pipeline quality metrics
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

st.set_page_config(page_title="RAG Evaluation", page_icon="🧪", layout="wide")
store.init_db()

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    background-color: #0D0E15 !important; color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] { background: #161927 !important; }
[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.score-pass { color: #86EFAC; font-weight: 700; }
.score-fail { color: #FCA5A5; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧪 RAG Pipeline Evaluation")
st.caption(
    "Test the retrieval and generation pipeline against known questions. "
    "Measures Faithfulness, Answer Relevance, and Context Recall."
)

# ── Load KB ──
chunks, vectors, bm25 = kb.load_index()
if not chunks:
    st.warning("No documents in the knowledge base. Upload documents first.")
    st.stop()

# ── Test Cases ──
st.markdown("### Define Test Cases")
st.caption("Add questions and (optionally) expected answers to evaluate the pipeline.")

if "eval_cases" not in st.session_state:
    st.session_state.eval_cases = [
        {"question": "What is your refund policy?", "expected": ""},
        {"question": "How do I contact support?", "expected": ""},
    ]

# Editable table
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
if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
    cases = [c for c in st.session_state.eval_cases if c["question"].strip()]
    if not cases:
        st.warning("Add at least one question.")
        st.stop()

    results = []
    progress = st.progress(0)
    status = st.status("Running evaluation …", expanded=True)

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    for idx, case in enumerate(cases):
        q = case["question"]
        status.write(f"📝 Evaluating: {q}")

        # Get RAG answer
        result = rag_engine.query_rag(q, chunks, vectors, bm25, API_KEY, top_k=CONFIG.get("top_k", 5))

        answer = result["clean_text"]
        confidence = result["confidence"]
        context = result.get("context_chunks", [])
        ctx_text = " ".join(c["text"] for c in context)

        # ── Faithfulness (LLM judge) ──
        faith_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                    "Score 0-10 how faithful this answer is to the provided context. "
                    "10 = every claim is supported by context. 0 = completely fabricated. "
                    "Return ONLY the number."},
                {"role": "user", "content":
                    f"Context: {ctx_text[:2000]}\n\nAnswer: {answer}"},
            ],
            temperature=0, max_tokens=5,
        )
        try:
            faithfulness = int(faith_resp.choices[0].message.content.strip()) / 10.0
        except ValueError:
            faithfulness = 0.5

        # ── Answer Relevance (LLM judge) ──
        rel_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                    "Score 0-10 how relevant this answer is to the question. "
                    "10 = perfectly answers the question. 0 = completely irrelevant. "
                    "Return ONLY the number."},
                {"role": "user", "content":
                    f"Question: {q}\n\nAnswer: {answer}"},
            ],
            temperature=0, max_tokens=5,
        )
        try:
            relevance = int(rel_resp.choices[0].message.content.strip()) / 10.0
        except ValueError:
            relevance = 0.5

        # ── Context Recall (vs expected, if provided) ──
        recall = None
        if case["expected"].strip():
            rec_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":
                        "Score 0-10 how much of the expected answer is covered by the context. "
                        "10 = all information is in the context. 0 = none is present. "
                        "Return ONLY the number."},
                    {"role": "user", "content":
                        f"Expected: {case['expected']}\n\nContext: {ctx_text[:2000]}"},
                ],
                temperature=0, max_tokens=5,
            )
            try:
                recall = int(rec_resp.choices[0].message.content.strip()) / 10.0
            except ValueError:
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

    status.update(label="Evaluation complete ✓", state="complete")

    # ── Results Table ──
    st.markdown("### Results")

    for r in results:
        f_class = "score-pass" if r["faithfulness"] >= 0.7 else "score-fail"
        r_class = "score-pass" if r["relevance"] >= 0.7 else "score-fail"
        recall_str = f'{r["recall"]:.0%}' if r["recall"] is not None else "—"
        rc_class = "score-pass" if r["recall"] and r["recall"] >= 0.7 else "score-fail"

        st.markdown(f"""
        <div style="background:#161927;border:1px solid #1e2235;border-radius:10px;padding:16px 20px;margin:8px 0;">
            <strong>{r['question']}</strong><br>
            <span style="color:#94A3B8;font-size:13px;">{r['answer']}…</span><br><br>
            <span>Confidence: <strong>{r['confidence']:.0%}</strong></span> ·
            <span>Faithfulness: <span class="{f_class}">{r['faithfulness']:.0%}</span></span> ·
            <span>Relevance: <span class="{r_class}">{r['relevance']:.0%}</span></span> ·
            <span>Recall: <span class="{rc_class}">{recall_str}</span></span>
        </div>
        """, unsafe_allow_html=True)

    # ── Aggregate ──
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["relevance"] for r in results) / len(results)
    recall_vals = [r["recall"] for r in results if r["recall"] is not None]
    avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else None

    st.markdown("### Aggregate Scores")
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("Avg Faithfulness", f"{avg_faith:.0%}")
    ac2.metric("Avg Relevance", f"{avg_rel:.0%}")
    ac3.metric("Avg Context Recall", f"{avg_recall:.0%}" if avg_recall else "—")

    # ── Export ──
    import csv, io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    st.download_button("📥 Download CSV", buf.getvalue(), "evaluation_results.csv", "text/csv")
