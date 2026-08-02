"""
rag_engine.py — Hybrid search (dense + BM25 + RRF), LLM re-ranking, and
streaming answer generation with citations.
"""

import json
import re

import numpy as np
from openai import OpenAI

from modules import knowledge_base as kb

# ── Real-Time Query Understanding ─────────────────────────────────────

def understand_query(query: str, conversation: list[dict], api_key: str) -> dict:
    """Analyze query in real-time to extract intent, expanded search query, key entities, and urgency."""
    client = OpenAI(api_key=api_key)

    history_snippet = "\n".join(
        f"{m['role']}: {m['content']}" for m in conversation[-4:]
    ) if conversation else "No prior history."

    sys_prompt = (
        "You are an expert NLP Query Understanding Engine for enterprise customer support.\n"
        "Analyze the user's latest query in context of the conversation and produce:\n"
        '1. "intent": concise intent category (e.g. Policy Inquiry, Refund Request, Delivery Issue, Escalation Request, Technical Support, General Question)\n'
        '2. "rewritten_query": an expanded, optimized search query incorporating domain keywords, policy terminology, and contextual references for vector search\n'
        '3. "entities": list of 2-5 key entities or domain terms extracted (e.g. ["UPI", "GST Invoice", "Refund Window"])\n'
        '4. "urgency": "Low", "Medium", or "High"\n'
        '5. "core_topic": 2-4 word summary of the user\'s specific issue\n'
        "Return ONLY valid JSON with these 5 keys. No markdown block."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Conversation History:\n{history_snippet}\n\nLatest User Query: {query}"},
            ],
            temperature=0.2,
            max_tokens=250,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return {
            "intent": "General Inquiry",
            "rewritten_query": query,
            "entities": [],
            "urgency": "Medium",
            "core_topic": query[:30],
        }


# ── Hybrid Search ────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    chunks: list[dict],
    vectors: np.ndarray,
    bm25,
    api_key: str,
    top_k: int = 10,
    search_query: str = None,
) -> list[dict]:
    """Combine dense (cosine) + sparse (BM25) via Reciprocal Rank Fusion."""
    if not chunks or vectors is None:
        return []

    target_query = search_query if search_query else query

    # 1. Dense search — cosine similarity
    qvec = kb.embed_texts([target_query], api_key)[0]
    norms = np.linalg.norm(vectors, axis=1)
    qnorm = np.linalg.norm(qvec)
    sims = vectors @ qvec / (norms * qnorm + 1e-10)
    dense_ranked = list(np.argsort(sims)[::-1][:20])

    # 2. Sparse search — BM25
    tokens = kb._tokenize(target_query)
    bm25_scores = bm25.get_scores(tokens) if bm25 else np.zeros(len(chunks))
    sparse_ranked = list(np.argsort(bm25_scores)[::-1][:20])

    # 3. Reciprocal Rank Fusion (k=60)
    rrf: dict[int, float] = {}
    k = 60
    for rank, idx in enumerate(dense_ranked):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_ranked):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank + 1)

    fused = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:top_k]

    return [chunks[i] for i in fused]


# ── Re-Ranking via LLM ──────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], api_key: str,
           top_n: int = 5) -> list[dict]:
    """Use gpt-4o-mini to score relevance of each candidate 0-10."""
    if len(candidates) <= top_n:
        return candidates

    client = OpenAI(api_key=api_key)

    numbered = "\n".join(
        f"[{i}] {c['text'][:200]}" for i, c in enumerate(candidates)
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevance scorer. Given a query and numbered text chunks, "
                    "score each chunk's relevance to the query from 0 (irrelevant) to 10 (perfect). "
                    "Return ONLY a JSON array of objects: [{\"idx\": 0, \"score\": 8}, ...] "
                    "for ALL chunks. No markdown."
                ),
            },
            {"role": "user", "content": f"Query: {query}\n\nChunks:\n{numbered}"},
        ],
        temperature=0.0,
        max_tokens=400,
    )

    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        scores = json.loads(text)
        scores.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_indices = [s["idx"] for s in scores[:top_n] if s["idx"] < len(candidates)]
        return [candidates[i] for i in top_indices]
    except (json.JSONDecodeError, KeyError):
        return candidates[:top_n]


# ── Streaming Answer Generation ──────────────────────────────────────

def generate_answer_stream(
    query: str,
    context_chunks: list[dict],
    conversation: list[dict],
    api_key: str,
):
    """Stream an answer grounded in the retrieved context.
    Yields token strings. After exhaustion, call parse_answer_metadata()
    on the full text to extract confidence and citations.
    """
    client = OpenAI(api_key=api_key)

    ctx_block = "\n\n".join(
        f"[Source: {c['source_file']}, Page {c['page_number']}]\n{c['text']}"
        for c in context_chunks
    )

    system = (
        "You are ZeroBT, an intelligent enterprise customer support chatbot. "
        "Answer the customer's question using ONLY the provided context from uploaded business documents.\n\n"
        "RULES:\n"
        "1. Ground every claim strictly in the context. Never invent or guess information.\n"
        "2. Include inline citations: mention the source file and page number "
        '   (e.g., "According to Refund_Policy.pdf Page 3, ...").\n'
        "3. If the context does not contain enough information or is missing details, say EXACTLY: "
        '"The policy documents do not contain the necessary information for your request. I am escalating this query to our Business Director and Founder for further review."\n'
        "4. Provide detailed, thorough, and well-explained answers in length with complete context and clear formatting so the customer receives a comprehensive response.\n"
        "5. At the VERY END of your response, on a new line, add a metadata line:\n"
        '   <!--META:{"confidence":<0.0-1.0>,"sources":[{"file":"...","page":N},...]}-->\n'
        "   Set confidence to 0.0 if information is missing from documents.\n\n"
        f"CONTEXT:\n{ctx_block}"
    )

    messages = [{"role": "system", "content": system}]
    # Include last few conversation turns for continuity
    for msg in conversation[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=800,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def parse_answer_metadata(full_text: str) -> dict:
    """Extract the <!--META:...--> JSON from the answer. Returns
    {confidence, sources, clean_text}."""
    meta_match = re.search(r"<!--META:(.*?)-->", full_text)
    clean = re.sub(r"\s*<!--META:.*?-->", "", full_text).strip()

    if meta_match:
        try:
            meta = json.loads(meta_match.group(1))
            return {
                "confidence": meta.get("confidence", 0.5),
                "sources": meta.get("sources", []),
                "clean_text": clean,
            }
        except json.JSONDecodeError:
            pass

    return {"confidence": 0.5, "sources": [], "clean_text": clean}


# ── Full RAG Pipeline (non-streaming, for evaluation) ───────────────

def query_rag(query: str, chunks, vectors, bm25, api_key: str,
              conversation: list[dict] = None, top_k: int = 5) -> dict:
    """Non-streaming version that returns the full answer + metadata."""
    candidates = hybrid_search(query, chunks, vectors, bm25, api_key, top_k * 2)
    reranked = rerank(query, candidates, api_key, top_k)

    client = OpenAI(api_key=api_key)

    ctx_block = "\n\n".join(
        f"[Source: {c['source_file']}, Page {c['page_number']}]\n{c['text']}"
        for c in reranked
    )

    system = (
        "You are a helpful customer support chatbot. Answer ONLY from the context.\n"
        "Include source citations (file and page). If unsure, say so.\n"
        "At the end, add: <!--META:{\"confidence\":<0-1>,\"sources\":[...]}-->\n\n"
        f"CONTEXT:\n{ctx_block}"
    )

    messages = [{"role": "system", "content": system}]
    if conversation:
        for m in conversation[-4:]:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.3, max_tokens=800
    )
    full = resp.choices[0].message.content.strip()
    meta = parse_answer_metadata(full)
    meta["context_chunks"] = reranked
    return meta
