"""
store.py — SQLite storage for conversations, escalations, documents, analytics.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "support.db"

# ── helpers ──────────────────────────────────────────────────────────

def _conn():
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            file_type   TEXT,
            status      TEXT DEFAULT 'active',
            chunk_count INTEGER DEFAULT 0,
            uploaded_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id       INTEGER NOT NULL,
            chunk_index  INTEGER,
            text         TEXT NOT NULL,
            source_file  TEXT,
            page_number  INTEGER,
            embedding    BLOB,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id          TEXT PRIMARY KEY,
            messages    TEXT DEFAULT '[]',
            frustration TEXT DEFAULT '[]',
            status      TEXT DEFAULT 'active',
            agent_id    TEXT,
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS escalations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            level       INTEGER,
            reason      TEXT,
            summary     TEXT,
            email_sent  TEXT,
            resolved    INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS ratings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            stars      INTEGER,
            feedback   TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS knowledge_gaps (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            query      TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    c.commit()
    c.close()
    seed_sample_conversations()


def seed_sample_conversations():
    """Ensure sample chat sessions varying from Low, Medium, to High frustration are seeded into database."""
    c = _conn()
    existing_count = c.execute("SELECT COUNT(*) as cnt FROM conversations").fetchone()["cnt"]
    if existing_count >= 5:
        c.close()
        return

    sample_sessions = [
        {
            "id": "sess_low_01",
            "messages": [
                {"role": "user", "content": "Hi, can you tell me what payment methods are accepted for online orders?"},
                {"role": "assistant", "content": "We accept UPI, Credit/Debit Cards, Net Banking, and ZeroBT Promo Cash wallet."}
            ],
            "frustration": [0.0, 10.0, 12.0],
            "status": "closed"
        },
        {
            "id": "sess_low_02",
            "messages": [
                {"role": "user", "content": "How do I check my remaining Promo Cash balance?"},
                {"role": "assistant", "content": "You can check your balance under Profile -> Wallet & Refunds."}
            ],
            "frustration": [0.0, 5.0, 15.0],
            "status": "closed"
        },
        {
            "id": "sess_med_01",
            "messages": [
                {"role": "user", "content": "I ordered lunch 30 mins ago during heavy rain, but delivery tracker shows delayed. What is happening?"},
                {"role": "assistant", "content": "During rain storms, delivery times may be extended up to 25 mins for rider safety. You receive a ₹30 delay voucher if delayed beyond 30 mins."}
            ],
            "frustration": [15.0, 35.0, 48.0],
            "status": "active"
        },
        {
            "id": "sess_med_02",
            "messages": [
                {"role": "user", "content": "Why was delivery fee added when I have FoodiePass subscription?"},
                {"role": "assistant", "content": "FoodiePass free delivery applies on orders over ₹199. Let me verify your cart total."}
            ],
            "frustration": [20.0, 42.0, 55.0],
            "status": "active"
        },
        {
            "id": "sess_high_01",
            "messages": [
                {"role": "user", "content": "I placed an order worth ₹280 for a Pure-Veg meal during a heavy rain storm today. Paid via ₹50 Promo Cash and rest via UPI. Food arrived 45 mins late, safety seal tape was TORN OPEN! My FoodiePass free delivery wasn't applied and extra delivery fee was charged! Process full refund across UPI and Promo Cash right now!"},
                {"role": "assistant", "content": "I deeply apologize for the torn safety seal and rain delay. I am escalating your ticket immediately to our Senior Support Agent and Quality team for a full refund."}
            ],
            "frustration": [45.0, 78.0, 92.0],
            "status": "active"
        },
        {
            "id": "sess_high_02",
            "messages": [
                {"role": "user", "content": "This service is completely awful! Fix my delivery fee dispute immediately and connect me to a Senior Agent or Manager!"},
                {"role": "assistant", "content": "I have escalated your ticket to Level 5 Senior Agent and Support Manager."}
            ],
            "frustration": [50.0, 82.0, 88.0],
            "status": "active"
        }
    ]

    for s in sample_sessions:
        c.execute(
            "INSERT OR IGNORE INTO conversations (id, messages, frustration, status) VALUES (?, ?, ?, ?)",
            (s["id"], json.dumps(s["messages"]), json.dumps(s["frustration"]), s["status"])
        )

    # Seed escalation queue records if empty
    esc_count = c.execute("SELECT COUNT(*) as cnt FROM escalations").fetchone()["cnt"]
    if esc_count == 0:
        sample_escs = [
            ("sess_high_01", 5, "Torn safety seal, 45-min rain delay, FoodiePass delivery fee dispute, split UPI + Promo Cash refund request", "High priority safety & billing escalation", "sanjograina50@gmail.com", 0),
            ("sess_high_02", 6, "Customer requested Senior Agent & Manager escalation due to repeated delivery fee dispute", "Managerial dispute escalation", "sanjograina50@gmail.com", 0),
            ("sess_med_01", 2, "Mild rain delay tracking inquiry and voucher request", "Logistics & Delivery Support", "sanjograina50@gmail.com", 1),
            ("sess_med_02", 3, "FoodiePass free delivery fee verification dispute", "Billing & Refunds Department", "sanjograina50@gmail.com", 0)
        ]
        for sess_id, lvl, reason, summary, email_sent, resolved in sample_escs:
            c.execute(
                "INSERT INTO escalations (session_id, level, reason, summary, email_sent, resolved) VALUES (?, ?, ?, ?, ?, ?)",
                (sess_id, lvl, reason, summary, email_sent, resolved)
            )

    # Seed CSAT ratings if empty
    rating_count = c.execute("SELECT COUNT(*) as cnt FROM ratings").fetchone()["cnt"]
    if rating_count == 0:
        sample_ratings = [
            ("sess_low_01", 5, "Fast and clear payment info!"),
            ("sess_low_02", 5, "Helped me find wallet balance."),
            ("sess_med_01", 3, "Delay voucher provided."),
            ("sess_high_01", 1, "Torn safety seal and delivery fee charge!")
        ]
        for sess_id, stars, feedback in sample_ratings:
            c.execute("INSERT INTO ratings (session_id, stars, feedback) VALUES (?, ?, ?)", (sess_id, stars, feedback))

    # Seed knowledge gaps if empty
    gap_count = c.execute("SELECT COUNT(*) as cnt FROM knowledge_gaps").fetchone()["cnt"]
    if gap_count == 0:
        sample_gaps = [
            "Torn safety seal refund policy for split payments (UPI + Promo Cash)",
            "FoodiePass free delivery minimum order threshold policy",
            "Rain storm delay compensation voucher rules"
        ]
        for g in sample_gaps:
            c.execute("INSERT INTO knowledge_gaps (query) VALUES (?)", (g,))

    c.commit()
    c.close()

# ── Documents ────────────────────────────────────────────────────────

def add_document_record(name: str, file_type: str) -> int:
    c = _conn()
    cur = c.execute(
        "INSERT INTO documents (name, file_type) VALUES (?, ?)",
        (name, file_type),
    )
    doc_id = cur.lastrowid
    c.commit()
    c.close()
    return doc_id


def update_document_chunks(doc_id: int, chunk_count: int):
    c = _conn()
    c.execute(
        "UPDATE documents SET chunk_count=?, status='active' WHERE id=?",
        (chunk_count, doc_id),
    )
    c.commit()
    c.close()


def soft_delete_document(doc_id: int):
    c = _conn()
    c.execute("UPDATE documents SET status='deleted' WHERE id=?", (doc_id,))
    c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    c.commit()
    c.close()


def deduplicate_documents():
    """Find active documents with duplicate names, keep the latest one, and soft-delete older duplicates."""
    c = _conn()
    docs = c.execute(
        "SELECT id, name, uploaded_at FROM documents WHERE status='active' ORDER BY uploaded_at DESC, id DESC"
    ).fetchall()

    seen_names = set()
    to_delete = []

    for d in docs:
        name = d["name"]
        if name in seen_names:
            to_delete.append(d["id"])
        else:
            seen_names.add(name)

    for doc_id in to_delete:
        c.execute("UPDATE documents SET status='deleted' WHERE id=?", (doc_id,))
        c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))

    c.commit()
    c.close()
    return len(to_delete)


def list_documents():
    c = _conn()
    rows = c.execute(
        "SELECT id, name, file_type, status, chunk_count, uploaded_at "
        "FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]



# ── Chunks ───────────────────────────────────────────────────────────

def save_chunks(doc_id: int, chunks: list[dict]):
    """Save chunks with embeddings.  Each chunk dict has text, source_file,
    page_number, embedding (bytes)."""
    c = _conn()
    c.executemany(
        "INSERT INTO chunks (doc_id, chunk_index, text, source_file, page_number, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (doc_id, i, ch["text"], ch["source_file"], ch["page_number"], ch["embedding"])
            for i, ch in enumerate(chunks)
        ],
    )
    c.commit()
    c.close()


def load_active_chunks() -> list[dict]:
    """Load all chunks belonging to active documents."""
    c = _conn()
    rows = c.execute(
        "SELECT c.id, c.text, c.source_file, c.page_number, c.embedding "
        "FROM chunks c JOIN documents d ON c.doc_id = d.id "
        "WHERE d.status = 'active' ORDER BY c.id"
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


# ── Conversations ────────────────────────────────────────────────────

def new_session_id() -> str:
    return str(uuid.uuid4())[:8]


def save_conversation(session_id: str, messages: list, frustration: list,
                      status: str = "active", agent_id: str = None):
    c = _conn()
    c.execute(
        """INSERT INTO conversations (id, messages, frustration, status, agent_id, updated_at)
           VALUES (?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT(id) DO UPDATE SET
             messages=excluded.messages, frustration=excluded.frustration,
             status=excluded.status, agent_id=excluded.agent_id,
             updated_at=datetime('now')""",
        (session_id, json.dumps(messages), json.dumps(frustration), status, agent_id),
    )
    c.commit()
    c.close()


def get_conversation(session_id: str) -> dict | None:
    c = _conn()
    row = c.execute("SELECT * FROM conversations WHERE id=?", (session_id,)).fetchone()
    c.close()
    return dict(row) if row else None


def get_active_conversations() -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT id, messages, frustration, status, agent_id, created_at, updated_at "
        "FROM conversations WHERE status IN ('active','takeover') "
        "ORDER BY updated_at DESC"
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_all_conversations() -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT id, messages, frustration, status, agent_id, created_at, updated_at "
        "FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    c.close()
    res = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("messages"), str):
            try:
                d["messages"] = json.loads(d["messages"])
            except Exception:
                d["messages"] = []
        if isinstance(d.get("frustration"), str):
            try:
                d["frustration"] = json.loads(d["frustration"])
            except Exception:
                d["frustration"] = []
        res.append(d)
    return res


def flag_takeover(session_id: str, agent_id: str):
    c = _conn()
    c.execute(
        "UPDATE conversations SET status='takeover', agent_id=? WHERE id=?",
        (agent_id, session_id),
    )
    c.commit()
    c.close()


def release_takeover(session_id: str):
    c = _conn()
    c.execute(
        "UPDATE conversations SET status='active', agent_id=NULL WHERE id=?",
        (session_id,),
    )
    c.commit()
    c.close()


# ── Escalations ──────────────────────────────────────────────────────

def log_escalation(session_id: str, level: int, reason: str,
                   summary: str, email_sent: str):
    c = _conn()
    c.execute(
        "INSERT INTO escalations (session_id, level, reason, summary, email_sent) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, level, reason, summary, email_sent),
    )
    c.commit()
    c.close()


def resolve_escalation(esc_id: int):
    c = _conn()
    c.execute("UPDATE escalations SET resolved=1 WHERE id=?", (esc_id,))
    c.commit()
    c.close()


def get_escalations(resolved: bool | None = None) -> list[dict]:
    c = _conn()
    q = "SELECT * FROM escalations"
    if resolved is not None:
        q += f" WHERE resolved={int(resolved)}"
    q += " ORDER BY created_at DESC"
    rows = c.execute(q).fetchall()
    c.close()
    return [dict(r) for r in rows]


# ── Ratings ──────────────────────────────────────────────────────────

def save_rating(session_id: str, stars: int, feedback: str = ""):
    c = _conn()
    c.execute(
        "INSERT INTO ratings (session_id, stars, feedback) VALUES (?, ?, ?)",
        (session_id, stars, feedback),
    )
    c.commit()
    c.close()


def get_avg_rating() -> float:
    c = _conn()
    row = c.execute("SELECT AVG(stars) as avg FROM ratings").fetchone()
    c.close()
    return round(row["avg"], 2) if row and row["avg"] else 0.0


def get_all_ratings() -> list[dict]:
    c = _conn()
    rows = c.execute("SELECT * FROM ratings ORDER BY created_at DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]


# ── Knowledge Gaps ───────────────────────────────────────────────────

def log_knowledge_gap(query: str):
    c = _conn()
    c.execute("INSERT INTO knowledge_gaps (query) VALUES (?)", (query,))
    c.commit()
    c.close()


def get_knowledge_gaps(limit: int = 20) -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT query, COUNT(*) as count FROM knowledge_gaps "
        "GROUP BY query ORDER BY count DESC LIMIT ?",
        (limit,),
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


# ── Analytics helpers ────────────────────────────────────────────────

def get_escalation_stats() -> dict:
    c = _conn()
    rows = c.execute(
        "SELECT level, COUNT(*) as count FROM escalations GROUP BY level"
    ).fetchall()
    c.close()
    return {r["level"]: r["count"] for r in rows}


def get_conversation_count() -> int:
    c = _conn()
    row = c.execute("SELECT COUNT(*) as cnt FROM conversations").fetchone()
    c.close()
    return row["cnt"] if row else 0
