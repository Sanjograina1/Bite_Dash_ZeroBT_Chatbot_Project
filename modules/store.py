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
