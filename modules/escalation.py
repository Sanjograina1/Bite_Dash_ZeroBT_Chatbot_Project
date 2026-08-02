"""
escalation.py — Gmail escalation with AI-drafted replies and action buttons.
"""

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from openai import OpenAI

CONFIG_PATH = Path(__file__).parent.parent / "config.json"

# ── Config ───────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def get_hierarchy_contact(level: int) -> dict:
    cfg = load_config()
    h = cfg.get("escalation_hierarchy", {})
    return h.get(str(level), h.get("1", {}))


# ── AI Summary & Suggested Reply ─────────────────────────────────────

def generate_escalation_summary(conversation: list[dict], api_key: str) -> dict:
    """Returns {summary, suggested_reply, key_facts}."""
    client = OpenAI(api_key=api_key)

    transcript = "\n".join(
        f"{'Customer' if m['role']=='user' else 'Bot'}: {m['content']}"
        for m in conversation
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyze this customer support conversation and produce:\n"
                    '1. "summary": 2-3 sentence problem summary\n'
                    '2. "suggested_reply": a ready-to-send reply template the agent can use\n'
                    '3. "key_facts": list of key facts (customer name if known, '
                    "order number, core issue)\n"
                    "Return ONLY JSON with these three keys. No markdown."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "summary": "Customer requires assistance.",
            "suggested_reply": "Dear Customer, thank you for reaching out. We are looking into your concern.",
            "key_facts": [],
        }


# ── HTML Email Builder ───────────────────────────────────────────────

_LEVEL_COLORS = {1: "#86EFAC", 2: "#FDE047", 3: "#FCA5A5", 4: "#FCA5A5"}
_LEVEL_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}


def build_escalation_email(
    conversation: list[dict],
    level: int,
    reason: str,
    summary_data: dict,
    session_id: str,
) -> str:
    """Build a rich HTML escalation email."""
    cfg = load_config()
    app_url = cfg.get("app_url", "http://localhost:8501")
    color = _LEVEL_COLORS.get(level, "#FCA5A5")
    label = _LEVEL_LABELS.get(level, "Unknown")

    transcript_html = ""
    for m in conversation:
        sender = "Customer" if m["role"] == "user" else "AI Bot"
        bg = "#2D3250" if m["role"] == "user" else "#1e2235"
        transcript_html += (
            f'<div style="background:{bg};color:#F1F5F9;padding:10px 14px;'
            f'border-radius:8px;margin:6px 0;font-size:14px;">'
            f"<strong>{sender}:</strong> {m['content']}</div>"
        )

    summary = summary_data.get("summary", "")
    suggested = summary_data.get("suggested_reply", "")
    facts = summary_data.get("key_facts", [])
    facts_html = "".join(f"<li>{f}</li>" for f in facts) if facts else "<li>N/A</li>"

    # Action button URLs
    resolve_url = f"{app_url}/Agent_Portal?action=resolve&ticket={session_id}"
    reassign_url = f"{app_url}/Agent_Portal?action=reassign&ticket={session_id}"

    return f"""
    <html><body style="background:#0D0E15;color:#F1F5F9;font-family:Arial,sans-serif;padding:24px;">
    <div style="max-width:640px;margin:0 auto;background:#161927;border-radius:12px;overflow:hidden;">
        <div style="background:{color};color:#0D0E15;padding:16px 24px;font-size:18px;font-weight:bold;">
            ⚠ Escalation — Level {level} ({label})
        </div>
        <div style="padding:24px;">
            <h3 style="color:#B8A1EA;margin-top:0;">Problem Summary</h3>
            <p>{summary}</p>

            <h3 style="color:#B8A1EA;">Key Facts</h3>
            <ul>{facts_html}</ul>

            <h3 style="color:#B8A1EA;">Reason for Escalation</h3>
            <p>{reason}</p>

            <h3 style="color:#B8A1EA;">Suggested Reply</h3>
            <div style="background:#2D3250;padding:14px;border-radius:8px;border-left:3px solid #86EFAC;">
                {suggested}
            </div>

            <h3 style="color:#B8A1EA;margin-top:24px;">Conversation Transcript</h3>
            {transcript_html}

            <div style="margin-top:24px;text-align:center;">
                <a href="{resolve_url}" style="display:inline-block;background:#86EFAC;color:#0D0E15;
                   padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:bold;margin:4px;">
                   ✓ Mark Resolved</a>
                <a href="{reassign_url}" style="display:inline-block;background:#FDE047;color:#0D0E15;
                   padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:bold;margin:4px;">
                   ↗ Re-assign</a>
            </div>
        </div>
        <div style="background:#0D0E15;padding:12px;text-align:center;font-size:12px;color:#94A3B8;">
            AI Customer Support System · Session {session_id}
        </div>
    </div>
    </body></html>
    """


# ── Send Gmail ───────────────────────────────────────────────────────

def send_gmail(to_email: str, subject: str, html_body: str,
               smtp_user: str, smtp_password: str) -> bool:
    """Send an HTML email via Gmail SMTP. Returns True on success."""
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False


# ── Full Escalation Pipeline ─────────────────────────────────────────

def escalate(
    session_id: str,
    conversation: list[dict],
    level: int,
    reason: str,
    api_key: str,
    gmail_user: str,
    gmail_password: str,
) -> dict:
    """Run the full escalation: summarize, build email, send, log."""
    from modules import store

    contact = get_hierarchy_contact(level)
    summary_data = generate_escalation_summary(conversation, api_key)

    html = build_escalation_email(conversation, level, reason, summary_data, session_id)

    subject = f"[Level {level}] Customer Escalation — {session_id}"
    to_email = contact.get("email", gmail_user)

    sent = send_gmail(to_email, subject, html, gmail_user, gmail_password)

    store.log_escalation(
        session_id, level, reason,
        summary_data.get("summary", ""),
        to_email if sent else "FAILED",
    )

    return {
        "sent": sent,
        "to": to_email,
        "contact": contact,
        "summary": summary_data,
        "level": level,
    }
