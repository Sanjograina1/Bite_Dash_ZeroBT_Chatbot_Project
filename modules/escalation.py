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

def generate_escalation_summary(conversation: list[dict], level: int, api_key: str) -> dict:
    """Returns {summary, suggested_reply, key_facts, executive_brief}."""
    client = OpenAI(api_key=api_key)

    transcript = "\n".join(
        f"{'Customer' if m['role']=='user' else 'Bot'}: {m['content']}"
        for m in conversation
    )

    is_executive = level >= 7

    sys_prompt = (
        "You are an enterprise escalation dossier engine for ZeroBT executive leadership.\n"
        "Analyze this full customer support conversation transcript and generate an exhaustive, multi-section dossier:\n"
        '1. "summary": comprehensive multi-paragraph operational breakdown covering the customer\'s core grievance, root causes, timeline of events, friction points, and financial/policy impact.\n'
        '2. "suggested_reply": an exhaustive, highly empathetic, professional, ready-to-send resolution email template for the support team or executive containing personalized customer greetings, policy explanation, action steps, compensation/waiver proposals, and direct contact details.\n'
        '3. "key_facts": comprehensive list of 5-10 key facts (customer name, contact, delivery location, order/SKU details, financial risk, legal mentions, frustration score, policy clauses involved).\n'
        '4. "executive_brief": an exhaustive, formal, objective executive briefing filtering out any profanity or raw aggression into constructive, high-level business intelligence, strategic risk analysis, brand impact, and policy recommendation.\n'
        "Return ONLY valid JSON with these four keys. No markdown block."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Escalation Level: {level}\nTranscript:\n{transcript}"},
        ],
        temperature=0.3,
        max_tokens=2500,
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
            "executive_brief": "Customer query escalated for executive review regarding policy coverage.",
        }


# ── HTML Email Builder ───────────────────────────────────────────────

_LEVEL_COLORS = {
    1: "#86EFAC",
    2: "#A7F3D0",
    3: "#FDE047",
    4: "#FACC15",
    5: "#FB923C",
    6: "#F87171",
    7: "#EF4444",
    8: "#DC2626",
}
_LEVEL_LABELS = {
    1: "Tier 1 Support",
    2: "Tier 2 Support",
    3: "Tier 3 Support",
    4: "Tier 4 Support",
    5: "Senior Agent",
    6: "Support Manager",
    7: "Business Director",
    8: "Founder",
}


def build_escalation_email(
    conversation: list[dict],
    level: int,
    reason: str,
    summary_data: dict,
    session_id: str,
) -> str:
    """Build a rich level-specific HTML escalation email."""
    cfg = load_config()
    app_url = cfg.get("app_url", "http://localhost:8501")
    contact = get_hierarchy_contact(level)
    role_name = contact.get("role", _LEVEL_LABELS.get(level, f"Level {level}"))
    color = _LEVEL_COLORS.get(level, "#EF4444")

    summary = summary_data.get("summary", "")
    suggested = summary_data.get("suggested_reply", "")
    facts = summary_data.get("key_facts", [])
    facts_html = "".join(f"<li>{f}</li>" for f in facts) if facts else "<li>N/A</li>"
    exec_brief = summary_data.get("executive_brief", summary)

    # Direct links
    reply_url = f"{app_url}/Agent_Portal?action=reply&ticket={session_id}"
    resolve_url = f"{app_url}/Agent_Portal?action=resolve&ticket={session_id}"
    reassign_url = f"{app_url}/Agent_Portal?action=reassign&ticket={session_id}"

    # Build Level-Specific Email Template
    if level >= 7:
        # Executive Template (Founder / Director) — No raw abusive text
        body_content = f"""
        <div style="background:#FFF5F5;border-left:4px solid #DC2626;padding:14px 18px;border-radius:8px;margin-bottom:20px;">
            <h3 style="color:#991B1B;margin-top:0;">👔 Executive Brief & Policy Review</h3>
            <p style="color:#7F1D1D;line-height:1.5;">{exec_brief}</p>
        </div>

        <h3 style="color:#4338CA;margin-top:0;">Core Issue & Business Impact</h3>
        <p style="color:#334155;line-height:1.5;">{summary}</p>

        <h3 style="color:#4338CA;">Key Incident Facts</h3>
        <ul style="color:#334155;">{facts_html}</ul>

        <h3 style="color:#4338CA;">Reason for Executive Escalation</h3>
        <p style="color:#334155;line-height:1.5;">{reason}</p>

        <h3 style="color:#4338CA;">Recommended Executive Reply</h3>
        <div style="background:#F1F5F9;color:#1E293B;padding:14px;border-radius:8px;border-left:4px solid #6366F1;">
            {suggested}
        </div>

        <div style="margin-top:28px;text-align:center;background:#F8FAFC;padding:16px;border-radius:10px;border:1px solid #E2E8F0;">
            <p style="color:#475569;margin-bottom:12px;font-weight:600;">Action Required: Respond to Customer Query</p>
            <a href="{reply_url}" style="display:inline-block;background:#4F46E5;color:#FFFFFF;
               padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:bold;margin:4px;">
               ✉ Respond to Customer Query</a>
            <a href="{resolve_url}" style="display:inline-block;background:#10B981;color:#FFFFFF;
               padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:bold;margin:4px;">
               ✓ Mark Resolved</a>
        </div>
        """
    elif level >= 5:
        # Senior Agent / Support Manager Template
        transcript_html = ""
        for m in conversation:
            sender = "Customer" if m["role"] == "user" else "ZeroBT AI"
            bg = "#F1F5F9" if m["role"] == "user" else "#E2E8F0"
            transcript_html += (
                f'<div style="background:{bg};color:#1E293B;padding:10px 14px;'
                f'border-radius:8px;margin:6px 0;font-size:14px;border:1px solid #CBD5E1;">'
                f"<strong>{sender}:</strong> {m['content']}</div>"
            )

        body_content = f"""
        <h3 style="color:#4338CA;margin-top:0;">Managerial Problem Summary</h3>
        <p style="color:#334155;line-height:1.5;">{summary}</p>

        <h3 style="color:#4338CA;">Key Incident Facts</h3>
        <ul style="color:#334155;">{facts_html}</ul>

        <h3 style="color:#4338CA;">Reason for Escalation</h3>
        <p style="color:#334155;line-height:1.5;">{reason}</p>

        <h3 style="color:#4338CA;">AI Suggested Managerial Reply</h3>
        <div style="background:#F1F5F9;color:#1E293B;padding:14px;border-radius:8px;border-left:4px solid #6366F1;">
            {suggested}
        </div>

        <h3 style="color:#4338CA;margin-top:24px;">Full Incident Transcript</h3>
        {transcript_html}

        <div style="margin-top:24px;text-align:center;">
            <a href="{reply_url}" style="display:inline-block;background:#4F46E5;color:#FFFFFF;
               padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:bold;margin:4px;">
               ✉ Respond to Customer Query</a>
            <a href="{resolve_url}" style="display:inline-block;background:#10B981;color:#FFFFFF;
               padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:bold;margin:4px;">
               ✓ Mark Resolved</a>
            <a href="{reassign_url}" style="display:inline-block;background:#F59E0B;color:#FFFFFF;
               padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:bold;margin:4px;">
               ↗ Re-assign</a>
        </div>
        """
    else:
        # Support Tier 1-4 Template
        transcript_html = ""
        for m in conversation:
            sender = "Customer" if m["role"] == "user" else "ZeroBT AI"
            bg = "#F1F5F9" if m["role"] == "user" else "#E2E8F0"
            transcript_html += (
                f'<div style="background:{bg};color:#1E293B;padding:10px 14px;'
                f'border-radius:8px;margin:6px 0;font-size:14px;border:1px solid #CBD5E1;">'
                f"<strong>{sender}:</strong> {m['content']}</div>"
            )

        body_content = f"""
        <h3 style="color:#4338CA;margin-top:0;">Support Ticket Summary</h3>
        <p style="color:#334155;line-height:1.5;">{summary}</p>

        <h3 style="color:#4338CA;">Key Facts</h3>
        <ul style="color:#334155;">{facts_html}</ul>

        <h3 style="color:#4338CA;">Escalation Trigger</h3>
        <p style="color:#334155;line-height:1.5;">{reason}</p>

        <h3 style="color:#4338CA;">AI Draft Response</h3>
        <div style="background:#F1F5F9;color:#1E293B;padding:14px;border-radius:8px;border-left:4px solid #6366F1;">
            {suggested}
        </div>

        <h3 style="color:#4338CA;margin-top:24px;">Customer Transcript</h3>
        {transcript_html}

        <div style="margin-top:24px;text-align:center;">
            <a href="{reply_url}" style="display:inline-block;background:#4F46E5;color:#FFFFFF;
               padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:bold;margin:4px;">
               ✉ Respond to Customer Query</a>
            <a href="{resolve_url}" style="display:inline-block;background:#10B981;color:#FFFFFF;
               padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:bold;margin:4px;">
               ✓ Mark Resolved</a>
        </div>
        """

    return f"""
    <html><body style="background:#F8FAFC;color:#0F172A;font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;padding:24px;">
    <div style="max-width:660px;margin:0 auto;background:#FFFFFF;border-radius:12px;overflow:hidden;box-shadow: 0 4px 16px rgba(0,0,0,0.08);border: 1px solid #E2E8F0;">
        <div style="background:{color};color:#FFFFFF;padding:18px 24px;font-size:18px;font-weight:bold;">
            ⚠ ZeroBT Escalation Alert — Level {level}: {role_name}
        </div>
        <div style="padding:24px;">
            {body_content}
        </div>
        <div style="background:#F1F5F9;padding:12px;text-align:center;font-size:12px;color:#64748B;border-top:1px solid #E2E8F0;">
            ZeroBT Enterprise Customer Support Engine · Session {session_id}
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
    summary_data = generate_escalation_summary(conversation, level, api_key)

    html = build_escalation_email(conversation, level, reason, summary_data, session_id)

    subject = f"[Level {level}] ZeroBT Escalation Alert — {session_id}"
    to_email = contact.get("email", "sanjograina50@gmail.com")

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
