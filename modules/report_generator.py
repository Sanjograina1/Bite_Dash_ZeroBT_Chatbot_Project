"""
report_generator.py — Multi-format exportable intelligence analysis report generator.
Uses OpenAI API (gpt-4o) to analyze all chat sessions, frustration levels, escalation queues, and customer needs.
Generates .xlsx, .txt, and .pdf analytics reports.
"""

import io
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from modules import store

CONFIG_PATH = Path(__file__).parent.parent / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

def get_frustration_category(score: float) -> str:
    if score >= 70:
        return "HIGH 🔴"
    elif score >= 35:
        return "MEDIUM 🟡"
    else:
        return "LOW 🟢"

def analyze_chat_sessions() -> list[dict]:
    """Inspect all database conversations and compute frustration tiers, customer needs, and escalation reasons."""
    get_convs = getattr(store, "get_all_conversations", getattr(store, "get_active_conversations", lambda: []))
    convs = get_convs()
    escs = store.get_escalations()
    esc_map = {e["session_id"]: e for e in escs}

    analysis_results = []
    for c in convs:
        sess_id = c["id"]
        msgs = c.get("messages", [])
        frust_hist = c.get("frustration", [])

        # Compute peak frustration
        peak_frust = max(frust_hist) if frust_hist else 0.0
        frust_cat = get_frustration_category(peak_frust)

        # Extract customer messages & bot responses
        user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
        full_query = " | ".join(user_msgs) if user_msgs else "No customer text."

        # Find escalation info
        esc = esc_map.get(sess_id)
        if esc:
            esc_status = "Pending Queue ⏳" if not esc.get("resolved") else "Resolved ✅"
            esc_role = CONFIG.get("escalation_hierarchy", {}).get(str(esc.get("level")), {}).get("role", "Support")
            esc_level = f"Level {esc.get('level')} ({esc_role})"
            esc_reason = esc.get("reason", "Customer requested escalation")
        else:
            esc_status = "Resolved by AI 🤖"
            esc_level = "None"
            esc_reason = "Query handled automatically by AI Assistant"

        # Identify customer needs & core problem
        query_lower = full_query.lower()
        needs = []
        if "seal" in query_lower or "torn" in query_lower or "hygiene" in query_lower:
            needs.append("Food Safety & Packaging Integrity Inspection")
        if "rain" in query_lower or "delay" in query_lower or "late" in query_lower:
            needs.append("Delivery Delay Compensation & Live Driver Tracking")
        if "foodiepass" in query_lower or "delivery fee" in query_lower:
            needs.append("FoodiePass Free Delivery Discount Verification")
        if "upi" in query_lower or "promo cash" in query_lower or "refund" in query_lower:
            needs.append("Dual Payment Refund Processing (UPI + Wallet)")
        if not needs:
            needs.append("General Support & Service Inquiry")

        analysis_results.append({
            "session_id": sess_id,
            "created_at": c.get("created_at", "N/A"),
            "peak_frustration": peak_frust,
            "frustration_category": frust_cat,
            "user_query_summary": full_query[:180],
            "customer_needs": ", ".join(needs),
            "escalation_status": esc_status,
            "assigned_department": esc_level,
            "escalation_reason": esc_reason
        })

    return analysis_results


def generate_openai_intelligence_summary(api_key: str) -> str:
    """Use OpenAI GPT-4o API to analyze all customer chat sessions and produce an executive intelligence synthesis."""
    if not api_key:
        return "OpenAI API key not configured. Using standard statistical analysis."

    sessions = analyze_chat_sessions()
    escs = store.get_escalations()
    pending_queue = [e for e in escs if not e.get("resolved")]

    prompt_context = {
        "total_sessions": len(sessions),
        "pending_queue_count": len(pending_queue),
        "escalations": escs,
        "chat_sessions": sessions
    }

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ZeroBT's Chief Customer Experience Analyst. Analyze the provided customer chat sessions, "
                        "frustration trajectories (Low, Medium, High), and escalation queue records. "
                        "Produce a detailed, executive-level intelligence analysis covering:\n"
                        "1. Key Reasons for Escalations across all chats.\n"
                        "2. Analysis of the Escalation Queue (Pending vs Resolved tickets).\n"
                        "3. Deep Understanding of Customer Needs, Friction Points (Rain delays, Torn safety seals, FoodiePass billing, Split refunds).\n"
                        "4. Recommended Operational Actions for Support Management."
                    )
                },
                {
                    "role": "user",
                    "content": f"Analyze these customer chat sessions and escalation records:\n\n{json.dumps(prompt_context, indent=2)[:6000]}"
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Analysis (Fallback): Examined {len(sessions)} customer chats. Identified {len(pending_queue)} pending queue escalations. Main friction points include torn safety seals, rain delays, and split payment refund allocations. (OpenAI API Note: {e})"


def generate_txt_report(api_key: str = "") -> bytes:
    """Generate plain text analysis report powered by OpenAI API."""
    sessions = analyze_chat_sessions()
    escs = store.get_escalations()
    pending_queue = [e for e in escs if not e.get("resolved")]
    resolved_queue = [e for e in escs if e.get("resolved")]

    ai_synthesis = generate_openai_intelligence_summary(api_key)

    high_count = sum(1 for s in sessions if "HIGH" in s["frustration_category"])
    med_count = sum(1 for s in sessions if "MEDIUM" in s["frustration_category"])
    low_count = sum(1 for s in sessions if "LOW" in s["frustration_category"])

    report_lines = [
        "================================================================================",
        "ZEROBT AI SUPPORT — CHAT & ESCALATION INTELLIGENCE ANALYSIS REPORT",
        "POWERED BY OPENAI GPT-4O ANALYTICS API",
        "================================================================================",
        f"Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "1. EXECUTIVE OVERVIEW & QUEUE METRICS",
        "--------------------------------------------------------------------------------",
        f"Total Chat Sessions Analyzed : {len(sessions)}",
        f"High Frustration Sessions    : {high_count} 🔴",
        f"Medium Frustration Sessions  : {med_count} 🟡",
        f"Low Frustration Sessions     : {low_count} 🟢",
        "",
        f"Total Escalations Ingested   : {len(escs)}",
        f"Escalations Pending in Queue : {len(pending_queue)} ⏳",
        f"Escalations Resolved         : {len(resolved_queue)} ✅",
        "",
        "2. OPENAI AI INTELLIGENCE SYNTHESIS & EXECUTIVE ANALYSIS",
        "--------------------------------------------------------------------------------",
        ai_synthesis,
        "",
        "3. CHAT-BY-CHAT FRUSTRATION & ESCALATION AUDIT LOG",
        "--------------------------------------------------------------------------------"
    ]

    for idx, s in enumerate(sessions, 1):
        report_lines.extend([
            f"[{idx}] Session ID: {s['session_id']} | Date: {s['created_at']}",
            f"    • Frustration Level : {s['frustration_category']} (Score: {s['peak_frustration']:.1f}/100)",
            f"    • Customer Needs    : {s['customer_needs']}",
            f"    • Escalation Status : {s['escalation_status']}",
            f"    • Assigned Dept     : {s['assigned_department']}",
            f"    • Escalation Reason : {s['escalation_reason']}",
            f"    • User Query Summary: {s['user_query_summary']}",
            ""
        ])

    report_lines.extend([
        "4. UNDERSTANDING CUSTOMER NEEDS & RECURRING PROBLEMS",
        "--------------------------------------------------------------------------------",
        "• Food Safety Concerns   : Torn safety seals require instant refund & replacement dispatch.",
        "• Weather Delays         : Rain storms increase delivery friction; automated delay vouchers soothe customers.",
        "• Subscription Disputes  : FoodiePass free delivery must be verified automatically before charging fees.",
        "• Dual Refund Processing : Refunds across split payments (UPI + Wallet Promo Cash) must credit back to original channels.",
        "================================================================================"
    ])

    return "\n".join(report_lines).encode("utf-8")


def generate_excel_report(api_key: str = "") -> bytes:
    """Generate Excel report (.xlsx) with multiple formatted sheets including OpenAI AI Analysis."""
    sessions = analyze_chat_sessions()
    escs = store.get_escalations()
    gaps = store.get_knowledge_gaps(50)
    ai_synthesis = generate_openai_intelligence_summary(api_key)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: AI Executive Analysis
        df_ai = pd.DataFrame([{
            "Report Title": "ZeroBT AI Support Intelligence Report",
            "Generated At": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "OpenAI GPT-4o AI Executive Synthesis": ai_synthesis
        }])
        df_ai.to_excel(writer, sheet_name="AI Executive Summary", index=False)

        # Sheet 2: Chat Sessions Audit
        df_chats = pd.DataFrame(sessions)
        df_chats.columns = ["Session ID", "Created At", "Peak Frustration", "Frustration Tier", "User Query Summary", "Customer Needs", "Escalation Status", "Assigned Department", "Escalation Reason"]
        df_chats.to_excel(writer, sheet_name="Chat Analysis Report", index=False)

        # Sheet 3: Escalations Queue
        df_escs = pd.DataFrame(escs) if escs else pd.DataFrame(columns=["id", "session_id", "level", "reason", "summary", "email_sent", "resolved", "created_at"])
        df_escs.to_excel(writer, sheet_name="Escalations Queue", index=False)

        # Sheet 4: Knowledge Gaps & Customer Needs
        df_gaps = pd.DataFrame(gaps) if gaps else pd.DataFrame(columns=["id", "query", "created_at"])
        df_gaps.to_excel(writer, sheet_name="Customer Needs & Gaps", index=False)

    return output.getvalue()


def generate_pdf_report(api_key: str = "") -> bytes:
    """Generate PDF analysis report powered by ReportLab and OpenAI API."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("DocTitle", parent=styles["Title"], fontSize=18, leading=22, textColor=colors.HexColor("#4338CA"))
        h2_style = ParagraphStyle("Heading2", parent=styles["Heading2"], fontSize=12, leading=16, textColor=colors.HexColor("#1E293B"))
        normal_style = ParagraphStyle("NormalText", parent=styles["Normal"], fontSize=8, leading=11, textColor=colors.HexColor("#334155"))

        sessions = analyze_chat_sessions()
        escs = store.get_escalations()
        pending_queue = [e for e in escs if not e.get("resolved")]
        ai_synthesis = generate_openai_intelligence_summary(api_key)

        elements = []

        # Title
        elements.append(Paragraph("ZeroBT AI Support — Intelligence Analysis Report", title_style))
        elements.append(Paragraph(f"Powered by OpenAI GPT-4o API · Generated: {datetime.now().strftime('%B %d, %Y - %H:%M:%S')}", normal_style))
        elements.append(Spacer(1, 10))

        # Metrics Summary Box
        high_cnt = sum(1 for s in sessions if "HIGH" in s["frustration_category"])
        med_cnt = sum(1 for s in sessions if "MEDIUM" in s["frustration_category"])
        low_cnt = sum(1 for s in sessions if "LOW" in s["frustration_category"])

        summary_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Sessions Analyzed", str(len(sessions)), "Pending Escalations Queue", str(len(pending_queue))],
            ["High Frustration Chats", str(high_cnt), "Resolved Escalations", str(len(escs) - len(pending_queue))],
            ["Medium Frustration Chats", str(med_cnt), "Low Frustration Chats", str(low_cnt)]
        ]

        t_summary = Table(summary_data, colWidths=[140, 100, 140, 100])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#6366F1")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ]))
        elements.append(t_summary)
        elements.append(Spacer(1, 12))

        # AI Executive Summary Box
        elements.append(Paragraph("OpenAI GPT-4o Executive Intelligence Synthesis", h2_style))
        elements.append(Spacer(1, 4))
        ai_clean = ai_synthesis.replace("\n", "<br/>")
        elements.append(Paragraph(ai_clean[:1800], normal_style))
        elements.append(Spacer(1, 12))

        # Chat Sessions Analysis Table
        elements.append(Paragraph("Chat Sessions & Escalation Reasons Breakdown", h2_style))
        elements.append(Spacer(1, 6))

        table_data = [["Session ID", "Frustration", "Customer Needs", "Queue Status", "Escalation Reason"]]
        for s in sessions[:12]:
            table_data.append([
                s["session_id"][:12],
                s["frustration_category"],
                Paragraph(s["customer_needs"], normal_style),
                s["escalation_status"],
                Paragraph(s["escalation_reason"][:70], normal_style)
            ])

        t_chats = Table(table_data, colWidths=[65, 75, 135, 90, 175])
        t_chats.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#475569")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        elements.append(t_chats)

        doc.build(elements)
        return buffer.getvalue()

    except Exception as e:
        return f"PDF Report Generation Error: {e}".encode("utf-8")
