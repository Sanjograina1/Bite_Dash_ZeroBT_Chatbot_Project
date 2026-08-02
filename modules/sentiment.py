"""
sentiment.py — Real-time frustration tracking and seriousness assessment.
"""

import json
from openai import OpenAI

# ── Single-Message Analysis ───────────────────────────────────────────

def analyze_message(message: str, api_key: str) -> dict:
    """Classify a customer message for sentiment, frustration, escalation signals, and profanity using gpt-4o.
    Returns dict with keys: sentiment, frustration_delta, escalation_signals, wants_human, has_profanity, sanitized_summary, detailed_tone_analysis.
    """
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a psychological and sentiment analysis system for enterprise customer support intelligence.\n"
                    "Analyze the user message thoroughly with deep behavioral, emotional, and linguistic profiling. Return ONLY valid JSON with these keys:\n"
                    '  "sentiment": one of "positive","neutral","negative","angry","distressed"\n'
                    '  "frustration_delta": integer from -20 to +35 indicating how much this message shifts frustration (positive = more frustrated)\n'
                    '  "escalation_signals": list of detected signals from ["demand_human","threat","urgency","financial_risk","legal_risk","abusive_language","profanity","repeated_complaint"]\n'
                    '  "wants_human": boolean, true if customer explicitly asks for a person\n'
                    '  "has_profanity": boolean, true if explicit profanity, vulgarity, swearing, curse words, or unparliamentary/abusive language is detected in the message\n'
                    '  "sanitized_summary": string, an objective, clean, polite 1-2 sentence summary of the customer\'s core grievance with ALL profanity, swearing, and offensive words completely removed/censored\n'
                    '  "detailed_tone_analysis": exhaustive 4-5 sentence analysis of the customer\'s underlying psychological state, emotional trajectory, micro-frustration markers, implicit subtext, and potential churn risk\n'
                    "Return ONLY valid JSON. No markdown."
                ),
            },
            {"role": "user", "content": message},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(text)
        if "has_profanity" not in data:
            data["has_profanity"] = "profanity" in data.get("escalation_signals", []) or "abusive_language" in data.get("escalation_signals", [])
        if "sanitized_summary" not in data:
            data["sanitized_summary"] = "Customer submitted a query regarding their order/service."
        return data
    except json.JSONDecodeError:
        return {
            "sentiment": "neutral",
            "frustration_delta": 0,
            "escalation_signals": [],
            "wants_human": False,
            "has_profanity": False,
            "sanitized_summary": "Customer submitted a query regarding their order/service.",
            "detailed_tone_analysis": "Standard customer tone.",
        }


# ── Frustration Tracker ─────────────────────────────────────────────

class FrustrationTracker:
    """Maintains a rolling frustration score (0–100) across a session."""

    def __init__(self):
        self.score: float = 0.0           # start at zero
        self.history: list[float] = []    # all deltas
        self.window: list[float] = []     # last 5 deltas

    def update(self, analysis: dict) -> float:
        delta = analysis.get("frustration_delta", 0)
        self.history.append(delta)
        self.window.append(delta)
        if len(self.window) > 5:
            self.window.pop(0)

        # Apply weighted delta (recent messages matter more)
        self.score += delta * 1.2
        self.score = max(0.0, min(100.0, self.score))

        # Gradual cooldown if positive signals
        if delta < 0:
            self.score = max(0.0, self.score + delta * 0.5)

        return self.score

    def should_auto_escalate(self) -> int | None:
        """Return escalation level if auto-escalation is warranted, else None."""
        if len(self.window) >= 3 and all(d > 5 for d in self.window[-3:]):
            if self.score >= 50:
                return 3

        if self.score >= 95:
            return 6
        if self.score >= 88:
            return 5
        if self.score >= 80:
            return 4
        if self.score >= 72:
            return 3
        if self.score >= 65:
            return 2
        if self.score >= 60:
            return 1
        return None

    def to_dict(self) -> dict:
        return {"score": self.score, "history": self.history}

    @classmethod
    def from_dict(cls, d: dict) -> "FrustrationTracker":
        t = cls()
        t.score = d.get("score", 0.0)
        t.history = d.get("history", [])
        t.window = t.history[-5:] if t.history else []
        return t


# ── Seriousness Assessment ───────────────────────────────────────────

def assess_seriousness(conversation: list[dict], frustration_score: float,
                       api_key: str) -> dict:
    """Full-context LLM assessment across 8 levels using gpt-4o. Returns {level: 1-8, reason: str, comprehensive_rationale: str}."""
    client = OpenAI(api_key=api_key)

    transcript = "\n".join(
        f"{'Customer' if m['role']=='user' else 'Bot'}: {m['content']}"
        for m in conversation[-30:]
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise escalation intelligence system for ZeroBT chatbot.\n"
                    "Analyze this full customer support conversation transcript comprehensively and determine the precise seriousness level from 1 to 8.\n\n"
                    "Consider in deep detail:\n"
                    "- Tone, linguistic distress markers, and emotional volatility\n"
                    "- Explicit demands for escalation or human agent intervention\n"
                    "- Financial, legal, regulatory, safety, or brand reputation risks\n"
                    "- Conversation length, repeated attempts, and friction without resolution\n"
                    "- Scope of customer impact and operational severity\n\n"
                    f"Current frustration score: {frustration_score}/100\n\n"
                    "Levels:\n"
                    "  1 = Tier 1 Support (Routine basic query)\n"
                    "  2 = Tier 2 Support (Mild issue / basic clarification)\n"
                    "  3 = Tier 3 Support (Technical or complex inquiry)\n"
                    "  4 = Tier 4 Support (Advanced technical / policy issue)\n"
                    "  5 = Senior Agent (Frustrated customer / repeated attempts)\n"
                    "  6 = Support Manager (Managerial escalation / dispute)\n"
                    "  7 = Business Director (High business/financial impact or missing policy information)\n"
                    "  8 = Founder (Critical threat, executive review, total knowledge gap)\n\n"
                    "Return ONLY valid JSON: {\"level\": <1-8>, \"reason\": \"<brief reason>\", \"comprehensive_rationale\": \"<exhaustive multi-paragraph detailed justification analyzing risk, escalation necessity, customer impact, and strategic resolution pathways>\"}"
                ),
            },
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        max_tokens=2500,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"level": 1, "reason": "Unable to assess", "comprehensive_rationale": "Standard routine query."}


