"""
sentiment.py — Real-time frustration tracking and seriousness assessment.
"""

import json
from openai import OpenAI

# ── Single-Message Analysis ──────────────────────────────────────────

def analyze_message(message: str, api_key: str) -> dict:
    """Classify a customer message for sentiment, frustration, and escalation signals.
    Returns dict with keys: sentiment, frustration_delta, escalation_signals, wants_human.
    """
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a sentiment classifier for customer support messages. "
                    "Analyze the message and return ONLY valid JSON with these keys:\n"
                    '  "sentiment": one of "positive","neutral","negative","angry","distressed"\n'
                    '  "frustration_delta": integer from -20 to +30 indicating how much this '
                    "message shifts frustration (positive = more frustrated)\n"
                    '  "escalation_signals": list of detected signals from '
                    '["demand_human","threat","urgency","financial_risk","legal_risk","abusive_language","repeated_complaint"]\n'
                    '  "wants_human": boolean, true if customer explicitly asks for a person\n'
                    "Return ONLY the JSON object, no markdown."
                ),
            },
            {"role": "user", "content": message},
        ],
        temperature=0.1,
        max_tokens=200,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "sentiment": "neutral",
            "frustration_delta": 0,
            "escalation_signals": [],
            "wants_human": False,
        }


# ── Frustration Tracker ─────────────────────────────────────────────

class FrustrationTracker:
    """Maintains a rolling frustration score (0–100) across a session."""

    def __init__(self):
        self.score: float = 20.0          # start slightly above zero
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
        # Three consecutive negative messages
        if len(self.window) >= 3 and all(d > 5 for d in self.window[-3:]):
            if self.score >= 50:
                return 2

        if self.score >= 95:
            return 4
        if self.score >= 85:
            return 3
        if self.score >= 70:
            return 2
        return None

    def to_dict(self) -> dict:
        return {"score": self.score, "history": self.history}

    @classmethod
    def from_dict(cls, d: dict) -> "FrustrationTracker":
        t = cls()
        t.score = d.get("score", 20.0)
        t.history = d.get("history", [])
        t.window = t.history[-5:] if t.history else []
        return t


# ── Seriousness Assessment ───────────────────────────────────────────

def assess_seriousness(conversation: list[dict], frustration_score: float,
                       api_key: str) -> dict:
    """Full-context LLM assessment. Returns {level: 1-4, reason: str}."""
    client = OpenAI(api_key=api_key)

    transcript = "\n".join(
        f"{'Customer' if m['role']=='user' else 'Bot'}: {m['content']}"
        for m in conversation
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an escalation intelligence system. Analyze this customer support "
                    "conversation and determine the seriousness level.\n\n"
                    "Consider:\n"
                    "- Tone and language (frustrated, abusive, calm, distressed)\n"
                    "- Explicit demands for escalation or human agent\n"
                    "- Financial, legal, or safety implications\n"
                    "- Conversation length without resolution\n"
                    "- Whether issue could affect multiple customers\n\n"
                    f"Current frustration score: {frustration_score}/100\n\n"
                    "Levels:\n"
                    "  1 = Routine (basic question or mild issue)\n"
                    "  2 = Elevated (frustrated customer, needs senior attention)\n"
                    "  3 = Serious (angry customer, financial/legal risk, repeated failure)\n"
                    "  4 = Critical (threats, safety concern, executive attention needed)\n\n"
                    "Return ONLY JSON: {\"level\": <1-4>, \"reason\": \"<brief reason>\"}"
                ),
            },
            {"role": "user", "content": transcript},
        ],
        temperature=0.1,
        max_tokens=150,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"level": 1, "reason": "Unable to assess"}
