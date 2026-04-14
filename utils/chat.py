from config import CHAT_SYSTEM_PROMPT

def build_messages(report, analysis, risk_level, history, user_message):
    """Constructs the full conversation transcript for Gemini chat."""
    context = f"Document: {report}\nAnalysis: {analysis}\nRisk Level: {risk_level}"
    messages = [{"role": "user", "parts": [f"{CHAT_SYSTEM_PROMPT}\n{context}"]}]
    for msg in history:
        messages.append({"role": "model" if msg["role"] == "assistant" else "user", "parts": [msg["content"]]})
    messages.append({"role": "user", "parts": [user_message]})
    return messages
