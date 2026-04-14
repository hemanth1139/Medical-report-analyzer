from google.genai import types
from config import CHAT_SYSTEM_PROMPT

def build_messages(report, analysis, risk_level, history, user_message):
    """Constructs the full conversation transcript for Gemini SDK v2."""
    context = f"Document: {report}\nAnalysis: {analysis}\nRisk Level: {risk_level}"
    messages = [
        types.Content(role="user", parts=[types.Part.from_text(text=f"{CHAT_SYSTEM_PROMPT}\n{context}")])
    ]
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        messages.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    
    messages.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
    return messages
