import json
from google import genai
from google.genai import types
from config import GEMINI_MODEL, LAB_PROMPT, PRESCRIPTION_PROMPT, MAX_RETRIES

def analyze_report(content, api_key, mode, mime_type=None):
    """Deeply diagnostic analyzer using new google-genai SDK."""
    client = genai.Client(api_key=api_key)
    p = LAB_PROMPT if mode == "lab" else PRESCRIPTION_PROMPT
    
    # Send all files as binary for better AI vision/context
    if isinstance(content, bytes):
        part = types.Part.from_bytes(data=content, mime_type=mime_type or "application/pdf")
        inp = [part, p]
    else:
        inp = [p, content]
    
    last_error = "Unknown Error"
    for i in range(MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=inp
            )
            if not resp.candidates:
                continue
            raw = resp.text.strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end+1])
        except Exception as e:
            last_error = str(e)
    return {"error": f"AI Error: {last_error}"}
