import json
from google import genai
from google.genai import types
from config import GEMINI_MODEL, LAB_PROMPT, PRESCRIPTION_PROMPT, MAX_RETRIES, CLASSIFICATION_PROMPT

def _build_content_part(content):
    """Builds a types.Part from raw bytes, detecting MIME type from file signature."""
    if content.startswith(b'%PDF'):
        mime = "application/pdf"
    elif content.startswith(b'\x89PNG'):
        mime = "image/png"
    elif content.startswith(b'\xff\xd8'):
        mime = "image/jpeg"
    elif content.startswith(b'GIF8'):
        mime = "image/gif"
    else:
        mime = "image/png"
    return types.Part.from_bytes(data=content, mime_type=mime)


def classify_document(content, api_key):
    """Pre-validates the document to check if it is a medical report or something else (e.g. a resume).
    Returns a dict with keys: is_medical (bool), document_category (str), confidence (str), reason (str)."""
    client = genai.Client(api_key=api_key)
    part = _build_content_part(content)
    inp = [part, CLASSIFICATION_PROMPT]
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=inp)
        if not resp.candidates:
            return {"is_medical": True, "document_category": "Unknown", "confidence": "Low", "reason": "Classification failed — proceeding with analysis."}
        raw = resp.text.strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end+1])
    except Exception as e:
        pass
    # On failure, allow analysis to proceed
    return {"is_medical": True, "document_category": "Unknown", "confidence": "Low", "reason": "Classification unavailable."}

def analyze_report(content, api_key, mode, mime_type=None):
    """Deeply diagnostic analyzer using new google-genai SDK."""
    client = genai.Client(api_key=api_key)
    p = LAB_PROMPT if mode == "lab" else PRESCRIPTION_PROMPT
    
    if isinstance(content, bytes):
        part = _build_content_part(content)
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
