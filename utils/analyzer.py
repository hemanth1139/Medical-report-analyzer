import json
import google.generativeai as genai
from config import GEMINI_MODEL, LAB_PROMPT, PRESCRIPTION_PROMPT, MAX_RETRIES

def analyze_report(content, api_key, mode, mime_type=None):
    """Deeply diagnostic analyzer to find the root cause of JSON failures."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    p = LAB_PROMPT if mode == "lab" else PRESCRIPTION_PROMPT
    # Send all files as binary for better AI vision/context
    inp = [{"mime_type": mime_type, "data": content}, p] if isinstance(content, bytes) else [p, content]
    
    last_error = "Unknown Error"
    for i in range(MAX_RETRIES + 1):
        try:
            resp = model.generate_content(inp)
            if not resp.candidates:
                print(f"Debug {i}: No response candidates. Blocked?")
                continue
            raw = resp.text.strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end+1])
            print(f"Debug {i}: Brackets missing. Raw head: {raw[:200]}")
        except Exception as e:
            print(f"Debug {i}: Exception: {e}")
            last_error = str(e)
    return {"error": f"AI Error: {last_error}"}
