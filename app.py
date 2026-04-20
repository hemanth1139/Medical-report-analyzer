import streamlit as st
import os
from dotenv import load_dotenv
from components.sidebar import render_sidebar
from components.report_view import render_report
from components.chat_view import render_chat
from utils.extractor import extract_content
from utils.analyzer import analyze_report, classify_document
from utils.validator import validate_schema
from utils.risk import detect_risk
from utils.terms import find_terms
from utils.knowledge_base import load_index

load_dotenv()
st.set_page_config(page_title="AI Medical Analyzer", layout="wide")

# Handle API Key for both local (.env) and Streamlit Cloud (st.secrets)
api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY is not set. Please configure it in your environment or Streamlit Secrets.")
    st.stop()

for key, default in {"report_content": None, "analysis": None, "risk_level": "UNKNOWN", 
                    "risk_keywords": [], "chat_history": []}.items():
    if key not in st.session_state: st.session_state[key] = default

if "kb_index" not in st.session_state:
    with st.spinner("Loading medical knowledge base..."):
        st.session_state.kb_index, st.session_state.kb_facts = load_index()

uploaded, analyze_clicked, mode = render_sidebar()

if analyze_clicked and uploaded:
    with st.spinner("Validating document..."):
        content = extract_content(uploaded)
        classification = classify_document(content, api_key)

    is_medical = classification.get("is_medical", True)
    doc_category = classification.get("document_category", "Unknown")
    reason = classification.get("reason", "")

    if not is_medical:
        st.error(
            f"🚫 **Invalid Document — This does not appear to be a medical report.**\n\n"
            f"**Detected:** {doc_category}\n\n"
            f"**Reason:** {reason}\n\n"
            f"Please upload a valid **Lab Report** or **Prescription** (PDF or image)."
        )
    else:
        with st.spinner("Analyzing your medical report..."):
            analysis = analyze_report(content, api_key, mode, mime_type=uploaded.type)
            if validate_schema(analysis, mode):
                st.session_state.report_content = content
                st.session_state.analysis = analysis
                st.session_state.risk_level, st.session_state.risk_keywords = detect_risk(analysis)
            else:
                st.error(analysis.get("error", "Invalid report format. Please upload a proper medical document."))

if st.session_state.analysis:
    analysis = st.session_state.analysis
    patient_name = analysis.get("patient_name", "Not Specified")
    if patient_name and patient_name != "Not Specified":
        st.markdown(f"### 👤 Patient: {patient_name}")
    render_report(analysis, st.session_state.risk_level, 
                  st.session_state.risk_keywords, find_terms(str(analysis)), mode)
    render_chat(api_key, st.session_state.report_content, analysis, 
                st.session_state.risk_level, st.session_state.kb_index, st.session_state.kb_facts)

