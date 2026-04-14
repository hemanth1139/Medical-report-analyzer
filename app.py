import streamlit as st
import os
from dotenv import load_dotenv
from components.sidebar import render_sidebar
from components.report_view import render_report
from components.chat_view import render_chat
from utils.extractor import extract_content
from utils.analyzer import analyze_report
from utils.validator import validate_schema
from utils.risk import detect_risk
from utils.terms import find_terms

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

uploaded, analyze_clicked, mode = render_sidebar()

if analyze_clicked and uploaded:
    with st.spinner("Analyzing..."):
        content = extract_content(uploaded)
        analysis = analyze_report(content, api_key, mode, mime_type=uploaded.type)
        if validate_schema(analysis, mode):
            st.session_state.report_content = content
            st.session_state.analysis = analysis
            st.session_state.risk_level, st.session_state.risk_keywords = detect_risk(analysis)
        else:
            st.error(analysis.get("error", "Invalid report format."))

if st.session_state.analysis:
    render_report(st.session_state.analysis, st.session_state.risk_level, 
                  st.session_state.risk_keywords, find_terms(str(st.session_state.analysis)), mode)
    render_chat(api_key, st.session_state.report_content, st.session_state.analysis, 
                st.session_state.risk_level)
