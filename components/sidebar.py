import streamlit as st
import os

def render_sidebar():
    """Renders all sidebar controls and returns their current states."""
    st.sidebar.title("🩺 AI Medical Analyzer")
    mode = st.sidebar.radio("Document Type", ["lab", "prescription"])
    uploaded = st.sidebar.file_uploader("Upload Report/Rx", type=["pdf", "png", "jpg", "jpeg"])
    analyze = st.sidebar.button("Analyze Document", type="primary")
    if st.sidebar.button("Clear History"): 
        st.session_state.chat_history = []
        st.rerun()
    return uploaded, analyze, mode
