import streamlit as st
from google import genai
from utils.chat import build_messages
from config import GEMINI_MODEL

def render_chat(api_key, report, analysis, risk):
    """Component for interactive chat with the analyzed document."""
    st.divider()
    st.subheader("Chat with your Report")
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
    
    user_input = st.chat_input("Ask a question...", key="chat_input_box")
    final_input = st.session_state.get("chat_input") or user_input
    
    if final_input:
        st.session_state.chat_input = None
        st.session_state.chat_history.append({"role": "user", "content": final_input})
        client = genai.Client(api_key=api_key)
        msgs = build_messages(report, analysis, risk, st.session_state.chat_history[:-1], final_input)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=msgs)
        st.session_state.chat_history.append({"role": "assistant", "content": resp.text})
        st.rerun()
