import streamlit as st
from utils.agent import run_agent

def render_chat(api_key, report, analysis, risk, kb_index, kb_facts):
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
        
        with st.spinner("Thinking..."):
            resp_text = run_agent(final_input, report, analysis, risk, kb_index, kb_facts, api_key)
            
        st.session_state.chat_history.append({"role": "assistant", "content": resp_text})
        st.rerun()
