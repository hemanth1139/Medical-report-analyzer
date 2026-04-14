import streamlit as st
import json

def render_report(data, risk, keywords, terms, mode):
    """Component to display the analyzed medical report data."""
    st.divider()
    trig = f"Triggers: {', '.join(keywords)}" if keywords else ""
    if risk == "HIGH": st.error(f"🚨 HIGH RISK: {trig}")
    elif risk == "MODERATE": st.warning(f"⚠️ MODERATE RISK: {trig}")
    elif risk == "NORMAL": st.success("✅ NORMAL RANGE")
    else: st.info("ℹ️ UNKNOWN RISK LEVEL")
    
    st.subheader("Summary")
    st.write(data.get("summary"))
    
    if mode == "lab":
        st.subheader("Key Findings")
        st.table(data.get("key_findings", []))
        st.warning(f"**Recommendation:** {data.get('recommendation')}")
        st.success("**Lifestyle Tips:** " + ", ".join(data.get("lifestyle_tips", [])))
    else:
        st.subheader("Medications")
        for m in data.get("medications", []):
            st.info(f"**{m.get('name')}** ({m.get('strength')}) - {m.get('instructions')}")
    
    if terms:
        with st.expander("Medical Terms Explained"):
            for t, e in terms.items(): st.markdown(f"**{t}**: {e}")
            
    st.download_button("Download JSON Results", json.dumps(data, indent=2), "analysis.json")
