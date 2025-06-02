import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from agents.crew_compliance_checker import CrewComplianceChecker
import os

st.set_page_config(page_title="Jurybee NDA Compliance Checker", page_icon="⚖️", layout="centered")
st.title("⚖️ Jurybee NDA Compliance Checker")
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #4F8BF9; color: white; border-radius: 8px;}
    .stTextInput>div>div>input {border-radius: 8px;}
    .stFileUploader>div>div {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.write("""
Upload an NDA clause as text or PDF/image file. The AI agent will analyze compliance and highlight issues.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    st.markdown("---")
    st.info("Your key is only used locally.")

# Input area
input_type = st.radio("Input Type", ["Text Clause", "PDF/Image File"])

if input_type == "Text Clause":
    clause = st.text_area("Paste NDA Clause", height=150)
    file = None
else:
    clause = ""
    file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"])

if st.button("Analyze Compliance", use_container_width=True):
    if not api_key:
        st.error("Please provide your Gemini API Key in the sidebar.")
    else:
        agent = CrewComplianceChecker(llm=None)  # LLM param not used in wrapper
        os.environ["GEMINI_API_KEY"] = api_key
        with st.spinner("Analyzing..."):
            try:
                if file:
                    # Save uploaded file to a temp location
                    import tempfile
                    suffix = os.path.splitext(file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(file.read())
                        tmp_path = tmp_file.name
                    from core.task import Task
                    result = agent._execute_task(Task(data={'file_path': tmp_path}))
                elif clause.strip():
                    from core.task import Task
                    result = agent._execute_task(Task(data={'clause': clause}))
                else:
                    st.warning("Please provide a clause or upload a file.")
                    st.stop()
                if isinstance(result, dict) and result.get("error"):
                    st.error(f"Error: {result['error']}")
                else:
                    st.success("Compliance Analysis Complete!")
                    st.json(result.model_dump() if hasattr(result, 'model_dump') else result)
            except Exception as e:
                st.error(f"Exception: {e}")
