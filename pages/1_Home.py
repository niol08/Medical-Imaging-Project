import streamlit as st

st.set_page_config(page_title="Medical Imaging AI")

st.title("Medical Imaging AI")
st.markdown("""
Welcome to **Medical Imaging AI** – a research platform for multimodal medical imaging analysis.

**Features:**
- Raman Spectroscopy classification
- X-ray Fluoroscopy analysis
- PET lesion quantification
- AI-powered clinical insights

Use the sidebar to navigate between imaging modalities and upload your data for instant predictions and explanations.

---
*For research use only. Not for clinical decision making.*
""")