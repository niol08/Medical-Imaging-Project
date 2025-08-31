import streamlit as st

st.title("Welcome")
st.write("Upload files on the other pages to run demo inference. Add real weights to enable true models.")
st.subheader("References (add these weights/code to go beyond the demo)")
st.markdown("- AutoPET public pretrained weights (FDG PET‑CT lesion segmentation, nnU‑Net v2).")
st.markdown("- DeepSA: pretrained subtraction & segmentation for coronary angiograms.")
st.markdown("- Raman frameworks: RSClassification (RamanSystem); DeepeR; RamanSPy toolbox.")
