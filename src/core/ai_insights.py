

import os
import google.generativeai as genai
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if api_key:
    genai.configure(api_key=api_key)
else:
    raise RuntimeError("GEMINI_API_KEY not found in environment or Streamlit secrets.")

def generate_gemini_insight(label: str, confidence: float) -> str:
    """
    Gemini insight for spectroscopy (kept exactly as before so spectroscopy page does not break).
    """
    prompt = f"""
You are a medical AI assistant. 
A Raman spectroscopy model predicted **{label}** with confidence {confidence:.2f}.
Provide a concise, human-friendly explanation and possible clinical interpretation.
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini AI insight failed: {e}"


def generate_insight(pred: Dict[str, Any], modality: str) -> str:
    """
    Gemini-powered insight generator for multiple imaging modalities.
    """
    try:
        if modality == "spectroscopy":
            return generate_gemini_insight(pred.get("label","unknown"), pred.get("confidence",0.0))

        elif modality == "fluoroscopy":
            pct = pred.get("lesion_percent", None)
            if pct is not None:
                prompt = f"""
You are a medical AI assistant.
A coronary angiogram fluoroscopy model estimated ~{pct:.1f}% vessel/lesion occupancy.
Explain what this means clinically in simple terms.
"""
            else:
                prompt = "Explain the result of a generic fluoroscopy segmentation in a clinical context."

        elif modality == "xray-fluoroscopy":
            label = pred.get("label", "unknown")
            conf = pred.get("confidence", 0.0)
            prompt = f"""
You are a medical AI assistant.
A chest X-ray (fluoroscopy demo) classifier predicted **{label}** with confidence {conf:.2f}.
Provide a concise clinical interpretation for this finding.
"""

        elif modality == "pet":
            vol = pred.get("lesion_ml", None)
            suv = pred.get("suvmax", None)
            if vol or suv:
                parts = []
                if vol is not None:
                    parts.append(f"lesion volume ≈ {vol:.1f} mL")
                if suv is not None:
                    parts.append(f"SUVmax ≈ {suv:.1f}")
                details = ", ".join(parts)
                prompt = f"""
You are a medical AI assistant.
An FDG PET-CT model produced: {details}.
Summarize this into a concise clinical interpretation.
"""
            else:
                prompt = "Explain the result of an FDG PET-CT analysis in a clinical context."

        else:
            return "No insight available."

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Gemini AI insight failed: {e}"
