
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import torchxrayvision as xrv
from transformers import AutoImageProcessor, AutoModelForImageClassification
from src.core.ai_insights import generate_insight

st.title("X-ray Fluoroscopy (Fluoroscopy Classifier Substitution)")


model_choice = st.sidebar.selectbox(
    "Select Chest X-ray model",
    ["Pneumonia (ViT binary)", "DenseNet121 (multilabel)"]
)


@st.cache_resource(show_spinner="Loading HuggingFace Pneumonia model…")
def load_pneumonia_assets():
    proc  = AutoImageProcessor.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
    model = AutoModelForImageClassification.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
    model.eval()
    return proc, model, ["Normal", "Pneumonia"]

@st.cache_resource(show_spinner="Loading DenseNet121 multilabel model…")
def load_densenet_assets():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model, xrv.datasets.default_pathologies


upl = st.file_uploader("Upload a chest X-ray (.png/.jpg)", type=["png","jpg","jpeg"], key="xray_up")

if upl:
    img = Image.open(upl).convert("RGB" if model_choice.startswith("Pneumonia") else "L")
    st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)

    if model_choice.startswith("Pneumonia"):
        proc, model, labels = load_pneumonia_assets()
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        for i, p in enumerate(probs):
            st.write(f"**{labels[i]}** — {p*100:.2f}%")

        top_idx = probs.argmax()
        res = {"label": labels[top_idx], "confidence": float(probs[top_idx])}
        st.success(f"Top finding: **{res['label']}** ({res['confidence']*100:.1f}%)")
        st.info(generate_insight(res, modality="xray-fluoroscopy"))

    else:
        model, labels = load_densenet_assets()

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255),
        ])
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            preds = model(img_tensor)[0]
        probs = preds.sigmoid().cpu().numpy()

        findings = []
        st.subheader("Findings")
        for i, p in enumerate(probs):
            if p > 0.3:
                st.write(f"**{labels[i]}** — {p*100:.1f}%")
                findings.append((labels[i], float(p)))

        if findings:
            top_label, top_prob = max(findings, key=lambda x: x[1])
            res = {"label": top_label, "confidence": top_prob}
            st.success(f"Top finding: **{res['label']}** ({res['confidence']*100:.1f}%)")
            st.info(generate_insight(res, modality="xray-fluoroscopy"))
        else:
            st.success("No significant abnormality detected (all < 30%).")
else:
    st.info("Upload an X-ray frame to run the demo classification.")
