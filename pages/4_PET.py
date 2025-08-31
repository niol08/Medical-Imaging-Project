import streamlit as st
import os, tempfile, numpy as np
import matplotlib.pyplot as plt
import pydicom
import tensorflow as tf
from keras.models import load_model
from src.core.ai_insights import generate_insight
import zipfile

st.title("PET Imaging")


MODEL_PATH = "src/models/pet/pet_classifier.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"PET model weights not found at {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_pet_model():
    return load_model(MODEL_PATH)

model = load_pet_model()


def load_dicom_series(folder_path):
    """Load PET DICOM slices and return volume + voxel spacing."""
    dcm_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".dcm")]
    if not dcm_files:
        raise ValueError("No .dcm files found in uploaded folder.")
    slices = [pydicom.dcmread(f) for f in dcm_files]

    slices.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))
    images = np.stack([s.pixel_array for s in slices]).astype(np.float32)

    images = (images - images.min()) / (images.max() - images.min() + 1e-6)


    try:
        px, py = slices[0].PixelSpacing  
        thickness = getattr(slices[0], "SliceThickness", 1.0)
        voxel_volume_ml = (float(px) * float(py) * float(thickness)) / 1000.0  
    except Exception:
        voxel_volume_ml = 1.0  

    return images, voxel_volume_ml

def preprocess_volume(volume, target_shape=(128,128)):
    """Resize slices for model input (2D CNN expects HxWx1)."""
    return np.array([
        tf.image.resize_with_pad(slice[..., np.newaxis], target_shape[0], target_shape[1]).numpy()
        for slice in volume
    ]) 

def compute_pet_metrics(volume, voxel_volume_ml, suv_threshold=0.5):
    """Compute SUVmax and lesion volume (very simplified)."""
    suvmax = float(volume.max())
    mask = volume > suv_threshold
    lesion_voxels = int(mask.sum())
    lesion_volume_ml = lesion_voxels * voxel_volume_ml
    return {"suvmax": suvmax, "lesion_ml": lesion_volume_ml}

def predict_pet(volume, model):
    """
    Run inference on PET volume using 2D CNN.
    Aggregates slice predictions into one patient-level result.
    """
    X = preprocess_volume(volume) 
    preds = []

    for i in range(X.shape[0]):
        slice_img = np.expand_dims(X[i], axis=0) 
        pred = model.predict(slice_img, verbose=0)[0]
        preds.append(pred)

    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)  

    if mean_pred.shape[0] == 1: 
        prob = float(mean_pred[0])
        label = "Cancer" if prob > 0.5 else "Healthy"
        confidence = prob if prob > 0.5 else 1 - prob
        return {"label": label, "confidence": confidence}
    else: 
        pred_class = int(np.argmax(mean_pred))
        confidence = float(np.max(mean_pred))
        return {"label": f"Class {pred_class}", "confidence": confidence}


uploaded = st.file_uploader("Upload PET DICOM folder (zip)", type=["zip"])

if uploaded is None:
    st.info("Upload a PET DICOM series (.zip with .dcm files).")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    zip_path = os.path.join(tmpdir, "upload.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir)


    subdirs = [os.path.join(tmpdir, d) for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
    if not subdirs:
        st.error("No DICOM folder found inside zip.")
        st.stop()

    dicom_folder = subdirs[0]
    volume, voxel_volume_ml = load_dicom_series(dicom_folder)

    st.write(f"Loaded PET volume with shape: {volume.shape}")


    mid_slice = volume[volume.shape[0] // 2]
    plt.imshow(mid_slice, cmap="hot")
    plt.title("Mid PET Slice")
    st.pyplot(plt)

    result = predict_pet(volume, model)

    metrics = compute_pet_metrics(volume, voxel_volume_ml)
    result.update(metrics)

    st.success(
        f"Prediction: **{result['label']}** (confidence {result['confidence']:.2f})\n\n"
        f"SUVmax: {result['suvmax']:.2f}, Lesion Volume: {result['lesion_ml']:.1f} mL"
    )

    try:
        insight_text = generate_insight(result, modality="pet")
        st.info(insight_text)
    except Exception as e:
        st.warning(f"Failed to generate AI insight: {e}")
