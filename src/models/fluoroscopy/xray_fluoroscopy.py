import torch
import torchvision.transforms as T
import torchxrayvision as xrv
import numpy as np
from PIL import Image


_model = xrv.models.DenseNet(weights="densenet121-res224-all")
_model.eval()

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])


_LABELS = _model.pathologies 


def predict(arr: np.ndarray):
    """
    Run inference on a chest X-ray frame.
    Args:
        arr: numpy array (H,W) grayscale
    Returns:
        dict: {"label": str, "confidence": float}
    """
    if arr.ndim == 2:  
        arr = np.expand_dims(arr, axis=2)
    img_tensor = _transform(arr).unsqueeze(0)

    with torch.no_grad():
        out = _model(img_tensor)[0].numpy()

 
    idx = int(np.argmax(out))
    return {
        "label": _LABELS[idx],
        "confidence": float(out[idx])
    }
