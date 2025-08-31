import numpy as np
from dataclasses import dataclass
import importlib

@dataclass
class RamanResult:
    label: str
    confidence: float

def _torch():
    return importlib.import_module("torch"), importlib.import_module("torch.nn")

class Tiny1DCNN:
    def __init__(self, n_classes=2):
        torch, nn = _torch()
        self.net = nn.Sequential(
            nn.Conv1d(1,16,9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, n_classes)
        )
        self.torch = torch
        self.net.eval()

    def __call__(self, x):
        with self.torch.no_grad():
            return self.net(x)

def load_model(weights_path: str | None = None, n_classes=2):
    torch, _ = _torch()
    m = Tiny1DCNN(n_classes)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        m.net.load_state_dict(state)
    return m

def predict(model, spectrum: np.ndarray) -> RamanResult:
    torch, _ = _torch()
    x = torch.from_numpy(spectrum[None,None,:].astype("float32"))
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    idx = int(probs.argmax())
    return RamanResult(label=f"class_{idx}", confidence=float(probs[idx]))
