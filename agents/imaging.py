from __future__ import annotations
from typing import TypedDict, Dict, Any,Optional
from langgraph.graph import StateGraph, START, END
import os
import math





try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    TORCH_OK = True
except Exception:
    TORCH_OK = False

classes=['normal','pneumonia','covid_suspect']
class Imagingagent:
    """
    Two modes:
      - mode="dummy" (default): deterministic fake probs from filename hash (fast, no deps)
      - mode="pretrained": load a local PyTorch model checkpoint and run real inference
        Expect 1x3x224x224 input; output logits for [normal, pneumonia, covid_suspect].
        Put your weights at: models/covid_cxr_resnet18.pt (or pass a custom path).
    """

    def __init__(self,mode: str = "dummy",ckpt_path: Optional[str] = "models/covid_cxr_resnet18.pt"):
        self.mode = mode
        self.ckpt_path = ckpt_path
        self.model = None
        self.transform = None
        if self.mode == "pretrained":
            if not TORCH_OK:
                raise RuntimeError("PyTorch/torchvision not available. Install torch/torchvision or use mode='dummy'.")
            if not (self.ckpt_path and os.path.exists(self.ckpt_path)):
                
                self.mode = "dummy"
            else:
                self._load_model()
    
    def predict_dummy(self,xray_path:str)->Dict[str,Any]:
        h = sum(bytearray(xray_path.encode())) % 100
        base = [0.30 + (h%10)/100.0, 0.30 + (h//10)/100.0, 0.40 - ((h%10)/100.0)]
        vals = [max(0.05, v) for v in base]
        s = sum(vals)
        probs = [v/s for v in vals]
        condition_probs = dict(zip(classes, probs))
        top = max(condition_probs, key=condition_probs.get)
        conf = condition_probs[top]
        severity = "mild" if condition_probs[top] < 0.5 else ("moderate" if condition_probs[top] < 0.7 else "severe")
        return {"condition_probs": condition_probs,"top_label":top,"decision_confidence":conf, "severity_hint": severity, "mode": "dummy"}
    
    def _load_model(self):
        
        import torchvision.models as models
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(classes))
        state = torch.load(self.ckpt_path, map_location="cpu")
        m.load_state_dict(state)
        m.eval()
        self.model = m
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            
        ])

    def _predict_pretrained(self, xray_path: str) -> Dict[str, Any]:
        if self.model is None:
            return self._predict_dummy(xray_path)
        img = Image.open(xray_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        condition_probs = dict(zip(classes, probs))
        top = max(condition_probs, key=condition_probs.get)
        severity = "mild" if condition_probs[top] < 0.5 else "moderate"
        return {"condition_probs": condition_probs, "severity_hint": severity, "mode": "pretrained"}

    def predict(self, xray_path: str) -> Dict[str, Any]:
        if self.mode == "pretrained":
            return self._predict_pretrained(xray_path)
        return self.predict_dummy(xray_path)
    



        



        
