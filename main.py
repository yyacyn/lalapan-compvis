import os
import io
import numpy as np
from PIL import Image
import onnxruntime as rt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ── Config ────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "cloud_densenet.onnx")
IMG_SIZE   = (256, 256)

CLASS_NAMES = [
    "1_cumulus", "2_altocumulus", "3_cirrus",
    "4_clearsky", "5_stratocumulus", "6_cumulonimbus"
]
CLASS_DISPLAY = [
    "Cumulus", "Altocumulus", "Cirrus",
    "Clear Sky", "Stratocumulus", "Cumulonimbus"
]
CLASS_EMOJI = ["🌤️", "🌥️", "🌫️", "☀️", "☁️", "⛈️"]

print(f"Loading ONNX model from {MODEL_PATH}...")
sess       = rt.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
print("Model loaded!")

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Cloud Classifier API",
    description="DenseNet121 cloud classification — 6 classes",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your Vercel URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────
class Prediction(BaseModel):
    name: str
    emoji: str
    confidence: float

class PredictionResponse(BaseModel):
    top: Prediction
    all: List[Prediction]

# ── Helpers ───────────────────────────────────────────────────
# def redistribute_mixed(probs: np.ndarray) -> np.ndarray:
#     """Redistribute Mixed probability proportionally to other classes."""
#     mixed_prob = probs[MIXED_IDX]
#     result = probs.copy()
#     result[MIXED_IDX] = 0.0
#     total = result.sum()
#     if total > 0:
#         result += (result / total) * mixed_prob
#     return result

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr   = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # (1, 256, 256, 3)

# ── Routes ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "classes": len(CLASS_NAMES)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    arr   = preprocess(image)
    probs = sess.run(None, {input_name: arr})[0][0].astype("float32")
    # probs = redistribute_mixed(probs)

    sorted_idx = np.argsort(probs)[::-1]
    all_preds  = [
        Prediction(
            name=CLASS_DISPLAY[i],
            emoji=CLASS_EMOJI[i],
            confidence=round(float(probs[i]) * 100, 2)
        )
        for i in sorted_idx
        # if i != MIXED_IDX  # exclude mixed from response entirely
    ]

    return PredictionResponse(top=all_preds[0], all=all_preds)
