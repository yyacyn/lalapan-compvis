import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import gdown

# ── Config ────────────────────────────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH", "best_cloud_densenet.keras")
MODEL_GDRIVE  = os.getenv("MODEL_GDRIVE_URL", "")   # set this env var in Leapcell
IMG_SIZE      = (256, 256)

CLASS_ABBR = ["Ac","As","Cb","Cc","Ci","Cs","Ct","Cu","Ns","Sc","St"]
CLASS_DISPLAY = [
    "Altocumulus","Altostratus","Cumulonimbus","Cirrocumulus",
    "Cirrus","Cirrostratus","Contrail","Cumulus",
    "Nimbostratus","Stratocumulus","Stratus"
]
CLASS_EMOJI = ["🌥️","🌫️","⛈️","🌤️","🌫️","☁️","✈️","⛅","🌧️","☁️","🌁"]

# ── Lazy model loader ─────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        # Download model if not present
        if not os.path.exists(MODEL_PATH):
            if not MODEL_GDRIVE:
                raise RuntimeError(
                    "Model file not found and MODEL_GDRIVE_URL env var is not set."
                )
            print(f"Downloading model from Google Drive...")
            gdown.download(MODEL_GDRIVE, MODEL_PATH, quiet=False, fuzzy=True)
            print("Download complete.")
        print(f"Loading model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded!")
    return _model

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Cloud Classifier API",
    description="DenseNet121 cloud classification — 11 classes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Schemas ───────────────────────────────────────────────────
class Prediction(BaseModel):
    abbr: str
    name: str
    emoji: str
    confidence: float

class PredictionResponse(BaseModel):
    top: Prediction
    all: List[Prediction]

# ── Preprocessing ─────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr   = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # (1, 256, 256, 3)

# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    html_path = "static/index.html"
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return HTMLResponse("<h1>Cloud Classifier API</h1><p>POST /predict with an image file.</p>")

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "classes": len(CLASS_ABBR)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read & preprocess
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = get_model()
    arr   = preprocess(image)
    probs = model(arr, training=False)[0].numpy().astype("float32")

    # Build response
    all_preds = [
        Prediction(
            abbr=CLASS_ABBR[i],
            name=CLASS_DISPLAY[i],
            emoji=CLASS_EMOJI[i],
            confidence=round(float(probs[i]) * 100, 2)
        )
        for i in np.argsort(probs)[::-1]
    ]

    return PredictionResponse(top=all_preds[0], all=all_preds)
