# Cloud Classifier API

FastAPI backend + HTML frontend untuk klasifikasi 11 jenis awan menggunakan DenseNet121.

## Struktur Folder

```
cloud_api/
├── main.py                    # FastAPI app
├── requirements.txt
├── Dockerfile
├── best_cloud_densenet.keras  # ← taruh model kamu di sini
└── static/
    └── index.html             # Frontend
```

## Setup Lokal

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Taruh model file di folder ini
cp /path/to/best_cloud_densenet.keras .

# 3. Jalankan server
uvicorn main:app --reload --port 8000
```

Buka http://localhost:8000

## API Endpoints

| Method | Path       | Description              |
|--------|------------|--------------------------|
| GET    | /          | Frontend HTML            |
| GET    | /health    | Status API + model info  |
| POST   | /predict   | Upload gambar, dapat prediksi |

### Contoh request /predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@foto_awan.jpg"
```

### Contoh response

```json
{
  "top": {
    "abbr": "Ci",
    "name": "Cirrus",
    "emoji": "🌫️",
    "confidence": 85.1
  },
  "all": [
    { "abbr": "Ci", "name": "Cirrus", "emoji": "🌫️", "confidence": 85.1 },
    { "abbr": "Cs", "name": "Cirrostratus", "emoji": "☁️", "confidence": 9.3 },
    ...
  ]
}
```

## Deploy ke Render (Gratis)

1. Push folder ini ke GitHub repo
2. Buka https://render.com → New → Web Service
3. Connect repo
4. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment variable:** `MODEL_PATH = best_cloud_densenet.keras`
5. Upload model file ke repo (atau pakai Render Disk)

## Deploy ke Railway (Gratis)

1. Push ke GitHub
2. Buka https://railway.app → New Project → Deploy from GitHub
3. Railway auto-detect Dockerfile
4. Selesai — dapat URL otomatis

## Deploy dengan Docker

```bash
# Build
docker build -t cloud-classifier .

# Run
docker run -p 8000:8000 cloud-classifier
```

## Catatan

- Model `best_cloud_densenet.keras` harus ada di folder yang sama dengan `main.py`
- Preprocessing: gambar di-resize ke 256×256, range 0-255 (densenet.preprocess_input ada di dalam model)
- `tensorflow-cpu` dipakai di requirements untuk ukuran image Docker yang lebih kecil
