FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY main.py .
COPY static/ ./static/

# Download model from Google Drive at build time
# Set MODEL_GDRIVE_URL as a build arg or hardcode the file ID below
ARG MODEL_GDRIVE_URL=""
RUN if [ -n "$MODEL_GDRIVE_URL" ]; then \
    python -c "\
    import re, gdown, sys; \
    url = '$MODEL_GDRIVE_URL'; \
    m = re.search(r'/d/([a-zA-Z0-9_-]+)', url); \
    fid = m.group(1) if m else url; \
    gdown.download(id=fid, output='best_cloud_densenet.keras', quiet=False)" ; \
    fi

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
