FROM python:3.11-slim

LABEL maintainer="Arth Vichpuria <avichpuria@gmail.com>"
LABEL description="Manufacturing Defect Detection — MobileNetV2 on MVTec-AD"

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose Flask port
EXPOSE 5000

# Environment
ENV FLASK_APP=app/app.py
ENV FLASK_DEBUG=false
ENV MODEL_PATH=models/defect_detection_mobilenetv2.keras
ENV LABELS_PATH=models/class_indices.json

# Run with gunicorn in production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", \
     "--timeout", "120", "app.app:app"]
