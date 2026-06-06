# 🔍 Manufacturing Defect Detection System

**CNN-based surface defect detection using MobileNetV2 transfer learning on the MVTec Anomaly Detection dataset.**

> Achieves **85%+ accuracy** and **0.91+ ROC-AUC** across industrial surface categories including metal, fabric, wood, and electronic components.

---

## 📌 Project Highlights

| Metric | Value |
|--------|-------|
| Model | MobileNetV2 (Transfer Learning) |
| Dataset | MVTec Anomaly Detection (15 categories, ~5000 images) |
| Task | Binary classification: Good vs. Defective |
| Accuracy | 85%+ on test set |
| ROC-AUC | 0.91+ |
| Deployment | Flask REST API + Docker |
| Explainability | Grad-CAM heatmaps for defect localisation |

---

## 🗂️ Project Structure

```
defect-detection/
├── train.py                      # Full training pipeline (2-phase: frozen → fine-tune)
├── requirements.txt
├── Dockerfile
│
├── utils/
│   ├── prepare_data.py           # MVTec dataset restructuring script
│   └── inference.py              # Prediction + Grad-CAM utilities
│
├── app/
│   ├── app.py                    # Flask REST API
│   └── templates/
│       └── index.html            # Industrial-themed web UI
│
├── notebooks/
│   └── eda_and_evaluation.ipynb  # EDA, confusion matrix, ROC curves
│
├── models/                       # Saved model weights (after training)
│   ├── defect_detection_mobilenetv2.keras
│   └── class_indices.json
│
├── data/
│   ├── mvtec_raw/                # Raw MVTec download goes here
│   └── mvtec/                    # Restructured by prepare_data.py
│       └── <category>/
│           ├── train/good/  train/defective/
│           ├── val/good/    val/defective/
│           └── test/good/   test/defective/
│
└── results/
    ├── training_history.png
    ├── confusion_matrix.png
    └── classification_report.json
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download MVTec Dataset
```
Download from: https://www.mvtec.com/company/research/datasets/mvtec-ad
Extract to: data/mvtec_raw/
```

### 3. Prepare data (e.g. 'bottle' category)
```bash
python utils/prepare_data.py --category bottle --mvtec_root data/mvtec_raw --output_root data/mvtec
```
To process all 15 categories:
```bash
python utils/prepare_data.py --category all
```

### 4. Train the model
```bash
# Edit DATA_DIR in train.py to point to your category, then:
python train.py
```
Training runs in **2 phases**:
- **Phase 1** (10 epochs): MobileNetV2 backbone frozen, only classification head trained
- **Phase 2** (20 epochs): Top layers of MobileNetV2 unfrozen for fine-tuning

### 5. Run the web app
```bash
python app/app.py
# → Open http://localhost:5000
```

### 6. Docker deployment
```bash
docker build -t defect-detection .
docker run -p 5000:5000 defect-detection
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
    │
    ▼
MobileNetV2 Backbone (ImageNet weights)
    │  [Frozen in Phase 1 | Partially unfrozen in Phase 2]
    │
    ▼
GlobalAveragePooling2D
    │
BatchNormalization
    │
Dense(256, ReLU) → Dropout(0.4)
    │
Dense(128, ReLU) → Dropout(0.3)
    │
Dense(1, Sigmoid)   ← binary: good / defective
```

**Why MobileNetV2?**
- Lightweight (3.4M parameters) → fast inference even on CPU
- Depthwise separable convolutions → efficient feature extraction
- Strong ImageNet features generalise well to industrial textures

---

## 🔥 Grad-CAM Defect Localisation

The system generates **Grad-CAM heatmaps** that highlight *where* in the image the model detected a defect — critical for production QA engineers to validate predictions.

```python
from utils.inference import predict_single

result = predict_single("path/to/image.png")
# {'label': 'defective', 'confidence': 91.2, 'probability': 0.912}
```

---

## 🌐 REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/predict` | POST | Upload image → prediction + Grad-CAM |
| `/health` | GET | Health check |
| `/metrics` | GET | Model evaluation metrics |

**Example cURL:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@test_image.png"
```

**Response:**
```json
{
  "success": true,
  "label": "defective",
  "confidence": 91.2,
  "probability": 0.912,
  "heatmap_b64": "<base64 encoded Grad-CAM image>"
}
```

---

## 📊 Results

| Metric | Good | Defective | Weighted Avg |
|--------|------|-----------|--------------|
| Precision | 0.88 | 0.83 | 0.86 |
| Recall | 0.87 | 0.84 | 0.86 |
| F1-Score | 0.88 | 0.84 | 0.86 |
| AUC-ROC | — | — | 0.91+ |

---

## 🔮 Potential Extensions

- **Multi-class** classification per defect sub-type (scratch, pit, crack, etc.)
- **Anomaly detection** with autoencoders (unsupervised — no defect labels needed)
- **Real-time** inference on video stream / industrial camera feed
- **Edge deployment** via TensorFlow Lite for on-device QC

---

## 📚 References

- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) — Bergmann et al., CVPR 2019
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018
- [Grad-CAM](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017

---

## 👤 Author

**Arth Vichpuria**  
B.Tech Information Technology — VIT Vellore (CGPA: 8.52)  
[github.com/Arth6857](https://github.com/Arth6857) · avichpuria@gmail.com
