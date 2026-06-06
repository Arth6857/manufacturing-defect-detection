"""
Inference Utilities
────────────────────
Single image and batch prediction with Grad-CAM heatmap visualisation.
Used by both the Flask API and standalone scripts.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for Flask)
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path


IMG_SIZE    = (224, 224)
MODEL_PATH  = os.environ.get("MODEL_PATH", "models/defect_detection_mobilenetv2.keras")
LABELS_PATH = os.environ.get("LABELS_PATH", "models/class_indices.json")
THRESHOLD   = 0.50           # Probability above this → defective


# ─── MODEL LOADING (singleton) ────────────────────────────────────────────────

_model       = None
_class_names = None   # {0: 'good', 1: 'defective'}  (inverted class_indices)


def get_model():
    global _model, _class_names
    if _model is None:
        print(f"[Inference] Loading model from {MODEL_PATH}...")
        _model = load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            class_indices = json.load(f)
        _class_names = {v: k for k, v in class_indices.items()}
        print(f"[Inference] Classes: {_class_names}")
    return _model, _class_names


# ─── PRE/POST PROCESSING ──────────────────────────────────────────────────────

def preprocess(img_path: str) -> np.ndarray:
    """Load, resize, normalise → (1, 224, 224, 3)"""
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_bytes(img_bytes: bytes) -> np.ndarray:
    """Preprocess raw bytes (from Flask upload)."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0), img   # (batch, H, W, 3) + raw for heatmap


def predict_single(img_path: str) -> dict:
    """Predict a single image file. Returns dict with label + confidence."""
    model, class_names = get_model()
    batch = preprocess(img_path)
    prob  = float(model.predict(batch, verbose=0)[0][0])
    label = "good" if prob >= THRESHOLD else "defective"
    confidence = prob if label == "good" else 1 - prob
    return {
        "label":       label,
        "confidence":  round(confidence * 100, 2),
        "probability": round(prob, 4),
    }


def predict_bytes(img_bytes: bytes) -> dict:
    """Predict from raw image bytes."""
    model, class_names = get_model()
    batch, raw_img = preprocess_bytes(img_bytes)
    prob  = float(model.predict(batch, verbose=0)[0][0])
    label = "good" if prob >= THRESHOLD else "defective"
    confidence = prob if label == "good" else 1 - prob
    heatmap_b64 = generate_gradcam_b64(model, batch, raw_img)
    return {
        "label":       label,
        "confidence":  round(confidence * 100, 2),
        "good_probability": round(prob * 100, 2),
        "defect_probability": round((1 - prob) * 100, 2),
        "heatmap_b64": heatmap_b64,
    }


# ─── GRAD-CAM ─────────────────────────────────────────────────────────────────

def generate_gradcam_b64(model, batch: np.ndarray, raw_img: np.ndarray) -> str:
    """
    Generate a Grad-CAM heatmap overlaid on the original image.
    Returns a base64-encoded PNG string (for embedding in HTML/JSON).
    """
    try:
        # Find last conv layer in MobileNetV2
        last_conv_layer = next(
            l for l in reversed(model.layers)
            if isinstance(l, tf.keras.layers.Conv2D)
        )
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(batch, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        original = np.uint8(raw_img * 255)
        overlay  = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

        # Encode to base64
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(
            axes,
            [original, heatmap_colored, overlay],
            ["Original", "Grad-CAM Heatmap", "Overlay"]
        ):
            ax.imshow(img)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        plt.suptitle("Defect Localisation — Grad-CAM", fontsize=13)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        print(f"[Grad-CAM] Warning: {e}")
        return ""


# ─── BATCH EVALUATION ─────────────────────────────────────────────────────────

def batch_predict(image_dir: str) -> list[dict]:
    """
    Predict all images in a directory.
    Returns list of dicts: [{filename, label, confidence, probability}]
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    results   = []
    for p in sorted(Path(image_dir).iterdir()):
        if p.suffix.lower() in supported:
            result = predict_single(str(p))
            result["filename"] = p.name
            results.append(result)
            print(f"  {p.name:40s} → {result['label']:10s}  ({result['confidence']:.1f}%)")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils/inference.py <image_path_or_directory>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        print(f"Batch prediction on directory: {target}")
        results = batch_predict(target)
        print(f"\nTotal: {len(results)} images processed.")
    else:
        result = predict_single(target)
        print(f"\nResult: {result}")
