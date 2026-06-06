"""
Flask Web Application — Manufacturing Defect Detection
────────────────────────────────────────────────────────
Routes:
    GET  /              → Upload UI
    POST /predict       → JSON prediction + Grad-CAM
    GET  /health        → Health check (for deployment)
    GET  /metrics       → Model performance metrics (from results/)
"""

import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Add project root to path so utils import works
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.inference import predict_bytes

app = Flask(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "classification_report.json"
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not supported. Allowed: {ALLOWED_EXTENSIONS}"
        }), 400

    img_bytes = file.read()
    result = predict_bytes(img_bytes)
    print("RESULT =", result)
    return jsonify({
        "success":     True,
        "label":       result["label"],
        "confidence":  result["confidence"],
        "good_probability": result["good_probability"],
        "defect_probability": result["defect_probability"],
        "heatmap_b64": result.get("heatmap_b64", ""),
        "filename":    secure_filename(file.filename),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "defect_detection_mobilenetv2"})


@app.route("/metrics")
def metrics():
    if not os.path.exists(RESULTS_PATH):
        return jsonify({"error": "No evaluation results found. Train the model first."}), 404
    with open(RESULTS_PATH) as f:
        report = json.load(f)
    return jsonify(report)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"  Starting Defect Detection API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
