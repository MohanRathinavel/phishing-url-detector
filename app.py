"""
app.py — Flask Backend for Phishing URL Detector
=================================================
Endpoints:
  GET  /          → serve the React SPA shell
  POST /predict   → JSON { "url": "..." } → prediction + confidence
  GET  /plots     → list available plot filenames
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from utils import extract_features, features_to_list

# ── app setup ────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# ── load trained model ───────────────────────────────────────────
MODEL_PATH = os.path.join("model", "model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Please run:  python model/train_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        artefact = pickle.load(f)
    return artefact["model"], artefact["features"], artefact["model_name"]

try:
    MODEL, FEATURE_NAMES, MODEL_NAME = load_model()
    print(f"✓ Loaded model: {MODEL_NAME}")
except FileNotFoundError as e:
    print(f"⚠  {e}")
    MODEL, FEATURE_NAMES, MODEL_NAME = None, None, "Not trained yet"


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html", model_name=MODEL_NAME)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body: { "url": "<string>" }
    Returns:
      {
        "url":        "<string>",
        "prediction": "Phishing" | "Safe",
        "confidence": <float 0-100>,
        "label":      1 | 0,
        "features":   { ... }
      }
    """
    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "Please provide a URL."}), 400

    # add scheme if missing so feature extractor works correctly
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    # ── extract & predict ──
    feat_dict  = extract_features(url)
    feat_list  = features_to_list(feat_dict, FEATURE_NAMES)
    feat_df    = pd.DataFrame([feat_list], columns=FEATURE_NAMES)

    label      = int(MODEL.predict(feat_df)[0])
    proba      = MODEL.predict_proba(feat_df)[0]  # [P(safe), P(phishing)]
    confidence = round(float(proba[label]) * 100, 2)

    return jsonify({
        "url":        url,
        "prediction": "Phishing" if label == 1 else "Safe",
        "confidence": confidence,
        "label":      label,
        "features":   feat_dict,
        "model_used": MODEL_NAME,
    })


@app.route("/plots")
def list_plots():
    """Return list of available visualisation filenames."""
    plot_dir = os.path.join("static", "plots")
    if not os.path.isdir(plot_dir):
        return jsonify({"plots": []})
    files = [f for f in os.listdir(plot_dir) if f.endswith(".png")]
    return jsonify({"plots": sorted(files)})


@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "model":      MODEL_NAME,
        "model_ready": MODEL is not None,
    })


# ── run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
