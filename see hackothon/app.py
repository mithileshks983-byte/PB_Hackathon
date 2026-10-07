"""
app.py
Production-ready Flask Backend for Phishing URL Detection
Exposes:
- POST /predict: Predicts phishing status, probability, confidence, reasons, features
- GET /health: Health check and model metadata
- GET /scans: In-memory history of recent scans for teammate's UI
Supports CORS from all origins (including file:// protocol).
"""

import datetime
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from features import FEATURE_NAMES, extract_features, generate_reasons

app = Flask(__name__)
# Enable CORS for all routes and origins (including file:// protocol)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global model bundle
MODEL_BUNDLE = None
RECENT_SCANS = []

def load_trained_model():
    """Loads the serialized model bundle saved by train.py."""
    global MODEL_BUNDLE
    model_path = "model.joblib"
    if not os.path.exists(model_path):
        print(f"[!] Warning: {model_path} not found. Running train.py...")
        import train
        train.main()
    MODEL_BUNDLE = joblib.load(model_path)
    print(f"[OK] Model '{MODEL_BUNDLE['model_name']}' loaded successfully.")

# Load model on startup
load_trained_model()

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Phishing URL Detection ML Service",
        "model_name": MODEL_BUNDLE.get("model_name", "Unknown"),
        "features_count": len(FEATURE_NAMES),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Expects JSON: {"url": "https://example.com"}
    Returns:
      - label: "Phishing" or "Legitimate"
      - is_phishing: boolean
      - confidence: float percentage (e.g. 98.5)
      - phishing_probability: float (0.0 to 1.0)
      - score: int (0 to 100 risk score)
      - reasons: top 3-5 human-readable explanation strings
      - features: dict of lexical features
    """
    data = request.get_json(silent=True) or {}
    raw_url = data.get("url", "").strip()

    if not raw_url:
        return jsonify({"error": "Missing or empty 'url' parameter"}), 400

    # 1. Feature Extraction
    features = extract_features(raw_url)
    feature_df = pd.DataFrame([features], columns=FEATURE_NAMES)

    # 2. Model Inference
    model = MODEL_BUNDLE["model"]
    scaler = MODEL_BUNDLE.get("scaler")

    if scaler is not None:
        X_input = scaler.transform(feature_df)
    else:
        X_input = feature_df

    # Predict probability of class 1 (phishing)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        # In scikit-learn, classes_ order corresponds to [0, 1]
        classes = list(model.classes_)
        phish_idx = classes.index(1) if 1 in classes else 1
        phishing_prob = float(probs[phish_idx])
    else:
        pred_raw = int(model.predict(X_input)[0])
        phishing_prob = 1.0 if pred_raw == 1 else 0.0

    # 3. Decision & Confidence
    is_phishing = phishing_prob >= 0.5
    label = "Phishing" if is_phishing else "Legitimate"

    # Confidence is probability for the chosen class
    chosen_prob = phishing_prob if is_phishing else (1.0 - phishing_prob)
    confidence_pct = round(chosen_prob * 100.0, 2)
    risk_score = int(round(phishing_prob * 100.0))

    # 4. Human-readable explainable reasons
    reasons = generate_reasons(raw_url, features, phishing_prob)

    response = {
        "url": raw_url,
        "label": label,
        "is_phishing": is_phishing,
        "confidence": confidence_pct,
        "phishing_probability": round(phishing_prob, 4),
        "score": risk_score,
        "reasons": reasons,
        "features": features
    }

    # Record scan in history for dashboard
    verdict_display = "Likely phishing" if is_phishing else "Looks safe"
    color = "#ff5d73" if is_phishing else "#2de2c0"
    RECENT_SCANS.insert(0, {
        "url": raw_url,
        "score": risk_score,
        "verdict": verdict_display,
        "color": color,
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })
    # Keep last 50 scans
    if len(RECENT_SCANS) > 50:
        RECENT_SCANS.pop()

    return jsonify(response), 200

@app.route("/scans", methods=["GET"])
def get_scans():
    """Returns recent scan history for teammate's frontend."""
    return jsonify(RECENT_SCANS), 200

if __name__ == "__main__":
    print("[*] Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
