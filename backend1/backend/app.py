import os
import sys

import joblib
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ml.features import extract_features  # noqa: E402

app = Flask(__name__)
CORS(app)

model = joblib.load(os.path.join(ROOT, "model", "model.pkl"))
PHISH_INDEX = list(model.classes_).index(1)


@app.route("/")
def home():
    return send_from_directory(os.path.join(ROOT, "frontend"), "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"error": "Please enter a URL."}), 400
    if len(url) > 2000:
        return jsonify({"error": "URL is too long."}), 400

    feats = extract_features(url)
    p_phish = float(model.predict_proba([feats])[0][PHISH_INDEX])
    is_phish = p_phish >= 0.5
    return jsonify({
        "url": url,
        "verdict": "phishing" if is_phish else "safe",
        "confidence": round((p_phish if is_phish else 1 - p_phish) * 100, 1),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)