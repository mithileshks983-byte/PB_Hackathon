# 3MK Phishing Detector

3MK analyzes URL structure with a trained machine-learning model and returns a phishing risk score, verdict, and human-readable signals. The frontend also keeps a local fallback if the API is unavailable.

## Problem statement

Phishing links often imitate trusted services or hide their real destination. This project uses URL-derived features to help identify suspicious links before a user opens them. The analysis is lexical and does not visit the submitted URL.

## Technology

- **Model:** scikit-learn Random Forest, serialized with joblib
- **Backend:** Python, Flask, Flask-CORS, SQLite
- **Frontend:** HTML, CSS, and vanilla JavaScript
- **Training data:** `backend1/data/phishing.csv`; the training script uses a stratified sample of 50,000 URLs

The model uses 18 URL features: URL length, host length, path length, dot count, hyphen count, host hyphen count, digit count, digit ratio, special-character count, `@` presence, IP-address presence, HTTPS presence, subdomain count, URL depth, suspicious-keyword count, keyword-in-host presence, risky-TLD presence, and URL entropy.

## Run locally

Python 3.10 or later is recommended. From the repository root, install the backend dependencies:

```sh
python -m pip install flask flask-cors joblib scikit-learn
```

Start the Flask application from its backend directory:

```sh
cd backend1/backend
python app.py
```

Open [http://localhost:5001](http://localhost:5001). The app serves `static/index.html`, loads the existing model from `backend1/model/model.pkl`, and creates `3mk_scans.db` at the repository root for prediction history.

On macOS, use `python3` instead of `python` if that is the Python 3 command on your system.

## API

- `GET /health` — service and model status
- `POST /predict` — classify a URL; send JSON such as `{"url":"https://github.com"}`
- `GET /api/scans` — return recent prediction history (up to 50 by default)

The prediction response contains the verdict, score, confidence, color, risk signals, and URL anatomy. The local SQLite history database is ignored by Git.

## Train the model (optional)

From the repository root, install the training dependencies and run:

```sh
python -m pip install pandas matplotlib
python backend1/ml/train.py
```

Training reads `backend1/data/phishing.csv` and replaces `backend1/model/model.pkl` with the best model selected by F1 score.