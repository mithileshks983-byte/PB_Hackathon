"""
3MK Flask Backend – serves static/index.html and exposes:
  POST /predict        → ML-powered phishing verdict + signals
  GET  /api/scans      → scan history (SQLite)
  GET  /health         → liveness check
"""

import os
import re
import sys
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlparse

import joblib
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Path resolution ─────────────────────────────────────────────────────────
# __file__ lives at  backend1/backend/app.py
# ROOT       →       backend1/
# PROJECT_ROOT →     PB_Hackathon/   (one level up from backend1)
# STATIC_DIR →       PB_Hackathon/static/
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(ROOT)
STATIC_DIR   = os.path.join(PROJECT_ROOT, "static")
DB_PATH      = os.path.join(PROJECT_ROOT, "3mk_scans.db")

sys.path.insert(0, ROOT)
from ml.features import extract_features  # noqa: E402

app = Flask(__name__)
CORS(app)

# ── Database ─────────────────────────────────────────────────────────────────
def get_db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

with get_db() as _boot:
    _boot.execute("""
        CREATE TABLE IF NOT EXISTS scans(
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            url        TEXT,
            score      INTEGER,
            verdict    TEXT,
            color      TEXT,
            confidence REAL,
            at         TEXT
        )
    """)

# ── ML Model ──────────────────────────────────────────────────────────────────
_model_path = os.path.join(ROOT, "model", "model.pkl")
model       = joblib.load(_model_path)
PHISH_IDX   = list(model.classes_).index(1)

# ── Signal / anatomy helpers ─────────────────────────────────────────────────
BRANDS = ["paypal","google","microsoft","apple","amazon","netflix","facebook",
          "instagram","sbi","hdfc","icici","paytm","whatsapp","dropbox","twitter",
          "linkedin","chase","wellsfargo","citibank","dhl","fedex","usps"]
KEYS   = ["login","signin","verify","secure","account","update","confirm","wallet",
          "password","bank","billing","suspend","alert","urgent","validate","reset"]
BADTLD = {"xyz","top","tk","ml","ga","cf","gq","click","zip","support","work",
          "loan","vip","site","online","fun","icu","live","buzz"}
SHORT  = {"bit.ly","tinyurl.com","t.co","goo.gl","is.gd","cutt.ly","rb.gy",
          "ow.ly","short.io","buff.ly","su.pr"}


def _reg_domain(host: str) -> str:
    """Return the registered (eTLD+1) part of a hostname."""
    p = host.split(".")
    if (len(p) > 2
            and len(p[-1]) == 2
            and re.match(r"^(co|com|org|gov|ac|net)$", p[-2])):
        return ".".join(p[-3:])
    return ".".join(p[-2:])


def build_signals_and_anatomy(url: str):
    """
    Return (signals_list, anatomy_dict) using rule-based heuristics so the
    frontend can highlight exactly *why* a link is suspicious – independently
    of the ML score.
    """
    raw = url.strip()
    if not re.match(r"^[a-z]+://", raw, re.I):
        raw = "http://" + raw

    try:
        u = urlparse(raw)
    except Exception:
        return [], {"protocol": "http:", "sub": "", "reg": url, "path": "", "is_secure": False}

    host     = (u.hostname or "").lower()
    reg      = _reg_domain(host)
    sub      = host[: len(host) - len(reg) - 1] if len(host) > len(reg) else ""
    tld      = host.split(".")[-1]
    full     = raw.lower()
    is_ip    = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))
    key_hits = [k for k in KEYS if k in full]
    brand    = next(
        (b for b in BRANDS
         if b in (sub + u.path.lower()) and not reg.startswith(b + ".")),
        None
    )
    rl = reg.split(".")[0]
    sub_parts = [s for s in sub.split(".") if s]

    checks = [
        ("No HTTPS Encryption",
         "Uses plain HTTP — data sent to this site can be read in transit.",
         u.scheme != "https", 15),

        ("Raw IP Address",
         f"Real services use domain names, not IP numbers like {host}.",
         is_ip, 30),

        ("@ Symbol in URL",
         "Everything before @ is ignored by the browser — it hides the real destination.",
         "@" in raw, 20),

        ("Brand Impersonation",
         f"Mentions \"{brand}\" but the real domain is {reg}." if brand
             else "No known brand is borrowed in the wrong place.",
         bool(brand), 25),

        ("Phishing Keywords",
         ("Found: " + ", ".join(key_hits) + ".") if key_hits
             else "No login / verify pressure words detected.",
         len(key_hits) > 0, min(24, len(key_hits) * 8)),

        ("Risky Domain Extension",
         f'".{tld}" is commonly used on throwaway or spam sites.',
         tld in BADTLD, 15),

        ("Excessive Subdomains",
         f'"{sub}" has {len(sub_parts)} subdomain layer(s) — layering confuses users.',
         len(sub_parts) > 2, 12),

        ("Hyphen Stacking",
         "Fraudulent domains chain words with hyphens to mimic legitimate brands.",
         (rl.count("-") >= 2 or host.count("-") >= 3), 10),

        ("Disguised Characters",
         "Uses punycode (xn--) encoding or digits swapped for letters.",
         host.startswith("xn--") or bool(re.search(r"[a-z]\d[a-z]", rl)), 20),

        ("Unusually Long URL",
         f"Length {len(full)} chars — very long URLs are used to bury the real domain.",
         len(full) > 75, 10),

        ("Link Shortener",
         "URL shorteners mask where you actually land.",
         host in SHORT, 12),
    ]

    signals = [
        {"name": name, "bad": bad, "detail": detail, "weight": w if bad else 0}
        for name, detail, bad, w in checks
    ]

    anatomy = {
        "protocol":  u.scheme + ":",
        "sub":       sub,
        "reg":       reg,
        "path":      u.path + (("?" + u.query) if u.query else ""),
        "is_secure": u.scheme == "https",
    }

    return signals, anatomy


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    """Serve the 3MK frontend from the project-level static/ directory."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "engine": "random_forest_ml", "brand": "3MK"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url  = str(data.get("url", "")).strip()

    if not url:
        return jsonify({"error": "Please enter a URL."}), 400
    if len(url) > 2000:
        return jsonify({"error": "URL is too long (max 2 000 chars)."}), 400

    # ── ML inference ──────────────────────────────────────────────────────
    feats    = extract_features(url)
    p_phish  = float(model.predict_proba([feats])[0][PHISH_IDX])
    score    = round(p_phish * 100)
    is_phish = p_phish >= 0.5
    verdict  = "phishing" if is_phish else "safe"
    color    = "#ff2d55" if is_phish else "#00e676"
    confidence = round((p_phish if is_phish else 1.0 - p_phish) * 100, 1)

    # ── Rule-based signals & URL anatomy ──────────────────────────────────
    signals, anatomy = build_signals_and_anatomy(url)

    # ── Persist to history DB ─────────────────────────────────────────────
    with get_db() as conn:
        conn.execute(
            "INSERT INTO scans(url,score,verdict,color,confidence,at)"
            " VALUES(?,?,?,?,?,?)",
            (url, score, verdict, color, confidence,
             datetime.now(timezone.utc).isoformat()),
        )

    return jsonify({
        "url":        url,
        "verdict":    verdict,
        "score":      score,
        "confidence": confidence,
        "color":      color,
        "signals":    signals,
        "anatomy":    anatomy,
    })


@app.route("/api/scans")
def api_scans():
    """Return recent scan history (newest first)."""
    limit = request.args.get("limit", 50, type=int)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT url,score,verdict,color,confidence,at"
            " FROM scans ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  3MK Phishing Detector")
    print(f"  ─────────────────────────────────────")
    print(f"  Static dir : {STATIC_DIR}")
    print(f"  ML model   : {_model_path}")
    print(f"  History DB : {DB_PATH}")
    print(f"  Running at : http://localhost:5001\n")
    app.run(debug=True, port=5001, host="0.0.0.0")