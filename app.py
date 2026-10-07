"""Linklens backend: FastAPI + SQLite. Run: uvicorn app:app --reload"""
import re, json, sqlite3, os
from datetime import datetime, timezone
from urllib.parse import urlparse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DB = "linklens.db"
app = FastAPI(title="Linklens API")

# ---- database -------------------------------------------------------------
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

with db() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT, score INTEGER, verdict TEXT, color TEXT,
        features TEXT, at TEXT)""")

# ---- model (optional) -----------------------------------------------------
# Put your trained scikit-learn model next to this file as model.pkl
# (joblib.dump(rf, "model.pkl")). Until then a simple rule fallback is used.
MODEL = None
if os.path.exists("model.pkl"):
    import joblib; MODEL = joblib.load("model.pkl")

KEYS = ["login","signin","verify","secure","account","update","confirm","wallet","password","bank"]

def extract_features(url: str) -> dict:
    # IMPORTANT: use the SAME features and ORDER as your training notebook.
    u = urlparse(url if "://" in url else "http://" + url)
    host = u.hostname or ""
    return {
        "url_length": len(url),
        "has_https": int(u.scheme == "https"),
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        "has_at": int("@" in url),
        "n_subdomains": max(0, host.count(".") - 1),
        "n_hyphens": host.count("-"),
        "n_digits_host": sum(ch.isdigit() for ch in host),
        "n_keywords": sum(k in url.lower() for k in KEYS),
    }

def predict_score(f: dict) -> int:
    if MODEL is not None:
        return round(MODEL.predict_proba([list(f.values())])[0][1] * 100)
    s = 15*(1-f["has_https"]) + 30*f["has_ip"] + 20*f["has_at"] + 8*min(3, f["n_keywords"]) \
        + 12*(f["n_subdomains"] > 2) + 10*(f["n_hyphens"] >= 2) + 10*(f["url_length"] > 75)
    return min(100, s)

def verdict(s):
    return ("Looks safe", "#2de2c0") if s < 25 else ("Suspicious", "#ffb347") if s < 55 else ("Likely phishing", "#ff5d73")

# ---- routes ---------------------------------------------------------------
class Req(BaseModel):
    url: str

@app.post("/api/predict")
def predict(r: Req):
    f = extract_features(r.url); score = predict_score(f); v, col = verdict(score)
    with db() as c:
        c.execute("INSERT INTO scans(url,score,verdict,color,features,at) VALUES(?,?,?,?,?,?)",
                  (r.url, score, v, col, json.dumps(f), datetime.now(timezone.utc).isoformat()))
    return {"score": score, "verdict": v, "features": f}

@app.get("/api/scans")
def scans(limit: int = 50):
    with db() as c:
        rows = c.execute("SELECT url,score,verdict,color,at FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(x) for x in rows]

# serve the front end from the same server (no CORS headaches)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
