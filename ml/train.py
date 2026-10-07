import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ml.features import extract_features  # noqa: E402

df = pd.read_csv(os.path.join(ROOT, "data", "phishing.csv"))
df = df.dropna().drop_duplicates(subset="URL")
df["y"] = df["Label"].map({"good": 0, "bad": 1})
df = df.dropna(subset=["y"])

# stratified sample of 50,000 rows for speed
df, _ = train_test_split(df, train_size=50000, stratify=df["y"], random_state=42)
print("Rows used:", len(df), "| phishing share:", round(df["y"].mean(), 3))

print("Extracting features...")
X = pd.DataFrame([extract_features(u) for u in df["URL"]])
y = df["y"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42),
}

os.makedirs(os.path.join(ROOT, "model"), exist_ok=True)
results = []
trained = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, pos_label=1, average="binary")
    results.append({"Model": name, "Accuracy": acc, "Precision": p, "Recall": r, "F1": f1})
    trained[name] = model

    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=["Safe", "Phishing"]))

    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Safe", "Phishing"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    out = os.path.join(ROOT, "model", f"confusion_{name.replace(' ', '_').lower()}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()

table = pd.DataFrame(results).round(4)
print("\n=== Model comparison (phishing class) ===")
print(table.to_string(index=False))

best = table.sort_values("F1", ascending=False).iloc[0]["Model"]
joblib.dump(trained[best], os.path.join(ROOT, "model", "model.pkl"))
print("\nBest model:", best, "-> saved to model/model.pkl")