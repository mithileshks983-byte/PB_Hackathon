"""
train.py
Phishing URL Detection - ML Training Pipeline
- Data Processing: Normalization, null/duplicate dropping, class balance
- Feature Extraction: Lexical features from URL strings
- Model Evaluation & Comparison: Logistic Regression, Random Forest, Gradient Boosting
- Bias & Data Leakage Analysis (e.g. HTTPS shortcut detection)
- Model Persistence: joblib
- Visual Reporting: Feature importance & Confusion Matrix for hackathon presentation slides
"""

import os
import sys
import joblib
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, extract_features

# Create reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

def load_and_preprocess_data(filepath="dataset.csv"):
    """
    Step 1: Data Processing
    - Loads dataset.csv
    - Cleans null values and duplicate URLs
    - Normalizes labels to 1 (phishing) and 0 (legitimate)
    - Verifies class balance
    """
    print("\n" + "=" * 60)
    print(" [STEP 1] DATA PROCESSING & NORMALIZATION")
    print("=" * 60)

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    df = pd.read_csv(filepath)
    initial_count = len(df)
    print(f"[*] Initial dataset loaded: {initial_count:,} records")

    # 1. Drop nulls
    df = df.dropna(subset=["url", "label"]).copy()
    post_null_count = len(df)
    print(f"[*] Dropped {initial_count - post_null_count} rows with null URL or label.")

    # 2. Normalize labels: 1 = phishing, 0 = legitimate
    def normalize_label(val):
        val_str = str(val).strip().lower()
        if val_str in {"1", "phishing", "phish", "bad", "malicious", "true"}:
            return 1
        elif val_str in {"0", "legitimate", "legit", "good", "benign", "false"}:
            return 0
        else:
            return None

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # 3. Drop duplicate URLs
    before_dup = len(df)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"[*] Dropped {before_dup - len(df)} duplicate URL entries.")
    print(f"[*] Clean dataset size: {len(df):,} unique URLs")

    # 4. Class Balance check
    class_counts = df["label"].value_counts()
    phish_count = class_counts.get(1, 0)
    legit_count = class_counts.get(0, 0)
    print(f"[*] Class Distribution: Phishing (1)={phish_count:,} ({phish_count/len(df)*100:.1f}%), Legitimate (0)={legit_count:,} ({legit_count/len(df)*100:.1f}%)")

    # Handle extreme imbalance if ratio > 3:1
    imbalance_ratio = max(phish_count, legit_count) / max(1, min(phish_count, legit_count))
    if imbalance_ratio > 3.0:
        print(f"[!] Warning: Class imbalance detected (ratio {imbalance_ratio:.2f}:1). Downsampling majority class.")
        min_size = min(phish_count, legit_count)
        df_phish = df[df["label"] == 1].sample(min_size, random_state=42)
        df_legit = df[df["label"] == 0].sample(min_size, random_state=42)
        df = pd.concat([df_phish, df_legit]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[*] Balanced dataset size: {len(df):,} records")
    else:
        print("[*] Dataset is well-balanced.")

    return df

def extract_all_features(df):
    """
    Step 2: Lexical Feature Extraction
    Applies extract_features to every URL in the dataframe.
    """
    print("\n" + "=" * 60)
    print(" [STEP 2] LEXICAL FEATURE EXTRACTION (~32 Features)")
    print("=" * 60)
    print("[*] Extracting lexical features from URLs (no network requests)...")

    feature_rows = []
    for url in df["url"]:
        feature_rows.append(extract_features(url))

    X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    y = df["label"].values

    print(f"[*] Feature matrix created with shape: {X.shape}")
    return X, y

def check_bias_and_leakage(X, y):
    """
    Audits dataset for potential data leakage or dangerous shortcuts (e.g., HTTPS bias).
    """
    print("\n" + "=" * 60)
    print(" [BIAS & DATA LEAKAGE AUDIT]")
    print("=" * 60)

    # Compute correlation with target
    correlations = {}
    for col in X.columns:
        if X[col].std() > 0:
            corr = np.corrcoef(X[col], y)[0, 1]
            correlations[col] = corr
        else:
            correlations[col] = 0.0

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("[*] Top 5 features correlated with phishing label:")
    for feat, corr in sorted_corr[:5]:
        print(f"    - {feat:25}: r = {corr:+.4f}")

    # Specific check for HTTPS bias
    https_corr = correlations.get("has_https", 0.0)
    print(f"\n[*] HTTPS Correlation with Label: r = {https_corr:+.4f}")
    if abs(https_corr) > 0.85:
        print("[!] WARNING / DATASET BIAS ALERT:")
        print("    'has_https' has an abnormally high correlation with the target (> 0.85).")
        print("    Risk: The model might learn a false heuristic that HTTPS = 100% safe.")
        print("    In modern phishing attacks, 60%+ of threat actors use valid SSL certificates.")
    else:
        print("[OK] HTTPS is balanced: threat actors and legitimate sites both utilize SSL/TLS.")

    # High correlation leakage warning
    high_leakage = [f for f, c in sorted_corr if abs(c) > 0.95]
    if high_leakage:
        print(f"[!] DANGER: Potential data leakage in features: {high_leakage} (correlation > 0.95)")
    else:
        print("[OK] No single feature exhibits >0.95 correlation leakage. Multi-feature lexical analysis confirmed.")

def train_and_evaluate_models(X, y):
    """
    Step 3: Model Training & Evaluation
    - Stratified 80/20 train/test split (random_state=42)
    - Compares Logistic Regression, Random Forest, Gradient Boosting
    - Calculates Accuracy, Precision, Recall, F1, ROC-AUC
    - Plots Feature Importance and Confusion Matrix
    - Saves best model to model.joblib
    """
    print("\n" + "=" * 60)
    print(" [STEP 3] MODEL TRAINING & COMPARATIVE BENCHMARK")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[*] Stratified Split: Train set = {len(X_train):,}, Test set = {len(X_test):,}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    results = []
    trained_objects = {}

    print("\n{:<22} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
    ))
    print("-" * 72)

    for name, clf in models.items():
        if name == "Logistic Regression":
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "confusion_matrix": cm,
            "clf": clf
        })
        trained_objects[name] = clf

        print("{:<22} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            name, acc, prec, rec, f1, roc
        ))

    # Determine Best Model based on F1 Score (preferring Random Forest on ties for robust non-linear boundaries)
    preference_order = {"Random Forest": 3, "Gradient Boosting": 2, "Logistic Regression": 1}
    best_res = max(results, key=lambda x: (x["f1"], preference_order.get(x["name"], 0)))
    best_name = best_res["name"]
    best_model = best_res["clf"]
    print("-" * 72)
    print(f"[*] BEST MODEL SELECTED: {best_name} (F1 = {best_res['f1']:.4f}, ROC-AUC = {best_res['roc_auc']:.4f})")

    # Save best model bundle
    bundle = {
        "model_name": best_name,
        "model": best_model,
        "scaler": scaler if best_name == "Logistic Regression" else None,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "accuracy": best_res["accuracy"],
            "precision": best_res["precision"],
            "recall": best_res["recall"],
            "f1": best_res["f1"],
            "roc_auc": best_res["roc_auc"]
        }
    }
    model_save_path = "model.joblib"
    joblib.dump(bundle, model_save_path)
    print(f"[OK] Saved model artifact to {model_save_path}")

    # Generate Reports for Slides
    save_reports(results, best_res, X, FEATURE_NAMES)

    return results, best_res

def save_reports(results, best_res, X, feature_names):
    """
    Saves presentation-ready visual reports:
    1. reports/feature_importance.png
    2. reports/confusion_matrix.png
    3. reports/model_comparison.png
    """
    print("\n" + "=" * 60)
    print(" [STEP 4] GENERATING PRESENTATION SLIDE CHARTS (/reports)")
    print("=" * 60)

    # 1. Feature Importance Chart
    plt.figure(figsize=(10, 6))
    if hasattr(best_res["clf"], "feature_importances_"):
        importances = best_res["clf"].feature_importances_
    else:
        # Use Random Forest feature importances as reference
        rf = [r for r in results if r["name"] == "Random Forest"][0]["clf"]
        importances = rf.feature_importances_

    indices = np.argsort(importances)[::-1][:12] # Top 12 features
    top_feats = [feature_names[i] for i in indices]
    top_scores = [importances[i] for i in indices]

    plt.barh(range(len(indices)), top_scores[::-1], color="#2de2c0", edgecolor="#06101d")
    plt.yticks(range(len(indices)), top_feats[::-1], fontsize=10)
    plt.xlabel("Gini Feature Importance Score", fontsize=11)
    plt.title(f"Top 12 Predictive Features ({best_res['name']})", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    feat_img_path = os.path.join("reports", "feature_importance.png")
    plt.savefig(feat_img_path, dpi=300)
    plt.close()
    print(f"[OK] Saved feature importance chart to {feat_img_path}")

    # 2. Confusion Matrix Chart
    cm = best_res["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=14, fontweight="bold")
    plt.title(f"Confusion Matrix: {best_res['name']}", fontsize=12, pad=16, fontweight="bold")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate (0)", "Phishing (1)"], fontsize=10)
    ax.set_yticklabels(["Legitimate (0)", "Phishing (1)"], fontsize=10)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual True Label", fontsize=11)
    plt.tight_layout()
    cm_img_path = os.path.join("reports", "confusion_matrix.png")
    plt.savefig(cm_img_path, dpi=300)
    plt.close()
    print(f"[OK] Saved confusion matrix image to {cm_img_path}")

    # 3. Model Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    names = [r["name"] for r in results]
    f1s = [r["f1"] * 100 for r in results]
    accs = [r["accuracy"] * 100 for r in results]

    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width/2, accs, width, label="Accuracy (%)", color="#4a90e2")
    plt.bar(x + width/2, f1s, width, label="F1 Score (%)", color="#2de2c0")
    plt.ylabel("Score (%)", fontsize=11)
    plt.title("Model Performance Benchmark Comparison", fontsize=13, fontweight="bold")
    plt.xticks(x, names, fontsize=11)
    plt.ylim(80, 100)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    comp_img_path = os.path.join("reports", "model_comparison.png")
    plt.savefig(comp_img_path, dpi=300)
    plt.close()
    print(f"[OK] Saved model comparison chart to {comp_img_path}")

def main():
    df = load_and_preprocess_data("dataset.csv")
    X, y = extract_all_features(df)
    check_bias_and_leakage(X, y)
    train_and_evaluate_models(X, y)
    print("\n[OK] Pipeline complete! Ready for Flask backend deployment.")

if __name__ == "__main__":
    main()
