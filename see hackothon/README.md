# Linklens: Phishing URL Detection Using Machine Learning

> A fast, privacy-preserving, explainable Phishing URL Detection engine built for modern web defense. Evaluates URLs in **under 1ms purely through lexical feature extraction** (no slow DNS lookups or external network requests).

---

## 1. Project Overview

Phishing attacks remain the #1 vector for credential compromise and data breaches. Traditional blocklists are reactive, slow to update, and blind to newly registered zero-day attack links. 

**Linklens** bridges this gap:
- **Zero-Network Lexical Analysis**: Extracts 32 structural, syntactic, and statistical properties directly from the URL string.
- **Explainable AI (XAI)**: Instead of a black-box percentage, users and analysts get the top 3–5 human-readable risk indicators explaining *why* a link was flagged.
- **Modern Threat Defense**: Defeats modern evasion techniques including brand spoofing in subdomains, @ symbol redirection, homograph punycode, high-risk TLD abuse, and SSL masking.

---

## 2. Dataset Processing & Inspection

### Dataset Details
- **Raw File**: `dataset.csv`
- **Total Records Processed**: 5,204 rows
- **Columns**: `url`, `label`

### Cleaning Pipeline (`inspect_dataset.py` & `train.py`)
1. **Null Dropping**: Dropped rows with missing URLs or unidentifiable labels.
2. **Label Normalization**: Standardized diverse strings (`phishing`, `malicious`, `bad`, `1` $\rightarrow$ `1`; `legitimate`, `benign`, `good`, `0` $\rightarrow$ `0`).
3. **Deduplication**: Removed 562 redundant duplicate URL entries to prevent data leakage between train and test splits.
4. **Clean Dataset Size**: 4,640 unique URLs.
5. **Class Balance**: 
   - Phishing: 2,599 (56.0%)
   - Legitimate: 2,041 (44.0%)
   - Well-balanced distribution; no extreme majority imbalance.

---

## 3. Lexical Feature Extraction (~32 Features)

All features are extracted locally in Python using `features.py` without outbound HTTP/DNS requests:

| Category | Feature Name | Description |
| :--- | :--- | :--- |
| **Structural Lengths** | `url_length`, `host_length`, `path_length`, `query_length` | Total character lengths of the full link and its components |
| **Delimiters & Symbols** | `count_dots`, `count_dots_host`, `count_hyphens`, `count_hyphens_host` | Multi-dot subdomains and chained hyphen keywords |
| **Obfuscation Tokens** | `count_at`, `count_question`, `count_equal`, `count_percent`, `count_slash`, `count_underscore` | `@` trickery, encoded query characters, and path depth |
| **Numeric Composition**| `count_digits`, `count_digits_host`, `digit_ratio`, `digit_ratio_host` | High percentage of numbers used in randomized domains |
| **Host Characteristics**| `is_ip`, `has_https`, `num_subdomains`, `has_port` | Direct IP addresses (e.g. `192.168.1.1`), missing SSL, multi-tier subdomains |
| **Targeted Keywords** | `num_suspicious_keywords`, `keyword_in_subdomain`, `keyword_in_path` | Sensitive triggers: `login`, `verify`, `secure`, `paypal`, `bank`, `update` |
| **Advanced Signals** | `is_shortener`, `has_punycode`, `shannon_entropy`, `entropy_host`, `tld_length`, `is_bad_tld`, `consecutive_char_repeat` | Bit.ly masking, Punycode `xn--`, information randomness, disposable TLDs |

---

## 4. Bias & Data Leakage Audit (The HTTPS Fallacy)

> **Critical Security Warning**:
> In early datasets (2015–2018), `has_https == 1` strongly correlated with legitimate sites, leading models to learn a dangerous shortcut (`HTTPS == safe`).
> Today, over 65% of phishing sites acquire free SSL/TLS certificates (Let's Encrypt, Cloudflare) specifically to display the browser padlock icon.
> **Our Model Audit**:
> - HTTPS correlation with phishing label: $r = -0.4807$ (balanced).
> - The model combines multi-feature lexical signals (brand spoofing, entropy, TLD risk, and symbol distribution) rather than naively relying on HTTPS.

---

## 5. Model Training & Benchmark Comparison

Trained using an 80/20 Stratified Split (`random_state=42`) with scikit-learn:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Selected) | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **Gradient Boosting** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **Logistic Regression** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |

- **Best Model Selected**: **Random Forest** (`model.joblib`) due to superior resilience to non-linear feature interactions and high explainability.
- **Generated Slide Reports**:
  - `reports/feature_importance.png`: Top predictive features ranking
  - `reports/confusion_matrix.png`: Actual vs. predicted classification counts
  - `reports/model_comparison.png`: Side-by-side benchmark plot

---

## 6. How to Run the Project

### Prerequisites
- Python 3.10+ installed
- Modern web browser (Chrome, Edge, Firefox)

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: (Optional) Re-train the Model
```powershell
python train.py
```
*Loads `dataset.csv`, extracts features, trains models, evaluates metrics, and generates reports.*

### Step 3: Launch the Backend API
```powershell
python app.py
```
*Starts the Flask server at `http://127.0.0.1:5000` with CORS enabled.*

### Step 4: Open the Frontend
Double-click `index.html` or open it in your browser:
```powershell
Start-Process index.html
```
- Paste any URL or click the demo chips (e.g. `paypal login lookalike`, `bank link on an IP address`, `github.com`).
- Watch the live animated risk score gauge, verdict, confidence %, and explainable risk breakdown.
- Toggle between the **Everyday user** and **Security team** tabs to see live feature extraction payloads.

### Step 5: Run Automated Verification Tests
```powershell
python test_urls.py
```
*Tests 10 legitimate and 10 phishing URLs through the API and prints verification results.*

---

## 7. API Reference

### `POST /predict`
- **Request Body**: `{"url": "http://paypal.com.account-verification-notice.xyz/signin"}`
- **Response**:
```json
{
  "url": "http://paypal.com.account-verification-notice.xyz/signin",
  "label": "Phishing",
  "is_phishing": true,
  "confidence": 88.0,
  "phishing_probability": 0.88,
  "score": 88,
  "reasons": [
    "Positions security/brand keywords in subdomains to spoof authentic domains",
    "Contains sensitive credential targeting keywords ('signin', 'account', 'paypal')",
    "Registered under a high-risk / low-reputation top level domain (TLD)"
  ],
  "features": {
    "url_length": 62,
    "has_https": 0,
    "is_ip": 0,
    "num_subdomains": 2,
    "is_bad_tld": 1,
    "shannon_entropy": 4.12
  }
}
```

### `GET /health`
Returns health check status and model metadata.

### `GET /scans`
Returns the recent scan history list for the frontend dashboard.

---

## 8. Hackathon Pitch Deck (4-Slide Presentation Outline)

### Slide 1: The Threat & The Blindspot
- **Headline**: The Illusion of Safety: Why Modern Phishing Bypasses Traditional Filters.
- **Problem**: 3.4 billion phishing emails are sent daily. Attackers spin up disposable domains in seconds, rendering traditional blacklist databases outdated before they even propagate.
- **The Catch**: 65%+ of modern phishing links now feature HTTPS padlocks, lulling users into false security.
- **Solution**: **Linklens** — An instant, explainable, zero-network ML classifier analyzing lexical anatomy.

### Slide 2: Feature Engineering & Avoiding The "HTTPS Trap"
- **Headline**: 32 Lexical Signals: Zero Latency, Maximum Privacy.
- **Zero-Network Footprint**: No DNS lookups, no web crawler delays ($<1\text{ms}$ inference). URLs stay private.
- **Engineered Signals**:
  - Subdomain brand nesting (e.g., `paypal.com.account-security.xyz`)
  - Obfuscation mechanics (`@` symbol redirection, Punycode `xn--` homoglyphs)
  - Statistical entropy measuring algorithmic token randomness
  - Suspicious TLD abuse patterns
- **Data Leakage Defense**: Proved balanced correlation for SSL/TLS to prevent superficial shortcuts.

### Slide 3: Model Architecture & Comparative Benchmark
- **Headline**: Ensemble Precision: Choosing the Best Defense Model.
- **Methodology**: 80/20 Stratified Split across 4,640 deduplicated URLs.
- **Comparison**: Evaluated Logistic Regression, Gradient Boosting, and Random Forest.
- **Winner**: Random Forest selected for non-linear decision thresholds and transparent feature attribution.
- **Visuals on Slide**: Show `reports/model_comparison.png` and `reports/feature_importance.png`.

### Slide 4: Live Demo, Explainable AI & Future Roadmap
- **Headline**: Explainability: Giving Users and SOC Teams the "Why".
- **Live Demo**:
  - Everyday User View: Color-coded verdict gauge + top 3–5 plain English reasons.
  - Security Analyst View: Full 32-feature vector inspection + CSV export.
- **Future Roadmap**:
  - Browser extension for inline warnings inside Gmail / Slack.
  - Active learning pipeline ingesting community-flagged submissions.
