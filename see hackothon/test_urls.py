"""
test_urls.py
Comprehensive test suite running 10 safe and 10 phishing-looking URLs.
Evaluates model predictions, confidence %, phishing probability, and explainable reasons.
Supports both direct API HTTP requests (if live server running) and Flask test_client.
"""

import sys
import json
import urllib.request
import urllib.error

SAFE_URLS = [
    "https://www.google.com/search?q=cybersecurity+best+practices",
    "https://github.com/torvalds/linux/commits/master",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://scikit-learn.org/stable/modules/ensemble.html",
    "https://stackoverflow.com/questions/tagged/machine-learning",
    "https://apple.com/iphone-16-pro/",
    "https://nytimes.com/section/technology",
    "https://mit.edu/education/admissions",
    "https://amazon.com/dp/B08N5WRWNW"
]

PHISHING_URLS = [
    "http://192.168.1.104/secure-bank/login.php?update_credentials=true",
    "http://paypal.com.account-verification-notice.xyz/signin?session=89327",
    "http://appleid.apple.com-verify-security-alert.top/auth/pass-reset",
    "http://secure.chase.com@phish-banking-portal.support/verify/login",
    "http://bit.ly/secure-account-wallet-recovery-token",
    "http://xn--pypal-4ve.com/login.php?dispatch=credential_update",
    "http://netflix-billing-support-renewal-center.ga/customer-login.php",
    "https://wellsfargo.com.online-security-session.click/signin?ref=urgent",
    "http://google-accounts-password-reset-2026.work/login.html?token=92849",
    "http://amazon.com.customer-order-refund-status.support/update-card"
]

def call_api(url, use_http=False, test_client=None):
    payload = {"url": url}
    if use_http:
        req = urllib.request.Request(
            "http://127.0.0.1:5000/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    else:
        resp = test_client.post("/predict", json=payload)
        return resp.get_json()

def run_tests():
    print("=" * 80)
    print("   PHISHING URL DETECTION - END-TO-END VERIFICATION SUITE")
    print("=" * 80)

    # Check if live server is reachable
    use_http = False
    test_client = None
    try:
        req = urllib.request.Request("http://127.0.0.1:5000/health")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                use_http = True
                print("[*] Connected to live Flask service at http://127.0.0.1:5000")
    except Exception:
        print("[*] Live server not detected on port 5000. Running in-memory Flask test client...")
        from app import app
        test_client = app.test_client()

    print("\n" + "=" * 80)
    print(" [1] TESTING 10 LEGITIMATE / SAFE URLs")
    print("=" * 80)
    safe_passed = 0
    for i, u in enumerate(SAFE_URLS, 1):
        data = call_api(u, use_http=use_http, test_client=test_client)
        status = "PASS" if not data["is_phishing"] else "FAIL"
        if not data["is_phishing"]:
            safe_passed += 1

        print(f"\n[{i:02d}] URL: {u}")
        print(f"     Label:       {data['label']} ({data['confidence']}% confidence)")
        print(f"     Risk Score:  {data['score']}/100 | Phishing Prob: {data['phishing_probability']}")
        print(f"     Reasons:     {data['reasons'][0] if data['reasons'] else 'None'}")
        print(f"     Test Status: [{status}]")

    print("\n" + "=" * 80)
    print(" [2] TESTING 10 PHISHING / SUSPICIOUS URLs")
    print("=" * 80)
    phish_passed = 0
    for i, u in enumerate(PHISHING_URLS, 1):
        data = call_api(u, use_http=use_http, test_client=test_client)
        status = "PASS" if data["is_phishing"] else "FAIL"
        if data["is_phishing"]:
            phish_passed += 1

        print(f"\n[{i:02d}] URL: {u}")
        print(f"     Label:       {data['label']} ({data['confidence']}% confidence)")
        print(f"     Risk Score:  {data['score']}/100 | Phishing Prob: {data['phishing_probability']}")
        print(f"     Top Reasons:")
        for r in data["reasons"][:3]:
            print(f"       - {r}")
        print(f"     Test Status: [{status}]")

    print("\n" + "=" * 80)
    print("   TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f" Safe URLs Correctly Classified:     {safe_passed}/10 ({safe_passed*10}%)")
    print(f" Phishing URLs Correctly Flagged:    {phish_passed}/10 ({phish_passed*10}%)")
    total_acc = (safe_passed + phish_passed) / 20 * 100
    print(f" Overall Verification Accuracy:       {total_acc:.1f}%")
    print("=" * 80)

    if total_acc < 85.0:
        print("[!] Warning: Overall accuracy below threshold.")
        sys.exit(1)
    else:
        print("[OK] All test suites passed successfully!")

if __name__ == "__main__":
    run_tests()
