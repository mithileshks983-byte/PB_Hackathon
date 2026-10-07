"""
Dataset Generator for Phishing URL Detection
Generates a realistic, balanced dataset of ~5,000 URLs with diverse
legitimate and phishing patterns to train robust ML models.
"""

import random
import csv

random.seed(42)

# --- Legitimate URL Components ---
LEGIT_DOMAINS = [
    "google.com", "microsoft.com", "github.com", "wikipedia.org", "amazon.com",
    "linkedin.com", "reddit.com", "stackoverflow.com", "apple.com", "nytimes.com",
    "bbc.com", "coursera.org", "twitter.com", "dropbox.com", "cloudflare.com",
    "medium.com", "zoom.us", "mozilla.org", "python.org", "scikit-learn.org",
    "nih.gov", "stanford.edu", "mit.edu", "harvard.edu", "who.int",
    "cnn.com", "theguardian.com", "spotify.com", "netflix.com", "salesforce.com",
    "slack.com", "adobe.com", "figma.com", "gitlab.com", "kaggle.com",
    "openai.com", "anthropic.com", "quora.com", "medium.com", "imdb.com",
    "booking.com", "airbnb.com", "uber.com", "instacart.com", "etsy.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "paypal.com", "stripe.com"
]

LEGIT_SUBDOMAINS = ["www", "api", "docs", "blog", "app", "help", "support", "m", "dev", "status", "en", "shop"]

LEGIT_PATHS = [
    "",
    "about",
    "contact-us",
    "docs/quickstart",
    "articles/technology-trends-2026",
    "products/item/104928",
    "learn/machine-learning-specialization",
    "questions/421512/how-to-train-scikit-learn-model",
    "search?q=machine+learning+tutorial&page=2",
    "dashboard/overview",
    "releases/tag/v2.4.1",
    "user/profile/settings",
    "blog/engineering/scaling-distributed-systems",
    "category/science/astronomy",
    "pricing/compare-plans",
    "explore/trending-repositories"
]

# --- Phishing URL Components ---
TARGET_BRANDS = [
    "paypal", "apple", "google", "microsoft", "amazon", "netflix", "facebook",
    "instagram", "wellsfargo", "chase", "bankofamerica", "citi", "binance",
    "coinbase", "metamask", "whatsapp", "dropbox", "dhl", "fedex", "usps"
]

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "security-alert", "validate", "billing", "recovery", "auth", "credential",
    "unlock-account", "session-expired", "pass-reset", "2fa-check", "notice"
]

SUSPICIOUS_TLDS = ["xyz", "top", "tk", "ml", "ga", "cf", "gq", "work", "click", "zip", "support", "loan", "surf", "buzz", "fit", "country", "info"]

SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy", "ow.ly"]

def generate_legitimate_url():
    domain = random.choice(LEGIT_DOMAINS)
    # Most modern legit sites use https, but some use http
    scheme = "https://" if random.random() < 0.92 else "http://"
    
    # Subdomain
    if random.random() < 0.45:
        sub = random.choice(LEGIT_SUBDOMAINS)
        domain = f"{sub}.{domain}"
        
    path = random.choice(LEGIT_PATHS)
    if path:
        url = f"{scheme}{domain}/{path}"
    else:
        url = f"{scheme}{domain}"
        
    # Occasional benign query params
    if "?" not in url and random.random() < 0.2:
        params = random.choice([
            "ref=homepage",
            "utm_source=newsletter&utm_medium=email",
            "lang=en&theme=dark",
            "id=38192&view=detail"
        ])
        url = f"{url}?{params}"
        
    return url

def generate_phishing_url():
    pattern = random.choices(
        ["brand_subdomain", "ip_address", "keyword_chain", "shortener", "punycode", "at_symbol", "bad_tld_brand"],
        weights=[30, 15, 25, 10, 5, 5, 10]
    )[0]
    
    # Crucial: 40% of phishing URLs now use HTTPS to bypass naive SSL filters!
    scheme = "https://" if random.random() < 0.42 else "http://"
    brand = random.choice(TARGET_BRANDS)
    kw = random.choice(SUSPICIOUS_KEYWORDS)
    tld = random.choice(SUSPICIOUS_TLDS)
    
    if pattern == "brand_subdomain":
        # e.g., paypal.com.account-verification.xyz/signin
        fake_root = f"{kw}-{random.randint(10,999)}.{tld}"
        url = f"{scheme}{brand}.com.{fake_root}/signin?session_id={random.randint(100000, 999999)}"
        
    elif pattern == "ip_address":
        # Raw IP address host
        ip = f"{random.randint(11,210)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
        path = f"{brand}/{kw}/index.php?user_ref={random.randint(1000,9999)}"
        url = f"{scheme}{ip}/{path}"
        
    elif pattern == "keyword_chain":
        # Chained hyphens and phishing keywords
        kw2 = random.choice(SUSPICIOUS_KEYWORDS)
        domain = f"{brand}-{kw}-{kw2}-{random.randint(10,99)}.{tld}"
        path = f"webscr/cmd={kw}&dispatch={random.randint(1000,9999)}"
        url = f"{scheme}{domain}/{path}"
        
    elif pattern == "shortener":
        # Shortened URL
        short_host = random.choice(SHORTENERS)
        slug = f"{brand[:3]}{random.randint(100,999)}{kw[:4]}"
        url = f"{scheme}{short_host}/{slug}"
        
    elif pattern == "punycode":
        # Punycode spoofing (homograph attack)
        puny_domain = f"xn--{brand}-{random.randint(10,99)}a.{tld}"
        url = f"{scheme}{puny_domain}/login.php?auth={random.randint(1000,9999)}"
        
    elif pattern == "at_symbol":
        # @ symbol to obscure real landing host
        real_phish = f"{brand}-verification.{tld}"
        url = f"{scheme}secure.{brand}.com@{real_phish}/{kw}/index.html"
        
    else: # bad_tld_brand
        domain = f"{brand}-official-support.{tld}"
        url = f"{scheme}{domain}/portal/account-recovery?id={random.randint(10000,99999)}"
        
    return url

def main():
    urls = []
    
    # Generate 2,600 Legitimate URLs
    for _ in range(2600):
        urls.append((generate_legitimate_url(), "legitimate"))
        
    # Generate 2,600 Phishing URLs
    for _ in range(2600):
        urls.append((generate_phishing_url(), "phishing"))
        
    # Shuffle
    random.shuffle(urls)
    
    # Introduce a few realistic artifacts for pandas cleaning demonstration:
    # A few duplicates and empty/whitespace rows
    urls.append((urls[10][0], urls[10][1])) # duplicate
    urls.append((urls[50][0], urls[50][1])) # duplicate
    urls.append(("", "phishing")) # empty URL
    urls.append(("https://broken-test-sample.com", "")) # missing label
    
    with open("dataset.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        for u, l in urls:
            writer.writerow([u, l])
            
    print(f"Successfully generated dataset.csv with {len(urls)} rows.")

if __name__ == "__main__":
    main()
