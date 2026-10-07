"""
features.py
Modular feature extraction library for Phishing URL Detection.
Computes ~32 lexical features from raw URLs without network requests.
Provides explainable reason generation for end users and security analysts.
"""

import math
import re
from urllib.parse import urlparse

# Suspicious keywords frequently found in credential harvesting campaigns
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "bank", "wallet", "password", "credential", "auth", "payment",
    "paypal", "apple", "google", "microsoft", "netflix", "amazon", "ebay",
    "chase", "support", "recover", "recovery", "billing", "security", "alert",
    "validate", "unlock", "session", "suspended"
]

# High-abuse Top Level Domains (TLDs)
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "work", "click", "zip",
    "support", "loan", "surf", "buzz", "fit", "country", "info", "live", "club"
}

# Known URL shortener hostnames
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly",
    "rb.gy", "ow.ly", "buff.ly", "rebrand.ly", "bl.ink", "tiny.cc"
}

FEATURE_NAMES = [
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "count_dots",
    "count_dots_host",
    "count_hyphens",
    "count_hyphens_host",
    "count_at",
    "count_question",
    "count_equal",
    "count_percent",
    "count_slash",
    "count_underscore",
    "count_digits",
    "count_digits_host",
    "digit_ratio",
    "digit_ratio_host",
    "is_ip",
    "has_https",
    "num_subdomains",
    "num_suspicious_keywords",
    "keyword_in_subdomain",
    "keyword_in_path",
    "is_shortener",
    "has_punycode",
    "shannon_entropy",
    "entropy_host",
    "tld_length",
    "is_bad_tld",
    "has_port",
    "consecutive_char_repeat"
]

def calculate_entropy(text: str) -> float:
    """Calculate the Shannon Entropy of a string to measure randomness/obfuscation."""
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    length = len(text)
    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
    return round(entropy, 4)

def max_consecutive_repeats(text: str) -> int:
    """Find maximum consecutive identical characters in a string."""
    if not text:
        return 0
    max_rep = 1
    curr_rep = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            curr_rep += 1
            if curr_rep > max_rep:
                max_rep = curr_rep
        else:
            curr_rep = 1
    return max_rep

def extract_registered_domain_and_sub(host: str):
    """Decompose host into registered domain, tld, and subdomains."""
    parts = host.split(".")
    if len(parts) <= 1:
        return host, "", ""
    tld = parts[-1]
    # Check for two-part country TLDs like .co.uk, .com.au
    if len(parts) > 2 and len(tld) == 2 and parts[-2] in {"co", "com", "org", "gov", "ac", "net", "edu"}:
        reg_domain = ".".join(parts[-3:])
        sub = ".".join(parts[:-3])
        tld = ".".join(parts[-2:])
    else:
        reg_domain = ".".join(parts[-2:])
        sub = ".".join(parts[:-2])
    return reg_domain, sub, tld

def extract_features(raw_url: str) -> dict:
    """
    Extract ~32 lexical features from a raw URL.
    Does not make any network calls.
    """
    url_str = str(raw_url).strip()
    if not url_str:
        return {feat: 0 for feat in FEATURE_NAMES}

    # Ensure URL has a scheme for urlparse
    has_explicit_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url_str))
    parse_target = url_str if has_explicit_scheme else f"http://{url_str}"

    try:
        parsed = urlparse(parse_target)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        host = ""
        path = ""
        query = ""

    url_lower = url_str.lower()
    reg_domain, subdomains, tld = extract_registered_domain_and_sub(host)

    # 1. Lengths
    url_length = len(url_str)
    host_length = len(host)
    path_length = len(path)
    query_length = len(query)

    # 2. Delimiter counts
    count_dots = url_str.count(".")
    count_dots_host = host.count(".")
    count_hyphens = url_str.count("-")
    count_hyphens_host = host.count("-")
    count_at = url_str.count("@")
    count_question = url_str.count("?")
    count_equal = url_str.count("=")
    count_percent = url_str.count("%")
    count_slash = url_str.count("/")
    count_underscore = url_str.count("_")

    # 3. Digits
    count_digits = sum(c.isdigit() for c in url_str)
    count_digits_host = sum(c.isdigit() for c in host)
    digit_ratio = count_digits / max(1, url_length)
    digit_ratio_host = count_digits_host / max(1, host_length)

    # 4. Host Properties
    # IPv4 regex: 4 octets
    is_ipv4 = bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", host))
    is_ipv6 = ":" in host and not host.startswith("xn--")
    is_ip = 1 if (is_ipv4 or is_ipv6) else 0

    has_https = 1 if url_lower.startswith("https://") else 0
    num_subdomains = len(subdomains.split(".")) if subdomains else 0

    # 5. Keyword presence
    kw_hits = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]
    num_suspicious_keywords = len(kw_hits)
    keyword_in_subdomain = 1 if any(kw in subdomains.lower() for kw in SUSPICIOUS_KEYWORDS) else 0
    keyword_in_path = 1 if any(kw in path.lower() for kw in SUSPICIOUS_KEYWORDS) else 0

    # 6. Specialized indicators
    is_shortener = 1 if host in SHORTENERS or any(host.endswith("." + s) for s in SHORTENERS) else 0
    has_punycode = 1 if "xn--" in host else 0
    shannon_entropy = calculate_entropy(url_str)
    entropy_host = calculate_entropy(host)
    tld_clean = tld.split(".")[-1] if tld else ""
    tld_length = len(tld_clean)
    is_bad_tld = 1 if tld_clean in SUSPICIOUS_TLDS else 0
    has_port = 1 if bool(parsed.port) else 0
    consecutive_char_repeat = max_consecutive_repeats(url_str)

    features = {
        "url_length": url_length,
        "host_length": host_length,
        "path_length": path_length,
        "query_length": query_length,
        "count_dots": count_dots,
        "count_dots_host": count_dots_host,
        "count_hyphens": count_hyphens,
        "count_hyphens_host": count_hyphens_host,
        "count_at": count_at,
        "count_question": count_question,
        "count_equal": count_equal,
        "count_percent": count_percent,
        "count_slash": count_slash,
        "count_underscore": count_underscore,
        "count_digits": count_digits,
        "count_digits_host": count_digits_host,
        "digit_ratio": round(digit_ratio, 4),
        "digit_ratio_host": round(digit_ratio_host, 4),
        "is_ip": is_ip,
        "has_https": has_https,
        "num_subdomains": num_subdomains,
        "num_suspicious_keywords": num_suspicious_keywords,
        "keyword_in_subdomain": keyword_in_subdomain,
        "keyword_in_path": keyword_in_path,
        "is_shortener": is_shortener,
        "has_punycode": has_punycode,
        "shannon_entropy": shannon_entropy,
        "entropy_host": entropy_host,
        "tld_length": tld_length,
        "is_bad_tld": is_bad_tld,
        "has_port": has_port,
        "consecutive_char_repeat": consecutive_char_repeat
    }
    return features

def generate_reasons(raw_url: str, features: dict, phishing_probability: float) -> list:
    """
    Generate 3-5 human-readable, context-aware reasons explaining
    why a URL was flagged as phishing or determined legitimate.
    """
    reasons = []
    url_str = str(raw_url).lower()
    
    # Check IP host
    if features.get("is_ip", 0) == 1:
        reasons.append("Uses a raw IP address instead of a legitimate domain name")

    # Check @ symbol
    if features.get("count_at", 0) > 0:
        reasons.append("Contains '@' symbol, an evasion technique used to obscure the true destination")

    # Check Punycode
    if features.get("has_punycode", 0) == 1:
        reasons.append("Contains Punycode ('xn--'), often used in homograph impersonation attacks")

    # Check URL Shortener
    if features.get("is_shortener", 0) == 1:
        reasons.append("Uses a URL shortener service which hides the destination server")

    # Check brand keyword in subdomain
    if features.get("keyword_in_subdomain", 0) == 1:
        reasons.append("Positions security/brand keywords in subdomains to spoof authentic domains")

    # Check suspicious keywords
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_str]
    if found_keywords:
        sample_kw = ", ".join(f"'{k}'" for k in found_keywords[:3])
        reasons.append(f"Contains sensitive credential targeting keywords ({sample_kw})")

    # Check excessive subdomains
    if features.get("num_subdomains", 0) >= 3:
        reasons.append(f"Unusually high subdomain count ({features['num_subdomains']} layers) disguising the real root domain")

    # Check suspicious TLD
    if features.get("is_bad_tld", 0) == 1:
        reasons.append("Registered under a high-risk / low-reputation top level domain (TLD)")

    # Check excessive hyphens in host
    if features.get("count_hyphens_host", 0) >= 2:
        reasons.append(f"Multiple hyphens in domain ({features['count_hyphens_host']}), characteristic of deceptive lookalikes")

    # Check digit ratio in host
    if features.get("digit_ratio_host", 0) > 0.25:
        reasons.append(f"Unusually high ratio of digits in domain name ({int(features['digit_ratio_host']*100)}%)")

    # Check Shannon entropy
    if features.get("shannon_entropy", 0) > 4.5:
        reasons.append(f"High character randomness/entropy ({features['shannon_entropy']:.2f}) indicates automated token generation")

    # Check abnormal length
    if features.get("url_length", 0) > 75:
        reasons.append(f"Abnormally long URL ({features['url_length']} characters) designed to conceal host details")

    # Check lack of HTTPS if risk is high
    if features.get("has_https", 0) == 0 and phishing_probability > 0.5:
        reasons.append("Unencrypted connection (HTTP): lacks SSL/TLS security certificate")

    # If Legitimate / Low Risk and few or no bad reasons found:
    if phishing_probability < 0.5:
        safe_reasons = []
        if features.get("has_https", 0) == 1:
            safe_reasons.append("Secure HTTPS connection with valid protocol structure")
        if features.get("num_subdomains", 0) <= 1:
            safe_reasons.append("Standard domain hierarchy without suspicious subdomain nesting")
        if features.get("is_bad_tld", 0) == 0:
            safe_reasons.append("Uses an established, reputable top-level domain")
        if features.get("num_suspicious_keywords", 0) == 0:
            safe_reasons.append("No credential harvesting or urgency keywords detected")
        if features.get("shannon_entropy", 0) < 4.2:
            safe_reasons.append("Natural lexical character distribution with low entropy")
        return safe_reasons[:4]

    # Return top 3 to 5 reasons for phishing
    if not reasons:
        reasons.append("Lexical pattern and token structure closely match known phishing signatures")
        if features.get("has_https", 0) == 0:
            reasons.append("Missing HTTPS transport encryption")

    return reasons[:5]
