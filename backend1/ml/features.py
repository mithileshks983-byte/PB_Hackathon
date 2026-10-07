import math
import re
from urllib.parse import urlparse

KEYWORDS = ["login", "verify", "secure", "account", "update", "bank", "paypal",
            "signin", "confirm", "password", "webscr", "ebay", "amazon",
            "apple", "microsoft", "support", "wallet", "billing"]
BAD_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "work", "zip", "loan"}

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "num_dots", "num_hyphens",
    "host_hyphens", "num_digits", "digit_ratio", "num_special", "has_at",
    "has_ip", "has_https", "num_subdomains", "url_depth", "keyword_count",
    "host_has_keyword", "bad_tld", "entropy",
]

IP_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _entropy(s):
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))


def extract_features(url):
    try:
        url = str(url).strip()
        has_https = int(url.lower().startswith("https://"))
        clean = re.sub(r"^https?://", "", url, flags=re.I)
        clean = re.sub(r"^www\.", "", clean, flags=re.I)

        host_part = re.split(r"[/?#]", clean, maxsplit=1)[0]
        rest = clean[len(host_part):]
        try:
            host = urlparse("http://" + clean).hostname or host_part
        except ValueError:
            host = host_part

        has_ip = int(bool(IP_PATTERN.match(host)))
        parts = host.split(".") if host else []
        num_sub = 0 if has_ip else max(len(parts) - 2, 0)
        tld = parts[-1].lower() if parts and not has_ip else ""
        lower = clean.lower()
        host_lower = host.lower()
        digits = sum(c.isdigit() for c in clean)

        return [
            len(clean), len(host_part), len(rest),
            clean.count("."), clean.count("-"), host_part.count("-"),
            digits, digits / max(len(clean), 1),
            sum(clean.count(c) for c in "?=&%_~"),
            int("@" in clean), has_ip, has_https, num_sub,
            clean.count("/"),
            sum(k in lower for k in KEYWORDS),
            int(any(k in host_lower for k in KEYWORDS)),
            int(tld in BAD_TLDS),
            round(_entropy(clean), 3),
        ]
    except Exception:
        return [0] * len(FEATURE_NAMES)


if __name__ == "__main__":
    for u in ["https://www.google.com/search", "paypal-verify-login.com/secure/update"]:
        print(u)
        print(dict(zip(FEATURE_NAMES, extract_features(u))))