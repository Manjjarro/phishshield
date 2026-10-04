# PhishShield feature extractor — deployment module.
# VERBATIM copy of the notebook feature cell (unchanged).
import re
import math
import ipaddress
from collections import Counter
from urllib.parse import urlsplit

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "cutt.ly", "rebrand.ly", "shorturl.at"
}

SUSPICIOUS_TOKENS = {
    "account", "auth", "bank", "confirm", "credential",
    "login", "password", "secure", "signin", "update",
    "verify", "wallet"
}

def get_url_parts(url):
    """Safely separate a URL into hostname, path, and scheme."""
    url = str(url).strip()
    candidate = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else f"//{url}"
    try:
        parsed = urlsplit(candidate)
        hostname = (parsed.hostname or "").lower()
        return url, hostname, parsed.path or "", parsed.scheme.lower()
    except ValueError:
        return url, "", "", ""

def is_ip_address(hostname):
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        return 1
    except ValueError:
        return 0

def calculate_entropy(text):
    """Measures how random/unusual the hostname characters appear."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )

def count_subdomains(hostname):
    labels = [item for item in hostname.split(".") if item]
    if len(labels) <= 2 or is_ip_address(hostname):
        return 0
    return len(labels) - 2

def extract_features(url):
    """Create explainable features from one raw URL only."""
    raw_url, hostname, path, scheme = get_url_parts(url)
    lower_url = raw_url.lower()
    digit_count = sum(character.isdigit() for character in raw_url)
    letter_count = sum(character.isalpha() for character in raw_url)
    words = set(re.findall(r"[a-z]+", lower_url))
    return {
        "url_length": len(raw_url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "dot_count": raw_url.count("."),
        "hyphen_count": raw_url.count("-"),
        "digit_count": digit_count,
        "at_count": raw_url.count("@"),
        "extra_double_slash_count": raw_url.split("://", 1)[-1].count("//"),
        "question_mark_count": raw_url.count("?"),
        "equals_count": raw_url.count("="),
        "has_ip_address_host": is_ip_address(hostname),
        "uses_https": int(scheme == "https"),
        "subdomain_count": count_subdomains(hostname),
        "digit_ratio": digit_count / max(digit_count + letter_count, 1),
        "has_shortener_domain": int(
            any(
                hostname == domain or hostname.endswith("." + domain)
                for domain in SHORTENER_DOMAINS
            )
        ),
        "hostname_entropy": calculate_entropy(hostname),
        "suspicious_token_count": len(words & SUSPICIOUS_TOKENS),
        "encoded_character_count": raw_url.count("%")
    }
