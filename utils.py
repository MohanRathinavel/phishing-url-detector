"""
utils.py — Feature Extraction Utilities
========================================
Single source of truth for URL feature extraction.
Used by both train_model.py and app.py so features
always stay in sync between training and inference.
"""

import re
import numpy as np

# ── keyword list must match the one in train_model.py ────────────
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "bank", "secure", "update",
    "account", "paypal", "password", "free", "prize",
    "billing", "confirm", "service", "support", "sign"
]

SUSPICIOUS_TLDS = [".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".click"]


def extract_features(url: str) -> dict:
    """
    Convert a raw URL string into a feature dictionary.

    Parameters
    ----------
    url : str
        The URL to analyse (e.g. 'https://google.com/search?q=hi')

    Returns
    -------
    dict
        13 numerical features keyed by name.
    """
    url_lower = url.lower()

    # ── 1. url_length ─────────────────────────────────────────────
    url_length = len(url)

    # ── 2. num_dots ───────────────────────────────────────────────
    num_dots = url.count(".")

    # ── 3. has_at ─────────────────────────────────────────────────
    has_at = int("@" in url)

    # ── 4. has_https ──────────────────────────────────────────────
    has_https = int(url_lower.startswith("https"))

    # ── 5. has_ip ─────────────────────────────────────────────────
    has_ip = int(bool(
        re.search(r"(https?://)?(\d{1,3}\.){3}\d{1,3}", url)
    ))

    # ── 6. num_hyphens ────────────────────────────────────────────
    num_hyphens = url.count("-")

    # ── 7. num_subdomains ─────────────────────────────────────────
    try:
        hostname      = re.split(r"https?://", url)[1].split("/")[0].split(":")[0]
        parts         = hostname.split(".")
        num_subdomains = max(len(parts) - 2, 0)
    except Exception:
        num_subdomains = 0

    # ── 8. url_depth ──────────────────────────────────────────────
    try:
        path  = re.split(r"https?://[^/]+", url, maxsplit=1)
        depth = len([p for p in path[-1].split("/") if p]) if len(path) > 1 else 0
    except Exception:
        depth = 0

    # ── 9. has_suspicious (binary) ────────────────────────────────
    has_suspicious = int(any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS))

    # ── 10. num_suspicious_kw (count) ────────────────────────────
    num_suspicious = sum(url_lower.count(kw) for kw in SUSPICIOUS_KEYWORDS)

    # ── 11. has_port ──────────────────────────────────────────────
    has_port = int(bool(re.search(r":\d{2,5}(/|$)", url)))

    # ── 12. tld_in_path ───────────────────────────────────────────
    tld_in_path = int(any(t in url_lower for t in SUSPICIOUS_TLDS))

    # ── 13. url_entropy (Shannon entropy) ────────────────────────
    freq    = {c: url.count(c) / len(url) for c in set(url)} if url else {}
    entropy = -sum(p * np.log2(p) for p in freq.values()) if freq else 0

    return {
        "url_length":        url_length,
        "num_dots":          num_dots,
        "has_at":            has_at,
        "has_https":         has_https,
        "has_ip":            has_ip,
        "num_hyphens":       num_hyphens,
        "num_subdomains":    num_subdomains,
        "url_depth":         depth,
        "has_suspicious":    has_suspicious,
        "num_suspicious_kw": num_suspicious,
        "has_port":          has_port,
        "tld_in_path":       tld_in_path,
        "url_entropy":       round(entropy, 4),
    }


def features_to_list(feature_dict: dict, feature_order: list) -> list:
    """
    Convert a feature dict → ordered list for model input.

    Parameters
    ----------
    feature_dict  : dict  — output of extract_features()
    feature_order : list  — feature names in training order

    Returns
    -------
    list of numeric values in the correct column order
    """
    return [feature_dict[k] for k in feature_order]


def describe_features(url: str) -> str:
    """Human-readable summary of features for a URL (useful for debugging)."""
    feats = extract_features(url)
    lines = [f"  URL : {url[:80]}{'…' if len(url)>80 else ''}",
             "  " + "─"*54]
    for k, v in feats.items():
        lines.append(f"  {k:<22} : {v}")
    return "\n".join(lines)


# ── quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "https://google.com/search?q=hello",
        "http://192.168.1.1/login/verify@secure",
        "http://paypal-secure-update.xyz/account/login.php?id=abc123",
    ]
    for t in tests:
        print(describe_features(t))
        print()
