"""
train_model.py — Phishing URL Detection: Model Training Pipeline
================================================================
Generates a realistic synthetic dataset, engineers features, trains
Logistic Regression / Decision Tree / Random Forest, compares metrics,
saves the best model with pickle, and exports visualisations.
"""

import os
import re
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# ── reproducibility ──────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── output dirs ──────────────────────────────────────────────────
os.makedirs("model",          exist_ok=True)
os.makedirs("static/plots",   exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATASET
# ══════════════════════════════════════════════════════════════════

SAFE_DOMAINS = [
    "google.com", "github.com", "amazon.com", "wikipedia.org",
    "stackoverflow.com", "microsoft.com", "apple.com", "youtube.com",
    "linkedin.com", "twitter.com", "reddit.com", "bbc.co.uk",
    "nytimes.com", "medium.com", "coursera.org", "udemy.com",
]

PHISHING_TEMPLATES = [
    "http://{ip}/login/verify?user={rand}",
    "http://secure-{rand}.{fake}/account/update",
    "http://{fake}/paypal-login/confirm.php?id={rand}",
    "http://{rand}.{fake}/banking/verify@secure",
    "http://{fake}/{rand}/login.htm",
    "http://{ip}/{rand}/bank-verify",
    "http://signin-{rand}.{fake}/update?ref={rand}",
    "http://{fake}/free-{rand}/claim.php",
    "http://{rand}-secure.{fake}/paypal/login.aspx",
    "http://{fake}:8080/secure/verify?session={rand}",
]

FAKE_TLDS   = ["xyz", "tk", "ml", "ga", "cf", "gq", "info", "biz", "click"]
FAKE_WORDS  = ["secure", "login", "bank", "update", "verify", "paypal",
               "account", "service", "support", "free", "prize", "win"]


def _rand_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _rand_str(n=6):
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choices(chars, k=n))


def _make_safe_url():
    domain  = random.choice(SAFE_DOMAINS)
    paths   = ["", "/about", "/docs", "/search?q=python", "/products/item",
               "/blog/post-title", "/contact", "/faq"]
    scheme  = "https://" if random.random() > 0.05 else "http://"
    return scheme + domain + random.choice(paths)


def _make_phishing_url():
    tpl  = random.choice(PHISHING_TEMPLATES)
    fake = _rand_str(8) + "." + random.choice(FAKE_TLDS)
    url  = tpl.format(ip=_rand_ip(), rand=_rand_str(), fake=fake)
    # sometimes add suspicious sub-path
    if random.random() > 0.5:
        url += "/" + random.choice(FAKE_WORDS) + _rand_str(3)
    return url


def build_dataset(n=2000):
    """Generate n/2 safe + n/2 phishing URLs."""
    half = n // 2
    urls   = [_make_safe_url()     for _ in range(half)]
    labels = [0] * half
    urls  += [_make_phishing_url() for _ in range(half)]
    labels+= [1] * half

    df = pd.DataFrame({"url": urls, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

SUSPICIOUS_KEYWORDS = ["login", "verify", "bank", "secure", "update",
                       "account", "paypal", "password", "free", "prize",
                       "billing", "confirm", "service", "support", "sign"]


def extract_features(url: str) -> dict:
    """
    Extract numerical features from a raw URL string.
    Must stay in sync with the feature order used during training.
    """
    url_lower = url.lower()

    # 1. url_length
    url_length = len(url)

    # 2. num_dots
    num_dots = url.count(".")

    # 3. has_at
    has_at = int("@" in url)

    # 4. has_https
    has_https = int(url_lower.startswith("https"))

    # 5. has_ip — simple IPv4 pattern
    has_ip = int(bool(
        re.search(r"(https?://)?(\d{1,3}\.){3}\d{1,3}", url)
    ))

    # 6. num_hyphens
    num_hyphens = url.count("-")

    # 7. num_subdomains
    try:
        hostname = re.split(r"https?://", url)[1].split("/")[0].split(":")[0]
        parts    = hostname.split(".")
        num_subdomains = max(len(parts) - 2, 0)
    except Exception:
        num_subdomains = 0

    # 8. url_depth  (number of path segments)
    try:
        path  = re.split(r"https?://[^/]+", url, maxsplit=1)
        depth = len([p for p in path[-1].split("/") if p]) if len(path) > 1 else 0
    except Exception:
        depth = 0

    # 9. has_suspicious_keywords
    has_suspicious = int(any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS))

    # 10. num_suspicious_keywords
    num_suspicious = sum(url_lower.count(kw) for kw in SUSPICIOUS_KEYWORDS)

    # 11. has_port
    has_port = int(bool(re.search(r":\d{2,5}(/|$)", url)))

    # 12. tld_in_path  (e.g. .com appears in the path portion)
    suspicious_tlds = [".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".click"]
    tld_in_path = int(any(t in url_lower for t in suspicious_tlds))

    # 13. url_entropy  (character diversity → randomness)
    freq    = {c: url.count(c) / len(url) for c in set(url)}
    entropy = -sum(p * np.log2(p) for p in freq.values()) if freq else 0

    return {
        "url_length":           url_length,
        "num_dots":             num_dots,
        "has_at":               has_at,
        "has_https":            has_https,
        "has_ip":               has_ip,
        "num_hyphens":          num_hyphens,
        "num_subdomains":       num_subdomains,
        "url_depth":            depth,
        "has_suspicious":       has_suspicious,
        "num_suspicious_kw":    num_suspicious,
        "has_port":             has_port,
        "tld_in_path":          tld_in_path,
        "url_entropy":          round(entropy, 4),
    }


def build_feature_matrix(df: pd.DataFrame):
    features = df["url"].apply(extract_features)
    X = pd.DataFrame(features.tolist())
    y = df["label"]
    return X, y


# ══════════════════════════════════════════════════════════════════
# 3.  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score (y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score   (y_test, y_pred), 4),
        "F1":        round(f1_score       (y_test, y_pred), 4),
    }


def plot_confusion_matrix(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Safe", "Phishing"],
        yticklabels=["Safe", "Phishing"],
        linewidths=0.5
    )
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    plt.tight_layout()
    fname = f"static/plots/cm_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  ✓ Saved {fname}")


def plot_feature_importance(rf_model, feature_names):
    importances = rf_model.named_steps["clf"].feature_importances_
    idx         = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.viridis(np.linspace(0.2, 0.85, len(feature_names)))
    ax.bar(range(len(feature_names)),
           importances[idx],
           color=colors[idx], edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha="right", fontsize=9)
    ax.set_title("Random Forest — Feature Importances", fontsize=13, pad=12)
    ax.set_ylabel("Importance", fontsize=11)
    plt.tight_layout()
    fname = "static/plots/feature_importance.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  ✓ Saved {fname}")


def plot_model_comparison(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x       = np.arange(len(results_df))
    width   = 0.2
    colors  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric],
               width, label=metric, color=colors[i], alpha=0.88, edgecolor="white")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13, pad=12)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fname = "static/plots/model_comparison.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  ✓ Saved {fname}")


# ══════════════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  PHISHING URL DETECTION — MODEL TRAINING")
    print("═"*60)

    # ── dataset ──
    print("\n[1/5] Generating synthetic dataset …")
    df = build_dataset(n=4000)
    print(f"      {len(df)} samples  |  {df['label'].sum()} phishing  |  {(df['label']==0).sum()} safe")

    # ── features ──
    print("\n[2/5] Engineering features …")
    X, y = build_feature_matrix(df)
    feature_names = X.columns.tolist()
    print(f"      {len(feature_names)} features: {feature_names}")

    # ── split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── models ──
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(max_depth=10, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=12,
                                               random_state=42, n_jobs=-1))
        ]),
    }

    print("\n[3/5] Training models …")
    results = []
    trained = {}
    for name, pipeline in models.items():
        print(f"\n  ▶ {name}")
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
        trained[name] = pipeline

        print(f"    Accuracy : {metrics['Accuracy']}")
        print(f"    Precision: {metrics['Precision']}")
        print(f"    Recall   : {metrics['Recall']}")
        print(f"    F1       : {metrics['F1']}")
        print(f"\n{classification_report(y_test, pipeline.predict(X_test), target_names=['Safe','Phishing'])}")

        plot_confusion_matrix(pipeline, name, X_test, y_test)

    # ── plots ──
    print("\n[4/5] Generating visualisations …")
    results_df = pd.DataFrame(results)
    plot_model_comparison(results_df)
    plot_feature_importance(trained["Random Forest"], feature_names)

    # ── best model ──
    best_row  = results_df.loc[results_df["F1"].idxmax()]
    best_name = best_row["Model"]
    best_model= trained[best_name]
    print(f"\n  🏆 Best model: {best_name}  (F1 = {best_row['F1']})")

    # ── save ──
    print("\n[5/5] Saving model and artefacts …")
    model_path = "model/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "features": feature_names,
                     "model_name": best_name}, f)
    print(f"  ✓ Model saved → {model_path}")

    results_df.to_csv("model/results.csv", index=False)
    print("  ✓ Results saved → model/results.csv")

    print("\n" + "═"*60)
    print("  TRAINING COMPLETE ✓")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
