"""PhishShield inference example.

Run in IT's environment with the three sibling files present:
model card, feature module, and the joblib model.
Requires the scikit-learn / joblib versions recorded in model_card.json.
"""
import json
import joblib
import pandas as pd

import phishshield_features  # sibling module in this bundle

with open("model_card.json") as file:
    model_card = json.load(file)

FEATURE_ORDER = model_card["features_in_training_order"]
DECISION_THRESHOLD = model_card["default_decision_threshold"]

model = joblib.load("phishshield_hgb_model.joblib")


def score_urls(urls, threshold=DECISION_THRESHOLD):
    """Per-URL phishing risk and flag decisions, for human review."""
    features = pd.DataFrame(
        [phishshield_features.extract_features(url) for url in urls]
    ).reindex(columns=FEATURE_ORDER)
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame({
        "url": urls,
        "phishing_risk": probabilities,
        "flag_for_review": probabilities >= threshold,
    })


if __name__ == "__main__":
    demo_urls = [
        "https://www.google.com",
        "http://192.168.1.1/verify-account",
        "https://bit.ly/login-update",
        "www.example.edu/student-portal",
    ]
    print(score_urls(demo_urls).to_string(index=False))
    print("\nDisplay scores near 1.0 as '>99.9%'.")
    print("Known blind spot: benign URLs with paths are over-flagged —")
    print("see model_card.json known_limitations.")
