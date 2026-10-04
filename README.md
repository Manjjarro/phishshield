PhishShield

Phishing URL risk scoring with plain-language explanations — from theURL text alone.

PhishShield reads a web address and estimates how likely it is to be aphishing link, in about a millisecond, without opening the page andwithout calling any outside service. Every score comes with the reasonsbehind it, in plain language.

Live demo: (https://phishshielder.streamlit.app/)
What it's for

A first check for a human: it prioritizes links for review. It shouldnever block anything on its own. On held-out testing it detects 98.4%of phishing URLs (F1 0.977, ROC-AUC 0.996) at a ~2% false-flag rate.

The model was trained on the PhiUSIIL Phishing URL Dataset; the fulltraining and audit pipeline lives in the project notebook. The app onlyloads the saved model — it never retrains.
Run it locally

Python 3.10 or newer:

python -m venv .venv.venv\Scripts\activate        (Mac/Linux: source .venv/bin/activate)pip install -r requirements.txtstreamlit run app.py

Self-check after starting (must match exactly): google.com → 0.4% risk;http://192.168.1.1/verify-account → >99.9%; bit.ly/login-update → >99.9%;www.example.edu/student-portal → >99.9%.
Files
File	Purpose
app.py	Streamlit web app
phishshield_hgb_model.joblib	Trained model (do not modify)
phishshield_features.py	Feature extractor (verbatim from training)
model_card.json	Versions, threshold, feature order, limitations
background_features.csv	SHAP reference sample
score_urls_example.py	Standalone scorer (no web app needed)
requirements.txt	Pinned dependencies
Known weaknesses (measured, not guessed)

    Over-flags links with paths (e.g. university portal pages) — thetraining data's safe sites were all bare homepages.
    Misses about 1 in 100 phishing links hosted on real, hacked websites.
    Most flags will be false alarms at real-world phishing rates — whichis exactly why a human stays in the loop.

Author

Divine Mfoniso Ukpong
