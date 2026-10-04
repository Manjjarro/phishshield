"""PhishShield — link checker.

Front-end for the locked deployment bundle. The model, features, and
threshold all come from the bundle; this file only presents them.
Run:  streamlit run app.py
"""
import json
import re
from urllib.parse import quote_plus

import joblib
import pandas as pd
import shap
import streamlit as st

import phishshield_features

st.set_page_config(page_title="PhishShield", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 1.5rem; padding-bottom: 3rem;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- bundle loading (cached, runs once) ----------

@st.cache_resource
def load_bundle():
    with open("model_card.json") as file:
        card = json.load(file)
    model = joblib.load("phishshield_hgb_model.joblib")
    background = pd.read_csv("background_features.csv").reindex(
        columns=card["features_in_training_order"]
    )
    explainer = shap.TreeExplainer(
        model,
        data=background,
        model_output="probability",
        feature_perturbation="interventional",
    )
    return card, model, explainer


CARD, MODEL, EXPLAINER = load_bundle()
FEATURE_ORDER = CARD["features_in_training_order"]
THRESHOLD = float(CARD["default_decision_threshold"])


# ---------- scoring and explanation ----------

def score_urls(urls):
    features = pd.DataFrame(
        [phishshield_features.extract_features(url) for url in urls]
    ).reindex(columns=FEATURE_ORDER)
    probabilities = MODEL.predict_proba(features)[:, 1]
    return pd.DataFrame({
        "url": urls,
        "phishing_risk": probabilities,
        "flagged": probabilities >= THRESHOLD,
    })


def explain_url(url):
    features = pd.DataFrame(
        [phishshield_features.extract_features(url)]
    ).reindex(columns=FEATURE_ORDER)
    probability = float(MODEL.predict_proba(features)[0, 1])
    shap_values = EXPLAINER.shap_values(features)
    if isinstance(shap_values, list):
        values = shap_values[1]
    elif shap_values.ndim == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values
    explanation = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "value": features.iloc[0].values,
        "contribution": values[0],
    })
    return probability, explanation


def format_risk(probability):
    return ">99.9%" if probability >= 0.9995 else f"{probability:.1%}"


# ---------- plain-language translation of every feature ----------

HUMAN_EXPLANATIONS = {
    "url_length": {
        "up": "The full address is {value} characters long. Safe site addresses are usually shorter, so this raises the risk score.",
        "down": "The full address is {value} characters long — short, like most safe sites. This lowers the risk score.",
    },
    "hostname_length": {
        "up": "The domain name is {value} characters long — an unusual size for a trusted site's address. This raises the risk score.",
        "down": "The domain name is {value} characters long — a typical size for a normal website. This lowers the risk score.",
    },
    "dot_count": {
        "up": "The address has {value} dots. Extra dots can mean stacked subdomains, which phishers use to imitate real companies.",
        "down": "The address has {value} dots — a simple, normal structure. This lowers the risk score.",
    },
    "hyphen_count": {
        "up": "The address contains {value} hyphen(s). Hyphens are far more common in phishing addresses than in real site names.",
        "down": "The address contains {value} hyphen(s) — a clean name, like most real sites.",
    },
    "digit_count": {
        "up": "The address contains {value} digits. Ordinary website addresses almost never contain digits.",
        "down": "The address contains {value} digit(s) — like almost every legitimate site address.",
    },
    "at_count": {
        "up": "The address contains an '@' symbol — a known trick for hiding the real destination in a link.",
        "down": "There is no '@' symbol hiding the real destination — a good sign.",
    },
    "extra_double_slash_count": {
        "up": "The address contains extra '//' sequences after the domain, which can obscure where the link really goes.",
        "down": "The address has a straightforward structure with no extra '//' sequences.",
    },
    "question_mark_count": {
        "up": "The address has {value} query parameter(s) ('?'). Fake login pages often carry these; safe homepages rarely do.",
        "down": "The address has no query parameters — like a plain site address.",
    },
    "equals_count": {
        "up": "The address carries {value} '=' sign(s) — data embedded in the link, which fake form pages often use.",
        "down": "The address carries no '=' signs — a plain, simple link.",
    },
    "has_ip_address_host": {
        "up": "The link points to a raw IP address instead of a domain name. Real organisations use domain names — this is a strong warning sign.",
        "down": "The link uses a normal domain name rather than a raw IP address.",
    },
    "subdomain_count": {
        "up": "Beyond the main domain, the address stacks {value} extra subdomain level(s) — a technique used to imitate real companies.",
        "down": "Beyond the main domain, the address has {value} extra subdomain level(s) — a simple, normal structure.",
    },
    "digit_ratio": {
        "up": "A large share of the address is made of digits, which is unusual for a real site name.",
        "down": "Almost none of the address is made of digits — typical of real site names.",
    },
    "has_shortener_domain": {
        "up": "The address uses a link shortener, which hides where a link really goes. Not proof of phishing, but worth a close look.",
        "down": "The address does not rely on a link shortener.",
    },
    "hostname_entropy": {
        "up": "The domain name reads like a random jumble of characters rather than a real word or brand.",
        "down": "The domain name reads like a real word or brand — normal for a trusted site.",
    },
    "suspicious_token_count": {
        "up": "The address contains {value} sensitive word(s) such as 'login', 'verify', 'account' or 'password' — exactly the words phishing pages rely on.",
        "down": "The address contains none of the sensitive words ('login', 'verify', 'account', 'password') that phishing pages rely on.",
    },
    "encoded_character_count": {
        "up": "The address contains {value} encoded character(s) ('%'), often used to disguise what a link really points to.",
        "down": "The address contains no encoded characters.",
    },
}


ZERO_VALUE_PHRASES = {
    "at_count": "no '@' symbol",
    "extra_double_slash_count": "no extra '//' sequences",
    "question_mark_count": "no query parameters",
    "equals_count": "no '=' signs",
    "hyphen_count": "no hyphens",
    "digit_count": "no digits",
    "encoded_character_count": "no encoded characters",
    "suspicious_token_count": "no sensitive words",
    "subdomain_count": "no 'www.' prefix",
    "has_ip_address_host": "a normal domain name rather than an IP",
    "has_shortener_domain": "no link shortener",
}


def build_human_reasons(explanation, url):
    matched_words = sorted(
        set(re.findall(r"[a-z]+", url.lower()))
        & phishshield_features.SUSPICIOUS_TOKENS
    )
    reasons = []
    for _, row in explanation.iterrows():
        spec = HUMAN_EXPLANATIONS.get(row["feature"])
        if not spec:
            continue
        template = spec["up"] if row["contribution"] > 0 else spec["down"]
        value = int(round(float(row["value"])))
        text = template.format(value=value)

        # Interaction effects: a factor can push risk up even when its
        # own value looks perfectly fine. Say so honestly instead of
        # producing a contradictory sentence.
        zero_phrase = ZERO_VALUE_PHRASES.get(row["feature"])
        if zero_phrase and value == 0 and row["contribution"] > 0:
            text = (f"The address has {zero_phrase}, which on its own "
                    "looks fine — but in combination with the other "
                    "factors here it nudges the risk score up slightly.")
        elif (row["feature"] == "url_length"
              and row["contribution"] > 0 and value <= 28):
            text = (f"The full address is {value} characters — not long "
                    "in itself, but an unusual length for the shape of "
                    "this address. This nudges the risk score up.")

        if row["feature"] == "suspicious_token_count" and matched_words:
            text += " Found here: " + ", ".join(matched_words) + "."
        reasons.append((text, float(row["contribution"])))
    strong = [r for r in reasons if abs(r[1]) >= 0.004] or reasons
    strong.sort(key=lambda item: -item[1])
    return strong[:5]


# ---------- URL extraction with de-fanging ----------

# Recognises normal schemes (http/https), de-fanged schemes (hxxp/hxxps,
# any capitalisation), and www.-prefixed links. Brackets and parentheses
# are allowed INSIDE the captured link, because de-fanged links like
# bit[.]ly contain them.
URL_PATTERN = re.compile(
    r"(?:(?:https?|hxxps?)://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)


def strip_trailing_junk(candidate):
    """Remove trailing punctuation and unmatched closing wrappers.

    A closing bracket/parenthesis is only removed when it has no
    matching opener inside the link — so '[.]' inside survives,
    while a stray ')' or ']' at the end (text wrappers) is dropped.
    """
    candidate = candidate.rstrip(".,;:!?\"'*")
    for closer, opener in ((")", "("), ("]", "["), ("}", "{")):
        while (candidate.endswith(closer)
               and candidate.count(opener) < candidate.count(closer)):
            candidate = candidate[:-1].rstrip(".,;:!?\"'*")
    return candidate


def defang_url(url):
    url = re.sub(r"hxxp", "http", url, flags=re.IGNORECASE)
    for pattern in ("[.]", "(.)", "[dot]"):
        url = url.replace(pattern, ".")
    url = url.replace("[:]", ":").replace("(:)", ":")
    return url


def extract_urls(text):
    found = []
    for match in URL_PATTERN.findall(text):
        candidate = defang_url(strip_trailing_junk(match))
        if candidate and candidate not in found:
            found.append(candidate)
    return found


# ---------- presentation ----------

def render_verdict(probability):
    flagged = probability >= THRESHOLD
    if flagged:
        color, title = "#b91c1c", "⚠ Warning signs in this address"
        detail = (f"Estimated phishing risk {format_risk(probability)} — "
                  f"at or above the {THRESHOLD:.0%} flag limit.")
    else:
        color, title = "#15803d", "✓ No warning signs found"
        detail = (f"Estimated phishing risk {format_risk(probability)} — "
                  f"below the {THRESHOLD:.0%} flag limit.")
    st.markdown(
        f"""<div style='background:{color}; color:#ffffff; padding:14px 18px;
             border-radius:10px; margin:6px 0 8px 0;'>
        <div style='font-size:20px; font-weight:700;'>{title}</div>
        <div style='font-size:14px; margin-top:2px;'>{detail}</div></div>""",
        unsafe_allow_html=True,
    )


def render_gauge(probability):
    position = min(probability, 1.0) * 100
    limit = THRESHOLD * 100
    st.markdown(
        f"""
        <div style='position:relative; height:16px; border-radius:8px;
             background:linear-gradient(90deg, #16a34a 0%, #eab308 45%, #dc2626 85%);
             margin:2px 0 18px 0;'>
          <div style='position:absolute; left:calc({limit:.1f}% - 1px); top:-3px;
               width:2px; height:22px; background:#0f172a;'></div>
          <div style='position:absolute; left:calc({position:.1f}% - 8px); top:1px;
               width:14px; height:14px; border:3px solid #0f172a;
               border-radius:50%; background:#ffffff;'></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption("Circle = this link's risk · dark line = the flag limit. "
               "Green side is safer, red side is riskier.")


def render_path_caveat(url):
    _, _, path, _ = phishshield_features.get_url_parts(url)
    if path and path.strip("/"):
        st.info(
            f"Heads-up: this address has a path after the domain name "
            f"(\"{path}\"). The training data contained no safe sites with "
            "paths, so the model leans suspicious on any address like this — "
            "its biggest known weakness. If this is a real page on a real "
            "site, check it manually."
        )


def render_reasons(explanation, url):
    st.subheader("What the model noticed")
    for text, contribution in build_human_reasons(explanation, url):
        if contribution > 0:
            style, label = "#b91c1c", "▲ raises risk"
        else:
            style, label = "#15803d", "▼ lowers risk"
        st.markdown(
            f"<div style='margin:6px 0;'><span style='color:{style}; "
            f"font-weight:700;'>{label}</span> — {text}</div>",
            unsafe_allow_html=True,
        )


def render_second_opinion(url):
    """Optional link-out: the USER searches public opinion themselves.
    The app sends nothing anywhere — these are plain hyperlinks."""
    _, hostname, _, _ = phishshield_features.get_url_parts(url)
    domain = hostname.strip("[]")
    if not domain:
        return
    d = quote_plus(domain)
    st.markdown(
        "<div style='font-size:0.85em; color:#64748b; margin-top:12px;'>"
        "Second opinion — see what real users say about "
        f"<b>{domain}</b> (opens a new tab; the app sends nothing "
        "unless you click): "
        f"<a href='https://www.google.com/search?q=%22{d}%22' target='_blank'>Google</a> · "
        f"<a href='https://www.google.com/search?q=site%3Areddit.com+{d}' target='_blank'>Reddit</a> · "
        f"<a href='https://www.trustpilot.com/review/{d}' target='_blank'>Trustpilot</a> · "
        f"<a href='https://www.scamadviser.com/check-website/{d}' target='_blank'>ScamAdviser</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------- UI ----------

st.title("PhishShield")
st.markdown("Paste a link. Get a risk score — and the reasons behind it, "
            "in plain language.")

tab_check, tab_message, tab_about = st.tabs(
    ["Check a link", "Check a message", "About"]
)

with tab_check:
    url = st.text_input("Link to check",
                        placeholder="https://example.com/login")
    if url.strip():
        probability, explanation = explain_url(url.strip())
        render_verdict(probability)
        render_gauge(probability)
        render_path_caveat(url.strip())
        render_reasons(explanation, url.strip())
        render_second_opinion(url.strip())
        with st.expander("Technical details"):
            top = (
                explanation.assign(weight=explanation["contribution"].abs())
                .sort_values("weight", ascending=False)
                .set_index("feature")
                .drop(columns="weight")
            )
            st.dataframe(top.round(4))
            st.caption("Raw model internals. 'contribution' is the SHAP "
                       "value — positive pushes the phishing probability up. "
                       "The plain-language list above is generated from this "
                       "table.")
            st.bar_chart(top["contribution"])

with tab_message:
    text = st.text_area(
        "Paste a message, email, or anything containing links",
        height=180,
        placeholder="e.g. 'URGENT: verify your account at "
                    "hxxps://bit[.]ly/login-update now…'",
    )
    uploaded = st.file_uploader("Or upload a CSV with a column of links",
                                type=["csv"])

    urls = []
    if text.strip():
        urls = extract_urls(text)
    elif uploaded is not None:
        frame = pd.read_csv(uploaded)
        url_column = next(
            (c for c in frame.columns if "url" in c.lower()), None
        )
        if url_column is None:
            for column in frame.columns:
                if frame[column].astype(str).str.match(
                        r"(?i)(https?://|www\.)").mean() > 0.5:
                    url_column = column
                    break
        if url_column is not None:
            urls = [str(v).strip()
                    for v in frame[url_column].dropna().tolist()]

    if urls:
        results = score_urls(urls).sort_values(
            "phishing_risk", ascending=False
        )
        col1, col2 = st.columns(2)
        col1.metric("Links found", len(results))
        col2.metric("Flagged for review", int(results["flagged"].sum()))

        display = results.assign(
            phishing_risk=results["phishing_risk"].map(format_risk),
            flagged=results["flagged"].map({True: "⚠ flag", False: "✓ pass"}),
        ).rename(columns={"url": "link", "phishing_risk": "risk",
                          "flagged": "verdict"})
        st.dataframe(display.set_index("link"), use_container_width=True)

        chosen = st.selectbox("See the reasons for a specific link",
                              results["url"].tolist())
        if chosen:
            probability, explanation = explain_url(chosen)
            render_verdict(probability)
            render_path_caveat(chosen)
            render_reasons(explanation, chosen)
            render_second_opinion(chosen)
    elif text.strip():
        st.info("No links found in that text.")

with tab_about:
    st.subheader("What this is")
    st.markdown(
        "PhishShield reads the text of a web address and estimates how "
        "likely it is to be a phishing link. It never opens the page and "
        "never calls an outside service — nothing you check leaves this "
        "app. It was built as a **first check for a human**: it helps "
        "decide what to look at first, and it should never block anything "
        "on its own."
    )
    st.caption("The 'second opinion' links under each result are plain "
               "hyperlinks — nothing is sent anywhere unless you click "
               "them yourself.")
    st.subheader("Where it does well")
    st.markdown(
        "- Detects 98.4% of phishing links in held-out testing\n"
        "- Gives plain-language reasons for every score\n"
        "- Catches disguised links written as hxxps:// or bit[.]ly"
    )
    st.subheader("Known weaknesses (measured, not guessed)")
    st.markdown(
        "- **Over-flags links with paths.** A link like a university portal "
        "page gets flagged, because the safe sites in the training data "
        "were all plain homepages. This is the model's biggest weakness.\n"
        "- **Misses about 1 in 100 phishing links** hosted on real, hacked "
        "websites — the address itself looks perfectly normal.\n"
        "- **Most flags will be false alarms at real-world volumes.** "
        "That's expected — which is exactly why a person stays in the loop."
    )
    st.caption("Built by Divine Mfoniso Ukpong")
    with st.expander("Technical summary (versions, features, metadata)"):
        st.json({key: CARD[key] for key in [
            "model_type", "feature_set", "default_decision_threshold",
            "training_rows", "source_dataset", "sklearn_version",
            "joblib_version", "intended_use"]})
        st.markdown("**The 16 features the model reads:** "
                    + ", ".join(FEATURE_ORDER))