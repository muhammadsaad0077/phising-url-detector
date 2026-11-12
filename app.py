import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from feature_extraction import extract_url_features, get_host

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")

st.set_page_config(page_title="Phishing URL Hunter", page_icon="💻", layout="centered")


@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


artifact = load_artifact()

st.title("Phishing URL Detector")
st.caption("AI-based phishing detection from the URL alone, no need to visit the site.")

if artifact is None:
    st.error(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Run the training script first:\n\n"
        "```\npython src/train_model.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv\n```"
    )
    st.stop()

model = artifact["model"]
model_name = artifact["model_name"]
scaler = artifact["scaler"]
needs_scaling = artifact["needs_scaling"]
feature_names = artifact["feature_names"]

st.info(f"Loaded model: **{model_name}**")

url_input = st.text_input("Enter a URL to analyze", placeholder="http://example-login-security.com")
analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked:
    if not url_input.strip():
        st.warning("Please enter a URL!")
    else:
        features = extract_url_features(url_input)
        feature_df = pd.DataFrame([features])[feature_names]

        X = scaler.transform(feature_df) if needs_scaling else feature_df.values

        proba = model.predict_proba(X)[0]
        phishing_prob = proba[1] * 100  # class 1 = phishing
        prediction = "Likely Phishing" if phishing_prob >= 50 else "Likely Legitimate"

        st.subheader("PHISHING URL HUNTER — Result")
        st.write(f"*URL:* {url_input}")

        if prediction == "Likely Phishing":
            st.error(f"*Prediction: {prediction}*")
        else:
            st.success(f"*Prediction: {prediction}*")

        st.metric("Phishing Probability", f"{phishing_prob:.1f}%")
        st.progress(min(max(phishing_prob / 100, 0.0), 1.0))
        st.caption(f"Model used: {model_name}")

        with st.expander("Extracted URL features"):
            st.dataframe(feature_df.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "This tool analyzes the structure of the URL text only — it never visits the "
    "website. Always exercise caution with unfamiliar links regardless of the score."
)

