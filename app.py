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


