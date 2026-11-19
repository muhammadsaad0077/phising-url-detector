import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(__file__))
from feature_extraction import FEATURE_NAMES, extract_url_features

URL_COL_CANDIDATES = ["URL", "url", "Url"]
LABEL_COL_CANDIDATES = ["label", "Label", "LABEL", "class", "Class", "Result", "status"]


def detect_column(df, candidates, kind):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not auto-detect the {kind} column. "
        f"Columns found: {list(df.columns)}. "
        f"Pass --{kind}-col explicitly."
    )


def normalize_label(series: pd.Series) -> pd.Series:
    """Map the label column to 1 = phishing, 0 = legitimate, regardless
    of how the source dataset encoded it."""
    if series.dtype == object:
        mapping = {
            "phishing": 1, "phish": 1, "bad": 1, "malicious": 1, "1": 1,
            "legitimate": 0, "legit": 0, "good": 0, "safe": 0, "0": 0,
        }
        return series.str.lower().map(mapping)

    unique_vals = sorted(series.dropna().unique().tolist())
    if set(unique_vals) == {0, 1}:
        # PhiUSIIL's official convention is 1 = legitimate, 0 = phishing.
        # We standardize to 1 = phishing for this project, so flip it.
        return 1 - series
    return series


def build_feature_dataframe(urls: pd.Series) -> pd.DataFrame:
    print(f"Extracting {len(FEATURE_NAMES)} lexical features from {len(urls):,} URLs...")
    start = time.time()
    records = [extract_url_features(u) for u in urls]
    feats = pd.DataFrame.from_records(records, columns=FEATURE_NAMES)
    print(f"  done in {time.time() - start:.1f}s")
    return feats


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Confusion Matrix [[TN FP][FN TP]]:\n{cm}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to PhiUSIIL CSV file")
    parser.add_argument("--sample", type=int, default=None, help="Optional row sample size")
    parser.add_argument("--url-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} ...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(df):,} rows.")

    url_col = args.url_col or detect_column(df, URL_COL_CANDIDATES, "url")
    label_col = args.label_col or detect_column(df, LABEL_COL_CANDIDATES, "label")
    print(f"Using URL column='{url_col}', label column='{label_col}'")

    df = df.dropna(subset=[url_col, label_col]).reset_index(drop=True)

    y = normalize_label(df[label_col]).astype(int)
    X = build_feature_dataframe(df[url_col].astype(str))

    print(f"\nClass balance -> phishing: {(y == 1).sum():,}, legitimate: {(y == 0).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    results.append((evaluate("Logistic Regression", lr, X_test_scaled, y_test), lr))

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)  # tree models don't need scaling
    results.append((evaluate("Random Forest", rf, X_test, y_test), rf))

    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    results.append((evaluate("XGBoost", xgb, X_test, y_test), xgb))

    # --- Comparison table ---
    comparison_df = pd.DataFrame([r[0] for r in results]).set_index("model")
    print("\n=== Model Comparison ===")
    print(comparison_df.round(4).to_string())

    os.makedirs(args.out_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(args.out_dir, "model_comparison.csv"))

    # --- Select best model by F1 ---
    best_name, best_model = max(
        zip([r[0]["model"] for r in results], [r[1] for r in results]),
        key=lambda pair: comparison_df.loc[pair[0], "f1"],
    )
    print(f"\nBest model selected: {best_name}")

    needs_scaling = best_name == "Logistic Regression"

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler,
        "needs_scaling": needs_scaling,
        "feature_names": FEATURE_NAMES,
    }
    out_path = os.path.join(args.out_dir, "best_model.pkl")
    joblib.dump(artifact, out_path)
    print(f"Saved best model artifact to {out_path}")


if __name__ == "__main__":
    main()
