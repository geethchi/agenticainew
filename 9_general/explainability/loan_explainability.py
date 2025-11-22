"""
Loan Explainability pipeline (full example)
- Loads loan_approval.csv
- Preprocesses data (drops name/city for modeling)
- Trains a RandomForestClassifier
- Computes global and local SHAP values (TreeExplainer)
- Finds simple counterfactual suggestions (greedy search)
- Generates a human-friendly explanation (template or LLM)
- CLI entrypoints: --row or --name
"""

import os
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import shap
import json

# Optional LLM explanation
from dotenv import load_dotenv
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
use_llm = OPENAI_API_KEY is not None

if use_llm:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
else:
    llm = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("loan-xai")

# ---------------------
# Configuration
# ---------------------
DATA_PATH = r"c:/code/agenticai/9_general/explainability/loan_approval.csv"
MODEL_PATH = r"c:/code/agenticai/9_general/explainability/loan_rf_model.joblib"
PIPE_PATH = r"c:/code/agenticai/9_general/explainability/preproc_pipeline.joblib"
SHAP_EXPLAINER_PATH = r"c:/code/agenticai/9_general/explainability/shap_explainer.joblib"

# Numeric features we'll use for modeling
NUMERIC_FEATURES = ["income", "credit_score", "loan_amount", "years_employed", "points"]
LABEL_COL = "loan_approved"


# ---------------------
# Utilities
# ---------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize label to binary
    if df[LABEL_COL].dtype == object:
        df[LABEL_COL] = df[LABEL_COL].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    df[LABEL_COL] = df[LABEL_COL].fillna(0).astype(int)
    return df


def build_pipeline():
    # For this dataset we only take numeric columns for modeling.
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    preproc = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES)
    ], remainder='drop')
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    pipe = Pipeline(steps=[("preproc", preproc), ("clf", clf)])
    return pipe


# ---------------------
# Training / Fit
# ---------------------
def train_and_save(df: pd.DataFrame, force_retrain: bool = False) -> Tuple[Pipeline, Dict[str, Any]]:
    if os.path.exists(MODEL_PATH) and os.path.exists(PIPE_PATH) and not force_retrain:
        logger.info("Loading existing model and pipeline...")
        pipe = joblib.load(PIPE_PATH)
        return pipe, {}

    logger.info("Preparing training data...")
    X = df[NUMERIC_FEATURES].copy()
    y = df[LABEL_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()
    logger.info("Training model...")
    pipe.fit(X_train, y_train)

    # Evaluate quickly
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    logger.info("Classification report:\n" + classification_report(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"ROC AUC: {auc:.3f}")
    except Exception:
        pass

    # Save pipeline
    joblib.dump(pipe, PIPE_PATH)
    logger.info(f"Saved pipeline -> {PIPE_PATH}")

    # ---------------------
    # MODIFIED FOR CORRECT SHAP FEATURE NAMES
    # ---------------------
    logger.info("Building SHAP explainer (full pipeline explainer)...")

    try:
        explainer = shap.Explainer(pipe, X_train)
        joblib.dump(explainer, SHAP_EXPLAINER_PATH)
        logger.info(f"Saved SHAP explainer -> {SHAP_EXPLAINER_PATH}")
    except Exception as e:
        logger.warning("Failed to build/save SHAP explainer: " + str(e))
        explainer = None

    return pipe, {"explainer": explainer}


# ---------------------
# SHAP helpers
# ---------------------
def load_explainer(pipe: Pipeline) -> Optional[Any]:
    if os.path.exists(SHAP_EXPLAINER_PATH):
        try:
            return joblib.load(SHAP_EXPLAINER_PATH)
        except Exception as e:
            logger.warning("Could not load saved explainer: " + str(e))
    return None


def explain_instance_shap(pipe: Pipeline, explainer, instance: pd.DataFrame) -> Dict[str, Any]:
    """Return SHAP values and expected_value for the instance. instance must be a one-row DF of numeric features."""
    model = pipe.named_steps["clf"]
    preproc = pipe.named_steps["preproc"]

    x_trans = preproc.transform(instance[NUMERIC_FEATURES])
    
    # Get individual tree predictions
    tree_predictions = [tree.predict(x_trans)[0] for tree in model.estimators_]
    n_approve = sum(tree_predictions)
    n_reject = len(tree_predictions) - n_approve
    logger.info(f"Tree votes: {n_approve} approve, {n_reject} reject")
    
    
    proba = model.predict_proba(x_trans)[:, 1][0]
    pred = model.predict(x_trans)[0]

    out = {"probability": float(proba), "prediction": int(pred)}

    if explainer is not None:
        try:
            shap_values = explainer(instance)

            # Extract positive-class SHAP values if classification
            if hasattr(shap_values, "values"):
                row_vals = shap_values.values[0]
                
                # FIXED: Handle multi-class output (shape: n_features, n_classes)
                if row_vals.ndim == 2:
                    # Take the positive class (index 1) values
                    row_vals = row_vals[:, 1]
                
                feature_names = shap_values.feature_names

                # Map back to a dictionary of all 5 features
                out["shap_values"] = {feature_names[i]: float(row_vals[i]) for i in range(len(feature_names))}
                
                # Handle base_values similarly
                base_val = shap_values.base_values[0]
                if isinstance(base_val, np.ndarray) and base_val.ndim == 1:
                    base_val = base_val[1]  # positive class
                out["expected_value"] = float(base_val)
            else:
                out["shap_values"] = None
        except Exception as e:
            logger.warning("Error computing shap values: " + str(e))
            out["shap_values"] = None

    return out


# ---------------------
# Counterfactual search (simple, greedy)
# ---------------------
def find_counterfactual(pipe: Pipeline, instance: pd.Series, target_label: int = 1,
                        max_steps: int = 5) -> Optional[Dict[str, Any]]:
    """
    A simple greedy counterfactual search:
    For numeric features: try monotonic changes in order of importance heuristics:
    1) credit_score (+)
    2) income (+)
    3) loan_amount (-)
    4) years_employed (+)
    5) points (+)
    We try incremental adjustments until prediction flips or max_steps exhausted.
    Returns the minimal adjustments that flip the model or None.
    """
    model = pipe.named_steps["clf"]
    preproc = pipe.named_steps["preproc"]

    base = instance[NUMERIC_FEATURES].astype(float).to_dict()
    curr = base.copy()
    step_counters = 0

    adjustments = {
        "credit_score": [10, 25, 50, 100],
        "income": [5000, 10000, 20000, 50000],
        "loan_amount": [-1000, -5000, -10000, -20000],
        "years_employed": [1, 2, 5],
        "points": [5, 10, 20]
    }

    order = ["credit_score", "income", "loan_amount", "years_employed", "points"]

    def predict_from_vals(vals: Dict[str, float]) -> Tuple[int, float]:
        df = pd.DataFrame([vals])
        x_trans = preproc.transform(df)
        p = model.predict_proba(x_trans)[:, 1][0]
        label = int(model.predict(x_trans)[0])
        return label, float(p)

    label0, prob0 = predict_from_vals(curr)
    if label0 == target_label:
        return {"flipped": False, "initial_label": label0, "initial_prob": prob0, "suggestions": []}

    from itertools import combinations, product
    for depth in range(1, max_steps + 1):
        for features in combinations(order, depth):
            grids = [adjustments[f] for f in features]
            for values in product(*grids):
                candidate = curr.copy()
                for f, delta in zip(features, values):
                    candidate[f] = float(candidate[f]) + float(delta)
                    if candidate[f] < 0:
                        candidate[f] = 0.0
                label_c, prob_c = predict_from_vals(candidate)
                if label_c == target_label:
                    deltas = {f: candidate[f] - base[f] for f in features}
                    return {
                        "flipped": True,
                        "initial_label": label0,
                        "initial_prob": prob0,
                        "final_label": label_c,
                        "final_prob": prob_c,
                        "suggested_changes": deltas
                    }
    return None


# ---------------------
# Natural-language explanation
# ---------------------
def build_nl_explanation(applicant_row: pd.Series, shap_info: Dict[str, Any], cf: Optional[Dict[str, Any]]) -> str:
    name = applicant_row.get("name", "Applicant")
    pred = shap_info.get("prediction")
    prob = shap_info.get("probability")
    svals = shap_info.get("shap_values") or {}

    if svals:
        sorted_feats = sorted(svals.items(), key=lambda kv: -abs(kv[1]))
        top = sorted_feats[:3]
        contributions = []
        for f, v in top:
            direction = "increased approval" if v > 0 else "reduced approval"
            contributions.append(f"{f} ({'+' if v>0 else ''}{v:.2f}) -> {direction}")
        contributions_text = "; ".join(contributions)
    else:
        contributions_text = "Model feature contributions unavailable."

    cf_text = ""
    if cf:
        if cf.get("flipped"):
            changes = cf.get("suggested_changes", {})
            change_texts = []
            for k, dv in changes.items():
                sign = "+" if dv >= 0 else ""
                change_texts.append(f"{k}: {sign}{dv:.0f}")
            cf_text = "Suggested minimal changes to flip outcome: " + ", ".join(change_texts)
        else:
            cf_text = "No counterfactual changes needed (already approved)."

    base_text = (
        f"{name}'s predicted probability of loan approval is {prob*100:.1f}% (label={pred}).\n"
        f"Top contributing features: {contributions_text}.\n"
        f"{cf_text}"
    )

    if use_llm and llm is not None:
        prompt = (
            "Rewrite the following technical explanation into a clear, empathetic, actionable note for a loan applicant:\n\n"
            f"Technical explanation:\n{base_text}\n\nKeep it short (3-4 sentences), polite, and provide one clear suggestion."
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            return resp.content
        except Exception as e:
            logger.warning("LLM rewrite failed, falling back to template: " + str(e))
            return base_text
    else:
        rec = ""
        if cf and cf.get("flipped"):
            changes = cf.get("suggested_changes", {})
            if changes:
                k, dv = max(changes.items(), key=lambda kv: abs(kv[1]))
                if k == "credit_score":
                    rec = f"Try improving your credit score by about {int(abs(dv))} points."
                elif k == "income":
                    rec = f"Consider increasing monthly income / adding a co-applicant to raise income by ~{int(abs(dv))}."
                elif k == "loan_amount":
                    rec = f"Consider requesting a loan about {int(abs(dv))} lower."
                else:
                    rec = f"Consider increasing {k} by {int(abs(dv))}."
        else:
            rec = "No straightforward improvement suggestion available."

        return base_text + "\nRecommendation: " + rec


# ---------------------
# Report assembly
# ---------------------
def explain_applicant(pipe: Pipeline, explainer, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    row = df.iloc[idx]
    model_input = row[NUMERIC_FEATURES].to_frame().T
    shap_info = explain_instance_shap(pipe, explainer, model_input)
    cf = find_counterfactual(pipe, row, target_label=1, max_steps=4)
    nl = build_nl_explanation(row, shap_info, cf)

    report = {
        "name": row.get("name"),
        "input": row[NUMERIC_FEATURES].to_dict(),
        "prediction": shap_info.get("prediction"),
        "probability": shap_info.get("probability"),
        "shap_values": shap_info.get("shap_values"),
        "expected_value": shap_info.get("expected_value"),  # ADD THIS LINE
        "counterfactual": cf,
        "natural_language": nl
    }
    return report


# ---------------------
# CLI / main
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Loan explainability demo")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--row", type=int, help="Row index (0-based) to explain")
    group.add_argument("--name", type=str, help="Applicant name to explain (exact match)")
    parser.add_argument("--retrain", action="store_true", help="Retrain model even if saved model exists")
    args = parser.parse_args()

    df = load_data(DATA_PATH)
    logger.info(f"Loaded data with {len(df)} rows")

    pipe, meta = train_and_save(df, force_retrain=args.retrain)

    explainer = load_explainer(pipe)
    if explainer is None and meta.get("explainer"):
        explainer = meta["explainer"]

    if args.row is not None:
        idx = args.row
        if idx < 0 or idx >= len(df):
            logger.error("Row index out of bounds")
            return
    elif args.name:
        matches = df.index[df["name"] == args.name].tolist()
        if not matches:
            logger.error(f"No applicant with name {args.name}")
            return
        idx = matches[0]
    else:
        print("Available applicants (first 10):")
        print(df[["name", "income", "credit_score", "loan_amount", LABEL_COL]].head(10).to_string(index=True))
        sel = input("Enter row index to explain: ").strip()
        idx = int(sel)

    report = explain_applicant(pipe, explainer, df, idx)

    print("\n=== EXPLAINABILITY REPORT ===")
    
    expected_val = report.get('expected_value')
    if expected_val is not None:
        print(f"\nSHAP baseline (expected value): {expected_val:.4f}")
    else:
        print(f"\nSHAP baseline (expected value): N/A")
    
    print("(This is the model's average prediction before considering features)")
    print(f"\nName: {report['name']}")
    print(f"Prediction (label): {report['prediction']}  |  Probability: {report['probability']*100:.2f}%")
    print("\nInput features:")
    for k, v in report["input"].items():
        print(f"  - {k}: {v}")
    print("\nSHAP feature contributions (positive -> increases approval):")
    if report["shap_values"]:
        for k, v in report["shap_values"].items():
            print(f"  - {k}: {v:.4f}")
    else:
        print("  - SHAP values not available")

    print("\nCounterfactual suggestion (simple greedy):")
    if report["counterfactual"] is None:
        print("  - No simple counterfactual found within search limits.")
    else:
        print(json.dumps(report["counterfactual"], indent=2))

    print("\nNatural-language explanation:")
    print(report["natural_language"])
    print("\n=============================\n")

    outpath = f"c://code//agenticai//9_general//explainability//explain_report_row{idx}.json"
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote report to {outpath}")


if __name__ == "__main__":
    main()
