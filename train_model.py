import os
import sys
import json
import argparse
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_DROP_COLS = {"_file", "_path", "_label", "_chars", "_error", "header_list"}

_BOOL_FEATURES = {
    "mixed_indent", "naming_uniform",
    "bits_stdc_present", "using_namespace_std", "ios_sync_present",
    "cin_tie_present", "freopen_present", "void_main",
    "error_handling_present", "naming_style_uniform",
}


def _load_and_prepare(csv_path: str):
    print(f"  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Raw rows: {len(df):,}")

    df = df.dropna(subset=["_label"])
    df["_label"] = df["_label"].astype(int)

    y = df["_label"].values

    feature_cols = [c for c in df.columns if c not in _DROP_COLS]
    X = df[feature_cols].copy()

    for col in X.columns:
        if col in _BOOL_FEATURES or X[col].dtype == "bool":
            X[col] = X[col].astype(int)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    dropped = set(X.columns) - set(numeric_cols)
    if dropped:
        print(f"  Dropped non-numeric columns: {dropped}")
    X = X[numeric_cols]

    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    print(f"  Features: {X.shape[1]}")
    print(f"  Samples:  {len(y):,}  (AI={sum(y==1):,}, Human={sum(y==0):,})")

    return X, y, numeric_cols


def _train_model(X, y, feature_names, model_type: str = "xgb") -> Dict[str, Any]:
    n_ai = sum(y == 1)
    n_human = sum(y == 0)
    total = len(y)
    w_ai = total / (2 * max(n_ai, 1))
    w_human = total / (2 * max(n_human, 1))
    sample_weights = np.where(y == 1, w_ai, w_human)

    print(f"\n  Class weights — AI: {w_ai:.2f}, Human: {w_human:.2f}")

    if model_type == "xgb":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                scale_pos_weight=n_human / max(n_ai, 1),
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            print("  Model: XGBoost (XGBClassifier)")
        except ImportError:
            print("  [WARN] xgboost not installed, falling back to GradientBoosting")
            model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )
            model_type = "gb"
            print("  Model: GradientBoostingClassifier (sklearn)")
    else:
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        print("  Model: Random Forest")

    print("\n  Running 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")
    cv_roc_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"  CV Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  CV F1:        {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  CV ROC-AUC:   {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")

    print("\n  Training final model on full dataset...")
    if model_type in ("xgb",):
        model.fit(X, y, sample_weight=sample_weights)
    elif model_type == "gb":
        model.fit(X, y, sample_weight=sample_weights)
    else:
        model.fit(X, y)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("\n  === Full Dataset Metrics ===")
    print(f"  Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"  Precision: {precision_score(y, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y, y_prob):.4f}")

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    print("\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=["Human", "AI"]))

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("  Top 20 Important Features:")
        for rank, idx in enumerate(sorted_idx[:20], 1):
            print(f"    {rank:2d}. {feature_names[idx]:40s}  {importances[idx]:.4f}")

    return {
        "model": model,
        "model_type": model_type,
        "feature_names": list(feature_names),
        "cv_metrics": {
            "accuracy_mean": float(cv_accuracy.mean()),
            "accuracy_std": float(cv_accuracy.std()),
            "f1_mean": float(cv_f1.mean()),
            "f1_std": float(cv_f1.std()),
            "roc_auc_mean": float(cv_roc_auc.mean()),
            "roc_auc_std": float(cv_roc_auc.std()),
        },
        "full_metrics": {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_prob)),
        },
        "n_samples": len(y),
        "n_ai": int(sum(y == 1)),
        "n_human": int(sum(y == 0)),
    }


def _run_pipeline(lang: str, model_type: str = "xgb"):
    config = {
        "c": {"csv": os.path.join(BASE_DIR, "features_c.csv"), "prefix": "c"},
        "cpp": {"csv": os.path.join(BASE_DIR, "features_cpp.csv"), "prefix": "cpp"},
    }

    cfg = config[lang]
    csv_path = cfg["csv"]
    prefix = cfg["prefix"]

    print(f"\n{'='*60}")
    print(f" Training ML Model for: {lang.upper()}")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"  [ERROR] Feature CSV not found: {csv_path}")
        print(f"  Run first: python extract_features_csv.py --lang {lang}")
        return

    X, y, feature_names = _load_and_prepare(csv_path)

    if len(y) < 10:
        print("  [ERROR] Not enough samples to train. Need at least 10.")
        return

    results = _train_model(X, y, feature_names, model_type)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{prefix}_model.pkl")
    joblib.dump(results["model"], model_path)
    print(f"\n  Model saved: {model_path}")

    scaler = StandardScaler()
    scaler.fit(X)
    scaler_path = os.path.join(MODELS_DIR, f"{prefix}_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    meta = {
        "language": lang,
        "model_type": results["model_type"],
        "feature_names": results["feature_names"],
        "n_features": len(results["feature_names"]),
        "n_samples": results["n_samples"],
        "n_ai": results["n_ai"],
        "n_human": results["n_human"],
        "cv_metrics": results["cv_metrics"],
        "full_metrics": results["full_metrics"],
    }
    meta_path = os.path.join(MODELS_DIR, f"{prefix}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ML model for AI code detection using extracted features."
    )
    parser.add_argument(
        "--lang",
        choices=["c", "cpp", "both"],
        default="both",
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "rf"],
        default="xgb",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" ML Training Pipeline")
    print(f" Language: {args.lang.upper()}")
    print(f" Model:    {args.model.upper()}")
    print("=" * 60)

    if args.lang in ("c", "both"):
        _run_pipeline("c", args.model)

    if args.lang in ("cpp", "both"):
        _run_pipeline("cpp", args.model)

    print("\n Training complete!")


if __name__ == "__main__":
    main()
