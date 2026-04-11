import os
import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class MLDetector:

    def __init__(self, lang: str = "c", threshold: float = 0.5,
                 model_dir: Optional[str] = None):
        self.lang = lang
        self.threshold = threshold
        self.model_dir = model_dir or MODELS_DIR
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}

        self._load()

    def _load(self):
        prefix = self.lang
        model_path = os.path.join(self.model_dir, f"{prefix}_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{prefix}_scaler.pkl")
        meta_path = os.path.join(self.model_dir, f"{prefix}_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run first:\n"
                f"  python extract_features_csv.py --lang {self.lang}\n"
                f"  python train_model.py --lang {self.lang}"
            )

        self.model = joblib.load(model_path)

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("feature_names", [])
        else:
            self.feature_names = []

    def _extract_features(self, code: str) -> Dict:
        if self.lang == "c":
            from .c_features import extract_c_features
            return extract_c_features(code)
        else:
            from .cpp_features import extract_cpp_ast_features
            return extract_cpp_ast_features(code)

    def _features_to_vector(self, features: Dict):
        if not self.feature_names:
            numeric_feats = {
                k: v for k, v in features.items()
                if isinstance(v, (int, float, bool, np.integer, np.floating))
            }
            self.feature_names = sorted(numeric_feats.keys())

        values = {}
        for name in self.feature_names:
            val = features.get(name, 0)
            if isinstance(val, bool):
                val = int(val)
            elif not isinstance(val, (int, float, np.integer, np.floating)):
                val = 0
            values[name] = float(val)

        df = pd.DataFrame([values], columns=self.feature_names)
        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        return df

    def analyze(self, code: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        thr = threshold if threshold is not None else self.threshold

        features = self._extract_features(code)
        X = self._features_to_vector(features)

        try:
            probabilities = self.model.predict_proba(X)
            p_ai = float(probabilities[0][1])
        except Exception:
            p_ai = 0.0

        p_ai = max(0.0, min(1.0, p_ai))
        score = int(round(p_ai * 100))
        flag = p_ai >= thr

        signals = self._get_top_signals(features, X)

        details = {
            **features,
            "ml_model_type": self.metadata.get("model_type", "unknown"),
            "ml_n_features": len(self.feature_names),
            "ml_cv_accuracy": self.metadata.get("cv_metrics", {}).get("accuracy_mean", 0),
            "ml_cv_f1": self.metadata.get("cv_metrics", {}).get("f1_mean", 0),
        }

        return {
            "p_ai": round(p_ai, 4),
            "score": score,
            "flag": flag,
            "signals": signals,
            "details": details,
        }

    def _get_top_signals(self, features: Dict, X) -> List[str]:
        signals = []

        if not hasattr(self.model, "feature_importances_"):
            return ["ML Model Prediction"]

        importances = self.model.feature_importances_
        if len(importances) != len(self.feature_names):
            return ["ML Model Prediction"]

        values = X.values.flatten() if hasattr(X, 'values') else np.asarray(X).flatten()
        contributions = importances * np.abs(values)

        sorted_idx = np.argsort(contributions)[::-1]

        for idx in sorted_idx[:6]:
            name = self.feature_names[idx]
            val = values[idx]
            imp = importances[idx]
            if imp > 0.005:
                signals.append(f"{name} = {val:.2f} (imp: {imp:.3f})")

        if not signals:
            signals = ["No strong signals"]

        return signals

    def detect(self, code: str) -> Dict[str, Any]:
        result = self.analyze(code)
        return {
            "is_ai_suspect": result["flag"],
            "confidence_score": result["score"],
            "reasons": result["signals"],
            "details": result["details"],
        }

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "language": self.lang,
            "model_type": self.metadata.get("model_type", "unknown"),
            "n_features": len(self.feature_names),
            "n_training_samples": self.metadata.get("n_samples", 0),
            "cv_accuracy": self.metadata.get("cv_metrics", {}).get("accuracy_mean", 0),
            "cv_f1": self.metadata.get("cv_metrics", {}).get("f1_mean", 0),
            "cv_roc_auc": self.metadata.get("cv_metrics", {}).get("roc_auc_mean", 0),
        }
