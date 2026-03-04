# ai_detector.py
import math
from typing import Dict, Any, List, Tuple

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_LM = True
except ImportError:
    HAS_LM = False

from .features import extract_features


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)

def _sig_indent_consistency(f: dict) -> float:
    if f.get('indent_inconsistency_count', 0) == 0:
        if f.get('total_lines', 0) > 10:
            return 1.0
        else:
            return 0.4
    ratio = f.get('indent_inconsistency_ratio', 0.0)
    return max(0.0, 1.0 - ratio * 10)


def _sig_operator_spacing(f: dict) -> float:
    rate = f.get('operator_spacing_rate', 0.0)
    if f.get('operator_count', 0) < 3:
        return 0.0
    if rate > 0.95:
        return 1.0
    elif rate > 0.85:
        return 0.6
    return 0.0


def _sig_comma_spacing(f: dict) -> float:
    if f.get('comma_count', 0) < 3:
        return 0.0
    rate = f.get('comma_space_rate', 0.0)
    return min(rate, 1.0) if rate > 0.9 else rate * 0.5


def _sig_docstrings(f: dict) -> float:
    func_count = f.get('function_count', 0)
    if func_count == 0:
        return 0.2 if f.get('docstring_present', False) else 0.0
    ratio = f.get('func_docstring_ratio', 0.0)
    if ratio >= 0.8:
        return 1.0
    elif ratio >= 0.5:
        return 0.6
    return 0.1


def _sig_tutorial_markers(f: dict) -> float:
    count = f.get('tutorial_markers_count', 0)
    if count >= 3:
        return 1.0
    elif count >= 1:
        return 0.5
    return 0.0


def _sig_comment_quality(f: dict) -> float:
    ratio = f.get('comment_line_ratio', 0.0)
    std = f.get('std_comment_len', 0.0)
    avg_len = f.get('avg_comment_len', 0.0)
    score = 0.0
    if 0.1 <= ratio <= 0.4:
        score += 0.4
    if 10 <= avg_len <= 60 and std < 25:
        score += 0.4
    if f.get('comment_count', 0) > 0 and std < 15:
        score += 0.2
    return min(score, 1.0)


def _sig_pythonic(f: dict) -> float:
    ratio = f.get('pythonic_construct_ratio', 0.0)
    count = f.get('pythonic_construct_count', 0)
    if count >= 5 and ratio > 0.05:
        return 1.0
    elif count >= 2:
        return 0.5
    return 0.0


def _sig_line_uniformity(f: dict) -> float:
    std = f.get('line_len_std', 999.0)
    mean = f.get('line_len_mean', 0.0)
    if mean < 5:
        return 0.0
    cv = std / mean if mean > 0 else 1.0
    if cv < 0.3:
        return 1.0
    elif cv < 0.5:
        return 0.5
    return 0.0


def _sig_pep8_compliance(f: dict) -> float:
    long_ratio = f.get('pep8_long_line_ratio', 1.0)
    trailing = f.get('trailing_whitespace_ratio', 1.0)
    score = 0.0
    if long_ratio < 0.05:
        score += 0.5
    if trailing < 0.02:
        score += 0.5
    return score


def _sig_naming_consistency(f: dict) -> float:
    if f.get('function_count', 0) == 0:
        return 0.3
    consistent = f.get('naming_convention_consistent', False)
    snake_ratio = f.get('naming_snake_case_ratio', 0.0)
    if consistent and snake_ratio >= 0.9:
        return 1.0
    elif consistent:
        return 0.6
    return 0.0


def _sig_type_hints(f: dict) -> float:
    ratio = f.get('type_hint_ratio', 0.0)
    if ratio >= 2.0:
        return 1.0
    elif ratio >= 1.0:
        return 0.6
    elif ratio > 0:
        return 0.3
    return 0.0


def _sig_error_handling(f: dict) -> float:
    rate = f.get('try_except_rate', 0.0)
    count = f.get('try_except_count', 0)
    if count >= 2 and rate > 0.02:
        return 0.8
    elif count >= 1:
        return 0.4
    return 0.0


def _sig_high_maintainability(f: dict) -> float:
    mi = f.get('maintainability_index', 0.0)
    if mi > 70:
        return 1.0
    elif mi > 50:
        return 0.5
    elif mi > 0:
        return 0.2
    return 0.0


def _sig_low_complexity(f: dict) -> float:
    cc = f.get('cyclomatic_mean', 0.0)
    if cc == 0:
        return 0.0
    if cc < 3:
        return 0.8
    elif cc < 5:
        return 0.4
    return 0.0


def _sig_blank_line_structure(f: dict) -> float:
    ratio = f.get('blank_line_ratio', 0.0)
    if 0.15 <= ratio <= 0.35:
        return 0.7
    elif 0.10 <= ratio <= 0.40:
        return 0.3
    return 0.0

SIGNALS: List[Tuple[str, Any, float]] = [
    ("Perfectly consistent indentation",  _sig_indent_consistency,  2.0),
    ("Consistent operator spacing",       _sig_operator_spacing,    1.5),
    ("Consistent comma spacing",          _sig_comma_spacing,       1.0),
    ("Docstrings present for functions",  _sig_docstrings,          2.0),
    ("Tutorial-style markers (Args/Returns/Example)", _sig_tutorial_markers, 2.5),
    ("High-quality, uniform comments",    _sig_comment_quality,     1.5),
    ("Pythonic constructs used",          _sig_pythonic,            1.5),
    ("Uniform line lengths",              _sig_line_uniformity,     1.5),
    ("PEP-8 compliance",                  _sig_pep8_compliance,     1.0),
    ("Consistent naming convention",      _sig_naming_consistency,  1.5),
    ("Type hints present",                _sig_type_hints,          1.5),
    ("Error handling patterns",           _sig_error_handling,      1.0),
    ("High maintainability index",        _sig_high_maintainability,1.5),
    ("Low cyclomatic complexity",         _sig_low_complexity,      1.0),
    ("Structured blank-line usage",       _sig_blank_line_structure,1.0),
]

_MAX_WEIGHT = sum(w for _, _, w in SIGNALS)


class AIDetector:

    def __init__(self, threshold: float = 0.60, use_perplexity: bool = True):
        self.threshold = threshold
        self.use_perplexity = use_perplexity and HAS_LM
        self.tokenizer = None
        self.model = None
        self.device = None

        if self.use_perplexity:
            self._load_lm()

    def _load_lm(self) -> None:
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[AIDetector] LM load failed (perplexity disabled): {e}")
            self.model = None
            self.use_perplexity = False

    def _calculate_perplexity(self, code: str) -> Dict[str, float]:
        if not self.use_perplexity or self.model is None or self.tokenizer is None:
            return {"perplexity": 0.0, "burstiness": 0.0}

        try:
            encodings = self.tokenizer(code, return_tensors="pt")
            max_length = self.model.config.n_positions
            stride = 512
            seq_len = encodings.input_ids.size(1)

            nlls = []
            prev_end = 0
            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                trg_len = end - prev_end
                input_ids = encodings.input_ids[:, begin:end].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    loss = self.model(input_ids, labels=target_ids).loss
                nlls.append(loss)
                prev_end = end
                if end == seq_len:
                    break

            if not nlls:
                return {"perplexity": 0.0, "burstiness": 0.0}

            ppl = torch.exp(torch.stack(nlls).mean()).item()
            burst = torch.stack(nlls).std().item() if len(nlls) > 1 else 0.0
            return {"perplexity": ppl, "burstiness": burst}
        except Exception:
            return {"perplexity": 0.0, "burstiness": 0.0}

    def analyze(self, code: str, threshold: float | None = None) -> Dict[str, Any]:
        thr = threshold if threshold is not None else self.threshold

        features = extract_features(code)

        signal_scores: List[Tuple[str, float, float]] = []  # (label, strength, weight)
        for label, fn, weight in SIGNALS:
            try:
                strength = float(fn(features))
            except Exception:
                strength = 0.0
            signal_scores.append((label, strength, weight))

        weighted_sum = sum(s * w for _, s, w in signal_scores)
        normalized = (weighted_sum / _MAX_WEIGHT) * 6.0 - 3.0
        p_ai = _sigmoid(normalized)

        ppl_data = self._calculate_perplexity(code)
        ppl_adj = 0.0
        if ppl_data["perplexity"] > 0:
            if ppl_data["perplexity"] < 20:
                ppl_adj = 0.08
            elif ppl_data["perplexity"] < 50:
                ppl_adj = 0.04
            elif ppl_data["perplexity"] > 200:
                ppl_adj = -0.05

            if ppl_data["burstiness"] < 1.5 and ppl_data["burstiness"] > 0:
                ppl_adj += 0.02

        p_ai = max(0.0, min(1.0, p_ai + ppl_adj))
        score = int(round(p_ai * 100))

        ranked = sorted(signal_scores, key=lambda x: x[1] * x[2], reverse=True)
        top_signals = [
            f"{label} ({strength:.0%})"
            for label, strength, _ in ranked
            if strength > 0.2
        ][:5]

        if not top_signals:
            top_signals = ["No strong AI signals detected"]

        details = {
            **features,
            **ppl_data,
            "signal_breakdown": {
                label: round(strength, 3) for label, strength, _ in signal_scores
            },
        }

        return {
            "p_ai": round(p_ai, 4),
            "score": score,
            "flag": p_ai >= thr,
            "signals": top_signals,
            "details": details,
        }

    def detect_ai_generated(self, code: str) -> Dict[str, Any]:
        result = self.analyze(code)
        return {
            "is_ai_suspect": result["flag"],
            "confidence_score": result["score"],
            "reasons": result["signals"],
            "details": result["details"],
        }
