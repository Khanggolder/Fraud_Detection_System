import math
import logging
from typing import Dict, Any, List, Tuple

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_LM = True
except ImportError:
    HAS_LM = False

from .features import extract_features

logger = logging.getLogger(__name__)


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


def _sig_comment_style(f: dict) -> float:
    perf_ratio = f.get('perfect_comment_ratio', 0.0)
    std_len = f.get('std_comment_len', 0.0)
    count = f.get('comment_count', 0)
    score = 0.0
    if count >= 5:
        if std_len < 5.0:
            score += 0.5
        elif std_len < 10.0:
            score += 0.3
        if perf_ratio > 0.8:
            score += 0.5
        elif perf_ratio > 0.5:
            score += 0.2
    elif count >= 3:
        if std_len < 5.0:
            score += 0.4
        if perf_ratio > 0.8:
            score += 0.3
    return min(score, 1.0)


def _sig_human_artifacts(f: dict) -> float:
    score = 0.0
    imb = f.get('imbalanced_spacing_count', 0)
    if imb >= 3:
        score -= 1.0
    elif imb >= 1:
        score -= 0.5
    rl = f.get('range_len_count', 0)
    if rl >= 2:
        score -= 0.8
    elif rl >= 1:
        score -= 0.4
    if f.get('dead_code_count', 0) > 0:
        score -= 0.5
    mcb = f.get('max_consecutive_blank_lines', 0)
    if mcb > 3:
        score -= 0.5
    elif mcb > 2:
        score -= 0.3
    if f.get('redundant_bool_count', 0) > 0:
        score -= 0.3
    return score


def _sig_clean_code(f: dict) -> float:
    score = 0.0
    if f.get('imbalanced_spacing_count', 0) == 0:
        score += 0.3
    if f.get('range_len_count', 0) == 0:
        score += 0.2
    if f.get('dead_code_count', 0) == 0:
        score += 0.2
    if f.get('redundant_bool_count', 0) == 0:
        score += 0.15
    if f.get('max_consecutive_blank_lines', 0) <= 2:
        score += 0.15
    if f.get('indent_inconsistency_count', 0) == 0 and f.get('total_lines', 0) > 5:
        score += 0.3
    if f.get('trailing_whitespace_ratio', 1.0) < 0.01:
        score += 0.2
    if f.get('pep8_long_line_ratio', 1.0) < 0.02:
        score += 0.2
    if f.get('naming_convention_consistent', False) and f.get('function_count', 0) > 0:
        score += 0.25
    return score


def _sig_over_perfection(f: dict) -> float:
    score = 0.0
    if f.get('indent_inconsistency_count', 0) == 0 and f.get('total_lines', 0) > 5:
        score += 0.3
    if f.get('naming_convention_consistent', False) and f.get('function_count', 0) > 0:
        score += 0.3
    if f.get('pep8_long_line_ratio', 1.0) < 0.02 and f.get('trailing_whitespace_ratio', 1.0) < 0.02:
        score += 0.2
    if f.get('imbalanced_spacing_count', 0) == 0 and f.get('operator_count', 0) >= 3:
        score += 0.2
    return min(score, 1.0)


SIGNALS: List[Tuple[str, Any, float]] = [
    ("Perfectly consistent indentation",  _sig_indent_consistency,  1.5),
    ("Consistent operator spacing",       _sig_operator_spacing,    1.2),
    ("Consistent comma spacing",          _sig_comma_spacing,       0.8),
    ("Docstrings present for functions",  _sig_docstrings,          1.8),
    ("Tutorial-style markers (Args/Returns/Example)", _sig_tutorial_markers, 2.5),
    ("High-quality, uniform comments",    _sig_comment_quality,     1.2),
    ("Pythonic constructs used",          _sig_pythonic,            1.2),
    ("Uniform line lengths",              _sig_line_uniformity,     1.2),
    ("PEP-8 compliance",                  _sig_pep8_compliance,     0.8),
    ("Consistent naming convention",      _sig_naming_consistency,  1.2),
    ("Type hints present",                _sig_type_hints,          1.2),
    ("Error handling patterns",           _sig_error_handling,      0.8),
    ("High maintainability index",        _sig_high_maintainability,1.2),
    ("Low cyclomatic complexity",         _sig_low_complexity,      0.8),
    ("Structured blank-line usage",       _sig_blank_line_structure,0.8),
    ("Comment Style Analysis",            _sig_comment_style,       3.0),
    ("Human Artifacts Penalty",           _sig_human_artifacts,     3.5),
    ("Clean Code Detection",              _sig_clean_code,          4.0),
    ("Over-perfection Detection",         _sig_over_perfection,     2.5),
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
        model_name = "Qwen/Qwen2.5-Coder-0.5B"
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading code model: %s on %s", model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.to(self.device)
            self.model.eval()
            logger.info("Code model loaded successfully")
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Failed to load code model %s: %s", model_name, e)
            self.model = None
            self.use_perplexity = False

    def _calculate_perplexity(self, code: str) -> Dict[str, float]:
        default = {"perplexity": 0.0, "burstiness": 0.0, "mean_entropy": 0.0}
        if not self.use_perplexity or self.model is None or self.tokenizer is None:
            return default

        try:
            encodings = self.tokenizer(
                code, return_tensors="pt", truncation=True, max_length=2048,
            )
            input_ids = encodings.input_ids.to(self.device)
            seq_len = input_ids.size(1)

            if seq_len < 2:
                return default

            max_length = min(
                getattr(self.model.config, 'max_position_embeddings', 2048), 2048,
            )
            stride = max_length // 2
            all_log_probs = []

            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                chunk_ids = input_ids[:, begin:end]

                with torch.no_grad():
                    outputs = self.model(chunk_ids)
                    logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = chunk_ids[:, 1:].contiguous()

                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                valid_start = (stride if begin > 0 else 0)
                valid_end = token_log_probs.size(1)
                if valid_start < valid_end:
                    all_log_probs.append(token_log_probs[:, valid_start:valid_end].squeeze(0))

                if end >= seq_len:
                    break

            if not all_log_probs:
                return default

            concat_log_probs = torch.cat(all_log_probs, dim=0)
            mean_nll = -concat_log_probs.mean().item()
            ppl = math.exp(min(mean_nll, 20.0))
            entropies = -concat_log_probs.detach().cpu().numpy()
            burst = float(np.std(entropies)) if len(entropies) > 1 else 0.0
            mean_ent = float(np.mean(entropies))

            return {
                "perplexity": round(ppl, 4),
                "burstiness": round(burst, 4),
                "mean_entropy": round(mean_ent, 4),
            }
        except (RuntimeError, IndexError, ValueError) as e:
            logger.debug("Perplexity calculation failed: %s", e)
            return default

    def analyze(self, code: str, threshold: float | None = None) -> Dict[str, Any]:
        thr = threshold if threshold is not None else self.threshold

        features = extract_features(code)

        signal_scores: List[Tuple[str, float, float]] = []
        for label, fn, weight in SIGNALS:
            try:
                strength = float(fn(features))
            except (KeyError, TypeError, ZeroDivisionError) as e:
                logger.debug("Signal '%s' failed: %s", label, e)
                strength = 0.0
            signal_scores.append((label, strength, weight))
        _SCALE_FACTOR = 6.0
        _OFFSET = -3.0
        weighted_sum = sum(s * w for _, s, w in signal_scores)
        normalized = (weighted_sum / _MAX_WEIGHT) * _SCALE_FACTOR + _OFFSET
        p_ai = _sigmoid(normalized)

        ppl_data = self._calculate_perplexity(code)
        ppl_adj = 0.0
        ppl_val = ppl_data["perplexity"]
        if ppl_val > 0:
            if ppl_val < 5:
                ppl_adj = 0.10
            elif ppl_val < 20:
                ppl_adj = 0.06
            elif ppl_val < 50:
                ppl_adj = 0.03
            elif ppl_val > 200:
                ppl_adj = -0.06
            elif ppl_val > 100:
                ppl_adj = -0.03

            burst = ppl_data["burstiness"]
            if 0 < burst < 0.8:
                ppl_adj += 0.03
            elif burst > 2.5:
                ppl_adj -= 0.03

            ent = ppl_data.get("mean_entropy", 0.0)
            if 0 < ent < 1.0:
                ppl_adj += 0.02
            elif ent > 4.0:
                ppl_adj -= 0.02

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
