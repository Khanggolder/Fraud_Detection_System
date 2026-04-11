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

from .c_features import extract_c_features

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


# ── Signal Functions ─────────────────────────────────────

def _sig_comment_perfection(f: dict) -> float:
    """AI writes well-structured comments: capitalized, ending with period, explanatory."""
    count = f.get('comment_count', 0)
    if count < 2:
        return 0.0

    score = 0.0

    perfect = f.get('perfect_comment_ratio', 0.0)
    if perfect >= 0.6:
        score += 0.35
    elif perfect >= 0.3:
        score += 0.18

    cap = f.get('capitalized_comment_ratio', 0.0)
    if cap >= 0.8:
        score += 0.25
    elif cap >= 0.5:
        score += 0.12

    expl = f.get('explanatory_comment_ratio', 0.0)
    if expl >= 0.7:
        score += 0.20
    elif expl >= 0.4:
        score += 0.10

    if f.get('doc_comment_count', 0) >= 2:
        score += 0.15
    elif f.get('doc_comment_count', 0) >= 1:
        score += 0.08

    if f.get('step_comment_count', 0) >= 2:
        score += 0.12

    clr = f.get('comment_line_ratio', 0.0)
    if 0.10 <= clr <= 0.40:
        score += 0.08

    std = f.get('std_comment_len', 100.0)
    avg = f.get('avg_comment_len', 0.0)
    if count >= 3 and avg > 0 and std < 10.0:
        score += 0.10

    return max(0.0, min(score, 1.0))


def _sig_naming_quality(f: dict) -> float:
    score = 0.0

    avg_len = f.get('avg_id_length', 0.0)
    if 5.0 <= avg_len <= 15.0:
        score += 0.20
    elif 4.0 <= avg_len <= 20.0:
        score += 0.10
    elif avg_len < 3.0:
        score -= 0.15

    if f.get('naming_uniform', False):
        score += 0.25
    elif f.get('naming_consistency', 0.0) >= 0.85:
        score += 0.15

    desc = f.get('func_descriptive_ratio', 0.0)
    if desc >= 0.7:
        score += 0.25
    elif desc >= 0.4:
        score += 0.12

    short = f.get('func_short_name_count', 0)
    if short >= 3:
        score -= 0.15
    elif short >= 1:
        score -= 0.08

    single = f.get('single_char_var_count', 0)
    total_ids = f.get('unique_id_count', 0)
    if total_ids > 0:
        single_ratio = single / total_ids
        if single_ratio > 0.30:
            score -= 0.20
        elif single_ratio > 0.15:
            score -= 0.10
        elif single_ratio < 0.05 and total_ids >= 5:
            score += 0.15

    return max(-0.4, min(score, 1.0))


def _sig_formatting_uniformity(f: dict) -> float:
    score = 0.0

    op_rate = f.get('op_spacing_rate', 0.0)
    op_total = f.get('op_spacing_total', 0)
    if op_total >= 3:
        if op_rate >= 0.95:
            score += 0.25
        elif op_rate >= 0.85:
            score += 0.15

    comma_rate = f.get('comma_space_rate', 0.0)
    if f.get('comma_total', 0) >= 3:
        if comma_rate >= 0.95:
            score += 0.15
        elif comma_rate >= 0.80:
            score += 0.08

    brace = f.get('brace_consistency', 0.0)
    if brace >= 0.95:
        score += 0.15

    if f.get('indent_inconsistency', 0) == 0 and f.get('total_lines', 0) > 10:
        score += 0.15
    elif f.get('indent_inconsistency_ratio', 1.0) < 0.05:
        score += 0.08

    if f.get('trailing_ws_ratio', 1.0) < 0.01:
        score += 0.10

    if not f.get('mixed_indent', False):
        score += 0.08

    return max(0.0, min(score, 1.0))


def _sig_structure_completeness(f: dict) -> float:
    score = 0.0

    alloc = f.get('total_alloc', 0)
    if alloc >= 1:
        pair = f.get('memory_pair_ratio', 0.0)
        if pair >= 0.9:
            score += 0.20
        elif pair >= 0.5:
            score += 0.10

        null_r = f.get('null_check_ratio', 0.0)
        if null_r >= 0.8:
            score += 0.15
        elif null_r >= 0.3:
            score += 0.08

    fopen = f.get('fopen_count', 0)
    if fopen >= 1:
        fpair = f.get('file_pair_ratio', 0.0)
        if fpair >= 0.9:
            score += 0.12

    if f.get('const_count', 0) >= 2:
        score += 0.10
    elif f.get('const_count', 0) >= 1:
        score += 0.05

    if f.get('typedef_count', 0) >= 1:
        score += 0.08

    func_c = f.get('function_count', 0)
    if func_c >= 3:
        lines_per = f.get('total_lines', 0) / func_c
        if 10 <= lines_per <= 40:
            score += 0.15
        elif lines_per <= 60:
            score += 0.08

    # Low nesting
    depth = f.get('max_nesting', 0)
    if depth <= 3:
        score += 0.10
    elif depth <= 5:
        score += 0.05

    return max(0.0, min(score, 1.0))


def _sig_memory_pattern(f: dict) -> float:
    """AI follows textbook memory management patterns."""
    alloc = f.get('total_alloc', 0)
    if alloc == 0:
        return 0.05  # No memory ops, slightly neutral

    score = 0.0

    # Complete malloc/free pairing
    pair = f.get('memory_pair_ratio', 0.0)
    if pair >= 0.9:
        score += 0.40
    elif pair >= 0.6:
        score += 0.20

    if f.get('calloc_count', 0) >= 1:
        score += 0.15

    # Always checks NULL after allocation
    null_r = f.get('null_check_ratio', 0.0)
    if null_r >= 0.8 and alloc >= 2:
        score += 0.30
    elif null_r >= 0.5:
        score += 0.15

    if f.get('fgets_count', 0) >= 1 and f.get('gets_count', 0) == 0:
        score += 0.10

    return max(0.0, min(score, 1.0))


def _sig_template_detection(f: dict) -> float:
    """AI generates code following common template patterns."""
    score = 0.0

    # Has main + helper functions (common AI pattern)
    func_c = f.get('function_count', 0)
    if 2 <= func_c <= 6:
        score += 0.15
    elif func_c >= 7:
        score += 0.08

    # Moderate complexity (not too simple, not too complex)
    avg_cyclo = f.get('avg_cyclomatic', 0.0)
    if 2 <= avg_cyclo <= 8:
        score += 0.12

    # Code size typical of AI exercises
    total_lines = f.get('total_lines', 0)
    if 30 <= total_lines <= 200:
        score += 0.10

    blank = f.get('blank_line_ratio', 0.0)
    if 0.10 <= blank <= 0.25:
        score += 0.10

    # Has both I/O and logic
    has_io = f.get('printf_count', 0) >= 1 or f.get('scanf_count', 0) >= 1
    has_logic = f.get('cyclomatic_branches', 0) >= 2
    if has_io and has_logic:
        score += 0.10

    # Comments present (AI almost always comments)
    if f.get('comment_count', 0) >= 3:
        score += 0.10
    elif f.get('comment_count', 0) >= 1:
        score += 0.05

    # Multiple standard headers
    if f.get('std_header_count', 0) >= 2:
        score += 0.08

    # No errors in AST
    if f.get('ast_error_count', 0) == 0 and f.get('ast_node_count', 0) > 30:
        score += 0.10

    return max(0.0, min(score, 1.0))


def _sig_line_uniformity(f: dict) -> float:
    """AI generates lines with very uniform lengths."""
    cv = f.get('nb_line_len_cv', 1.0)
    mean = f.get('nb_line_len_mean', 0.0)
    if mean < 5:
        return 0.0

    if cv < 0.25:
        return 0.65
    if cv < 0.40:
        return 0.35
    if cv < 0.55:
        return 0.15
    return 0.0


def _sig_over_perfection(f: dict) -> float:
    """Detects "too perfect" code that's unlikely to be human."""
    indicators = 0.0

    if f.get('indent_inconsistency', 0) == 0 and f.get('total_lines', 0) > 15:
        indicators += 1.0

    if f.get('naming_uniform', False) and f.get('unique_id_count', 0) >= 5:
        indicators += 1.0

    if f.get('op_spacing_rate', 0.0) >= 0.98 and f.get('op_spacing_total', 0) >= 5:
        indicators += 1.0

    if f.get('comma_space_rate', 0.0) >= 0.98 and f.get('comma_total', 0) >= 3:
        indicators += 1.0

    if f.get('brace_consistency', 0.0) >= 0.98:
        indicators += 0.5

    if f.get('trailing_ws_ratio', 1.0) == 0.0 and f.get('total_lines', 0) > 10:
        indicators += 0.5

    if f.get('ast_error_count', 0) == 0 and f.get('ast_node_count', 0) > 50:
        indicators += 0.5

    if f.get('dead_code_count', 0) == 0:
        indicators += 0.5

    if indicators >= 5.0:
        return 1.0
    if indicators >= 3.5:
        return 0.70
    if indicators >= 2.5:
        return 0.45
    if indicators >= 1.5:
        return 0.20
    return 0.0


def _sig_human_penalty(f: dict) -> float:
    """Penalize patterns typical of human/competitive programmers."""
    score = 0.0

    if f.get('single_char_var_count', 0) >= 8:
        score -= 0.30
    elif f.get('single_char_var_count', 0) >= 5:
        score -= 0.18
    elif f.get('single_char_var_count', 0) >= 3:
        score -= 0.08

    if f.get('global_var_count', 0) >= 3:
        score -= 0.25
    elif f.get('global_var_count', 0) >= 1:
        score -= 0.10

    if f.get('goto_count', 0) >= 1:
        score -= 0.15

    if f.get('mixed_indent', False):
        score -= 0.12

    max_blank = f.get('max_consec_blank', 0)
    if max_blank > 4:
        score -= 0.15
    elif max_blank > 2:
        score -= 0.05

    if f.get('dead_code_count', 0) >= 3:
        score -= 0.15
    elif f.get('dead_code_count', 0) >= 1:
        score -= 0.08

    if f.get('gets_count', 0) >= 1:
        score -= 0.10

    # Very short avg identifier -> human shortcut style
    if f.get('avg_id_length', 10.0) < 3.0:
        score -= 0.15

    return max(-1.0, min(score, 0.0))


def _sig_clean_aggregate(f: dict) -> float:
    """Aggregate clean code indicators."""
    score = 0.0

    if f.get('dead_code_count', 0) == 0:
        score += 0.10

    if f.get('trailing_ws_ratio', 1.0) < 0.01:
        score += 0.10

    if not f.get('mixed_indent', False):
        score += 0.08

    if f.get('indent_inconsistency', 0) == 0 and f.get('total_lines', 0) > 5:
        score += 0.10

    if f.get('naming_uniform', False):
        score += 0.10

    if f.get('max_consec_blank', 0) <= 2:
        score += 0.08

    if f.get('global_var_count', 0) == 0:
        score += 0.08

    if f.get('ast_error_count', 0) == 0 and f.get('ast_node_count', 0) > 20:
        score += 0.10

    if f.get('halstead_bugs', 1.0) < 0.1:
        score += 0.08

    mi = f.get('maintainability_index', 0.0)
    if mi >= 70:
        score += 0.10
    elif mi >= 50:
        score += 0.05

    return max(0.0, min(score, 1.0))


def _sig_token_distribution(perplexity_data: dict) -> float:
    ppl = perplexity_data.get('perplexity', 0.0)
    burst = perplexity_data.get('burstiness', 0.0)
    entropy = perplexity_data.get('mean_entropy', 0.0)

    if ppl <= 0:
        return 0.0

    score = 0.0

    if ppl < 1.5:
        score += 0.45
    elif ppl < 2.0:
        score += 0.35
    elif ppl < 3.0:
        score += 0.25
    elif ppl < 5.0:
        score += 0.15
    elif ppl < 10.0:
        score += 0.08
    elif ppl > 100.0:
        score -= 0.15
    elif ppl > 50.0:
        score -= 0.08

    if 0 < burst < 0.6:
        score += 0.15
    elif burst < 1.0:
        score += 0.10
    elif burst < 1.5:
        score += 0.05
    elif burst > 2.5:
        score -= 0.10

    if 0 < entropy < 0.5:
        score += 0.12
    elif entropy < 1.0:
        score += 0.08
    elif entropy < 2.0:
        score += 0.03
    elif entropy > 4.0:
        score -= 0.08

    return max(-0.3, min(score, 0.7))


# ── Signal Registry ──────────────────────────────────────

C_SIGNALS: List[Tuple[str, Any, float]] = [
    ("Comment Perfection",         _sig_comment_perfection,      3.5),
    ("Naming Quality",             _sig_naming_quality,           3.0),
    ("Formatting Uniformity",      _sig_formatting_uniformity,    2.5),
    ("Structure Completeness",     _sig_structure_completeness,   2.5),
    ("Memory Management Pattern",  _sig_memory_pattern,           2.0),
    ("Code Template Pattern",      _sig_template_detection,       3.0),
    ("Line Length Uniformity",     _sig_line_uniformity,          1.5),
    ("Over-Perfection Detection",  _sig_over_perfection,          2.5),
    ("Human Artifact Penalty",     _sig_human_penalty,            3.5),
    ("Clean Code Aggregate",       _sig_clean_aggregate,          2.0),
]

_C_TOTAL_WEIGHT = sum(w for _, _, w in C_SIGNALS)
_C_TOKEN_SIGNAL_WEIGHT = 1.0


# ── Detector Class ────────────────────────────────────────

class CAIDetector:
    """Detects AI-generated C code using AST-based signal analysis."""

    def __init__(self, threshold: float = 0.35, use_perplexity: bool = False,
                 model_name: str = "Qwen/Qwen2.5-Coder-0.5B"):
        self.threshold = threshold
        self.use_perplexity = use_perplexity and HAS_LM
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None

        if self.use_perplexity:
            self._load_code_model()

    def _load_code_model(self) -> None:
        """Load Qwen2.5-Coder for perplexity scoring."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.to(self.device)
            self.model.eval()
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Failed to load code model: %s", e)
            self.model = None
            self.use_perplexity = False

    def _calculate_code_perplexity(self, code: str) -> Dict[str, float]:
        """Calculate perplexity, burstiness, and mean entropy of code."""
        if not self.use_perplexity or self.model is None or self.tokenizer is None:
            return {"perplexity": 0.0, "burstiness": 0.0, "mean_entropy": 0.0}

        try:
            encodings = self.tokenizer(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            input_ids = encodings.input_ids.to(self.device)
            seq_len = input_ids.size(1)

            if seq_len < 2:
                return {"perplexity": 0.0, "burstiness": 0.0, "mean_entropy": 0.0}

            max_length = min(
                getattr(self.model.config, 'max_position_embeddings', 2048),
                2048,
            )
            stride = max_length // 2

            all_log_probs: List[torch.Tensor] = []

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
                    valid_log_probs = token_log_probs[:, valid_start:valid_end]
                    all_log_probs.append(valid_log_probs.squeeze(0))

                if end >= seq_len:
                    break

            if not all_log_probs:
                return {"perplexity": 0.0, "burstiness": 0.0, "mean_entropy": 0.0}

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
            return {"perplexity": 0.0, "burstiness": 0.0, "mean_entropy": 0.0}

    def analyze(self, code: str, threshold: float | None = None) -> Dict[str, Any]:
        thr = threshold if threshold is not None else self.threshold

        features = extract_c_features(code)

        signal_scores: List[Tuple[str, float, float]] = []
        for label, fn, weight in C_SIGNALS:
            try:
                strength = float(fn(features))
            except (KeyError, TypeError, ZeroDivisionError) as e:
                logger.debug("Signal '%s' failed: %s", label, e)
                strength = 0.0
            signal_scores.append((label, strength, weight))

        ppl_data = self._calculate_code_perplexity(code)
        token_strength = _sig_token_distribution(ppl_data)
        signal_scores.append(("Token Distribution (Code Model)", token_strength, _C_TOKEN_SIGNAL_WEIGHT))

        _SCALE_FACTOR = 7.0
        _OFFSET = -2.5
        total_weight = _C_TOTAL_WEIGHT + _C_TOKEN_SIGNAL_WEIGHT
        weighted_sum = sum(s * w for _, s, w in signal_scores)
        normalized = (weighted_sum / total_weight) * _SCALE_FACTOR + _OFFSET
        p_ai = _sigmoid(normalized)
        p_ai = max(0.0, min(1.0, p_ai))
        score = int(round(p_ai * 100))

        ranked = sorted(signal_scores, key=lambda x: x[1] * x[2], reverse=True)
        top_signals = [
            f"{label} ({strength:.0%})"
            for label, strength, _ in ranked
            if strength > 0.10
        ][:6]

        neg_signals = [
            f"{label} ({strength:.0%})"
            for label, strength, _ in ranked
            if strength < -0.08
        ]

        if not top_signals and not neg_signals:
            top_signals = ["No strong AI signals detected"]

        all_signals = top_signals + neg_signals

        details = {
            **features,
            **ppl_data,
            "signal_breakdown": {
                label: round(strength, 4) for label, strength, _ in signal_scores
            },
            "weight_breakdown": {
                label: round(strength * weight, 4) for label, strength, weight in signal_scores
            },
        }

        return {
            "p_ai": round(p_ai, 4),
            "score": score,
            "flag": p_ai >= thr,
            "signals": all_signals,
            "details": details,
        }

    def detect(self, code: str) -> Dict[str, Any]:
        result = self.analyze(code)
        return {
            "is_ai_suspect": result["flag"],
            "confidence_score": result["score"],
            "reasons": result["signals"],
            "details": result["details"],
        }
