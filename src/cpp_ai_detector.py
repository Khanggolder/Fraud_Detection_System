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

from .cpp_features import extract_cpp_ast_features

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)


def _sig_ast_modernity(f: dict) -> float:
    modern_total = f.get('modern_cpp_total', 0)
    n_lines = max(f.get('total_lines', 1), 1)
    density = modern_total / n_lines

    feature_count = 0
    score = 0.0

    if f.get('auto_count', 0) >= 2:
        score += 0.15
        feature_count += 1
    elif f.get('auto_count', 0) >= 1:
        score += 0.12
        feature_count += 1

    if f.get('nullptr_count', 0) >= 1:
        score += 0.08
        feature_count += 1

    if f.get('lambda_count', 0) >= 1:
        score += 0.12
        feature_count += 1

    if f.get('range_for_count', 0) >= 1:
        score += 0.10
        feature_count += 1

    if f.get('constexpr_count', 0) >= 1:
        score += 0.08
        feature_count += 1

    if f.get('noexcept_count', 0) >= 1:
        score += 0.10
        feature_count += 1

    if f.get('structured_binding_count', 0) >= 1:
        score += 0.10
        feature_count += 1

    if f.get('enum_class_count', 0) >= 1:
        score += 0.08
        feature_count += 1

    if f.get('move_semantics_count', 0) >= 1:
        score += 0.10
        feature_count += 1

    if f.get('emplace_count', 0) >= 1:
        score += 0.08
        feature_count += 1

    if f.get('override_count', 0) >= 1:
        score += 0.08
        feature_count += 1

    if f.get('using_alias_count', 0) >= 1:
        score += 0.08
        feature_count += 1

    if f.get('modern_algorithm_count', 0) >= 1:
        score += 0.10
        feature_count += 1

    if f.get('concept_count', 0) >= 1 or f.get('requires_clause_count', 0) >= 1:
        score += 0.12
        feature_count += 1

    if f.get('consteval_count', 0) >= 1 or f.get('constinit_count', 0) >= 1:
        score += 0.08
        feature_count += 1

    if feature_count >= 8:
        score += 0.20
    elif feature_count >= 5:
        score += 0.12
    elif feature_count >= 3:
        score += 0.06

    if density >= 0.12:
        score += 0.10
    elif density >= 0.05:
        score += 0.05

    return max(0.0, min(score, 1.0))


def _sig_raii_conformance(f: dict) -> float:
    score = 0.0

    smart_ptr = f.get('smart_ptr_count', 0)
    new_expr = f.get('new_expression_count', 0)
    delete_expr = f.get('delete_expression_count', 0)

    if smart_ptr >= 2:
        score += 0.30
    elif smart_ptr >= 1:
        score += 0.20

    raii_ratio = f.get('raii_memory_ratio', 0.0)
    if raii_ratio >= 0.9:
        score += 0.10
    elif raii_ratio >= 0.5:
        score += 0.05

    if new_expr >= 2 and smart_ptr == 0:
        score -= 0.15
    if delete_expr >= 1:
        score -= 0.10

    if f.get('raii_guard_count', 0) >= 1:
        score += 0.12
    if f.get('raii_type_variety', 0) >= 2:
        score += 0.08

    if f.get('const_ref_count', 0) >= 2:
        score += 0.10
    elif f.get('const_ref_count', 0) >= 1:
        score += 0.06

    if f.get('const_method_count', 0) >= 1:
        score += 0.10

    if f.get('noexcept_count', 0) >= 2:
        score += 0.10
    elif f.get('noexcept_count', 0) >= 1:
        score += 0.06

    if f.get('deleted_function_count', 0) >= 1 or f.get('defaulted_function_count', 0) >= 1:
        score += 0.12

    if f.get('error_handling_present', False):
        score += 0.08

    if f.get('static_assert_count', 0) >= 1:
        score += 0.06

    if f.get('goto_count', 0) >= 1:
        score -= 0.15

    return max(-0.5, min(score, 1.0))


def _sig_human_artifact_penalty(f: dict) -> float:
    score = 0.0

    if f.get('using_namespace_std', False):
        score -= 0.30

    if f.get('bits_stdc_present', False):
        score -= 0.25

    if f.get('ios_sync_present', False):
        score -= 0.20
    if f.get('cin_tie_present', False):
        score -= 0.15

    if f.get('large_array_count', 0) >= 1:
        max_sz = f.get('large_array_max_size', 0)
        if max_sz >= 100000:
            score -= 0.30
        else:
            score -= 0.15

    if f.get('global_var_count', 0) >= 3:
        score -= 0.25
    elif f.get('global_var_count', 0) >= 1:
        score -= 0.10

    if f.get('define_shortcut_count', 0) >= 2:
        score -= 0.20
    elif f.get('define_shortcut_count', 0) >= 1:
        score -= 0.10

    scv = f.get('single_char_var_count', 0)
    tpl = f.get('template_count', 0)
    adjusted_scv = max(0, scv - tpl * 2)
    if adjusted_scv >= 8:
        score -= 0.25
    elif adjusted_scv >= 5:
        score -= 0.15
    elif adjusted_scv >= 3:
        score -= 0.08

    if f.get('void_main', False):
        score -= 0.10

    if f.get('freopen_present', False):
        score -= 0.10

    cp = f.get('competitive_pattern_count', 0)
    if cp >= 5:
        score -= 0.30
    elif cp >= 3:
        score -= 0.15

    if f.get('dead_code_count', 0) >= 3:
        score -= 0.15
    elif f.get('dead_code_count', 0) >= 1:
        score -= 0.08

    if f.get('mixed_indent', False):
        score -= 0.10

    max_blank = f.get('max_consecutive_blank_lines', 0)
    if max_blank > 4:
        score -= 0.15
    elif max_blank > 2:
        score -= 0.05

    return max(-1.0, min(score, 0.0))


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


def _sig_naming_consistency(f: dict) -> float:
    total_styled = f.get('snake_case_count', 0) + f.get('camel_case_count', 0) + f.get('pascal_case_count', 0)
    if total_styled < 2:
        return 0.10

    consistency = f.get('naming_consistency', 0.0)
    uniform = f.get('naming_style_uniform', False)
    func_consistency = f.get('func_naming_consistency', 0.0)

    score = 0.0
    if uniform:
        score += 0.45
    elif consistency >= 0.85:
        score += 0.35
    elif consistency >= 0.70:
        score += 0.20
    elif consistency >= 0.55:
        score += 0.10

    if func_consistency >= 0.90:
        score += 0.25
    elif func_consistency >= 0.70:
        score += 0.15

    avg_len = f.get('avg_identifier_length', 0.0)
    if 4.0 <= avg_len <= 18.0:
        score += 0.15
    elif avg_len < 3.0:
        score -= 0.10

    if f.get('unique_identifier_count', 0) >= 8 and consistency >= 0.80:
        score += 0.10

    return max(0.0, min(score, 1.0))


def _sig_formatting_precision(f: dict) -> float:
    score = 0.0

    op_rate = f.get('binary_op_spacing_rate', 0.0)
    op_total = f.get('binary_op_total', 0)
    if op_total >= 3:
        if op_rate >= 0.95:
            score += 0.25
        elif op_rate >= 0.85:
            score += 0.15
        elif op_rate >= 0.70:
            score += 0.08

    comma_rate = f.get('comma_space_rate', 0.0)
    comma_total = f.get('comma_total', 0)
    if comma_total >= 3:
        if comma_rate >= 0.95:
            score += 0.15
        elif comma_rate >= 0.80:
            score += 0.08

    decl_cons = f.get('decl_spacing_consistency', 0.0)
    if f.get('decl_spacing_total', 0) >= 5:
        if decl_cons >= 0.90:
            score += 0.10

    brace_cons = f.get('brace_style_consistency', 0.0)
    if brace_cons >= 0.95:
        score += 0.15
    elif brace_cons >= 0.80:
        score += 0.08

    if f.get('indent_inconsistency_count', 0) == 0 and f.get('total_lines', 0) > 10:
        score += 0.15
    elif f.get('indent_inconsistency_ratio', 1.0) < 0.05:
        score += 0.08

    if f.get('trailing_whitespace_ratio', 1.0) < 0.01:
        score += 0.08

    if not f.get('mixed_indent', False):
        score += 0.06

    return max(0.0, min(score, 1.0))


def _sig_comment_quality(f: dict) -> float:
    count = f.get('comment_count', 0)
    if count < 2:
        return 0.0

    score = 0.0
    perfect_ratio = f.get('perfect_comment_ratio', 0.0)
    cap_ratio = f.get('capitalized_comment_ratio', 0.0)

    if perfect_ratio >= 0.7:
        score += 0.35
    elif perfect_ratio >= 0.4:
        score += 0.18

    if cap_ratio >= 0.85:
        score += 0.25
    elif cap_ratio >= 0.6:
        score += 0.12

    std_len = f.get('std_comment_len', 0.0)
    if count >= 3 and std_len < 8.0:
        score += 0.15

    avg_len = f.get('avg_comment_len', 0.0)
    if 15.0 <= avg_len <= 60.0:
        score += 0.10

    clr = f.get('comment_line_ratio', 0.0)
    if 0.10 <= clr <= 0.35:
        score += 0.10

    return max(0.0, min(score, 1.0))


def _sig_structure_quality(f: dict) -> float:
    score = 0.0

    if f.get('namespace_count', 0) >= 1:
        score += 0.08

    if f.get('class_count', 0) >= 1 and f.get('virtual_count', 0) >= 1:
        score += 0.08

    if f.get('template_count', 0) >= 1:
        score += 0.08

    avg_cyclo = f.get('avg_cyclomatic_per_function', 0.0)
    if 0 < avg_cyclo <= 5:
        score += 0.15
    elif avg_cyclo <= 10:
        score += 0.08
    elif avg_cyclo > 15:
        score -= 0.10

    func_c = f.get('function_count', 0)
    if func_c >= 2:
        lines_per_func = f.get('total_lines', 0) / func_c
        if 10 <= lines_per_func <= 40:
            score += 0.12
        elif lines_per_func <= 60:
            score += 0.05

    depth = f.get('max_nesting_depth', 0)
    if depth <= 3:
        score += 0.10
    elif depth <= 5:
        score += 0.05
    elif depth > 7:
        score -= 0.10

    density = f.get('code_density', 0.0)
    if 0.65 <= density <= 0.85:
        score += 0.08

    blank_ratio = f.get('blank_line_ratio', 0.0)
    if 0.10 <= blank_ratio <= 0.30:
        score += 0.08

    return max(-0.2, min(score, 1.0))


def _sig_line_uniformity(f: dict) -> float:
    cv = f.get('nb_line_len_cv', 1.0)
    mean = f.get('nb_line_len_mean', 0.0)
    if mean < 3:
        return 0.0

    if cv < 0.35:
        return 0.60
    if cv < 0.55:
        return 0.40
    if cv < 0.70:
        return 0.20
    if cv < 0.85:
        return 0.08
    return 0.0


def _sig_over_perfection(f: dict) -> float:
    indicators = 0.0

    if f.get('indent_inconsistency_count', 0) == 0 and f.get('total_lines', 0) > 15:
        indicators += 1.0

    if f.get('naming_style_uniform', False) and f.get('unique_identifier_count', 0) >= 5:
        indicators += 1.0

    if f.get('binary_op_spacing_rate', 0.0) >= 0.98 and f.get('binary_op_total', 0) >= 5:
        indicators += 1.0

    if f.get('comma_space_rate', 0.0) >= 0.98 and f.get('comma_total', 0) >= 3:
        indicators += 1.0

    if f.get('brace_style_consistency', 0.0) >= 0.98:
        indicators += 0.5

    if f.get('trailing_whitespace_ratio', 1.0) == 0.0 and f.get('total_lines', 0) > 10:
        indicators += 0.5

    if f.get('ast_error_count', 0) == 0 and f.get('ast_node_count', 0) > 50:
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


def _sig_maintainability_index(f: dict) -> float:
    mi = f.get('maintainability_index', 0.0)
    if mi >= 80:
        return 0.80
    if mi >= 70:
        return 0.60
    if mi >= 55:
        return 0.35
    if mi >= 40:
        return 0.15
    if mi > 0:
        return -0.10
    return 0.0


def _sig_clean_code_aggregate(f: dict) -> float:
    score = 0.0

    if f.get('dead_code_count', 0) == 0:
        score += 0.12

    if f.get('trailing_whitespace_ratio', 1.0) < 0.01:
        score += 0.10

    if not f.get('mixed_indent', False):
        score += 0.08

    if f.get('indent_inconsistency_count', 0) == 0 and f.get('total_lines', 0) > 5:
        score += 0.12

    if f.get('naming_style_uniform', False):
        score += 0.12

    if f.get('max_consecutive_blank_lines', 0) <= 2:
        score += 0.08

    if not f.get('using_namespace_std', False):
        score += 0.08

    if not f.get('bits_stdc_present', False):
        score += 0.08

    if f.get('global_var_count', 0) == 0:
        score += 0.08

    if f.get('ast_error_count', 0) == 0 and f.get('ast_node_count', 0) > 20:
        score += 0.10

    if f.get('halstead_bugs_estimate', 1.0) < 0.1:
        score += 0.08

    return max(0.0, min(score, 1.0))


SIGNALS: List[Tuple[str, Any, float]] = [
    ("AST Modernity (C++11/14/17/20)",     _sig_ast_modernity,          5.0),  # Boosted
    ("RAII Conformance",                    _sig_raii_conformance,        4.0),  # Boosted
    ("Naming Consistency",                  _sig_naming_consistency,      2.0),
    ("Formatting Precision",               _sig_formatting_precision,    2.0),
    ("Comment Quality",                     _sig_comment_quality,         1.5),
    ("Code Structure Quality",             _sig_structure_quality,        1.5),
    ("Line Length Uniformity",             _sig_line_uniformity,          1.0),
    ("Over-Perfection Detection",          _sig_over_perfection,          2.5),
    ("Maintainability Index",              _sig_maintainability_index,    1.0),  # Halved
    ("Clean Code Aggregate",               _sig_clean_code_aggregate,     1.5),  # Halved
    ("Human Artifact Penalty",             _sig_human_artifact_penalty,   4.0),
]

_TOTAL_WEIGHT = sum(w for _, _, w in SIGNALS)
_TOKEN_SIGNAL_WEIGHT = 1.0  # Dilution reduced


class CppAIDetector:

    def __init__(self, threshold: float = 0.60, use_perplexity: bool = True,
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
            nlls: List[float] = []

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

                overlap = max(0, begin - 0) if begin > 0 else 0
                valid_start = (stride if begin > 0 else 0)
                valid_end = token_log_probs.size(1)
                if valid_start < valid_end:
                    valid_log_probs = token_log_probs[:, valid_start:valid_end]
                    all_log_probs.append(valid_log_probs.squeeze(0))
                    nll = -valid_log_probs.mean().item()
                    nlls.append(nll)

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

        features = extract_cpp_ast_features(code)

        signal_scores: List[Tuple[str, float, float]] = []
        for label, fn, weight in SIGNALS:
            try:
                strength = float(fn(features))
            except (KeyError, TypeError, ZeroDivisionError) as e:
                logger.debug("Signal '%s' failed: %s", label, e)
                strength = 0.0
            signal_scores.append((label, strength, weight))

        ppl_data = self._calculate_code_perplexity(code)
        token_strength = _sig_token_distribution(ppl_data)
        signal_scores.append(("Token Distribution (Code Model)", token_strength, _TOKEN_SIGNAL_WEIGHT))

        _SCALE_FACTOR = 10.0
        _OFFSET = -3.5
        total_weight = _TOTAL_WEIGHT + _TOKEN_SIGNAL_WEIGHT
        weighted_sum = sum(s * w for _, s, w in signal_scores)
        normalized = (weighted_sum / total_weight) * _SCALE_FACTOR + _OFFSET
        p_ai = _sigmoid(normalized)

        p_ai = max(0.0, min(1.0, p_ai))
        score = int(round(p_ai * 100))

        ranked = sorted(signal_scores, key=lambda x: x[1] * x[2], reverse=True)
        top_signals = [
            f"{label} ({strength:.0%})"
            for label, strength, _ in ranked
            if strength > 0.15
        ][:6]

        negative_signals = [
            f"{label} ({strength:.0%})"
            for label, strength, _ in ranked
            if strength < -0.1
        ]

        if not top_signals and not negative_signals:
            top_signals = ["No strong AI signals detected"]

        all_signals = top_signals + negative_signals

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

    def detect_ai_generated(self, code: str) -> Dict[str, Any]:
        result = self.analyze(code)
        return {
            "is_ai_suspect": result["flag"],
            "confidence_score": result["score"],
            "reasons": result["signals"],
            "details": result["details"],
        }
