# features.py
import re
import ast
import math
import statistics
from typing import Dict, Union, List, Tuple

try:
    import radon.complexity as radon_cc
    from radon.metrics import h_visit, mi_visit
    HAS_RADON = True
except ImportError:
    HAS_RADON = False


FeatureDict = Dict[str, Union[float, int, bool]]

_RE_COMMENT = re.compile(r'^\s*#(.*)$')
_RE_OPERATOR = re.compile(r'(?<!=)\s*([+\-*/%&|^~<>]=?|==|!=|<=|>=|<<|>>|\*\*|//)\s*')
_RE_OPERATOR_SPACED = re.compile(r'(?<!=)\s([+\-*/%&|^~<>]=?|==|!=|<=|>=|<<|>>|\*\*|//)\s')
_RE_OPERATOR_NOSPACE = re.compile(r'\S([+\-*/%]=?|==|!=|<=|>=)\S')
_RE_COMMA = re.compile(r',')
_RE_COMMA_SPACE = re.compile(r',\s')
_RE_TRAILING_WS = re.compile(r'[ \t]+$')
_RE_IMPORT = re.compile(r'^\s*(import |from \S+ import )')
_RE_COMPREHENSION = re.compile(r'\[.+\bfor\b.+\bin\b.+\]')
_RE_ENUMERATE = re.compile(r'\benumerate\s*\(')
_RE_ZIP = re.compile(r'\bzip\s*\(')
_RE_ANY_ALL = re.compile(r'\b(any|all)\s*\(')
_RE_FSTRING = re.compile(r'f["\']')
_RE_TRY = re.compile(r'^\s*try\s*:')
_RE_EXCEPT = re.compile(r'^\s*except\b')
_RE_TUTORIAL_MARKERS = re.compile(
    r'\b(Step\s+\d|Example|Edge\s+case|Returns|Args|Parameters|Raises|Note[s]?)\b',
    re.IGNORECASE,
)
_RE_SINGLE_CHAR_VAR = re.compile(r'\b([a-z_])\s*=')
_RE_SNAKE_CASE_FUNC = re.compile(r'^def\s+[a-z_][a-z0-9_]*\s*\(')
_RE_CAMEL_CASE_FUNC = re.compile(r'^def\s+[a-z]+[A-Z]')
_RE_CLASS_DEF = re.compile(r'^\s*class\s+')
_RE_DECORATOR = re.compile(r'^\s*@')
_RE_LAMBDA = re.compile(r'\blambda\b')
_RE_WITH = re.compile(r'^\s*with\b')
_RE_ASSERT = re.compile(r'^\s*assert\b')
_RE_TYPE_HINT = re.compile(r'(:\s*\w+(\[.*?\])?|->)')
_RE_WALRUS = re.compile(r':=')
_RE_GLOBAL = re.compile(r'^\s*global\b')

def _get_indent_widths(lines: List[str]) -> List[int]:
    widths: List[int] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        n_spaces = len(line) - len(stripped)
        if n_spaces > 0:
            widths.append(n_spaces)
    return widths


def _detect_indent_mode(widths: List[int]) -> int:
    if not widths:
        return 4
    counts = {2: 0, 4: 0, 8: 0}
    for w in widths:
        for step in (2, 4, 8):
            if w % step == 0:
                counts[step] += 1
    return max(counts, key=counts.get)

def _extract_comments(lines: List[str]) -> List[str]:
    comments: List[str] = []
    for line in lines:
        m = _RE_COMMENT.match(line)
        if m:
            comments.append(m.group(1).strip())
        else:
            # Inline comment
            idx = line.find(' #')
            if idx != -1:
                after = line[idx + 2:].strip()
                if after:
                    comments.append(after)
    return comments

def _ast_function_info(code: str) -> Tuple[int, List[int], int, int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0, [], 0, 0

    func_count = 0
    func_lens: List[int] = []
    class_count = 0
    decorator_count = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_count += 1
            if hasattr(node, 'end_lineno') and node.end_lineno and node.lineno:
                func_lens.append(node.end_lineno - node.lineno + 1)
            decorator_count += len(node.decorator_list)
        elif isinstance(node, ast.ClassDef):
            class_count += 1
            decorator_count += len(node.decorator_list)

    return func_count, func_lens, class_count, decorator_count


def _ast_docstring_info(code: str) -> Tuple[bool, int, int, float]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, 0, 0, 0.0

    total_ds = 0
    func_count = 0
    func_ds_count = 0

    if ast.get_docstring(tree) is not None:
        total_ds += 1

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_count += 1
            if ast.get_docstring(node) is not None:
                total_ds += 1
                func_ds_count += 1
        elif isinstance(node, ast.ClassDef):
            if ast.get_docstring(node) is not None:
                total_ds += 1

    func_ds_ratio = func_ds_count / max(func_count, 1)
    return total_ds > 0, total_ds, func_ds_count, func_ds_ratio


def _count_unique_names(code: str) -> Tuple[int, int, int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0, 0, 0

    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.arg):
            names.append(node.arg)

    total = len(names)
    unique = len(set(names))
    single_char = sum(1 for n in set(names) if len(n) == 1)
    return total, unique, single_char

def _radon_metrics(code: str) -> Dict[str, float]:
    defaults = {
        'cyclomatic_mean': 0.0,
        'cyclomatic_max': 0.0,
        'maintainability_index': 0.0,
        'halstead_volume': 0.0,
        'halstead_difficulty': 0.0,
        'halstead_effort': 0.0,
    }
    if not HAS_RADON:
        return defaults

    try:
        blocks = radon_cc.cc_visit(code)
        complexities = [b.complexity for b in blocks] if blocks else [0]
        defaults['cyclomatic_mean'] = float(statistics.mean(complexities))
        defaults['cyclomatic_max'] = float(max(complexities))
    except Exception:
        pass

    try:
        mi = mi_visit(code, multi=False)
        defaults['maintainability_index'] = float(mi)
    except Exception:
        pass

    try:
        h = h_visit(code)
        if h.total:
            defaults['halstead_volume'] = float(h.total.volume) if h.total.volume else 0.0
            defaults['halstead_difficulty'] = float(h.total.difficulty) if h.total.difficulty else 0.0
            defaults['halstead_effort'] = float(h.total.effort) if h.total.effort else 0.0
    except Exception:
        pass

    return defaults

def extract_features(code: str) -> FeatureDict:

    features: FeatureDict = {}

    lines = code.split('\n')
    total_lines = len(lines)
    non_blank_lines = [l for l in lines if l.strip()]
    n_non_blank = max(len(non_blank_lines), 1)
    total_chars = max(len(code), 1)

    tab_count = code.count('\t')
    space_indent_lines = sum(1 for l in lines if l and l[0] == ' ')
    tab_indent_lines = sum(1 for l in lines if l and l[0] == '\t')
    indent_lines = max(space_indent_lines + tab_indent_lines, 1)

    features['tab_ratio'] = tab_count / total_chars
    features['space_indent_ratio'] = space_indent_lines / indent_lines
    features['tab_indent_ratio'] = tab_indent_lines / indent_lines

    indent_widths = _get_indent_widths(lines)
    indent_mode = _detect_indent_mode(indent_widths)
    features['indent_width_mode'] = indent_mode

    inconsistencies = sum(1 for w in indent_widths if w % indent_mode != 0) if indent_widths else 0
    features['indent_inconsistency_count'] = inconsistencies
    features['indent_inconsistency_ratio'] = inconsistencies / max(len(indent_widths), 1)

    op_total = len(_RE_OPERATOR.findall(code))
    op_spaced = len(_RE_OPERATOR_SPACED.findall(code))
    op_nospace = len(_RE_OPERATOR_NOSPACE.findall(code))
    op_denom = max(op_total, 1)

    features['operator_spacing_rate'] = op_spaced / op_denom
    features['missing_operator_spacing_rate'] = op_nospace / op_denom
    features['operator_count'] = op_total

    comma_total = len(_RE_COMMA.findall(code))
    comma_spaced = len(_RE_COMMA_SPACE.findall(code))
    features['comma_space_rate'] = comma_spaced / max(comma_total, 1)
    features['comma_count'] = comma_total

    trailing_ws_lines = sum(1 for l in lines if _RE_TRAILING_WS.search(l))
    features['trailing_whitespace_ratio'] = trailing_ws_lines / max(total_lines, 1)

    blank_lines = total_lines - n_non_blank
    features['blank_line_ratio'] = blank_lines / max(total_lines, 1)
    features['blank_line_count'] = blank_lines

    line_lengths = [len(l) for l in lines]
    features['line_len_mean'] = statistics.mean(line_lengths) if line_lengths else 0.0
    features['line_len_std'] = statistics.pstdev(line_lengths) if len(line_lengths) > 1 else 0.0
    features['line_len_max'] = max(line_lengths) if line_lengths else 0
    features['line_len_median'] = statistics.median(line_lengths) if line_lengths else 0.0

    long_lines = sum(1 for l in line_lengths if l > 79)
    features['pep8_long_line_ratio'] = long_lines / max(total_lines, 1)

    comments = _extract_comments(lines)
    comment_lines_count = sum(1 for l in lines if _RE_COMMENT.match(l))

    features['comment_line_ratio'] = comment_lines_count / n_non_blank
    features['comment_count'] = len(comments)

    ds_present, ds_count, func_ds_count, func_ds_ratio = _ast_docstring_info(code)
    features['docstring_present'] = ds_present
    features['docstring_count'] = ds_count
    features['func_docstring_ratio'] = func_ds_ratio

    if comments:
        comment_lens = [len(c) for c in comments]
        features['avg_comment_len'] = statistics.mean(comment_lens)
        features['std_comment_len'] = statistics.pstdev(comment_lens) if len(comment_lens) > 1 else 0.0
        features['max_comment_len'] = max(comment_lens)
    else:
        features['avg_comment_len'] = 0.0
        features['std_comment_len'] = 0.0
        features['max_comment_len'] = 0

    tutorial_hits = len(_RE_TUTORIAL_MARKERS.findall(code))
    features['tutorial_markers_present'] = tutorial_hits > 0
    features['tutorial_markers_count'] = tutorial_hits

    comprehension_count = len(_RE_COMPREHENSION.findall(code))
    enumerate_count = len(_RE_ENUMERATE.findall(code))
    zip_count = len(_RE_ZIP.findall(code))
    any_all_count = len(_RE_ANY_ALL.findall(code))
    fstring_count = len(_RE_FSTRING.findall(code))
    lambda_count = len(_RE_LAMBDA.findall(code))
    with_count = sum(1 for l in lines if _RE_WITH.match(l))
    walrus_count = len(_RE_WALRUS.findall(code))

    pythonic_total = (comprehension_count + enumerate_count + zip_count +
                      any_all_count + fstring_count + lambda_count +
                      with_count + walrus_count)
    features['pythonic_construct_count'] = pythonic_total
    features['pythonic_construct_ratio'] = pythonic_total / n_non_blank
    features['comprehension_count'] = comprehension_count
    features['fstring_count'] = fstring_count

    try_count = sum(1 for l in lines if _RE_TRY.match(l))
    except_count = sum(1 for l in lines if _RE_EXCEPT.match(l))
    features['try_except_count'] = try_count
    features['try_except_rate'] = try_count / n_non_blank

    func_count, func_lens, class_count, deco_count = _ast_function_info(code)
    features['function_count'] = func_count
    features['avg_function_len'] = statistics.mean(func_lens) if func_lens else 0.0
    features['std_function_len'] = statistics.pstdev(func_lens) if len(func_lens) > 1 else 0.0
    features['class_count'] = class_count
    features['decorator_count'] = deco_count

    import_lines = sum(1 for l in lines if _RE_IMPORT.match(l))
    features['import_count'] = import_lines

    total_names, unique_names, single_char_names = _count_unique_names(code)
    features['unique_var_name_ratio'] = unique_names / max(total_names, 1)
    features['single_char_var_ratio'] = single_char_names / max(unique_names, 1)
    features['total_name_count'] = total_names

    snake_funcs = sum(1 for l in lines if _RE_SNAKE_CASE_FUNC.match(l.strip()))
    camel_funcs = sum(1 for l in lines if _RE_CAMEL_CASE_FUNC.match(l.strip()))
    features['naming_snake_case_ratio'] = snake_funcs / max(func_count, 1)
    features['naming_convention_consistent'] = (camel_funcs == 0) if func_count > 0 else True

    type_hint_count = len(_RE_TYPE_HINT.findall(code))
    features['type_hint_count'] = type_hint_count
    features['type_hint_ratio'] = type_hint_count / max(func_count, 1)

    assert_count = sum(1 for l in lines if _RE_ASSERT.match(l))
    features['assert_count'] = assert_count

    global_count = sum(1 for l in lines if _RE_GLOBAL.match(l))
    features['global_usage_count'] = global_count

    code_only_lines = n_non_blank - comment_lines_count
    features['code_density'] = code_only_lines / max(total_lines, 1)

    if indent_widths and indent_mode > 0:
        features['max_nesting_depth'] = max(w // indent_mode for w in indent_widths)
        features['avg_nesting_depth'] = statistics.mean(w / indent_mode for w in indent_widths)
    else:
        features['max_nesting_depth'] = 0
        features['avg_nesting_depth'] = 0.0

    features['total_lines'] = total_lines
    features['total_chars'] = len(code)

    radon = _radon_metrics(code)
    features.update(radon)

    return features
