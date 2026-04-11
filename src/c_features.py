"""
C Code Feature Extraction using Tree‑sitter.
Completely independent from cpp_features.py.
"""
import re
import math
from typing import Dict, Union, List, Set

import tree_sitter_c
from tree_sitter import Language, Parser, Node

CFeatureDict = Dict[str, Union[float, int, bool]]

C_LANGUAGE = Language(tree_sitter_c.language())
_PARSER = Parser(C_LANGUAGE)

_SNAKE_RE = re.compile(r'^[a-z][a-z0-9]*(_[a-z0-9]+)+$')
_CAMEL_RE = re.compile(r'^[a-z][a-z0-9]*([A-Z][a-z0-9]*)+$')
_UPPER_RE = re.compile(r'^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$')
_SINGLE_RE = re.compile(r'^[a-zA-Z]$')

_CODE_IN_COMMENT_RE = re.compile(
    r'^\s*(int |for |if |while |return |printf|scanf|#|void |struct |char |float |double )'
)

_C_STD_HEADERS = frozenset({
    'stdio.h', 'stdlib.h', 'string.h', 'math.h', 'ctype.h', 'time.h',
    'assert.h', 'errno.h', 'float.h', 'limits.h', 'locale.h', 'setjmp.h',
    'signal.h', 'stdarg.h', 'stddef.h', 'stdbool.h', 'stdint.h', 'inttypes.h',
    'complex.h', 'fenv.h', 'iso646.h', 'tgmath.h', 'wchar.h', 'wctype.h',
})

_OPERATOR_TYPES = frozenset({
    'binary_expression', 'unary_expression', 'assignment_expression',
    'update_expression', 'compound_assignment_expr', 'pointer_expression',
    'subscript_expression', 'call_expression', 'field_expression',
    'conditional_expression', 'cast_expression', 'sizeof_expression',
})
_OPERAND_TYPES = frozenset({
    'identifier', 'field_identifier', 'number_literal', 'string_literal',
    'char_literal', 'true', 'false', 'null', 'type_identifier',
})


# ── Helpers ──────────────────────────────────────────────

def _text(node: Node, cb: bytes) -> str:
    return cb[node.start_byte:node.end_byte].decode('utf-8', errors='replace')


def _collect(root: Node) -> Dict[str, List[Node]]:
    result: Dict[str, List[Node]] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        result.setdefault(n.type, []).append(n)
        for c in n.children:
            stack.append(c)
    return result


def _max_depth(root: Node) -> int:
    best = 0
    def _w(n: Node, d: int):
        nonlocal best
        dd = d + 1 if n.type == 'compound_statement' else d
        if dd > best:
            best = dd
        for c in n.children:
            _w(c, dd)
    _w(root, 0)
    return best


def _avg_depth(root: Node) -> float:
    depths: List[int] = []
    def _w(n: Node, d: int):
        dd = d + 1 if n.type == 'compound_statement' else d
        if n.type == 'compound_statement':
            depths.append(dd)
        for c in n.children:
            _w(c, dd)
    _w(root, 0)
    return sum(depths) / max(len(depths), 1)


# ── Analysis Modules ─────────────────────────────────────

def _analyze_structure(nbt: Dict[str, List[Node]], cb: bytes, root: Node) -> CFeatureDict:
    r: CFeatureDict = {}
    r['function_count'] = len(nbt.get('function_definition', []))
    r['struct_count'] = len(nbt.get('struct_specifier', []))
    r['union_count'] = len(nbt.get('union_specifier', []))
    r['enum_count'] = len(nbt.get('enum_specifier', []))

    typedef_c = 0
    for n in nbt.get('type_definition', []):
        typedef_c += 1
    r['typedef_count'] = typedef_c

    # Global variables
    g = 0
    for child in root.children:
        if child.type == 'declaration':
            if not any(sub.type == 'compound_statement' for sub in child.children):
                g += 1
    r['global_var_count'] = g

    r['goto_count'] = len(nbt.get('goto_statement', []))
    r['return_count'] = len(nbt.get('return_statement', []))
    r['switch_count'] = len(nbt.get('switch_statement', []))
    r['case_count'] = len(nbt.get('case_statement', []))
    r['ternary_count'] = len(nbt.get('conditional_expression', []))

    # Static keyword usage
    static_c = 0
    for n in nbt.get('storage_class_specifier', []):
        if _text(n, cb) == 'static':
            static_c += 1
    r['static_count'] = static_c

    # Const usage
    const_c = 0
    for n in nbt.get('type_qualifier', []):
        if _text(n, cb) == 'const':
            const_c += 1
    r['const_count'] = const_c

    return r


def _analyze_memory(nbt: Dict[str, List[Node]], cb: bytes) -> CFeatureDict:
    r: CFeatureDict = {}
    malloc_c = calloc_c = realloc_c = free_c = 0

    for n in nbt.get('call_expression', []):
        t = _text(n, cb)
        if 'malloc(' in t or 'malloc (' in t:
            malloc_c += 1
        if 'calloc(' in t or 'calloc (' in t:
            calloc_c += 1
        if 'realloc(' in t or 'realloc (' in t:
            realloc_c += 1
        if 'free(' in t or 'free (' in t:
            free_c += 1

    r['malloc_count'] = malloc_c
    r['calloc_count'] = calloc_c
    r['realloc_count'] = realloc_c
    r['free_count'] = free_c
    r['total_alloc'] = malloc_c + calloc_c + realloc_c
    r['memory_pair_ratio'] = min(free_c, r['total_alloc']) / max(r['total_alloc'], 1)

    # NULL checks after allocation
    null_check = 0
    for n in nbt.get('if_statement', []):
        t = _text(n, cb)
        if 'NULL' in t or 'null' in t or '== 0)' in t:
            null_check += 1
    r['null_check_count'] = null_check
    r['null_check_ratio'] = null_check / max(r['total_alloc'], 1)

    return r


def _analyze_io(nbt: Dict[str, List[Node]], cb: bytes) -> CFeatureDict:
    r: CFeatureDict = {}
    printf_c = scanf_c = fprintf_c = fscanf_c = fopen_c = fclose_c = 0
    puts_c = gets_c = fgets_c = 0

    for n in nbt.get('call_expression', []):
        t = _text(n, cb)
        if 'printf(' in t and 'fprintf(' not in t and 'sprintf(' not in t:
            printf_c += 1
        if 'scanf(' in t and 'fscanf(' not in t and 'sscanf(' not in t:
            scanf_c += 1
        if 'fprintf(' in t: fprintf_c += 1
        if 'fscanf(' in t: fscanf_c += 1
        if 'fopen(' in t: fopen_c += 1
        if 'fclose(' in t: fclose_c += 1
        if 'puts(' in t: puts_c += 1
        if 'gets(' in t: gets_c += 1
        if 'fgets(' in t: fgets_c += 1

    r['printf_count'] = printf_c
    r['scanf_count'] = scanf_c
    r['fprintf_count'] = fprintf_c
    r['fscanf_count'] = fscanf_c
    r['fopen_count'] = fopen_c
    r['fclose_count'] = fclose_c
    r['puts_count'] = puts_c
    r['gets_count'] = gets_c
    r['fgets_count'] = fgets_c
    r['file_pair_ratio'] = min(fopen_c, fclose_c) / max(fopen_c, 1)
    return r


def _analyze_includes(nbt: Dict[str, List[Node]], cb: bytes) -> CFeatureDict:
    r: CFeatureDict = {}
    includes = nbt.get('preproc_include', [])
    r['include_count'] = len(includes)

    headers: List[str] = []
    for node in includes:
        for child in node.children:
            if child.type in ('system_lib_string', 'string_literal'):
                h = _text(child, cb).strip('<>"').strip()
                headers.append(h)

    std_c = sum(1 for h in headers if h in _C_STD_HEADERS)
    r['std_header_count'] = std_c
    r['custom_header_count'] = len(headers) - std_c
    r['header_diversity'] = len(set(headers))
    return r


def _analyze_preprocessor(nbt: Dict[str, List[Node]], cb: bytes) -> CFeatureDict:
    r: CFeatureDict = {}
    defs = nbt.get('preproc_def', [])
    func_defs = nbt.get('preproc_function_def', [])
    r['define_count'] = len(defs) + len(func_defs)
    r['define_func_count'] = len(func_defs)
    r['ifdef_count'] = len(nbt.get('preproc_ifdef', []))
    r['ifndef_count'] = len(nbt.get('preproc_ifndef', []))
    r['pragma_count'] = len(nbt.get('preproc_pragma', []))

    # Check if defines have descriptive names
    desc_names = 0
    for n in defs + func_defs:
        nn = n.child_by_field_name('name')
        if nn:
            name = _text(nn, cb)
            if len(name) >= 4 and '_' in name:
                desc_names += 1
    r['define_descriptive_ratio'] = desc_names / max(len(defs) + len(func_defs), 1)
    return r


def _analyze_naming(nbt: Dict[str, List[Node]], cb: bytes) -> CFeatureDict:
    r: CFeatureDict = {}
    ids: Set[str] = set()
    single_c = 0
    all_lens: List[int] = []

    for ntype in ('identifier', 'field_identifier'):
        for n in nbt.get(ntype, []):
            name = _text(n, cb)
            if len(name) > 1:
                ids.add(name)
            all_lens.append(len(name))
            if _SINGLE_RE.match(name):
                single_c += 1

    r['single_char_var_count'] = single_c
    r['unique_id_count'] = len(ids)
    r['avg_id_length'] = sum(all_lens) / max(len(all_lens), 1)

    snake = sum(1 for n in ids if _SNAKE_RE.match(n))
    camel = sum(1 for n in ids if _CAMEL_RE.match(n))
    upper = sum(1 for n in ids if _UPPER_RE.match(n))
    r['snake_case_count'] = snake
    r['camel_case_count'] = camel
    r['upper_case_count'] = upper

    styled = snake + camel
    if styled > 0:
        dominant = max(snake, camel)
        r['naming_consistency'] = dominant / styled
        r['naming_uniform'] = (snake == 0 or camel == 0)
    else:
        r['naming_consistency'] = 1.0
        r['naming_uniform'] = True

    # Function naming
    fnames: List[str] = []
    for n in nbt.get('function_declarator', []):
        for child in n.children:
            if child.type == 'identifier':
                fnames.append(_text(child, cb))
                break

    r['func_count_named'] = len(fnames)
    descriptive = sum(1 for fn in fnames if len(fn) >= 5 and '_' in fn)
    r['func_descriptive_ratio'] = descriptive / max(len(fnames), 1)

    short_funcs = sum(1 for fn in fnames if len(fn) <= 3 and fn != 'main')
    r['func_short_name_count'] = short_funcs

    return r


def _analyze_comments(nbt: Dict[str, List[Node]], cb: bytes, lines: List[str]) -> CFeatureDict:
    r: CFeatureDict = {}
    comments = nbt.get('comment', [])
    r['comment_count'] = len(comments)

    texts: List[str] = []
    for node in comments:
        t = _text(node, cb)
        if t.startswith('//'):
            c = t[2:].strip()
        elif t.startswith('/*') and t.endswith('*/'):
            c = t[2:-2].strip()
        else:
            c = t.strip()
        if c:
            texts.append(c)

    if texts:
        # Perfect = starts uppercase, ends with period
        perfect = sum(1 for c in texts if c and c[0].isupper() and c.rstrip().endswith('.'))
        r['perfect_comment_ratio'] = perfect / len(texts)

        cap = sum(1 for c in texts if c and c[0].isupper())
        r['capitalized_comment_ratio'] = cap / len(texts)

        lens = [len(c) for c in texts]
        m = sum(lens) / len(lens)
        r['avg_comment_len'] = m
        r['std_comment_len'] = (sum((l - m) ** 2 for l in lens) / len(lens)) ** 0.5 if len(lens) > 1 else 0.0

        # Explanatory comments (long, sentence-like)
        explanatory = sum(1 for c in texts if len(c) >= 15 and ' ' in c)
        r['explanatory_comment_ratio'] = explanatory / len(texts)

        # Step-by-step comments (numbered or sequential)
        step_patterns = sum(1 for c in texts if
            c[:2] in ('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.') or
            c.lower().startswith(('step ', 'first', 'then', 'next', 'finally'))
        )
        r['step_comment_count'] = step_patterns

        # Function documentation style
        doc_style = sum(1 for c in texts if
            any(kw in c.lower() for kw in ['param', 'return', 'description', 'purpose', 'input', 'output', 'args', 'function'])
        )
        r['doc_comment_count'] = doc_style
    else:
        r['perfect_comment_ratio'] = 0.0
        r['capitalized_comment_ratio'] = 0.0
        r['avg_comment_len'] = 0.0
        r['std_comment_len'] = 0.0
        r['explanatory_comment_ratio'] = 0.0
        r['step_comment_count'] = 0
        r['doc_comment_count'] = 0

    # Dead code in comments
    dead = 0
    for node in comments:
        t = _text(node, cb)
        content = t[2:].strip() if t.startswith('//') else t
        if _CODE_IN_COMMENT_RE.match(content):
            dead += 1
    r['dead_code_count'] = dead

    # Comment lines
    clines: Set[int] = set()
    for node in comments:
        for ln in range(node.start_point[0], node.end_point[0] + 1):
            clines.add(ln)
    r['comment_line_count'] = len(clines)
    n_nb = max(sum(1 for l in lines if l.strip()), 1)
    r['comment_line_ratio'] = len(clines) / n_nb
    return r


def _analyze_formatting(nbt: Dict[str, List[Node]], cb: bytes) -> CFeatureDict:
    r: CFeatureDict = {}

    # Binary operator spacing
    binaries = nbt.get('binary_expression', [])[:200]
    spaced = unspaced = mixed = total = 0
    for node in binaries:
        ch = node.children
        if len(ch) >= 3:
            left, op, right = ch[0], ch[1], ch[2]
            lg = op.start_byte - left.end_byte
            rg = right.start_byte - op.end_byte
            total += 1
            if lg > 0 and rg > 0: spaced += 1
            elif lg == 0 and rg == 0: unspaced += 1
            else: mixed += 1

    r['op_spacing_total'] = total
    r['op_spacing_rate'] = spaced / max(total, 1)

    # Comma spacing
    arglists = nbt.get('argument_list', []) + nbt.get('parameter_list', [])
    comma_t = comma_s = 0
    for a in arglists[:80]:
        for i, ch in enumerate(a.children):
            if ch.type == ',':
                comma_t += 1
                if i + 1 < len(a.children):
                    nxt = a.children[i + 1]
                    if nxt.start_byte - ch.end_byte > 0:
                        comma_s += 1
    r['comma_total'] = comma_t
    r['comma_space_rate'] = comma_s / max(comma_t, 1)

    return r


def _analyze_lines(lines: List[str]) -> CFeatureDict:
    r: CFeatureDict = {}
    total = len(lines)
    non_blank = [l for l in lines if l.strip()]
    r['total_lines'] = total
    r['total_chars'] = sum(len(l) for l in lines) + max(total - 1, 0)

    lens = [len(l) for l in lines]
    m = sum(lens) / max(len(lens), 1)
    r['line_len_mean'] = m
    r['line_len_std'] = (sum((x - m) ** 2 for x in lens) / len(lens)) ** 0.5 if len(lens) > 1 else 0.0
    r['line_len_max'] = max(lens) if lens else 0

    nb_lens = [len(l) for l in non_blank]
    if nb_lens:
        nb_m = sum(nb_lens) / len(nb_lens)
        nb_std = (sum((x - nb_m) ** 2 for x in nb_lens) / len(nb_lens)) ** 0.5 if len(nb_lens) > 1 else 0.0
        r['nb_line_len_mean'] = nb_m
        r['nb_line_len_std'] = nb_std
        r['nb_line_len_cv'] = nb_std / nb_m if nb_m > 0 else 0.0
    else:
        r['nb_line_len_mean'] = 0.0
        r['nb_line_len_std'] = 0.0
        r['nb_line_len_cv'] = 0.0

    blank = total - len(non_blank)
    r['blank_line_ratio'] = blank / max(total, 1)

    # Max consecutive blank lines
    mx = cur = 0
    for l in lines:
        if not l.strip():
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    r['max_consec_blank'] = mx

    # Indent analysis
    tab_l = space_l = 0
    widths: List[int] = []
    for l in lines:
        s = l.lstrip()
        if not s:
            continue
        ns = len(l) - len(s)
        if ns > 0:
            widths.append(ns)
        if l and l[0] == '\t':
            tab_l += 1
        elif l and l[0] == ' ':
            space_l += 1

    r['tab_indent_lines'] = tab_l
    r['space_indent_lines'] = space_l
    r['mixed_indent'] = tab_l > 0 and space_l > 0

    if widths:
        counts = {2: 0, 4: 0, 8: 0}
        for w in widths:
            for step in (2, 4, 8):
                if w % step == 0:
                    counts[step] += 1
        imode = max(counts, key=counts.get)
        incon = sum(1 for w in widths if w % imode != 0)
    else:
        imode = 4
        incon = 0
    r['indent_mode'] = imode
    r['indent_inconsistency'] = incon
    r['indent_inconsistency_ratio'] = incon / max(len(widths), 1)

    r['trailing_ws_ratio'] = sum(1 for l in lines if l != l.rstrip()) / max(total, 1)

    # Brace style
    bsame = bnext = 0
    for i, l in enumerate(lines):
        s = l.rstrip()
        if s.endswith('{') and ')' in s:
            bsame += 1
        elif s.endswith(')') and i + 1 < total and lines[i + 1].strip().startswith('{'):
            bnext += 1
    tb = bsame + bnext
    r['brace_same_line'] = bsame
    r['brace_next_line'] = bnext
    r['brace_consistency'] = max(bsame, bnext) / max(tb, 1) if tb > 0 else 1.0

    r['code_density'] = len(non_blank) / max(total, 1)
    return r


def _analyze_complexity(nbt: Dict[str, List[Node]], root: Node) -> CFeatureDict:
    r: CFeatureDict = {}
    branches = (
        len(nbt.get('if_statement', [])) +
        len(nbt.get('for_statement', [])) +
        len(nbt.get('while_statement', [])) +
        len(nbt.get('do_statement', [])) +
        len(nbt.get('switch_statement', [])) +
        len(nbt.get('case_statement', [])) +
        len(nbt.get('conditional_expression', []))
    )
    funcs = max(len(nbt.get('function_definition', [])), 1)
    r['cyclomatic_branches'] = branches
    r['avg_cyclomatic'] = branches / funcs
    r['max_nesting'] = _max_depth(root)
    r['avg_nesting'] = _avg_depth(root)
    return r


def _analyze_halstead(nbt: Dict[str, List[Node]], cb: bytes) -> CFeatureDict:
    r: CFeatureDict = {}
    operators: Dict[str, int] = {}
    operands: Dict[str, int] = {}

    for ntype in _OPERATOR_TYPES:
        for node in nbt.get(ntype, []):
            for child in node.children:
                if child.type in (
                    '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
                    '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--',
                    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=',
                    '->', '.', '?', ':', ',', ';',
                ):
                    operators[child.type] = operators.get(child.type, 0) + 1

    for ntype in _OPERAND_TYPES:
        for node in nbt.get(ntype, []):
            txt = _text(node, cb)
            operands[txt] = operands.get(txt, 0) + 1

    n1, n2 = len(operators), len(operands)
    N1, N2 = sum(operators.values()), sum(operands.values())
    vocab = n1 + n2
    length = N1 + N2

    r['halstead_n1'] = n1
    r['halstead_n2'] = n2
    r['halstead_N1'] = N1
    r['halstead_N2'] = N2
    r['halstead_vocabulary'] = vocab
    r['halstead_length'] = length

    volume = length * math.log2(max(vocab, 2)) if vocab > 0 and length > 0 else 0.0
    r['halstead_volume'] = round(volume, 2)

    difficulty = (n1 / 2.0) * (N2 / n2) if n2 > 0 and N2 > 0 else 0.0
    r['halstead_difficulty'] = round(difficulty, 2)

    r['halstead_effort'] = round(volume * difficulty, 2)
    r['halstead_bugs'] = round(volume / 3000.0, 4) if difficulty > 0 else 0.0
    return r


def _calc_maintainability(volume: float, cyclo: float, loc: int, comment_ratio: float) -> float:
    if loc <= 0 or volume <= 0:
        return 0.0
    lnv = math.log(max(volume, 1.0))
    lnloc = math.log(max(loc, 1))
    mi = 171.0 - 5.2 * lnv - 0.23 * cyclo - 16.2 * lnloc
    mi += 50.0 * math.sin(math.sqrt(2.4 * comment_ratio))
    return round(max(0.0, min(mi, 100.0)), 2)


# ── Main Entry Point ─────────────────────────────────────

def extract_c_features(code: str) -> CFeatureDict:
    """Extract ~65 features from C source code using tree-sitter AST."""
    cb = code.encode('utf-8')
    tree = _PARSER.parse(cb)
    root = tree.root_node
    lines = code.split('\n')
    n_nb = max(sum(1 for l in lines if l.strip()), 1)

    nbt = _collect(root)
    features: CFeatureDict = {}

    features.update(_analyze_structure(nbt, cb, root))
    features.update(_analyze_memory(nbt, cb))
    features.update(_analyze_io(nbt, cb))
    features.update(_analyze_includes(nbt, cb))
    features.update(_analyze_preprocessor(nbt, cb))
    features.update(_analyze_naming(nbt, cb))
    features.update(_analyze_comments(nbt, cb, lines))
    features.update(_analyze_formatting(nbt, cb))
    features.update(_analyze_lines(lines))
    features.update(_analyze_complexity(nbt, root))

    halstead = _analyze_halstead(nbt, cb)
    features.update(halstead)

    features['maintainability_index'] = _calc_maintainability(
        volume=halstead['halstead_volume'],
        cyclo=features.get('cyclomatic_branches', 0),
        loc=n_nb,
        comment_ratio=features.get('comment_line_ratio', 0.0),
    )

    features['ast_node_count'] = sum(len(v) for v in nbt.values())
    features['ast_error_count'] = len(nbt.get('ERROR', []))
    features['ast_error_ratio'] = features['ast_error_count'] / max(features['ast_node_count'], 1)

    return features
