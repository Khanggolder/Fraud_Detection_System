import re
import math
from typing import Dict, Union, List, Set, Tuple

import tree_sitter_cpp
from tree_sitter import Language, Parser, Node

CppFeatureDict = Dict[str, Union[float, int, bool]]

CPP_LANGUAGE = Language(tree_sitter_cpp.language())

_SMART_PTR_NAMES = frozenset({'unique_ptr', 'shared_ptr', 'weak_ptr', 'make_unique', 'make_shared'})
_CP_SHORTCUT_MACROS = frozenset({'ll', 'ull', 'pb', 'mp', 'fi', 'se', 'all', 'sz', 'rep', 'FOR', 'FAST', 'endl'})
_KNOWN_HEADERS = frozenset({
    'vector', 'string', 'algorithm', 'memory', 'map', 'set', 'queue', 'stack',
    'list', 'deque', 'array', 'numeric', 'functional', 'iostream', 'fstream',
    'sstream', 'iomanip', 'cmath', 'cstdio', 'cstdlib', 'cstring', 'cassert',
    'stdexcept', 'utility', 'tuple', 'optional', 'variant', 'any', 'chrono',
    'thread', 'mutex', 'condition_variable', 'future', 'regex', 'filesystem',
    'unordered_map', 'unordered_set', 'bitset', 'complex', 'random', 'limits',
    'type_traits', 'iterator', 'initializer_list', 'numeric', 'ranges',
    'concepts', 'span', 'format', 'source_location', 'coroutine',
})
_RAII_TYPES = frozenset({
    'lock_guard', 'unique_lock', 'scoped_lock', 'shared_lock',
    'unique_ptr', 'shared_ptr', 'weak_ptr',
    'fstream', 'ifstream', 'ofstream', 'stringstream',
    'thread', 'jthread',
})
_MODERN_CPP_ALGORITHMS = frozenset({
    'std::transform', 'std::for_each', 'std::accumulate', 'std::reduce',
    'std::find_if', 'std::count_if', 'std::any_of', 'std::all_of', 'std::none_of',
    'std::sort', 'std::stable_sort', 'std::partial_sort',
    'std::copy_if', 'std::remove_if', 'std::partition',
})

_SNAKE_RE = re.compile(r'^[a-z][a-z0-9]*(_[a-z0-9]+)+$')
_CAMEL_RE = re.compile(r'^[a-z][a-z0-9]*([A-Z][a-z0-9]*)+$')
_PASCAL_RE = re.compile(r'^[A-Z][a-z0-9]+([A-Z][a-z0-9]*)+$')
_UPPER_SNAKE_RE = re.compile(r'^[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$')
_SINGLE_RE = re.compile(r'^[a-zA-Z]$')
_CODE_IN_COMMENT_RE = re.compile(
    r'^\s*(int |for |if |while |return |cout|cin|printf|scanf|#|void |class |struct |auto |std::)'
)


def _create_parser() -> Parser:
    return Parser(CPP_LANGUAGE)


def _node_text(node: Node, code_bytes: bytes) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')


def _collect_nodes(root: Node) -> Dict[str, List[Node]]:
    result: Dict[str, List[Node]] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        ntype = node.type
        if ntype not in result:
            result[ntype] = []
        result[ntype].append(node)
        for child in node.children:
            stack.append(child)
    return result


def _max_compound_depth(root: Node) -> int:
    best = 0

    def _walk(node: Node, depth: int):
        nonlocal best
        d = depth
        if node.type == 'compound_statement':
            d += 1
            if d > best:
                best = d
        for child in node.children:
            _walk(child, d)

    _walk(root, 0)
    return best


def _avg_compound_depth(root: Node) -> float:
    depths: List[int] = []

    def _walk(node: Node, depth: int):
        d = depth
        if node.type == 'compound_statement':
            d += 1
            depths.append(d)
        for child in node.children:
            _walk(child, d)

    _walk(root, 0)
    return sum(depths) / max(len(depths), 1)


def _leaf_compound_depths(root: Node) -> List[int]:
    results: List[int] = []

    def _walk(node: Node, depth: int):
        d = depth
        if node.type == 'compound_statement':
            d += 1
        has_compound_child = False
        for child in node.children:
            if child.type == 'compound_statement':
                has_compound_child = True
            _walk(child, d)
        if node.type == 'compound_statement' and not has_compound_child:
            results.append(d)

    _walk(root, 0)
    return results


def _analyze_modern_cpp(nbt: Dict[str, List[Node]], cb: bytes) -> Dict[str, int]:
    r: Dict[str, int] = {}

    auto_count = len(nbt.get('auto', []))
    for n in nbt.get('placeholder_type_specifier', []):
        auto_count += 1
    for n in nbt.get('type_identifier', []):
        if _node_text(n, cb) == 'auto':
            auto_count += 1
    r['auto_count'] = auto_count

    r['nullptr_count'] = len(nbt.get('nullptr', []))
    r['template_declaration_count'] = len(nbt.get('template_declaration', []))
    r['lambda_count'] = len(nbt.get('lambda_expression', []))
    r['range_for_count'] = len(nbt.get('for_range_loop', []))
    r['traditional_for_count'] = len(nbt.get('for_statement', []))

    constexpr_c = 0
    for ntype in ('type_qualifier', 'storage_class_specifier'):
        for n in nbt.get(ntype, []):
            if _node_text(n, cb) == 'constexpr':
                constexpr_c += 1
    r['constexpr_count'] = constexpr_c

    consteval_c = 0
    constinit_c = 0
    for ntype in ('type_qualifier', 'storage_class_specifier'):
        for n in nbt.get(ntype, []):
            txt = _node_text(n, cb)
            if txt == 'consteval':
                consteval_c += 1
            elif txt == 'constinit':
                constinit_c += 1
    r['consteval_count'] = consteval_c
    r['constinit_count'] = constinit_c

    noexcept_c = len(nbt.get('noexcept', []))
    for n in nbt.get('function_declarator', []):
        if 'noexcept' in _node_text(n, cb):
            noexcept_c += 1
    r['noexcept_count'] = noexcept_c

    r['structured_binding_count'] = len(nbt.get('structured_binding_declarator', []))

    enum_class_c = 0
    for n in nbt.get('enum_specifier', []):
        t = _node_text(n, cb)
        if 'enum class' in t or 'enum struct' in t:
            enum_class_c += 1
    r['enum_class_count'] = enum_class_c

    override_c = 0
    virtual_c = 0
    for n in nbt.get('virtual_function_specifier', []):
        t = _node_text(n, cb)
        if t == 'override':
            override_c += 1
        elif t == 'virtual':
            virtual_c += 1
    virtual_c += len(nbt.get('virtual', []))
    r['override_count'] = override_c
    r['virtual_count'] = virtual_c

    r['using_alias_count'] = len(nbt.get('alias_declaration', []))

    move_c = 0
    emplace_c = 0
    algo_c = 0
    for n in nbt.get('call_expression', []):
        t = _node_text(n, cb)
        if 'std::move' in t or '::move(' in t:
            move_c += 1
        if 'emplace' in t:
            emplace_c += 1
        for algo in _MODERN_CPP_ALGORITHMS:
            if algo in t:
                algo_c += 1
                break
    r['move_semantics_count'] = move_c
    r['emplace_count'] = emplace_c
    r['modern_algorithm_count'] = algo_c

    concept_c = len(nbt.get('concept_definition', []))
    requires_c = len(nbt.get('requires_clause', []))
    r['concept_count'] = concept_c
    r['requires_clause_count'] = requires_c

    r['if_statement_count'] = len(nbt.get('if_statement', []))
    init_if = 0
    for n in nbt.get('if_statement', []):
        t = _node_text(n, cb)
        if t.startswith('if') and 'if (' not in t[:10] and 'if(' not in t[:10]:
            for child in n.children:
                if child.type == 'init_statement':
                    init_if += 1
                    break
    r['init_if_count'] = init_if

    r['modern_cpp_total'] = (
        r['auto_count'] + r['nullptr_count'] + r['template_declaration_count'] +
        r['lambda_count'] + r['range_for_count'] + r['constexpr_count'] +
        r['noexcept_count'] + r['structured_binding_count'] + r['enum_class_count'] +
        r['override_count'] + r['using_alias_count'] + r['move_semantics_count'] +
        r['emplace_count'] + r['modern_algorithm_count'] + r['concept_count'] +
        r['requires_clause_count'] + r['consteval_count'] + r['constinit_count']
    )
    return r


def _analyze_memory(nbt: Dict[str, List[Node]], cb: bytes) -> Dict[str, Union[int, float]]:
    r: Dict[str, Union[int, float]] = {}
    r['new_expression_count'] = len(nbt.get('new_expression', []))
    r['delete_expression_count'] = len(nbt.get('delete_expression', []))

    smart = 0
    smart_types: Set[str] = set()
    for n in nbt.get('template_type', []):
        t = _node_text(n, cb)
        for name in _SMART_PTR_NAMES:
            if name in t:
                smart += 1
                smart_types.add(name)
                break
    for n in nbt.get('call_expression', []):
        t = _node_text(n, cb)
        if 'make_unique' in t:
            smart += 1
            smart_types.add('make_unique')
        elif 'make_shared' in t:
            smart += 1
            smart_types.add('make_shared')
    r['smart_ptr_count'] = smart
    r['smart_ptr_variety'] = len(smart_types)

    raw_mem = r['new_expression_count'] + r['delete_expression_count']
    total_mem = raw_mem + smart
    r['raii_memory_ratio'] = smart / max(total_mem, 1)

    code_text = cb.decode('utf-8', errors='replace')
    r['deleted_function_count'] = len(re.findall(r'=\s*delete\s*;', code_text))
    r['defaulted_function_count'] = len(re.findall(r'=\s*default\s*;', code_text))
    return r


def _analyze_includes(nbt: Dict[str, List[Node]], cb: bytes) -> Dict[str, Union[int, float, bool]]:
    r: Dict[str, Union[int, float, bool]] = {}
    includes = nbt.get('preproc_include', [])
    r['total_include_count'] = len(includes)

    bits_stdc = False
    headers: List[str] = []
    for node in includes:
        text = _node_text(node, cb)
        if 'bits/stdc++.h' in text:
            bits_stdc = True
        for child in node.children:
            if child.type == 'system_lib_string':
                h = _node_text(child, cb).strip('<>').strip()
                headers.append(h)
            elif child.type == 'string_literal':
                h = _node_text(child, cb).strip('"').strip()
                headers.append(h)

    r['bits_stdc_present'] = bits_stdc
    specific = [h for h in headers if h in _KNOWN_HEADERS]
    r['specific_header_count'] = len(specific)
    r['unique_headers'] = len(set(specific))
    r['specific_header_ratio'] = len(specific) / max(len(headers), 1)
    r['header_list'] = headers
    return r


def _analyze_formatting(nbt: Dict[str, List[Node]], cb: bytes) -> Dict[str, Union[int, float]]:
    r: Dict[str, Union[int, float]] = {}
    binary_nodes = nbt.get('binary_expression', [])[:200]

    spaced = 0
    unspaced = 0
    mixed_spacing = 0
    total = 0
    for node in binary_nodes:
        children = node.children
        if len(children) >= 3:
            left, op, right = children[0], children[1], children[2]
            lg = op.start_byte - left.end_byte
            rg = right.start_byte - op.end_byte
            total += 1
            if lg > 0 and rg > 0:
                spaced += 1
            elif lg == 0 and rg == 0:
                unspaced += 1
            else:
                mixed_spacing += 1

    r['binary_op_total'] = total
    r['binary_op_spaced'] = spaced
    r['binary_op_unspaced'] = unspaced
    r['binary_op_mixed'] = mixed_spacing
    r['binary_op_spacing_rate'] = spaced / max(total, 1)
    r['binary_op_nospace_rate'] = unspaced / max(total, 1)

    arg_lists = nbt.get('argument_list', []) + nbt.get('parameter_list', [])
    comma_total = 0
    comma_spaced = 0
    for anode in arg_lists[:80]:
        for i, child in enumerate(anode.children):
            if child.type == ',':
                comma_total += 1
                if i + 1 < len(anode.children):
                    nxt = anode.children[i + 1]
                    if nxt.start_byte - child.end_byte > 0:
                        comma_spaced += 1

    r['comma_total'] = comma_total
    r['comma_spaced'] = comma_spaced
    r['comma_space_rate'] = comma_spaced / max(comma_total, 1)

    decl_nodes = nbt.get('declaration', [])[:100]
    decl_spacing_consistent = 0
    decl_spacing_total = 0
    for node in decl_nodes:
        children = node.children
        for i in range(len(children) - 1):
            c1, c2 = children[i], children[i + 1]
            gap = c2.start_byte - c1.end_byte
            if gap >= 0:
                decl_spacing_total += 1
                if gap == 1:
                    decl_spacing_consistent += 1
    r['decl_spacing_total'] = decl_spacing_total
    r['decl_spacing_consistency'] = decl_spacing_consistent / max(decl_spacing_total, 1)
    return r


def _analyze_naming(nbt: Dict[str, List[Node]], cb: bytes) -> Dict[str, Union[int, float, bool]]:
    r: Dict[str, Union[int, float, bool]] = {}
    ids: Set[str] = set()
    single_c = 0
    all_id_lengths: List[int] = []
    for ntype in ('identifier', 'field_identifier'):
        for n in nbt.get(ntype, []):
            name = _node_text(n, cb)
            if len(name) > 1:
                ids.add(name)
            all_id_lengths.append(len(name))
            if _SINGLE_RE.match(name):
                single_c += 1

    r['single_char_var_count'] = single_c
    r['unique_identifier_count'] = len(ids)
    r['avg_identifier_length'] = sum(all_id_lengths) / max(len(all_id_lengths), 1)

    snake_c = sum(1 for n in ids if _SNAKE_RE.match(n))
    camel_c = sum(1 for n in ids if _CAMEL_RE.match(n))
    pascal_c = sum(1 for n in ids if _PASCAL_RE.match(n))
    upper_snake_c = sum(1 for n in ids if _UPPER_SNAKE_RE.match(n))
    r['snake_case_count'] = snake_c
    r['camel_case_count'] = camel_c
    r['pascal_case_count'] = pascal_c
    r['upper_snake_case_count'] = upper_snake_c

    total_styled = snake_c + camel_c + pascal_c
    if total_styled > 0:
        dominant = max(snake_c, camel_c, pascal_c)
        r['naming_consistency'] = dominant / total_styled
        r['naming_style_uniform'] = (
            (snake_c == 0 or camel_c == 0) and
            (snake_c == 0 or pascal_c == 0) and
            (camel_c == 0 or pascal_c == 0)
        )
    else:
        r['naming_consistency'] = 1.0
        r['naming_style_uniform'] = True

    func_names: List[str] = []
    for n in nbt.get('function_declarator', []):
        for child in n.children:
            if child.type == 'identifier':
                func_names.append(_node_text(child, cb))
                break
            elif child.type == 'field_identifier':
                func_names.append(_node_text(child, cb))
                break

    func_snake = sum(1 for n in func_names if _SNAKE_RE.match(n))
    func_camel = sum(1 for n in func_names if _CAMEL_RE.match(n))
    func_total = func_snake + func_camel
    r['func_naming_consistency'] = max(func_snake, func_camel) / max(func_total, 1)
    return r


def _analyze_comments(nbt: Dict[str, List[Node]], cb: bytes, lines: List[str]) -> Dict[str, Union[int, float]]:
    r: Dict[str, Union[int, float]] = {}
    comment_nodes = nbt.get('comment', [])
    r['comment_count'] = len(comment_nodes)

    texts: List[str] = []
    for node in comment_nodes:
        t = _node_text(node, cb)
        if t.startswith('//'):
            c = t[2:].strip()
        elif t.startswith('/*') and t.endswith('*/'):
            c = t[2:-2].strip()
        else:
            c = t.strip()
        if c:
            texts.append(c)

    if texts:
        perfect = sum(1 for c in texts if c and c[0].isupper() and c.rstrip().endswith('.'))
        r['perfect_comment_ratio'] = perfect / len(texts)
        cap = sum(1 for c in texts if c and c[0].isupper())
        r['capitalized_comment_ratio'] = cap / len(texts)
        lens = [len(c) for c in texts]
        m = sum(lens) / len(lens)
        r['avg_comment_len'] = m
        r['std_comment_len'] = (sum((l - m) ** 2 for l in lens) / len(lens)) ** 0.5 if len(lens) > 1 else 0.0
    else:
        r['perfect_comment_ratio'] = 0.0
        r['capitalized_comment_ratio'] = 0.0
        r['avg_comment_len'] = 0.0
        r['std_comment_len'] = 0.0

    dead = 0
    inline_between_code = 0
    sorted_comments = sorted(comment_nodes, key=lambda n: n.start_point[0])
    for node in sorted_comments:
        t = _node_text(node, cb)
        content = t[2:].strip() if t.startswith('//') else t
        if _CODE_IN_COMMENT_RE.match(content):
            dead += 1

        line_no = node.start_point[0]
        parent = node.parent
        if parent and parent.type not in ('translation_unit', 'comment'):
            siblings = parent.children
            node_idx = None
            for idx, sib in enumerate(siblings):
                if sib.id == node.id:
                    node_idx = idx
                    break
            if node_idx is not None:
                has_code_before = any(
                    siblings[j].type != 'comment' and siblings[j].type != '{'
                    and siblings[j].type != '}'
                    for j in range(0, node_idx)
                )
                has_code_after = any(
                    siblings[j].type != 'comment' and siblings[j].type != '{'
                    and siblings[j].type != '}'
                    for j in range(node_idx + 1, len(siblings))
                )
                if has_code_before and has_code_after:
                    inline_between_code += 1

    r['dead_code_count'] = dead
    r['interleaved_comment_count'] = inline_between_code

    clines: Set[int] = set()
    for node in comment_nodes:
        for ln in range(node.start_point[0], node.end_point[0] + 1):
            clines.add(ln)
    r['comment_line_count'] = len(clines)
    n_nb = max(sum(1 for l in lines if l.strip()), 1)
    r['comment_line_ratio'] = len(clines) / n_nb
    return r


def _analyze_structure(nbt: Dict[str, List[Node]], cb: bytes) -> Dict[str, Union[int, float, bool]]:
    r: Dict[str, Union[int, float, bool]] = {}
    r['function_count'] = len(nbt.get('function_definition', []))
    r['class_count'] = len(nbt.get('class_specifier', [])) + len(nbt.get('struct_specifier', []))
    r['namespace_count'] = len(nbt.get('namespace_definition', []))
    r['template_count'] = len(nbt.get('template_declaration', []))
    r['try_catch_count'] = len(nbt.get('try_statement', []))
    r['catch_count'] = len(nbt.get('catch_clause', []))
    r['throw_count'] = len(nbt.get('throw_statement', []))
    r['error_handling_present'] = r['try_catch_count'] > 0

    raii = 0
    raii_types_found: Set[str] = set()
    for n in nbt.get('declaration', []):
        t = _node_text(n, cb)
        for g in _RAII_TYPES:
            if g in t:
                raii += 1
                raii_types_found.add(g)
                break
    r['raii_guard_count'] = raii
    r['raii_type_variety'] = len(raii_types_found)

    cref = 0
    for n in nbt.get('parameter_declaration', []):
        t = _node_text(n, cb)
        if 'const' in t and '&' in t:
            cref += 1
    r['const_ref_count'] = cref

    cm = 0
    for n in nbt.get('function_declarator', []):
        p = n.parent
        if p and ') const' in _node_text(p, cb):
            cm += 1
    r['const_method_count'] = cm

    static_assert_c = 0
    for n in nbt.get('static_assert_declaration', []):
        static_assert_c += 1
    r['static_assert_count'] = static_assert_c

    return_c = len(nbt.get('return_statement', []))
    r['return_statement_count'] = return_c

    r['ternary_count'] = len(nbt.get('conditional_expression', []))

    goto_c = len(nbt.get('goto_statement', []))
    r['goto_count'] = goto_c
    return r


def _analyze_competitive(nbt: Dict[str, List[Node]], cb: bytes, bits_stdc: bool) -> Dict[str, Union[int, bool]]:
    r: Dict[str, Union[int, bool]] = {}

    using_ns_std = False
    for n in nbt.get('using_declaration', []):
        if 'namespace std' in _node_text(n, cb):
            using_ns_std = True
            break
    r['using_namespace_std'] = using_ns_std

    la_count = 0
    la_max = 0
    for n in nbt.get('array_declarator', []):
        for child in n.children:
            if child.type == 'number_literal':
                try:
                    v = int(_node_text(child, cb))
                    if v >= 10000:
                        la_count += 1
                        la_max = max(la_max, v)
                except ValueError:
                    pass
    r['large_array_count'] = la_count
    r['large_array_max_size'] = la_max

    pdefs = nbt.get('preproc_def', []) + nbt.get('preproc_function_def', [])
    r['define_macro_count'] = len(pdefs)
    sc = 0
    for n in pdefs:
        nn = n.child_by_field_name('name')
        if nn and _node_text(nn, cb) in _CP_SHORTCUT_MACROS:
            sc += 1
    r['define_shortcut_count'] = sc

    ios_sync = False
    cin_tie = False
    freopen = False
    for n in nbt.get('call_expression', []):
        t = _node_text(n, cb)
        if 'sync_with_stdio' in t:
            ios_sync = True
        if 'cin.tie' in t or 'cin .tie' in t:
            cin_tie = True
        if 'freopen' in t:
            freopen = True
    r['ios_sync_present'] = ios_sync
    r['cin_tie_present'] = cin_tie
    r['freopen_present'] = freopen

    void_main = False
    for n in nbt.get('function_definition', []):
        t = _node_text(n, cb)
        if t.lstrip().startswith('void main') or t.lstrip().startswith('void\nmain'):
            void_main = True
            break
    r['void_main'] = void_main

    r['competitive_pattern_count'] = (
        int(bits_stdc) + int(using_ns_std) + int(freopen) +
        int(ios_sync) + int(cin_tie) + sc + la_count + int(void_main)
    )
    return r


def _analyze_line_metrics(lines: List[str]) -> Dict[str, Union[int, float, bool]]:
    r: Dict[str, Union[int, float, bool]] = {}
    total = len(lines)
    non_blank = [l for l in lines if l.strip()]
    r['total_lines'] = total
    r['total_chars'] = sum(len(l) for l in lines) + max(total - 1, 0)

    lens = [len(l) for l in lines]
    m = sum(lens) / max(len(lens), 1)
    r['line_len_mean'] = m
    r['line_len_std'] = (sum((l - m) ** 2 for l in lens) / len(lens)) ** 0.5 if len(lens) > 1 else 0.0
    r['line_len_max'] = max(lens) if lens else 0

    nb_lens = [len(l) for l in non_blank]
    if nb_lens:
        nb_m = sum(nb_lens) / len(nb_lens)
        r['nb_line_len_mean'] = nb_m
        r['nb_line_len_std'] = (sum((l - nb_m) ** 2 for l in nb_lens) / len(nb_lens)) ** 0.5 if len(nb_lens) > 1 else 0.0
        r['nb_line_len_cv'] = r['nb_line_len_std'] / nb_m if nb_m > 0 else 0.0
    else:
        r['nb_line_len_mean'] = 0.0
        r['nb_line_len_std'] = 0.0
        r['nb_line_len_cv'] = 0.0

    blank_c = total - len(non_blank)
    r['blank_line_ratio'] = blank_c / max(total, 1)

    mx = 0
    cur = 0
    for l in lines:
        if not l.strip():
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    r['max_consecutive_blank_lines'] = mx

    widths: List[int] = []
    tab_l = 0
    space_l = 0
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
    r['indent_width_mode'] = imode
    r['indent_inconsistency_count'] = incon
    r['indent_inconsistency_ratio'] = incon / max(len(widths), 1)

    r['trailing_whitespace_ratio'] = sum(1 for l in lines if l != l.rstrip()) / max(total, 1)

    bsame = 0
    bnext = 0
    for i, l in enumerate(lines):
        s = l.rstrip()
        if s.endswith('{') and ')' in s:
            bsame += 1
        elif s.endswith(')') and i + 1 < total and lines[i + 1].strip().startswith('{'):
            bnext += 1
    tb = bsame + bnext
    r['brace_same_line_count'] = bsame
    r['brace_next_line_count'] = bnext
    r['brace_style_consistency'] = max(bsame, bnext) / max(tb, 1) if tb > 0 else 1.0
    return r


def _count_global_declarations(root: Node, cb: bytes) -> int:
    c = 0
    for child in root.children:
        if child.type == 'declaration':
            has_body = any(sub.type == 'compound_statement' for sub in child.children)
            if not has_body:
                c += 1
    return c


def _analyze_complexity(nbt: Dict[str, List[Node]], root: Node) -> Dict[str, Union[int, float]]:
    r: Dict[str, Union[int, float]] = {}
    branch_nodes = (
        len(nbt.get('if_statement', [])) +
        len(nbt.get('for_statement', [])) +
        len(nbt.get('for_range_loop', [])) +
        len(nbt.get('while_statement', [])) +
        len(nbt.get('do_statement', [])) +
        len(nbt.get('switch_statement', [])) +
        len(nbt.get('catch_clause', [])) +
        len(nbt.get('conditional_expression', []))
    )
    func_count = max(len(nbt.get('function_definition', [])), 1)
    r['cyclomatic_branch_count'] = branch_nodes
    r['avg_cyclomatic_per_function'] = branch_nodes / func_count

    leaf_depths = _leaf_compound_depths(root)
    r['leaf_nesting_mean'] = sum(leaf_depths) / max(len(leaf_depths), 1)
    return r


_OPERATOR_NODE_TYPES = frozenset({
    'binary_expression', 'unary_expression', 'assignment_expression',
    'update_expression', 'compound_assignment_expr', 'pointer_expression',
    'subscript_expression', 'call_expression', 'field_expression',
    'conditional_expression', 'cast_expression', 'sizeof_expression',
    'new_expression', 'delete_expression', 'throw_statement',
    'co_await_expression', 'co_yield_expression',
})
_OPERAND_NODE_TYPES = frozenset({
    'identifier', 'field_identifier', 'number_literal', 'string_literal',
    'char_literal', 'true', 'false', 'nullptr', 'this',
    'type_identifier', 'namespace_identifier',
})


def _analyze_halstead(nbt: Dict[str, List[Node]], cb: bytes) -> Dict[str, float]:
    r: Dict[str, float] = {}

    operators: Dict[str, int] = {}
    operands: Dict[str, int] = {}

    for ntype in _OPERATOR_NODE_TYPES:
        for node in nbt.get(ntype, []):
            for child in node.children:
                if child.type in (
                    '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
                    '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--',
                    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=',
                    '->', '.', '::', '?', ':', ',', ';',
                ):
                    op_text = child.type
                    operators[op_text] = operators.get(op_text, 0) + 1

    for ntype in _OPERAND_NODE_TYPES:
        for node in nbt.get(ntype, []):
            txt = _node_text(node, cb)
            operands[txt] = operands.get(txt, 0) + 1

    n1 = len(operators)
    n2 = len(operands)
    cap_n1 = sum(operators.values())
    cap_n2 = sum(operands.values())

    r['halstead_unique_operators'] = n1
    r['halstead_unique_operands'] = n2
    r['halstead_total_operators'] = cap_n1
    r['halstead_total_operands'] = cap_n2

    vocabulary = n1 + n2
    length = cap_n1 + cap_n2
    r['halstead_vocabulary'] = vocabulary
    r['halstead_length'] = length

    if vocabulary > 0 and length > 0:
        volume = length * math.log2(max(vocabulary, 2))
    else:
        volume = 0.0
    r['halstead_volume'] = round(volume, 2)

    if n2 > 0 and cap_n2 > 0:
        difficulty = (n1 / 2.0) * (cap_n2 / n2)
    else:
        difficulty = 0.0
    r['halstead_difficulty'] = round(difficulty, 2)

    effort = volume * difficulty
    r['halstead_effort'] = round(effort, 2)

    if difficulty > 0:
        bugs_estimate = volume / 3000.0
    else:
        bugs_estimate = 0.0
    r['halstead_bugs_estimate'] = round(bugs_estimate, 4)

    return r


def _calculate_maintainability_index(
    halstead_volume: float,
    cyclomatic: float,
    loc: int,
    comment_ratio: float,
) -> float:
    if loc <= 0 or halstead_volume <= 0:
        return 0.0

    ln_v = math.log(max(halstead_volume, 1.0))
    ln_loc = math.log(max(loc, 1))

    mi = 171.0 - 5.2 * ln_v - 0.23 * cyclomatic - 16.2 * ln_loc
    mi += 50.0 * math.sin(math.sqrt(2.4 * comment_ratio))
    mi = max(0.0, min(mi, 100.0))
    return round(mi, 2)


def extract_cpp_ast_features(code: str) -> CppFeatureDict:
    parser = _create_parser()
    cb = code.encode('utf-8')
    tree = parser.parse(cb)
    root = tree.root_node
    lines = code.split('\n')
    n_nb = max(sum(1 for l in lines if l.strip()), 1)

    nbt = _collect_nodes(root)
    features: CppFeatureDict = {}

    features['max_nesting_depth'] = _max_compound_depth(root)
    features['avg_nesting_depth'] = _avg_compound_depth(root)

    modern = _analyze_modern_cpp(nbt, cb)
    features.update(modern)
    features['modern_cpp_ratio'] = modern['modern_cpp_total'] / n_nb

    features.update(_analyze_memory(nbt, cb))
    includes = _analyze_includes(nbt, cb)
    header_list = includes.pop('header_list', [])
    features.update(includes)
    features.update(_analyze_formatting(nbt, cb))
    features.update(_analyze_naming(nbt, cb))
    features.update(_analyze_comments(nbt, cb, lines))
    features.update(_analyze_structure(nbt, cb))

    cp = _analyze_competitive(nbt, cb, includes.get('bits_stdc_present', False))
    features.update(cp)
    features['global_var_count'] = _count_global_declarations(root, cb)
    features.update(_analyze_line_metrics(lines))
    features.update(_analyze_complexity(nbt, root))

    halstead = _analyze_halstead(nbt, cb)
    features.update(halstead)

    features['maintainability_index'] = _calculate_maintainability_index(
        halstead_volume=halstead['halstead_volume'],
        cyclomatic=features.get('cyclomatic_branch_count', 0),
        loc=n_nb,
        comment_ratio=features.get('comment_line_ratio', 0.0),
    )

    features['ast_node_count'] = sum(len(v) for v in nbt.values())
    features['ast_error_count'] = len(nbt.get('ERROR', []))

    tf = features.get('range_for_count', 0) + features.get('traditional_for_count', 0)
    features['range_for_ratio'] = features.get('range_for_count', 0) / max(tf, 1)

    features['ast_error_ratio'] = features['ast_error_count'] / max(features['ast_node_count'], 1)
    features['code_density'] = n_nb / max(features['total_lines'], 1)
    return features
