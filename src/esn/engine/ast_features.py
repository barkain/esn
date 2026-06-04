# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Generic AST-based structural features for branch identity.

Replacement for the per-domain regex/keyword classifiers
(``aspect_motifs``, ``family_classifier``, ``classify_family_generic``).

The contract: given any Python program, return a deterministic, domain-free
fingerprint describing the program's *algorithmic shape*. Two programs that
implement the same algorithm with renamed variables produce the same
fingerprint. Two programs with structurally distinct control flow produce
different fingerprints. No domain-specific patterns. No regex on raw source.

Output:

    {
      "features": list[str],   # sorted feature tags (deterministic)
      "family":   str,         # one of the coarse structural buckets
      "cfhash":   str,         # 8-char sha256 prefix of control-flow shape
    }

Family buckets (mutually exclusive, exhaustive):

    recursive-multi    program contains a function with >=2 self-calls
    recursive-tail     program contains a recursive function (1 self-call)
    iterative-nested   no recursion, max loop nesting depth >= 2
    iterative-flat     no recursion, max loop nesting depth == 1
    straight-line      no recursion, no loops
    unparseable        ast.parse raised SyntaxError
"""

from __future__ import annotations

import ast
import hashlib
from typing import Any

# Control-flow node types contributing to the structural hash. Anything not in
# this set is ignored for hashing purposes — variable names, constants, and
# expression-level nodes do not affect the hash.
_CFLOW_TYPES: tuple[type[ast.AST], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.Try,
    ast.Return,
    ast.Break,
    ast.Continue,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
)


def _bucket_count(n: int, edges: tuple[int, ...]) -> str:
    """Bucket integer ``n`` according to ``edges`` (sorted, non-decreasing)."""
    for i, edge in enumerate(edges):
        if n <= edge:
            if i == 0:
                return f"{n}" if edge == 1 else f"<={edge}"
            return f"{edges[i - 1] + 1}-{edge}"
    return f"{edges[-1] + 1}+"


def _max_loop_depth(node: ast.AST, current: int = 0) -> int:
    """Deepest nesting of For/While inside ``node``."""
    deepest = current
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
            deepest = max(deepest, _max_loop_depth(child, current + 1))
        else:
            deepest = max(deepest, _max_loop_depth(child, current))
    return deepest


def _self_call_count(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count direct calls to ``fn`` from within its own body."""
    count = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == fn.name:
                count += 1
    return count


def _all_function_defs(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _has_node(tree: ast.AST, types: tuple[type[ast.AST], ...]) -> bool:
    return any(isinstance(node, types) for node in ast.walk(tree))


def _has_loop(tree: ast.AST) -> bool:
    return _has_node(tree, (ast.For, ast.AsyncFor, ast.While))


def _has_early_return(tree: ast.AST) -> bool:
    """Return statement nested inside a loop."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    return True
    return False


def _is_sorted_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "sorted":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "sort":
        return True
    return False


def _sort_directions(tree: ast.AST) -> tuple[bool, bool]:
    """Return (has_ascending, has_descending) sort calls."""
    asc = False
    desc = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_sorted_call(node):
            reverse = False
            for kw in node.keywords:
                if kw.arg == "reverse":
                    val = kw.value
                    if isinstance(val, ast.Constant) and val.value is True:
                        reverse = True
                    elif isinstance(val, ast.Name) and val.id == "True":
                        reverse = True
            if reverse:
                desc = True
            else:
                asc = True
    return asc, desc


def _has_pairwise_loop(tree: ast.AST) -> bool:
    """Nested For where inner references the outer loop variable."""
    for outer in ast.walk(tree):
        if not isinstance(outer, (ast.For, ast.AsyncFor)):
            continue
        if not isinstance(outer.target, ast.Name):
            continue
        outer_var = outer.target.id
        for inner in ast.walk(outer):
            if inner is outer or not isinstance(inner, (ast.For, ast.AsyncFor)):
                continue
            for ref in ast.walk(inner):
                if isinstance(ref, ast.Name) and ref.id == outer_var:
                    return True
    return False


def _has_accumulator(tree: ast.AST, kind: str) -> bool:
    """Detect dict/list accumulator pattern: structure built/mutated inside a loop.

    ``kind`` is "dict" or "list".
    """
    target_method = "append" if kind == "list" else None
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.AsyncFor, ast.While)):
            continue
        for node in ast.walk(loop):
            if kind == "list":
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == target_method
                ):
                    return True
            else:  # dict
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Subscript):
                            return True
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
                    return True
    return False


def _imported_modules(tree: ast.Module) -> set[str]:
    """Top-level module names from ``import`` and ``from`` statements."""
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module.split(".")[0])
    return mods


def _calls_any(tree: ast.AST, names: frozenset[str]) -> bool:
    """True if any Call references one of ``names`` (Name or Attribute)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in names:
                return True
            if isinstance(func, ast.Attribute) and func.attr in names:
                return True
    return False


_HEAP_FNS = frozenset({"heappush", "heappop", "heapify", "heapreplace", "heappushpop"})
_DEQUE_NAMES = frozenset({"deque"})
_MEMO_DECORATORS = frozenset({"cache", "lru_cache"})


def _has_memoization(tree: ast.Module) -> bool:
    for fn in _all_function_defs(tree):
        for dec in fn.decorator_list:
            target = dec
            if isinstance(target, ast.Call):
                target = target.func
            if isinstance(target, ast.Name) and target.id in _MEMO_DECORATORS:
                return True
            if isinstance(target, ast.Attribute) and target.attr in _MEMO_DECORATORS:
                return True
    return False


def _reads_stdin(tree: ast.Module) -> bool:
    """Detects sys.stdin reads or input() calls."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "input":
                return True
            if isinstance(func, ast.Attribute):
                # sys.stdin.read / .readline / .readlines
                if func.attr in {"read", "readline", "readlines"}:
                    obj = func.value
                    if isinstance(obj, ast.Attribute) and obj.attr == "stdin":
                        return True
                    if isinstance(obj, ast.Name) and obj.id == "stdin":
                        return True
        if isinstance(node, ast.Attribute) and node.attr == "stdin":
            return True
    return False


def _writes_stdout(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "write":
                obj = func.value
                if isinstance(obj, ast.Attribute) and obj.attr == "stdout":
                    return True
                if isinstance(obj, ast.Name) and obj.id == "stdout":
                    return True
    return False


def _output_in_loop(tree: ast.Module) -> bool:
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.AsyncFor, ast.While)):
            continue
        for node in ast.walk(loop):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    return True
                if isinstance(func, ast.Attribute) and func.attr == "write":
                    return True
    return False


def _entry_function(tree: ast.Module) -> ast.AST:
    """Pick the function whose control-flow defines the structural hash.

    Preference order: ``solve``, ``main``, the first ``FunctionDef``, else the
    module body itself. Falling back to the module body keeps the hash defined
    even for scripts with no functions.
    """
    fns = _all_function_defs(tree)
    by_name = {fn.name: fn for fn in fns}
    if "solve" in by_name:
        return by_name["solve"]
    if "main" in by_name:
        return by_name["main"]
    if fns:
        return fns[0]
    return tree


def _control_flow_sequence(node: ast.AST) -> list[str]:
    """Pre-order DFS emitting control-flow node type names only.

    Uses explicit recursive pre-order traversal (visit-then-recurse) so the
    resulting sequence reflects the static structural shape of the program.
    ``ast.walk`` is BFS and loses the parent->child ordering the hash relies
    on, so it is intentionally NOT used here.
    """
    out: list[str] = []

    def visit(n: ast.AST) -> None:
        if isinstance(n, _CFLOW_TYPES):
            out.append(type(n).__name__)
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return out


def _compute_cfhash(tree: ast.Module) -> str:
    entry = _entry_function(tree)
    seq = _control_flow_sequence(entry)
    if not seq:
        return "0" * 8
    payload = "|".join(seq).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def _loop_depth_bucket(depth: int) -> str:
    if depth <= 0:
        return "loop_depth=0"
    if depth == 1:
        return "loop_depth=1"
    if depth == 2:
        return "loop_depth=2"
    return "loop_depth=3+"


def _classify_family(*, multi_recursive: bool, recursive: bool, loop_depth: int) -> str:
    if multi_recursive:
        return "recursive-multi"
    if recursive:
        return "recursive-tail"
    if loop_depth >= 2:
        return "iterative-nested"
    if loop_depth == 1:
        return "iterative-flat"
    return "straight-line"


def extract_ast_features(code: str) -> dict[str, Any]:
    """Extract a domain-agnostic structural fingerprint from Python source.

    Returns a dict with three keys:

    - ``features``: sorted ``list[str]`` of structural tags (see module
      docstring for the vocabulary).
    - ``family``: one coarse structural bucket.
    - ``cfhash``: 8-char hex prefix of sha256 over the control-flow node
      sequence of the entry function.

    Always returns a valid dict — never raises. Unparseable input yields
    ``family="unparseable"`` and a single ``"unparseable"`` feature tag.
    """
    if not code or not code.strip():
        return {"features": ["empty"], "family": "straight-line", "cfhash": "0" * 8}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"features": ["unparseable"], "family": "unparseable", "cfhash": "0" * 8}

    features: set[str] = set()

    # --- Recursion ---
    fn_defs = _all_function_defs(tree)
    is_any_recursive = False
    is_multi_recursive = False
    for fn in fn_defs:
        n = _self_call_count(fn)
        if n >= 1:
            is_any_recursive = True
        if n >= 2:
            is_multi_recursive = True
    if is_any_recursive:
        features.add("recursive_fn")
    if is_multi_recursive:
        features.add("multi_recursive")

    # --- Loop nesting ---
    depth = _max_loop_depth(tree)
    features.add(_loop_depth_bucket(depth))

    # --- Loop kinds ---
    if _has_node(tree, (ast.For, ast.AsyncFor)):
        features.add("has_for")
    if _has_node(tree, (ast.While,)):
        features.add("has_while")

    # --- Control flow extras ---
    if _has_early_return(tree):
        features.add("early_return")
    if _has_node(tree, (ast.Break,)):
        features.add("has_break")
    if _has_node(tree, (ast.Continue,)):
        features.add("has_continue")
    if _has_node(tree, (ast.Try,)):
        features.add("try_except")

    # --- Function structure ---
    num_fns = len(fn_defs)
    features.add(f"num_fns={_bucket_count(num_fns, (1, 3, 6))}")
    # Nested function (closure pattern): a FunctionDef inside another FunctionDef
    for fn in fn_defs:
        for child in ast.walk(fn):
            if child is fn:
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features.add("nested_fn")
                break
        if "nested_fn" in features:
            break
    if _has_node(tree, (ast.Lambda,)):
        features.add("has_lambda")
    if any(fn.decorator_list for fn in fn_defs):
        features.add("has_decorator")

    # --- Sort + iteration patterns ---
    asc, desc = _sort_directions(tree)
    if asc:
        features.add("sort_asc")
    if desc:
        features.add("sort_desc")
    if _has_pairwise_loop(tree):
        features.add("pairwise_loop")
    if _has_accumulator(tree, "dict"):
        features.add("accumulator_dict")
    if _has_accumulator(tree, "list"):
        features.add("accumulator_list")

    # --- Library shape ---
    mods = _imported_modules(tree)
    if "heapq" in mods and _calls_any(tree, _HEAP_FNS):
        features.add("uses_heap")
    if "collections" in mods and _calls_any(tree, _DEQUE_NAMES):
        features.add("uses_deque")
    if _has_memoization(tree):
        features.add("memoization")

    # --- I/O shape ---
    if _reads_stdin(tree):
        features.add("reads_stdin")
    if _writes_stdout(tree):
        features.add("writes_stdout")
    if _output_in_loop(tree):
        features.add("output_in_loop")

    sorted_features = sorted(features)
    return {
        "features": sorted_features,
        "family": _classify_family(
            multi_recursive=is_multi_recursive,
            recursive=is_any_recursive,
            loop_depth=depth,
        ),
        "cfhash": _compute_cfhash(tree),
    }


# ---------------------------------------------------------------------------
# Deterministic feature vector for branch centroid geometry
# ---------------------------------------------------------------------------

# Family buckets — order matters (defines one-hot positions).
_FAMILIES: tuple[str, ...] = (
    "recursive-multi",
    "recursive-tail",
    "iterative-nested",
    "iterative-flat",
    "straight-line",
    "unparseable",
)

# Binary feature flags — order matters (defines vector positions).
_BINARY_FEATURES: tuple[str, ...] = (
    "recursive_fn",
    "multi_recursive",
    "has_for",
    "has_while",
    "early_return",
    "has_break",
    "has_continue",
    "try_except",
    "nested_fn",
    "has_lambda",
    "has_decorator",
    "sort_asc",
    "sort_desc",
    "pairwise_loop",
    "accumulator_dict",
    "accumulator_list",
    "uses_heap",
    "uses_deque",
    "memoization",
    "reads_stdin",
    "writes_stdout",
    "output_in_loop",
)

# Loop-depth buckets → scalar encoding.
_LOOP_DEPTH_MAP: dict[str, float] = {
    "loop_depth=0": 0.0,
    "loop_depth=1": 1.0,
    "loop_depth=2": 2.0,
    "loop_depth=3+": 3.0,
}

# num_fns buckets → scalar encoding.
_NUM_FNS_MAP: dict[str, float] = {
    "num_fns=0": 0.0,
    "num_fns=1": 1.0,
    "num_fns=2-3": 2.5,
    "num_fns=4-6": 5.0,
    "num_fns=7+": 8.0,
}

# cfhash encoding: 8 hex chars → 8 dims via deterministic hash-to-float.
_CFHASH_DIMS = 8
# Weight applied to cfhash dims before L2 normalisation. Controls how much
# control-flow shape matters relative to coarse structural features. At 0.5,
# two same-family/same-feature programs with different cfhash separate by
# ~0.05-0.10 cosine distance — enough for split decisions without dominating.
_CFHASH_WEIGHT = 0.5

# Vector length: 6 (family) + 22 (binary) + 1 (loop_depth) + 1 (num_fns) + 8 (cfhash) = 38
FEATURE_VECTOR_DIM = len(_FAMILIES) + len(_BINARY_FEATURES) + 2 + _CFHASH_DIMS


def _cfhash_to_floats(cfhash: str) -> list[float]:
    """Map an 8-char hex cfhash to 8 deterministic floats in [-1, 1].

    Each hex digit (4 bits, value 0-15) is mapped to a float via
    ``(val - 7.5) / 7.5``.  Since cfhash is a SHA-256 prefix, any
    structural change cascades to all digits, giving near-orthogonal
    vectors for different control-flow shapes.
    """
    floats: list[float] = []
    for ch in cfhash[:_CFHASH_DIMS]:
        try:
            val = int(ch, 16)
        except ValueError:
            val = 0
        floats.append((val - 7.5) / 7.5)
    # Pad if cfhash is shorter than expected
    while len(floats) < _CFHASH_DIMS:
        floats.append(0.0)
    return floats


def features_to_vector(ast_result: dict[str, Any]) -> list[float]:
    """Convert ``extract_ast_features`` output to a fixed-length float vector.

    The vector is L2-normalised so cosine distance is meaningful.
    Deterministic: same input always produces the same output.

    Encodes: family (one-hot), binary feature flags, loop depth, num_fns,
    and cfhash (weighted control-flow structure fingerprint).
    """
    vec: list[float] = []

    # One-hot family (6 dims)
    family = ast_result.get("family", "unparseable")
    for f in _FAMILIES:
        vec.append(1.0 if f == family else 0.0)

    # Binary feature flags (22 dims)
    feature_set = set(ast_result.get("features", []))
    for f in _BINARY_FEATURES:
        vec.append(1.0 if f in feature_set else 0.0)

    # Loop depth scalar (1 dim)
    loop_val = 0.0
    for tag in feature_set:
        if tag in _LOOP_DEPTH_MAP:
            loop_val = _LOOP_DEPTH_MAP[tag]
            break
    vec.append(loop_val / 3.0)  # normalise to [0, 1]

    # num_fns scalar (1 dim)
    fns_val = 0.0
    for tag in feature_set:
        if tag in _NUM_FNS_MAP:
            fns_val = _NUM_FNS_MAP[tag]
            break
    vec.append(fns_val / 8.0)  # normalise to [0, 1]

    # cfhash — fine-grained control-flow structure (8 dims, weighted)
    cfhash = ast_result.get("cfhash", "0" * 8)
    for v in _cfhash_to_floats(cfhash):
        vec.append(v * _CFHASH_WEIGHT)

    # L2-normalise
    norm = 0.0
    for x in vec:
        norm += x * x
    if norm > 1e-24:
        norm = norm**0.5
        vec = [x / norm for x in vec]

    return vec
