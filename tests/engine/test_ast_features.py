"""Unit tests for src/esn/engine/ast_features.py."""

from __future__ import annotations

import textwrap

from esn.engine.ast_features import extract_ast_features


def _features(code: str) -> list[str]:
    return extract_ast_features(textwrap.dedent(code))["features"]


def _family(code: str) -> str:
    return extract_ast_features(textwrap.dedent(code))["family"]


def _cfhash(code: str) -> str:
    return extract_ast_features(textwrap.dedent(code))["cfhash"]


# ---------------------------------------------------------------------------
# Family bucket tests — every Python program falls into exactly one bucket.
# ---------------------------------------------------------------------------


def test_family_straight_line():
    code = """
    x = 1
    y = 2
    print(x + y)
    """
    assert _family(code) == "straight-line"


def test_family_iterative_flat():
    code = """
    def main():
        for i in range(10):
            print(i)
    """
    assert _family(code) == "iterative-flat"


def test_family_iterative_nested():
    code = """
    def main():
        for i in range(10):
            for j in range(10):
                print(i, j)
    """
    assert _family(code) == "iterative-nested"


def test_family_recursive_tail():
    code = """
    def f(n):
        if n <= 0:
            return 0
        return f(n - 1)
    f(5)
    """
    assert _family(code) == "recursive-tail"


def test_family_recursive_multi():
    code = """
    def f(n):
        if n <= 1:
            return n
        return f(n - 1) + f(n - 2)
    f(10)
    """
    assert _family(code) == "recursive-multi"


# ---------------------------------------------------------------------------
# Edge cases — never raise, always return a valid bucket.
# ---------------------------------------------------------------------------


def test_empty_code():
    result = extract_ast_features("")
    assert result["family"] == "straight-line"
    assert "empty" in result["features"]


def test_whitespace_only():
    result = extract_ast_features("   \n\n\t\n")
    assert result["family"] == "straight-line"


def test_unparseable_code():
    result = extract_ast_features("def foo( :::")
    assert result["family"] == "unparseable"
    assert "unparseable" in result["features"]


def test_imports_only():
    code = """
    import sys
    import math
    """
    assert _family(code) == "straight-line"


def test_comment_only():
    code = """
    # this is a comment
    # another comment
    """
    assert _family(code) == "straight-line"


# ---------------------------------------------------------------------------
# Feature detection — specific tags fire on the patterns they describe.
# ---------------------------------------------------------------------------


def test_sort_ascending():
    code = "x = sorted([3, 1, 2])"
    feats = _features(code)
    assert "sort_asc" in feats
    assert "sort_desc" not in feats


def test_sort_descending_kwarg():
    code = "x = sorted([3, 1, 2], reverse=True)"
    feats = _features(code)
    assert "sort_desc" in feats
    assert "sort_asc" not in feats


def test_sort_method_call():
    code = """
    xs = [3, 1, 2]
    xs.sort()
    """
    assert "sort_asc" in _features(code)


def test_pairwise_loop_inner_uses_outer_var():
    code = """
    def f(n):
        for i in range(n):
            for j in range(i):
                print(i, j)
    """
    assert "pairwise_loop" in _features(code)


def test_pairwise_loop_inner_independent():
    # Inner loop does NOT reference outer loop variable → not pairwise.
    code = """
    def f(n):
        for i in range(n):
            for k in range(5):
                print(k)
    """
    assert "pairwise_loop" not in _features(code)


def test_uses_heap():
    code = """
    import heapq
    h = []
    heapq.heappush(h, 1)
    """
    assert "uses_heap" in _features(code)


def test_uses_heap_requires_actual_call():
    code = "import heapq"
    assert "uses_heap" not in _features(code)


def test_uses_deque():
    code = """
    from collections import deque
    q = deque()
    """
    assert "uses_deque" in _features(code)


def test_memoization_lru_cache():
    code = """
    import functools
    @functools.lru_cache
    def f(n):
        return n
    """
    assert "memoization" in _features(code)


def test_memoization_bare_cache():
    code = """
    from functools import cache
    @cache
    def f(n):
        return n
    """
    assert "memoization" in _features(code)


def test_accumulator_list():
    code = """
    def f():
        out = []
        for i in range(10):
            out.append(i)
        return out
    """
    assert "accumulator_list" in _features(code)


def test_accumulator_dict():
    code = """
    def f():
        out = {}
        for i in range(10):
            out[i] = i * 2
        return out
    """
    assert "accumulator_dict" in _features(code)


def test_reads_stdin():
    code = """
    import sys
    data = sys.stdin.read()
    """
    assert "reads_stdin" in _features(code)


def test_writes_stdout_via_print():
    code = "print('hello')"
    assert "writes_stdout" in _features(code)


def test_output_in_loop():
    code = """
    for i in range(5):
        print(i)
    """
    assert "output_in_loop" in _features(code)


def test_output_outside_loop_only():
    code = """
    xs = [1, 2, 3]
    print(xs)
    """
    feats = _features(code)
    assert "writes_stdout" in feats
    assert "output_in_loop" not in feats


def test_nested_fn_detected():
    code = """
    def outer():
        def inner():
            return 1
        return inner()
    """
    assert "nested_fn" in _features(code)


def test_lambda_detected():
    code = "f = lambda x: x + 1"
    assert "has_lambda" in _features(code)


def test_decorator_detected():
    code = """
    import functools
    @functools.lru_cache
    def f(n):
        return n
    """
    assert "has_decorator" in _features(code)


def test_loop_depth_buckets():
    one = "for i in range(5):\n    print(i)"
    two = "for i in range(5):\n    for j in range(5):\n        print(i, j)"
    three = (
        "for i in range(5):\n"
        "    for j in range(5):\n"
        "        for k in range(5):\n"
        "            print(i, j, k)"
    )
    assert "loop_depth=1" in _features(one)
    assert "loop_depth=2" in _features(two)
    assert "loop_depth=3+" in _features(three)


def test_try_except_detected():
    code = """
    try:
        x = 1
    except Exception:
        x = 0
    """
    assert "try_except" in _features(code)


# ---------------------------------------------------------------------------
# cfhash stability and sensitivity — the load-bearing claim.
# ---------------------------------------------------------------------------


def test_cfhash_stable_under_variable_rename():
    a = """
    def f(n):
        if n <= 0:
            return 0
        for i in range(n):
            print(i)
        return n
    """
    b = """
    def f(count):
        if count <= 0:
            return 0
        for index in range(count):
            print(index)
        return count
    """
    assert _cfhash(a) == _cfhash(b)


def test_cfhash_stable_under_constant_change():
    a = "def f():\n    for i in range(10):\n        print(i)"
    b = "def f():\n    for i in range(99999):\n        print(i)"
    assert _cfhash(a) == _cfhash(b)


def test_cfhash_changes_when_for_becomes_while():
    a = """
    def f():
        for i in range(10):
            print(i)
    """
    b = """
    def f():
        i = 0
        while i < 10:
            print(i)
            i += 1
    """
    assert _cfhash(a) != _cfhash(b)


def test_cfhash_changes_when_recursion_added():
    a = """
    def f(n):
        return n + 1
    """
    b = """
    def f(n):
        if n <= 0:
            return 0
        return f(n - 1)
    """
    assert _cfhash(a) != _cfhash(b)


def test_cfhash_solve_preferred_over_main():
    # If both solve and main exist, solve should drive the hash.
    a = """
    def solve():
        for i in range(5):
            print(i)
    def main():
        pass
    """
    b = """
    def solve():
        for i in range(5):
            print(i)
    """
    assert _cfhash(a) == _cfhash(b)


# ---------------------------------------------------------------------------
# Real elite programs — guard against future regressions.
# ---------------------------------------------------------------------------


def test_real_seed_program_is_iterative_flat():
    """Trivial seed: print 1x1 rectangles in a single loop."""
    code = """
    import sys
    def main():
        data = sys.stdin.read().split()
        idx = 0
        n = int(data[idx]); idx += 1
        for _ in range(n):
            x = int(data[idx]); idx += 1
            y = int(data[idx]); idx += 1
            _ = data[idx]; idx += 1
            ax = x if x < 10000 else 9999
            ay = y if y < 10000 else 9999
            print(ax, ay, ax + 1, ay + 1)
    if __name__ == '__main__':
        main()
    """
    result = extract_ast_features(textwrap.dedent(code))
    assert result["family"] == "iterative-flat"
    assert "reads_stdin" in result["features"]
    assert "output_in_loop" in result["features"]


def test_real_bsp_program_is_recursive_multi():
    """A common recursive-partition (BSP) strategy."""
    code = """
    import sys
    def main():
        n = 5
        rects = [None] * n
        def partition(ids, x1, y1, x2, y2):
            if len(ids) <= 1:
                return
            mid = len(ids) // 2
            partition(ids[:mid], x1, y1, x2, y2)
            partition(ids[mid:], x1, y1, x2, y2)
        partition(list(range(n)), 0, 0, 100, 100)
    """
    result = extract_ast_features(textwrap.dedent(code))
    assert result["family"] == "recursive-multi"
    assert "multi_recursive" in result["features"]
    assert "recursive_fn" in result["features"]
    assert "nested_fn" in result["features"]


def test_real_greedy_program_separates_from_bsp():
    """Greedy expand: distinct fingerprint from BSP."""
    greedy = """
    def main():
        n, rs = 10, list(range(10))
        order = sorted(range(n), key=lambda i: rs[i], reverse=True)
        rects = []
        for i in order:
            for j in range(i):
                if i != j:
                    pass
            rects.append((0, 0, 1, 1))
    """
    bsp = """
    def main():
        def partition(ids, x1, y1, x2, y2):
            if len(ids) <= 1:
                return
            partition(ids[:1], x1, y1, x2, y2)
            partition(ids[1:], x1, y1, x2, y2)
        partition([0, 1, 2], 0, 0, 100, 100)
    """
    g = extract_ast_features(textwrap.dedent(greedy))
    b = extract_ast_features(textwrap.dedent(bsp))
    assert g["family"] != b["family"]
    assert g["cfhash"] != b["cfhash"]
    assert "sort_desc" in g["features"]
    assert "multi_recursive" in b["features"]
