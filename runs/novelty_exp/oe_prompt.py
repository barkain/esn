"""OpenEvolve-spirit mutator prompt: removes ALL anti/forbidding/strategy-
limiting instructions and instead ENCOURAGES powerful methods (numerical
optimization e.g. scipy.optimize, variable-sized elements, full rewrites,
fundamentally different strategies) — matching how OpenEvolve reaches SOTA.

Only kept: (1) the TASK validity contract (the problem's own rules — not a
strategy bias), (2) mechanical output format (return solve(), ASCII, valid
Python), (3) a soft time-guard so long optimizers finish within the budget
(OpenEvolve uses a 90s cap + self-guarding too). No 'avoid', 'never', 'prefer
greedy', 'validity first', or 'preserve the approach' language.

Install by importing and calling install(); gated in run_specdim via
OPENEVOLVE_PROMPT=1.
"""
from esn.engine import mutator as _mut
from esn.engine.mutator import _runtime_budget_hint

OE_STYLES = {
    "refine": (
        "Improve the current program to increase the score. You may change "
        "parameters, algorithms, or structure. Consider stronger methods — "
        "numerical optimization, variable-sized elements, or other techniques "
        "that could score higher."
    ),
    "explore": (
        "Try a qualitatively different and more powerful approach than the parent. "
        "Strong options include numerical optimization, variable-sized elements, "
        "or hybrid constructions. Aim to maximize the score."
    ),
    "repair": (
        "The current program has issues. Fix the specific problems (constraint "
        "violations, runtime errors, numerical issues, or invalid output) while "
        "keeping the working logic."
    ),
    "radical": (
        "Write a completely different solver using whatever approach is most "
        "powerful for this problem. One strong direction is to treat all free "
        "variables (positions and sizes together) as a numerical optimization "
        "problem with a good initialization. Use full rewrites and fundamentally "
        "different strategies rather than small tweaks. Maximize the score."
    ),
    "diverge": (
        "The search has plateaued on one kind of approach. Invent a categorically "
        "different algorithm — a different construction or a different optimization "
        "method. Start fresh from the problem description and maximize the score."
    ),
    "synthesize": "Combine the strongest ideas from the provided parent programs into one coherent, higher-scoring solver.",
    "recombine": (
        "You are given two parent programs from different search branches. Produce "
        "ONE complete solver that combines one concrete strength from each (e.g. "
        "A's initialization with B's optimizer). Synthesize a single coherent "
        "program and briefly note the merge in the JSON metadata."
    ),
}


def build_system_prompt(self, style: str) -> str:
    instruction = self._domain.style_overrides.get(style) or OE_STYLES.get(style, OE_STYLES["refine"])
    interface = (
        "Return ONLY the solve() function and any helper functions it needs "
        "(no __main__ block, no test code, no prints). Use ASCII characters only "
        "and output syntactically valid Python."
    )
    return (
        f"You are an expert at solving the '{self._domain.name}' problem with code. "
        "Aim for the best possible score.\n"
        f"Problem: {self._domain.description}\n"
        "\n"
        "Use whatever approach produces the highest score — including numerical "
        "optimization, variable-sized elements, hybrid methods, or full rewrites. "
        "Prefer fundamentally stronger strategies over small parameter tweaks.\n"
        "\n"
        "If your code runs an iterative or search loop, add a time.time() guard so "
        "it finishes within the time limit and returns a valid result.\n"
        + interface
        + "\n"
        f"Mutation style: {style}\n"
        f"Instruction: {instruction}\n"
        "\n"
        'Return raw Python code or JSON with fields {"code": "...", "diff_summary": '
        '"...", "intended_effect": "..."}. Output only the program content / JSON.\n'
        f"Allowed imports: {'any (unrestricted)' if self._domain.allowed_imports is None else ', '.join(sorted(self._domain.allowed_imports))}\n"
        f"Validity requirements (the solution must satisfy these): {list(self._domain.hard_constraints or [])}\n"
    )


def install():
    _mut.LLMMutator._build_system_prompt = build_system_prompt
    _mut._STYLE_INSTRUCTIONS = OE_STYLES


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "h2h_bf"))
    from biasfree_nz import biasfree_nz_domain
    install()
    m = _mut.LLMMutator(lambda s, u: "", biasfree_nz_domain())
    for style in ("refine", "radical"):
        print(f"\n{'='*30} SYSTEM PROMPT (style={style}) {'='*30}")
        print(m._build_system_prompt(style))
