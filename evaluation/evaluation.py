import re

"""
Normalization and equivalence checking for LaTeX math answers.

Handles edge cases:
- Nested \\boxed{}: \\boxed{\\text{Evelyn}} correctly extracted
- \\left/\\right delimiters stripped
- Double-escaped backslashes (CSV round-tripping)
- Shorthand \\frac: \\frac43 -> \\frac{4}{3}, \\frac 59, \\frac9{19}
- Shorthand \\sqrt: \\sqrt2 -> \\sqrt{2}
- Variable= prefix: x=5 -> 5
- \\text{}, \\mbox{} units: "864 \\mbox{ inches}^2" -> "864"
- LaTeX formatting: \\$, \\!, \\,, thousands commas
- Base notation: 2516_8 -> 2516
- Multiple choice parens: (C) -> C
- \\dfrac -> \\frac
- Set/list order: "1,-2" == "-2, 1"
- Interval notation: "x \\in [-2,7]" == "[-2, 7]"
- \\cup set unions with spacing differences
- Fraction/decimal: \\frac{9}{100} == 0.09
- Algebraic equivalence via sympy: \\frac{11+9a}{20} == \\frac{9a+11}{20}
"""


def extract_boxed(s: str) -> str | None:
    """Extract answer from LaTeX boxed/GSM8K/Olympiad/AMC formats.

    Handles nested braces (e.g. \\boxed{\\text{Evelyn}}) by parsing brace depth.
    Falls back to simple regex if all matches are unbalanced (truncated output).
    """
    if not s:
        return None

    # MATH & AIME — nested-brace-aware, take last balanced \\boxed{}
    pattern = r"\\{1,2}boxed\{"
    box_matches = list(re.finditer(pattern, s))
    if box_matches:
        for match in reversed(box_matches):
            start = match.end()
            depth = 1
            i = start
            while i < len(s) and depth > 0:
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                i += 1

            if depth != 0:
                continue  # Unbalanced — try previous match

            content = s[start : i - 1].strip()

            # Unwrap \text{...}, \textbf{...}, etc.
            text_match = re.match(r"\\text(?:bf|it|rm|sf)?\{(.+)\}$", content)
            if text_match:
                content = text_match.group(1).strip()

            return content

        # All unbalanced — fall back to simple regex
        simple = re.findall(r"\\{1,2}boxed\{([^}]*)\}", s)
        if simple:
            return simple[-1].strip()

    # GSM8K: #### <answer>
    matches = re.findall(r"(?m)^[ \t]####[ \t]([^\n\r#]+?)[ \t]*$", s)
    if matches:
        return matches[-1].strip()

    # Olympiad: last $...$
    matches = re.findall(r"\$([^$]*)\$", s)
    if matches:
        return matches[-1].strip()

    # AMC: last standalone number
    matches = re.findall(r"(?m)^[ \t]([+-]?\d+(?:\.\d+)?)[ \t]$", s)
    if matches:
        return matches[-1].strip()

    return s


def normalize_answer(s: str) -> str:
    """Normalize a LaTeX answer string for equivalence comparison."""
    if not s or s == "nan":
        return s

    # Fix double-escaped backslashes (e.g. from CSV round-tripping)
    while "\\\\" in s:
        s = s.replace("\\\\", "\\")

    # Strip \left / \right delimiters
    s = s.replace("\\left(", "(").replace("\\right)", ")")
    s = s.replace("\\left[", "[").replace("\\right]", "]")
    s = s.replace("\\left", "").replace("\\right", "")

    # Strip \text{}, \mbox{} with optional trailing exponent (e.g. ^2)
    s = re.sub(
        r"\s*\\(?:text|mbox|textbf|mathrm)\{[^}]\}(?:\^\d+)?\s$", "", s
    ).strip()
    s = re.sub(
        r"\s*\\(?:text|mbox|textbf|mathrm)\{[^}]*\}(?:\^\d+)?", "", s
    ).strip()

    # Strip \$ (LaTeX literal dollar sign)
    s = s.replace("\\$", "")

    # Strip \! (thin neg space) and \, (thin space)
    s = s.replace("\\!", "").replace("\\,", "")

    # Strip "x \in" prefix from intervals
    s = re.sub(r"^[a-zA-Z]\s*\\in\s*", "", s).strip()

    # Strip ^\circ (degree symbol)
    s = re.sub(r"\^\\circ\s*$", "", s).strip()

    # Strip base notation suffix: 2516_8 -> 2516, 4210_{5} -> 4210
    s = re.sub(r"_\{?\d+\}?\s*$", "", s).strip()

    # Strip variable= prefix: x=5 -> 5
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s).strip()

    # Unwrap single-letter parens: (C) -> C
    m = re.match(r"^\(([A-Za-z])\)$", s)
    if m:
        s = m.group(1)

    # \dfrac -> \frac
    s = s.replace("\\dfrac", "\\frac")

    # Normalize shorthand \sqrt: \sqrt2 -> \sqrt{2} (single non-brace char)
    s = re.sub(r"\\sqrt([^{\s\\])", r"\\sqrt{\1}", s)

    # Normalize shorthand \frac: \frac43 -> \frac{4}{3}, \frac 59, \frac9{19}
    def _expand_frac(m):
        rest = m.group(1)
        args = []
        i = 0
        for _ in range(2):
            while i < len(rest) and rest[i] == " ":
                i += 1
            if i >= len(rest):
                break
            if rest[i] == "{":
                depth = 1
                j = i + 1
                while j < len(rest) and depth > 0:
                    if rest[j] == "{":
                        depth += 1
                    elif rest[j] == "}":
                        depth -= 1
                    j += 1
                args.append(rest[i:j])
                i = j
            else:
                args.append("{" + rest[i] + "}")
                i += 1
        if len(args) == 2:
            return "\\frac" + args[0] + args[1]
        return m.group(0)

    s = re.sub(r"\\frac(.*)", _expand_frac, s)

    # Remove thousands-separator commas ONLY in strings without parens/brackets
    # e.g. "58,500" -> "58500" but NOT "(2,12)" or "-2,1"
    if not any(c in s for c in "()[]\\"):
        s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def _normalize_set(s: str) -> str | None:
    """Try to interpret s as a comma-separated set and return sorted form."""
    inner = s.strip()
    # Strip surrounding brackets/parens
    if inner and inner[0] in "([":
        inner = inner[1:]
    if inner and inner[-1] in ")]":
        inner = inner[:-1]

    parts = [p.strip() for p in inner.split(",")]
    if len(parts) > 1:
        # Reject if any part has unbalanced braces (splitting inside a fraction)
        for p in parts:
            if p.count("{") != p.count("}"):
                return None
        return ",".join(sorted(parts))
    return None


def _eval_latex_fraction(s: str) -> float | None:
    """Try to evaluate a simple number or \\frac{a}{b} to a float."""
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r"^\\frac\{([^}]+)\}\{([^}]+)\}$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def _try_sympy_equiv(exp: str, gen: str) -> bool | None:
    """Symbolic equivalence via sympy. Returns None if parsing fails."""
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify

        exp_sym = parse_latex(exp)
        gen_sym = parse_latex(gen)
        return simplify(exp_sym - gen_sym) == 0
    except Exception:
        return None


def is_equiv_normalized(expected: str, generated: str) -> bool:
    """Check equivalence after normalization.

    Layers (in order):
    1. Exact match after normalization
    2. Exact match ignoring spaces
    3. Set/list comparison (order-independent)
    4. Numeric fraction/decimal comparison
    5. Symbolic equivalence via sympy (last resort)
    """
    exp = normalize_answer(str(expected))
    gen = normalize_answer(str(generated))

    # 1. Exact
    if exp == gen:
        return True

    # 2. Ignore spaces
    if exp.replace(" ", "") == gen.replace(" ", ""):
        return True

    # 3. Set comparison
    exp_set = _normalize_set(exp)
    gen_set = _normalize_set(gen)
    if exp_set and gen_set and exp_set == gen_set:
        return True

    # 4. Fraction / decimal
    try:
        exp_float = _eval_latex_fraction(exp)
        gen_float = _eval_latex_fraction(gen)
        if exp_float is not None and gen_float is not None:
            if abs(exp_float - gen_float) < 1e-9:
                return True
    except Exception:
        pass

    # 5. Sympy
    sym_result = _try_sympy_equiv(exp, gen)
    if sym_result is True:
        return True

    return False


def evaluate_answer(expected_answer: str, generated_answer: str) -> bool:
    exp_val = extract_boxed(expected_answer)
    gen_val = extract_boxed(generated_answer)
    if exp_val is None or gen_val is None:
        return False, exp_val, gen_val
    return is_equiv_normalized(gen_val, exp_val), exp_val, gen_val