"""
Numeric answer checker (C1.3).

Treats 0.05, 1/20, 5E-02, ०.०५ (Devanagari digits), "5%" vs 0.05 is NOT merged
(percent is only stripped when the gold answer itself is a percentage), and
"1,000" == 1000.  Values are compared as exact Fractions when possible, with a
relative tolerance of 1e-6 as a fallback for long decimals.
"""
import re
from fractions import Fraction

DEVANAGARI = str.maketrans("०१२३४५६७८९", "0123456789")
NUM_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?"        # fraction  a/b
    r"|[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?\.?\d+(?:[eE][-+]?\d+)?"   # int / decimal / sci
)


def to_fraction(s):
    """String → Fraction, or None."""
    if s is None:
        return None
    s = str(s).translate(DEVANAGARI).strip().replace(",", "").replace(" ", "")
    s = s.rstrip(".।")
    try:
        if "/" in s:
            a, b = s.split("/")
            return Fraction(a) / Fraction(b)
        return Fraction(s)          # Fraction accepts "5E-02", "0.05", "7"
    except (ValueError, ZeroDivisionError):
        try:
            return Fraction(float(s))
        except ValueError:
            return None


def extract_answer(text):
    """Pick the final numeric answer from model output.

    Priority: \\boxed{..} → number after the last 'उत्तर'/'answer' marker → last number.
    """
    t = text.translate(DEVANAGARI)
    m = re.findall(r"\\boxed\{([^}]*)\}", t)
    if m:
        nums = NUM_RE.findall(m[-1])
        if nums:
            return nums[-1]
    for marker in ["अंतिम उत्तर", "उत्तर", "Final answer", "final answer", "Answer", "answer"]:
        i = t.rfind(marker)
        if i != -1:
            nums = NUM_RE.findall(t[i + len(marker):])
            if nums:
                return nums[0]
    nums = NUM_RE.findall(t)
    return nums[-1] if nums else None


def same_number(a, b, rel_tol=1e-6):
    fa, fb = to_fraction(a), to_fraction(b)
    if fa is None or fb is None:
        return False
    if fa == fb:
        return True
    return abs(fa - fb) <= rel_tol * max(abs(fa), abs(fb), Fraction(1))


def is_correct(model_output, gold):
    return same_number(extract_answer(model_output), gold)


if __name__ == "__main__":
    # self-test required by the brief
    assert same_number("0.05", "1/20") and same_number("1/20", "5E-02") and same_number("5e-2", "0.05")
    assert same_number("०.०५", "0.05") and same_number("1,000", "1000") and not same_number("0.5", "0.05")
    assert extract_answer("तो कुल 36 केले। उत्तर: 36") == "36"
    assert extract_answer("... \\boxed{1/20}") == "1/20"
    assert is_correct("उत्तर: ५E-02", "0.05")
    print("checker self-test passed")
