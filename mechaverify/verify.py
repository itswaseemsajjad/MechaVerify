"""Deterministic answer parsing and verification.

Models are prompted to emit a structured response that separates reasoning from
the final answer using ``<think>...</think>`` and ``<answer>...</answer>`` fields.
This module extracts the answer payload, normalizes it, and compares it to the
closed-form ground truth:

    * categorical answers  -> exact match after synonym normalization
    * numeric answers      -> acceptance within a relative/absolute tolerance band

The result carries a binary ``correct`` signal (used both for benchmark scoring
and as the RLVR reward) and a separate ``format_ok`` signal (the format reward).
Binary, programmatic rewards of this kind are reported to reduce reward hacking
relative to learned reward models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from mechaverify.specs import Answer, Instance

# --- canonical synonym tables for categorical answers ------------------------
_CATEGORICAL_SYNONYMS: dict[str, str] = {
    # rotation sense
    "cw": "clockwise",
    "clockwise": "clockwise",
    "clock-wise": "clockwise",
    "ccw": "counterclockwise",
    "counterclockwise": "counterclockwise",
    "counter-clockwise": "counterclockwise",
    "anticlockwise": "counterclockwise",
    "anti-clockwise": "counterclockwise",
    # yes / no (machinability, full-rotation questions)
    "yes": "yes",
    "true": "yes",
    "machinable": "yes",
    "feasible": "yes",
    "no": "no",
    "false": "no",
    "not machinable": "no",
    "unmachinable": "no",
    "infeasible": "no",
    # Grashof classification
    "crank-rocker": "crank-rocker",
    "crank rocker": "crank-rocker",
    "crankrocker": "crank-rocker",
    "double-crank": "double-crank",
    "double crank": "double-crank",
    "drag-link": "double-crank",
    "drag link": "double-crank",
    "double-rocker": "double-rocker",
    "double rocker": "double-rocker",
    "triple-rocker": "triple-rocker",
    "triple rocker": "triple-rocker",
    "change-point": "change-point",
    "change point": "change-point",
}

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_FALLBACK_RE = re.compile(
    r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)", re.IGNORECASE
)
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class ParsedResponse:
    think: str | None
    answer_text: str | None
    format_ok: bool


@dataclass
class VerifyResult:
    correct: bool
    format_ok: bool
    parsed_answer: Any
    expected: Any
    reason: str

    @property
    def reward(self) -> float:
        """Binary correctness reward used for RLVR."""
        return 1.0 if self.correct else 0.0


def parse_response(text: str) -> ParsedResponse:
    """Extract think/answer fields. ``format_ok`` requires a well-formed
    ``<answer>`` tag (the format reward); a looser fallback still recovers an
    answer for scoring so a missing tag is penalized but not unrecoverable."""
    if text is None:
        return ParsedResponse(None, None, False)
    think_m = _THINK_RE.search(text)
    ans_m = _ANSWER_RE.search(text)
    think = think_m.group(1).strip() if think_m else None
    if ans_m:
        return ParsedResponse(think, ans_m.group(1).strip(), True)
    # Fallback: "Answer: ..." on its own, or the last non-empty line.
    fb = _ANSWER_FALLBACK_RE.search(text)
    if fb:
        return ParsedResponse(think, fb.group(1).strip(), False)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return ParsedResponse(think, lines[-1] if lines else None, False)


# --- numeric parsing ---------------------------------------------------------
def parse_number(s: str) -> float | None:
    """Parse a scalar from free text, tolerating fractions and ``a:b`` ratios."""
    if s is None:
        return None
    s = s.strip().lower()
    s = s.replace("×", "x").replace("≈", "").replace("~", "")
    # ratio "a:b" -> a/b ;  also "a to b"
    ratio = re.match(r"^\s*([-+]?\d*\.?\d+)\s*[:/]\s*([-+]?\d*\.?\d+)\s*$", s)
    if ":" in s and ratio:
        a, b = float(ratio.group(1)), float(ratio.group(2))
        return a / b if b != 0 else None
    # bare fraction "a/b"
    if "/" in s and ratio:
        a, b = float(ratio.group(1)), float(ratio.group(2))
        return a / b if b != 0 else None
    try:
        if "/" in s:
            return float(Fraction(s.split()[0]))
    except (ValueError, ZeroDivisionError):
        pass
    m = _NUM_RE.search(s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def normalize_categorical(s: str, choices: list[str] | None = None) -> str | None:
    """Map a free-text categorical answer to its canonical form."""
    if s is None:
        return None
    raw = s.strip().lower()
    raw = raw.strip(".,;:!\"'()[]")
    raw = re.sub(r"\s+", " ", raw)
    if raw in _CATEGORICAL_SYNONYMS:
        return _CATEGORICAL_SYNONYMS[raw]
    # Look for any known synonym as a substring (handles "the output is clockwise").
    best = None
    for key, canon in _CATEGORICAL_SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}\b", raw):
            # Prefer longer keys (more specific) to avoid e.g. matching nothing.
            if best is None or len(key) > best[0]:
                best = (len(key), canon)
    if best:
        return best[1]
    if choices:
        for c in choices:                       # word-boundary match, not substring
            if re.search(rf"\b{re.escape(c.lower())}\b", raw):
                return c.lower()
    return raw or None


def _numbers_match(pred: float, exp: float, ans: Answer) -> bool:
    if ans.abs_tolerance is not None:
        return abs(pred - exp) <= ans.abs_tolerance
    tol = ans.tolerance if ans.tolerance is not None else 0.01
    denom = max(abs(exp), 1e-9)
    return abs(pred - exp) / denom <= tol


def check_answer(answer: Answer, answer_text: str | None) -> tuple[bool, Any, str]:
    """Compare a parsed answer string against the ground-truth ``Answer``."""
    if answer_text is None:
        return False, None, "no answer found"
    if answer.kind == "numeric":
        pred = parse_number(answer_text)
        if pred is None:
            return False, None, "could not parse a number"
        ok = _numbers_match(pred, float(answer.value), answer)
        return ok, pred, ("within tolerance" if ok else "outside tolerance")
    # categorical
    pred = normalize_categorical(answer_text, answer.choices)
    exp = normalize_categorical(str(answer.value), answer.choices)
    ok = pred == exp
    return ok, pred, ("exact match" if ok else f"expected {exp!r}, got {pred!r}")


def verify(instance: Instance, model_text: str) -> VerifyResult:
    """End-to-end: parse a model response and score it against ``instance``."""
    parsed = parse_response(model_text)
    correct, pred, reason = check_answer(instance.answer, parsed.answer_text)
    return VerifyResult(
        correct=correct,
        format_ok=parsed.format_ok,
        parsed_answer=pred,
        expected=instance.answer.value,
        reason=reason,
    )


def reward_fn(instance: Instance, model_text: str, format_weight: float = 0.1) -> dict[str, float]:
    """RLVR reward: binary correctness plus a small well-formedness bonus."""
    r = verify(instance, model_text)
    return {
        "correct": r.reward,
        "format": 1.0 if r.format_ok else 0.0,
        "total": r.reward + (format_weight if r.format_ok else 0.0),
    }
