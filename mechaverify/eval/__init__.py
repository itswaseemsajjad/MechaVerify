"""Evaluation harness: run VLMs over MechaVerify under the three diagnostic
input conditions (D / S / D+S), score with the deterministic verifier, and
aggregate accuracy and the perception gap."""

from mechaverify.eval.providers import get_provider  # noqa: F401
from mechaverify.eval.runner import evaluate, perception_gap  # noqa: F401
