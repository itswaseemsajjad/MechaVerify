"""Tests for the deterministic parser / verifier / reward."""

from mechaverify.rng import SeededRNG
from mechaverify.specs import Answer, Instance
from mechaverify.tasks import generate
from mechaverify.verify import (
    normalize_categorical,
    parse_number,
    parse_response,
    reward_fn,
    verify,
)


def _instance(answer: Answer) -> Instance:
    return Instance(id="t", family="kinematics", task="gear_sense", split="id_test",
                    difficulty=1, seed=0, spec={"type": "gear_train"},
                    question="q", answer=answer, rationale="", symbolic_description="")


def test_parse_response_fields():
    p = parse_response("<think>reason here</think>\n<answer>clockwise</answer>")
    assert p.think == "reason here"
    assert p.answer_text == "clockwise"
    assert p.format_ok is True


def test_parse_response_fallback_not_format_ok():
    p = parse_response("blah blah\nAnswer: counterclockwise")
    assert p.answer_text == "counterclockwise"
    assert p.format_ok is False


def test_categorical_synonyms():
    assert normalize_categorical("CW") == "clockwise"
    assert normalize_categorical("counter-clockwise") == "counterclockwise"
    assert normalize_categorical("the gear is Clockwise.") == "clockwise"
    assert normalize_categorical("Yes") == "yes"
    assert normalize_categorical("drag-link") == "double-crank"


def test_substring_does_not_false_match_choice():
    # "unknown" contains the substring "no" but must NOT be scored as "no"
    assert normalize_categorical("unknown", ["yes", "no"]) == "unknown"
    inst = _instance(Answer(value="no", kind="categorical", choices=["yes", "no"]))
    assert not verify(inst, "<answer>unknown</answer>").correct


def test_parse_number_forms():
    assert parse_number("2.5") == 2.5
    assert parse_number("2:1") == 2.0
    assert parse_number("1/4") == 0.25
    assert parse_number("ratio is 0.333") == 0.333


def test_numeric_tolerance():
    inst = _instance(Answer(value=0.3333, kind="numeric", tolerance=0.02))
    assert verify(inst, "<answer>0.33</answer>").correct
    assert not verify(inst, "<answer>0.5</answer>").correct


def test_categorical_match_and_reward():
    inst = _instance(Answer(value="clockwise", kind="categorical",
                            choices=["clockwise", "counterclockwise"]))
    r = verify(inst, "<think>2 meshes, even</think><answer>CW</answer>")
    assert r.correct and r.format_ok
    rw = reward_fn(inst, "<answer>clockwise</answer>")
    assert rw["correct"] == 1.0 and rw["total"] > 1.0


def test_wrong_answer_zero_reward():
    inst = _instance(Answer(value="clockwise", kind="categorical"))
    rw = reward_fn(inst, "<answer>counterclockwise</answer>")
    assert rw["correct"] == 0.0


def test_end_to_end_on_generated_instance():
    gr = generate("gear_ratio", SeededRNG(5), 3, "mixed")
    inst = Instance(id="g", family="kinematics", task="gear_ratio", split="id_test",
                    difficulty=3, seed=5, spec=gr.spec, question=gr.question,
                    answer=gr.answer, rationale=gr.rationale,
                    symbolic_description=gr.symbolic_description)
    good = f"<answer>{gr.answer.value}</answer>"
    assert verify(inst, good).correct
