import pytest

from reasoners.benchmark.gsm8k import GSM8KEvaluator


def _bare_evaluator(init_prompt=None, sample_prompt_type="l2m"):
    # __init__ unconditionally calls datasets.load_dataset(), a network call this
    # unit test has no business making - bypass it, sample_prompt()/eval_output()
    # only need self.init_prompt and self.sample_prompt_type.
    evaluator = GSM8KEvaluator.__new__(GSM8KEvaluator)
    evaluator.init_prompt = init_prompt
    evaluator.sample_prompt_type = sample_prompt_type
    return evaluator


def test_eval_output_none_is_false():
    evaluator = _bare_evaluator()
    assert evaluator.eval_output("42", None) is False


def test_eval_output_int_match_and_mismatch():
    evaluator = _bare_evaluator()
    assert evaluator.eval_output("42", "42") is True
    assert evaluator.eval_output("42", "43") is False


def test_eval_output_float_match():
    evaluator = _bare_evaluator()
    assert evaluator.eval_output("3.5", "3.5") is True


def test_eval_output_falls_back_to_string_equality():
    evaluator = _bare_evaluator()
    assert evaluator.eval_output("cannot be determined", "cannot be determined") is True
    assert evaluator.eval_output("3", "three") is False


def test_sample_prompt_cot_joins_pool_and_appends_prefix():
    init_prompt = {"cot_pool": ["ex1\n", "ex2\n"], "prefix": "Q: final question"}
    evaluator = _bare_evaluator(init_prompt, sample_prompt_type="cot")

    prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=2)

    assert prompt["cot"] == "ex1\nex2\nQ: final question"


def test_sample_prompt_l2m_builds_three_variants():
    init_prompt = {
        "decomposition_pool": ["d1\n", "d2\n"],
        "solving_pool": ["s1\n", "s2\n"],
        "composition_prefix": "COMPOSE",
        "overall_prefix": "OVERALL",
        "solving_prefix": "SOLVE",
    }
    evaluator = _bare_evaluator(init_prompt, sample_prompt_type="l2m")

    prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=2)

    assert prompt["decomposition"] == "d1\nd2\nCOMPOSE"
    assert prompt["overall"] == "d1\nd2\nOVERALL"
    assert prompt["solving"] == "s1\ns2\nSOLVE"


def test_sample_prompt_rap_preserves_pairing_after_sampling():
    # interactive_examples[i] and useful_examples[i] must stay paired together
    # even after random.sample() reorders/subsets them - a zip/unzip bug here
    # would silently mismatch which "useful" example goes with which example.
    init_prompt = {
        "interactive_examples": ["a0", "a1", "a2", "a3"],
        "useful_examples": ["b0", "b1", "b2", "b3"],
        "unrelated_key": "unchanged",
    }
    evaluator = _bare_evaluator(init_prompt, sample_prompt_type="rap")

    prompt = evaluator.sample_prompt(shuffle_prompt=True, num_shot=2)

    assert len(prompt["interactive_examples"]) == 2
    for interactive, useful in zip(prompt["interactive_examples"], prompt["useful_examples"]):
        assert interactive[-1] == useful[-1]  # "a2" pairs only with "b2", etc.
    assert prompt["unrelated_key"] == "unchanged"
    # original untouched - sample_prompt() must not mutate the shared init_prompt
    assert init_prompt["interactive_examples"] == ["a0", "a1", "a2", "a3"]


def test_sample_prompt_unknown_type_raises():
    # also covers the now-removed "grace" branch - GRACE (examples/Grace) was its
    # only caller and has been removed, so it falls through to this same error
    evaluator = _bare_evaluator(sample_prompt_type="not-a-real-type")
    with pytest.raises(NotImplementedError):
        evaluator.sample_prompt()
