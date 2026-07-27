from types import SimpleNamespace

import pytest

from reasoners.benchmark.prontoqa import ProntoQAEvaluatorFinal, get_cot_prompt

COT_ENTRY = {
    "Facts": "Cats are mammals. Mammals are animals.",
    "Query": "True or false: Fae is not an animal.",
    "claims": ["Fae is a cat.", "Fae is a mammal.", "Fae is an animal."],
    "next_steps": ["Cats are mammals.", "Mammals are animals.", "Finish."],
}


def _fake_example(query, answer):
    return SimpleNamespace(test_example=SimpleNamespace(query=query, answer=answer))


def test_get_cot_prompt_formats_question_and_steps():
    prompt = get_cot_prompt([COT_ENTRY])

    assert prompt.startswith(
        "Q: Cats are mammals. Mammals are animals. Fae is a cat. True or false: Fae is not an animal.\n"
    )
    assert "A: Fae is a cat. " in prompt
    assert "Cats are mammals. So Fae is a mammal." in prompt
    assert "Mammals are animals. So Fae is an animal." in prompt


def test_get_cot_prompt_false_when_claim_and_query_disagree_on_negation():
    # final claim "...is an animal." has no "not"; query "...is not an animal." does
    # -> polarities disagree -> the query's negated claim is false
    prompt = get_cot_prompt([COT_ENTRY])
    assert "The answer is false." in prompt


def test_get_cot_prompt_true_when_claim_and_query_agree_on_negation():
    entry = dict(COT_ENTRY, claims=["Fae is a cat.", "Fae is a mammal.", "Fae is not an animal."])
    prompt = get_cot_prompt([entry])
    assert "The answer is true." in prompt


def test_init_derives_queries_and_answers_from_dataset():
    dataset = [
        _fake_example("True or false: Fae is not an animal.", "true"),
        _fake_example("True or false: Rex is small.", "false"),
    ]

    evaluator = ProntoQAEvaluatorFinal(dataset=dataset, disable_tqdm=True)

    # query text after the "True or false:" prefix is stripped for display
    assert evaluator.queries == ["Fae is not an animal.", "Rex is small."]
    assert evaluator.answers == ["true", "false"]
    assert evaluator.full_dataset == dataset


def test_sample_prompt_cot_delegates_to_get_cot_prompt():
    evaluator = ProntoQAEvaluatorFinal(dataset=[], init_prompt=[COT_ENTRY], sample_prompt_type="cot", disable_tqdm=True)

    prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=1)

    assert prompt == get_cot_prompt([COT_ENTRY])


def test_sample_prompt_rap_returns_raw_sampled_examples():
    evaluator = ProntoQAEvaluatorFinal(dataset=[], init_prompt=[COT_ENTRY], sample_prompt_type="rap", disable_tqdm=True)

    prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=1)

    assert prompt == [COT_ENTRY]


def test_sample_prompt_unknown_type_raises():
    evaluator = ProntoQAEvaluatorFinal(dataset=[], init_prompt=[COT_ENTRY], sample_prompt_type="not-a-real-type", disable_tqdm=True)

    with pytest.raises(NotImplementedError):
        evaluator.sample_prompt(shuffle_prompt=False, num_shot=1)


def test_eval_output_none_is_false():
    evaluator = ProntoQAEvaluatorFinal(dataset=[], disable_tqdm=True)
    assert evaluator.eval_output("true", None) is False


def test_eval_output_string_equality():
    evaluator = ProntoQAEvaluatorFinal(dataset=[], disable_tqdm=True)
    assert evaluator.eval_output("Fae is not an animal.", "Fae is not an animal.") is True
    assert evaluator.eval_output("Fae is not an animal.", "Fae is an animal.") is False
