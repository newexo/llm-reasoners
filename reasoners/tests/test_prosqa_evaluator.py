import json

from reasoners.benchmark.prosqa import ProsQAEvaluator, load_ProsQA


def _bare_evaluator(init_prompt=None):
    # __init__ hardcodes loading 'data/prosqa_test.json' relative to cwd, with no
    # way to inject a dataset - bypass it, sample_prompt()/eval_output() only need
    # self.init_prompt.
    evaluator = ProsQAEvaluator.__new__(ProsQAEvaluator)
    evaluator.init_prompt = init_prompt
    return evaluator


def test_load_prosqa_keeps_only_known_fields(tmp_path):
    entry = {
        "question": "Q1",
        "answer": "A1",
        "steps": ["s1"],
        "idx_to_symbol": {"0": "x"},
        "edges": [[0, 1]],
        "root": 0,
        "target": 1,
        "neg_target": 2,
        "extra_unused_field": "dropped",
    }
    json_file = tmp_path / "prosqa.json"
    json_file.write_text(json.dumps([entry]))

    data = load_ProsQA(str(json_file))

    assert data == [
        {
            "question": "Q1",
            "answer": "A1",
            "steps": ["s1"],
            "idx_to_symbol": {"0": "x"},
            "edges": [[0, 1]],
            "root": 0,
            "target": 1,
            "neg_target": 2,
        }
    ]


def test_sample_prompt_returns_init_prompt_unchanged():
    evaluator = _bare_evaluator(init_prompt={"pool": ["ex1", "ex2"]})
    assert evaluator.sample_prompt(shuffle_prompt=True, num_shot=1) == {"pool": ["ex1", "ex2"]}


def test_eval_output_none_is_false():
    evaluator = _bare_evaluator()
    assert evaluator.eval_output("A1", None) is False


def test_eval_output_string_equality():
    evaluator = _bare_evaluator()
    assert evaluator.eval_output("A1", "A1") is True
    assert evaluator.eval_output("A1", "A2") is False
