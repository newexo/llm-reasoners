from types import SimpleNamespace

import pytest

from reasoners.benchmark.blocksworld import BWEvaluator, get_icl, rap_bw_extractor


def test_rap_bw_extractor_none_trace_returns_empty_string():
    assert rap_bw_extractor(SimpleNamespace(trace=None)) == ""


def test_rap_bw_extractor_joins_the_plan_actions():
    algo_output = SimpleNamespace(trace=(["state0", "state1"], ["pick A", "stack A B"]))
    assert rap_bw_extractor(algo_output) == "pick A\nstack A B"


def test_rap_bw_extractor_returns_empty_string_on_error_instead_of_raising():
    class BoomOutput:
        @property
        def trace(self):
            raise RuntimeError("boom")

    assert rap_bw_extractor(BoomOutput()) == ""


def test_get_icl_builds_statement_goal_and_plan_template():
    init_prompt = {"intro": "INTRO\n"}
    examples = [{"init": "A is on B", "goal": "B is on A", "plan": "\npick A\nstack A B"}]

    icl = get_icl(init_prompt, examples)

    assert icl.startswith("INTRO\n")
    assert "As initial conditions I have that, A is on B." in icl
    assert "My goal is to have that B is on A." in icl
    assert "pick A\nstack A B" in icl
    # trailing template for the live query, appended after all examples
    assert icl.endswith(
        "[STATEMENT]\nAs initial conditions I have that, <init_state>\n"
        "My goal is to <goals>\n\nMy plan is as follows:\n\n[PLAN]\n<action>"
    )


def _bare_evaluator(init_prompt):
    # __init__ unconditionally calls bw_utils.load_blocksworld(), which needs real
    # PDDL domain/instance files - bypass it, sample_prompt() only needs init_prompt.
    evaluator = BWEvaluator.__new__(BWEvaluator)
    evaluator.init_prompt = init_prompt
    evaluator.sample_prompt_type = "rap"
    return evaluator


EXAMPLE_WITH_FOUR_STATES = {
    "init": "A is on B",
    "goal": "B is on A",
    "plan": "\npick A\nstack A B\nunstack B\nstack B A",
    "states": [
        "A is on B",
        "A is clear, B is on table",
        "A is on table, B is clear",
        "B is on A",
    ],
}


def test_sample_prompt_rap_builds_six_progressively_shorter_examples():
    evaluator = _bare_evaluator({"intro": "INTRO\n", "example_pool": [EXAMPLE_WITH_FOUR_STATES]})

    prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=1)

    # 1 initial ICL + 5 rounds of state-stripping, regardless of how many states exist
    assert len(prompt["icl_list"]) == 6
    assert prompt["icl"] == prompt["icl_list"][0]


def test_sample_prompt_rap_strips_leading_state_and_plan_lines_each_round():
    evaluator = _bare_evaluator({"intro": "INTRO\n", "example_pool": [EXAMPLE_WITH_FOUR_STATES]})

    prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=1)

    # Round 1 (icl_list[1]): the plan's first two actions are dropped, but "init"
    # lags one round behind the states shift, so it's still the original state.
    assert "A is on B" in prompt["icl_list"][1]
    assert "pick A" not in prompt["icl_list"][1]
    assert "unstack B\nstack B A" in prompt["icl_list"][1]

    # Round 2 (icl_list[2]): "init" catches up to the state the previous round
    # dropped, and the plan has emptied out entirely.
    assert "A is clear, B is on table" in prompt["icl_list"][2]
    assert "A is on B" not in prompt["icl_list"][2]


def test_sample_prompt_rap_stops_shortening_once_one_state_remains():
    evaluator = _bare_evaluator({"intro": "INTRO\n", "example_pool": [EXAMPLE_WITH_FOUR_STATES]})

    prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=1)

    # 4 states -> shortens for 3 rounds (indices 1,2,3), then holds steady once only
    # one state remains, so the last two entries must be identical
    assert prompt["icl_list"][3] == prompt["icl_list"][4] == prompt["icl_list"][5]


def test_sample_prompt_unknown_type_raises():
    evaluator = _bare_evaluator({"intro": "INTRO\n", "example_pool": []})
    evaluator.sample_prompt_type = "not-a-real-type"

    with pytest.raises(NotImplementedError):
        evaluator.sample_prompt()
