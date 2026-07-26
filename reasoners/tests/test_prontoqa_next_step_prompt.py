import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "RAP" / "prontoqa"))

from search_config import build_next_step_query  # noqa: E402
from world_model import ProntoQAState  # noqa: E402


def _chain_of(*bodies_and_actions):
    """Build a ProntoQAState chain from (body, action_that_led_here) pairs;
    action is None for the initial state."""
    state = None
    for body, action in bodies_and_actions:
        state = ProntoQAState(body=body, last_state=state, last_action=action)
    return state


def test_next_step_query_accumulates_full_history_not_just_current_state():
    """Regression test for the bug where CLAIM_FORMAT/NEXT_STEP_PREFIX hardcoded
    the sub-step index as a literal ".1", so the live query always rendered as
    if it were the first reasoning step regardless of true search depth - and
    the few-shot demonstrations only ever show "Finish." at the highest
    sub-step index, never at ".1", making it unreachable in practice."""
    state = _chain_of(
        ("Sally is a painted lady.", None),
        ("Sally is a butterfly.", "Each painted lady is a butterfly."),
        ("Sally is an arthropod.", "Butterflies are lepidopterans."),
    )
    sampled_data = [
        {"Facts": "f1", "Query": "q1", "claims": ["c1"], "next_steps": ["n1"]},
        {"Facts": "f2", "Query": "q2", "claims": ["c2"], "next_steps": ["n2"]},
    ]

    prompt = build_next_step_query(sampled_data, ["fact a", "fact b"], "True or false: Sally is not bony.", state)

    problem_idx = len(sampled_data) + 1
    assert f"Claim {problem_idx}.1: Sally is a painted lady." in prompt
    assert f"Next {problem_idx}.1: Each painted lady is a butterfly." in prompt
    assert f"Claim {problem_idx}.2: Sally is a butterfly." in prompt
    assert f"Next {problem_idx}.2: Butterflies are lepidopterans." in prompt
    assert f"Claim {problem_idx}.3: Sally is an arthropod." in prompt
    assert prompt.rstrip("\n").endswith(f"Next {problem_idx}.3:")


def test_next_step_query_first_step_uses_real_sub_index_not_hardcoded():
    """Even the very first call (no history yet) must use the real sub-step
    index (1) rather than a value baked into the format string, so later
    steps in the same search correctly advance past it."""
    state = _chain_of(("Sally is a painted lady.", None))
    sampled_data = [{"Facts": "f1", "Query": "q1", "claims": ["c1"], "next_steps": ["n1"]}]

    prompt = build_next_step_query(sampled_data, ["fact a"], "True or false: Sally is not bony.", state)

    problem_idx = len(sampled_data) + 1
    assert f"Claim {problem_idx}.1: Sally is a painted lady." in prompt
    assert prompt.rstrip("\n").endswith(f"Next {problem_idx}.1:")
