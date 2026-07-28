import pytest

from reasoners.benchmark.bw_utils import (
    apply_change,
    extract_goals,
    extract_init_state,
    fill_template,
    generate_all_actions,
    get_ordered_objects,
    goal_check,
    read_config,
)


def test_fill_template_builds_statement_and_replaces_ontable_and_dashes():
    text = fill_template("the red block is ontable", "the red-block is clear", "")
    assert "As initial conditions I have that, the red block is on the table." in text
    assert "My goal is to have that the red block is clear." in text
    assert "ontable" not in text
    assert "-" not in text


def test_fill_template_omits_missing_sections():
    text = fill_template("", "", "\n[PLAN END]\n")
    assert "[STATEMENT]" not in text
    assert "My goal" not in text
    assert "[PLAN]" in text


def test_get_ordered_objects_orders_by_position_in_line():
    objs = get_ordered_objects(
        ["red block", "blue block"], "stack the blue block on top of the red block"
    )
    assert objs == ["blue block", "red block"]


def test_generate_all_actions_when_hand_empty():
    state = (
        "the red block is clear, the blue block is clear, the hand is empty, "
        "the red block is on the table, and the blue block is on the table."
    )
    actions = generate_all_actions(state)
    assert "pick up the red block" in actions
    assert "pick up the blue block" in actions


def test_generate_all_actions_unstack_when_block_is_on_another():
    state = (
        "the red block is clear, the hand is empty, the blue block is on the table, "
        "and the red block is on top of the blue block."
    )
    actions = generate_all_actions(state)
    assert actions == ["unstack the red block from on top of the blue block"]


def test_generate_all_actions_when_holding_a_block():
    state = "the blue block is clear, the hand is holding the red block, and the blue block is on the table."
    actions = generate_all_actions(state)
    assert "stack the red block on top of the blue block" in actions
    assert "put down the red block" in actions


def test_apply_change_hand_branch():
    state = "the red block is clear, the hand is empty, and the red block is on the table."
    change = "the hand was empty and is now holding the red block"
    assert apply_change(change, state) == (
        "the red block is clear, the hand is holding the red block, and the red block is on the table."
    )


def test_apply_change_color_was_now_branch():
    state = "the red block is clear, the hand is empty, and the red block is on the table."
    change = "the red block was clear and is now in the hand"
    assert apply_change(change, state) == (
        "the red block is in the hand, the hand is empty, and the red block is on the table."
    )


def test_apply_change_no_longer_and_priority_sorting():
    # "clear" sorts before "on top of" which sorts before "on the table" regardless
    # of the original state's ordering.
    state = "the red block is clear, the red block is in the hand, and the blue block is on the table."
    change = "the red block is no longer in the hand, the red block is now on top of the blue block"
    assert apply_change(change, state) == (
        "the red block is clear, the red block is on top of the blue block, "
        "and the blue block is on the table."
    )


def test_apply_change_raises_on_unrecognized_change():
    # apply_change only ever raises a bare Exception("ERROR") - not a specific type
    state = "the red block is clear, the hand is empty, and the red block is on the table."
    with pytest.raises(Exception, match="ERROR"):
        apply_change("something completely unrelated to any block", state)


def test_goal_check_all_goals_met():
    blocks_state = "the red block is on top of the blue block, and the hand is empty."
    assert goal_check(["the red block is on top of the blue block"], blocks_state) == (True, 1.0)


def test_goal_check_partial_goals_met():
    blocks_state = "the red block is on top of the blue block, and the hand is empty."
    goals = ["the red block is on top of the blue block", "the blue block is on the table"]
    met, fraction = goal_check(goals, blocks_state)
    assert met is False
    assert fraction == 0.5


QUESTION = (
    "\n[STATEMENT]\nAs initial conditions I have that, the red block is on the table."
    "\nMy goal is to have that the red block is on top of the blue block."
    "\n\nMy plan is as follows:\n\n[PLAN]\npick up the red block\n[PLAN END]\n"
)


def test_extract_goals_finds_block_relations():
    assert extract_goals({"question": QUESTION}) == ["the red block is on top of the blue block"]


def test_extract_init_state():
    assert extract_init_state({"question": QUESTION}) == "the red block is on the table."


def test_read_config_loads_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("encoded_objects:\n  a: red block\n  b: blue block\n")

    config = read_config(str(config_file))

    assert config == {"encoded_objects": {"a": "red block", "b": "blue block"}}
