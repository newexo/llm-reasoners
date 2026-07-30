import pytest

from reasoners import SearchConfig, WorldModel
from reasoners.algorithm import BeamSearch


class ToyWorldModel(WorldModel):
    """State is the action sequence so far. Terminal once 3 actions deep."""

    def init_state(self):
        return ()

    def step(self, state, action):
        return state + (action,), {}

    def is_terminal(self, state):
        return len(state) >= 3


class TwoActionConfig(SearchConfig):
    """'a' is always strictly better than 'b'."""

    def get_actions(self, state):
        return ["a", "b"]

    def fast_reward(self, state, action):
        return (1.0 if action == "a" else 0.0), {}

    def reward(self, state, action, **kwargs):
        return (1.0 if action == "a" else 0.0), {}


class ThreeActionConfig(SearchConfig):
    def get_actions(self, state):
        return ["a", "b", "c"]

    def fast_reward(self, state, action):
        return 0.0, {}

    def reward(self, state, action, **kwargs):
        return 0.0, {}


class NoActionConfig(SearchConfig):
    def get_actions(self, state):
        return []

    def fast_reward(self, state, action):
        return 0.0, {}

    def reward(self, state, action, **kwargs):
        return 0.0, {}


def test_argmax_beam_search_finds_best_path():
    bs = BeamSearch(beam_size=2, max_depth=3, sampling_strategy="argmax")
    result = bs(ToyWorldModel(), TwoActionConfig())
    assert result.terminal_state == ("a", "a", "a")


def test_reward_aggregator_variants_produce_different_cum_reward():
    # same winning path each time ("a" always wins), but the *aggregated* value
    # over its 3 rewards of 1.0 each must differ by aggregator
    common = {"beam_size": 2, "max_depth": 3, "sampling_strategy": "argmax"}
    cumulative = BeamSearch(reward_aggregator="cumulative", **common)(ToyWorldModel(), TwoActionConfig())
    mean = BeamSearch(reward_aggregator="mean", **common)(ToyWorldModel(), TwoActionConfig())
    last = BeamSearch(reward_aggregator="last", **common)(ToyWorldModel(), TwoActionConfig())

    assert cumulative.cum_reward == pytest.approx(3.0)
    assert mean.cum_reward == pytest.approx(1.0)
    assert last.cum_reward == pytest.approx(1.0)


def test_unrecognized_reward_aggregator_raises():
    with pytest.raises(NotImplementedError):
        BeamSearch(beam_size=2, max_depth=3, reward_aggregator="not-a-real-aggregator")


def test_beam_size_caps_the_returned_beam():
    bs = BeamSearch(beam_size=2, max_depth=1, sampling_strategy="argmax", return_beam=True)
    beam = bs(ToyWorldModel(), ThreeActionConfig())
    assert len(beam) == 2


def test_no_available_actions_returns_empty_result_without_crashing():
    bs = BeamSearch(beam_size=2, max_depth=3, sampling_strategy="argmax")
    result = bs(ToyWorldModel(), NoActionConfig())
    assert result.terminal_state is None
    assert result.terminal_node is None
    assert result.trace == []


def test_temperature_zero_forces_argmax_even_when_stochastic_requested():
    # Regression test: temperature=0.0 is falsy in Python, so the original
    # `self.temperature and self.temperature < 1e-4` guard silently skipped
    # forcing argmax for exactly 0.0 (while still catching e.g. 0.00001) -
    # meaning softmax() would later divide by zero if actually invoked.
    with pytest.warns(UserWarning, match="Temperature is set to 0"):
        bs = BeamSearch(beam_size=2, max_depth=1, sampling_strategy="stochastic", temperature=0.0)
    assert bs.sampling_strategy == "argmax"


def test_stochastic_without_a_temperature_forces_argmax():
    # Regression test for the adjacent gap: sampling_strategy="stochastic" with no
    # temperature at all (the default) used to sail through construction and only
    # fail later with a TypeError inside softmax() (float / None). This is
    # concretely reachable through examples/ToT/blocksworld/tot_inference.py: its
    # own --temperature CLI flag is consumed by the LLM's generation config, so
    # BeamSearch always gets constructed with temperature=None regardless of what
    # the user passes, unless sampling_strategy stays "argmax".
    with pytest.warns(UserWarning, match="Stochastic sampling requires a temperature"):
        bs = BeamSearch(beam_size=2, max_depth=1, sampling_strategy="stochastic")
    assert bs.sampling_strategy == "argmax"


def test_invalid_sampling_strategy_falls_back_to_argmax_with_warning():
    with pytest.warns(UserWarning, match="Sampling strategy only supports"):
        bs = BeamSearch(beam_size=2, max_depth=1, sampling_strategy="bogus")
    assert bs.sampling_strategy == "argmax"


def test_early_terminate_false_forces_return_beam_with_warning():
    with pytest.warns(UserWarning, match="early_terminate is set to False"):
        bs = BeamSearch(beam_size=2, max_depth=1, early_terminate=False)
    assert bs.return_beam is True


def test_early_terminate_stops_expansion_once_world_is_terminal():
    bs = BeamSearch(beam_size=2, max_depth=10, sampling_strategy="argmax", early_terminate=True)
    result = bs(ToyWorldModel(), TwoActionConfig())
    # ToyWorldModel.is_terminal() fires at length 3, well short of max_depth=10
    assert len(result.terminal_state) == 3
