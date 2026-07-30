from reasoners import SearchConfig, WorldModel
from reasoners.algorithm import DFS


class ToyWorldModel(WorldModel):
    """State is the action sequence so far. Terminal once 5 actions deep - deep
    enough that every test below hits its own depth/count cap first."""

    def init_state(self):
        return ()

    def step(self, state, action):
        return state + (action,), {}

    def is_terminal(self, state):
        return len(state) >= 5


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


class SingleActionConfig(SearchConfig):
    def get_actions(self, state):
        return ["only"]

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


def _trace_actions(node):
    return [a for a, s, r in node.get_trace()]


def test_depth_caps_search_and_prior_selects_best_path():
    dfs = DFS(depth=2, total_states=100, max_per_state=2, prior=True)
    result = dfs(ToyWorldModel(), TwoActionConfig())

    # depth=2 with 2 actions/state -> exactly 2^2 = 4 terminal leaves
    assert len(result.terminal_nodes) == 4
    # prior=True sorts by fast_reward, so the all-"a" path is explored (and wins) first
    assert result.terminal_state == ("a", "a")
    assert _trace_actions(result.terminal_nodes[0])[1:] == ["a", "a"]


def test_max_per_state_limits_branching():
    dfs = DFS(depth=2, max_per_state=1, prior=False)
    result = dfs(ToyWorldModel(), ThreeActionConfig())

    # only the first action per state is explored -> a single linear path to depth 2
    assert len(result.terminal_nodes) == 1


def test_max_terminal_nodes_caps_results():
    dfs = DFS(depth=2, max_per_state=3, max_terminal_nodes=2, prior=False)
    result = dfs(ToyWorldModel(), ThreeActionConfig())

    # without the cap, 3 actions/state at depth 2 would produce 3^2 = 9 leaves
    assert len(result.terminal_nodes) == 2


def test_return_if_single_first_action_short_circuits():
    dfs = DFS(depth=3, return_if_single_first_action=True)
    result = dfs(ToyWorldModel(), SingleActionConfig())

    assert len(result.terminal_nodes) == 1
    assert result.terminal_nodes[0].action == "only"
    assert result.terminal_nodes[0].depth == 1


def test_no_available_actions_returns_empty_result_instead_of_crashing():
    # Regression test: if get_actions() is empty at the root before any terminal
    # node is ever found, self.terminals stays empty and sorted_terminals[0] used
    # to raise IndexError. Now returns a None-terminal_state sentinel, matching
    # BeamSearch's handling of its own empty-terminal-beam case.
    dfs = DFS(depth=3)
    result = dfs(ToyWorldModel(), NoActionConfig())

    assert result.terminal_state is None
    assert result.terminal_nodes == []
