from reasoners import Reasoner, SearchConfig, WorldModel
from reasoners.algorithm import MCTS


class ToyWorldModel(WorldModel):
    def init_state(self):
        return 0

    def step(self, state, action):
        return state + 1, {}

    def is_terminal(self, state):
        return state >= 3


class ToyConfig(SearchConfig):
    def get_actions(self, state):
        return ["a", "b"]

    def reward(self, state, action, **kwargs):
        return 1.0, {}


def _run(n_iters, log_tree_stats=False):
    mcts = MCTS(
        n_iters=n_iters, depth_limit=3, disable_tqdm=True, log_tree_stats=log_tree_stats
    )
    reasoner = Reasoner(
        world_model=ToyWorldModel(), search_config=ToyConfig(), search_algo=mcts
    )
    reasoner("toy-example")
    return mcts


def test_tree_stats_reports_growing_node_count():
    mcts = _run(n_iters=5)
    node_count, mean_visits = mcts._tree_stats()
    # root + at least one expansion per iteration
    assert node_count > 1
    assert mean_visits > 0


def test_log_tree_stats_prints_one_line_per_iteration(capsys):
    n_iters = 3
    _run(n_iters=n_iters, log_tree_stats=True)
    lines = [
        l
        for l in capsys.readouterr().out.splitlines()
        if l.startswith("MCTS iteration")
    ]
    assert len(lines) == n_iters
    assert "tree_nodes=" in lines[0]
    assert "mean_visits_per_node=" in lines[0]


def test_log_tree_stats_disabled_by_default_is_silent(capsys):
    _run(n_iters=3, log_tree_stats=False)
    lines = [
        l
        for l in capsys.readouterr().out.splitlines()
        if l.startswith("MCTS iteration")
    ]
    assert len(lines) == 0
