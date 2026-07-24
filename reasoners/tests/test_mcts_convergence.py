from reasoners import Reasoner, SearchConfig, WorldModel
from reasoners.algorithm import MCTS, MCTSAggregation


class ConvergingWorldModel(WorldModel):
    """State is the action sequence so far. Terminal after 3 actions."""

    def init_state(self):
        return ()

    def step(self, state, action):
        return state + (action,), {}

    def is_terminal(self, state):
        return len(state) >= 3


class ConvergingConfig(SearchConfig):
    """'a' is always strictly better than 'b', so search should quickly and
    consistently favor the ('a', 'a', 'a') path."""

    def get_actions(self, state):
        return ["a", "b"]

    def fast_reward(self, state, action):
        return (1.0 if action == "a" else 0.0), {}

    def reward(self, state, action, **kwargs):
        return (1.0 if action == "a" else 0.0), {}


def _run(
    n_iters,
    convergence_stop_iters=None,
    convergence_stop_share=0.8,
    with_aggregator=True,
    log_tree_stats=True,
):
    aggregator = (
        MCTSAggregation(retrieve_answer=lambda state: state)
        if with_aggregator
        else None
    )
    mcts = MCTS(
        n_iters=n_iters,
        depth_limit=3,
        disable_tqdm=True,
        log_tree_stats=log_tree_stats,
        convergence_stop_iters=convergence_stop_iters,
        convergence_stop_share=convergence_stop_share,
        aggregator=aggregator,
    )
    reasoner = Reasoner(
        world_model=ConvergingWorldModel(),
        search_config=ConvergingConfig(),
        search_algo=mcts,
    )
    reasoner("toy-example")
    return mcts


def _captured_lines(capsys):
    return capsys.readouterr().out.splitlines()


def _iteration_lines(lines):
    return [l for l in lines if l.startswith("MCTS iteration")]


def test_without_convergence_stop_runs_full_budget(capsys):
    n_iters = 20
    _run(n_iters=n_iters, convergence_stop_iters=None)
    lines = _captured_lines(capsys)
    assert len(_iteration_lines(lines)) == n_iters


def test_convergence_stop_ends_search_early(capsys):
    n_iters = 20
    _run(n_iters=n_iters, convergence_stop_iters=3, convergence_stop_share=0.8)
    lines = _captured_lines(capsys)
    # dominant path should be found and held well before the full budget
    assert len(_iteration_lines(lines)) < n_iters
    assert any("converged" in line for line in lines)


def test_convergence_stop_ignored_without_aggregator(capsys):
    n_iters = 10
    _run(
        n_iters=n_iters,
        convergence_stop_iters=2,
        convergence_stop_share=0.8,
        with_aggregator=False,
    )
    lines = _captured_lines(capsys)
    # no aggregator means _leading_answer_share() always returns None -> never converges
    assert len(_iteration_lines(lines)) == n_iters
    assert not any("converged" in line for line in lines)
