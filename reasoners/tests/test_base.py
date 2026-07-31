import ast
import os
import pickle
from types import SimpleNamespace

from reasoners import (
    DefaultWorldModel,
    Environment,
    Evaluator,
    Reasoner,
    SearchConfig,
    WorldModel,
)
from reasoners.algorithm import MCTS
from reasoners.base import Tool, create_directory_if_not_exists


def test_create_directory_if_not_exists_creates_missing_directory(tmp_path):
    target = tmp_path / "new_dir"
    assert not target.exists()

    create_directory_if_not_exists(str(target))

    assert target.is_dir()


def test_create_directory_if_not_exists_is_a_noop_when_already_present(tmp_path):
    target = tmp_path / "existing_dir"
    target.mkdir()

    create_directory_if_not_exists(str(target))  # must not raise

    assert target.is_dir()


def test_default_world_model_tracks_action_sequence_as_state():
    world_model = DefaultWorldModel(base_model=None)

    assert world_model.init_state() == []
    state, aux = world_model.step([], "a")
    assert state == ["a"]
    assert aux == {}
    state2, _ = world_model.step(state, "b")
    assert state2 == ["a", "b"]
    assert world_model.is_terminal(state2) is False


def test_environment_init_has_no_env_by_default():
    class ConcreteEnvironment(Environment):
        def init_state(self):
            return None

        def step(self, state, action):
            return state, {}

        def is_terminal(self, state):
            return True

    assert ConcreteEnvironment().env is None


def test_tool_call_invokes_the_wrapped_function_with_kwargs():
    tool = Tool(func=lambda x, y: x + y, name="add", description="adds two numbers")

    assert tool.name == "add"
    assert tool.description == "adds two numbers"
    assert tool(x=1, y=2) == 3


class ToyWorldModel(WorldModel):
    def init_state(self):
        return ()

    def step(self, state, action):
        return state + (action,), {}

    def is_terminal(self, state):
        return len(state) >= 1


class ToyConfig(SearchConfig):
    def get_actions(self, state):
        return ["a"]

    def reward(self, state, action, **kwargs):
        return 1.0, {}


def test_reasoner_forwards_prompt_to_world_model_and_search_config():
    world_model = ToyWorldModel()
    search_config = ToyConfig()
    reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=MCTS(disable_tqdm=True, n_iters=2, depth_limit=1))

    reasoner("example", prompt="a real prompt")

    assert world_model.prompt == "a real prompt"
    assert search_config.prompt == "a real prompt"
    assert world_model.example == "example"
    assert search_config.example == "example"


class ToyEvaluator(Evaluator):
    def __init__(self, dataset, disable_log=False, disable_tqdm=True):
        self.full_dataset = dataset
        self._dataset_name = "toy"
        self.input_processor = lambda x: x
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm

    def sample_prompt(self, shuffle_prompt=True, num_shot=4, sample_prompt_type=None):
        return None

    def eval_output(self, answer, output):
        return answer == output


def _toy_reasoner():
    return Reasoner(
        world_model=ToyWorldModel(),
        search_config=ToyConfig(),
        search_algo=MCTS(disable_tqdm=True, n_iters=2, depth_limit=1),
    )


def test_evaluate_computes_accuracy_and_writes_logs(tmp_path):
    evaluator = ToyEvaluator(dataset=["ex1", "ex2"])
    evaluator.output_extractor = lambda algo_output: algo_output.terminal_state
    evaluator.answer_extractor = lambda x: ("a",)

    log_dir = str(tmp_path / "mylog")
    accuracy = evaluator.evaluate(_toy_reasoner(), log_dir=log_dir)

    assert accuracy == 1.0
    assert os.path.isfile(os.path.join(log_dir, "args.txt"))
    with open(os.path.join(log_dir, "result.log")) as f:
        result_log = f.read()
    assert "Case #1" in result_log
    assert "Case #2" in result_log
    with open(os.path.join(log_dir, "algo_output", "1.pkl"), "rb") as f:
        pickle.load(f)  # must not raise


def test_evaluate_disable_log_skips_per_example_logging_but_still_creates_log_dir(tmp_path):
    evaluator = ToyEvaluator(dataset=["ex1"], disable_log=True)
    evaluator.output_extractor = lambda algo_output: algo_output.terminal_state
    evaluator.answer_extractor = lambda x: ("a",)

    log_dir = str(tmp_path / "mylog")
    accuracy = evaluator.evaluate(_toy_reasoner(), log_dir=log_dir)

    assert accuracy == 1.0
    assert os.path.isfile(os.path.join(log_dir, "args.txt"))
    assert not os.path.exists(os.path.join(log_dir, "result.log"))


class SubQAWorldModel(WorldModel):
    """States carry .sub_question/.sub_answer, as evaluate_sc()'s path-building
    assumes of whatever terminal_state ends up being."""

    def init_state(self):
        return (SimpleNamespace(sub_question="q0", sub_answer="a0"),)

    def step(self, state, action):
        return state + (SimpleNamespace(sub_question="q", sub_answer="a"),), {}

    def is_terminal(self, state):
        return len(state) >= 2


def test_evaluate_sc_majority_votes_across_samples_and_writes_paths(tmp_path):
    evaluator = ToyEvaluator(dataset=["ex1"])
    evaluator.output_extractor = lambda algo_output: "final"
    evaluator.answer_extractor = lambda x: "final"

    reasoner = Reasoner(
        world_model=SubQAWorldModel(),
        search_config=ToyConfig(),
        search_algo=MCTS(disable_tqdm=True, n_iters=2, depth_limit=1),
    )

    log_dir = str(tmp_path / "mylog")
    accuracy = evaluator.evaluate_sc(reasoner, n_sc=3, log_dir=log_dir)

    assert accuracy == 1.0
    with open(os.path.join(log_dir, "algo_output.txt")) as f:
        saved = ast.literal_eval(f.read())
    assert saved == ["q0 a0 q a "] * 3
