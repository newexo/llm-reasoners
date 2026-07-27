"""Functional (Tier 3) test: exercises a real LM, needs GPU. Not run in CI - see
`make test-functional`. Guards against the regression fixed in PR #6, where
ProntoQA RAP's MCTS search could never reach a terminal "Finish." state at all
(every case returned empty output), regardless of iteration budget.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PRONTOQA_RAP_DIR = Path(__file__).resolve().parents[3] / "examples" / "RAP" / "prontoqa"
sys.path.insert(0, str(PRONTOQA_RAP_DIR))

N_CASES = 2


@pytest.mark.functional
def test_prontoqa_rap_reaches_terminal_state():
    from dataset import ProntoQADataset
    from search_config import ProntoQAConfig
    from world_model import ProntoQAWorldModel

    from reasoners import Reasoner
    from reasoners.algorithm import MCTS
    from reasoners.benchmark import ProntoQAEvaluatorFinal
    from reasoners.lm import HFModel

    def rap_answer_extractor(mcts_result):
        if mcts_result.trace is None:
            return ""
        return "\n".join(mcts_result.trace[0][i].body for i in range(1, len(mcts_result.trace[0]) - 1))

    dataset = ProntoQADataset.from_file(str(PRONTOQA_RAP_DIR.parent.parent / "CoT" / "prontoqa" / "data" / "345hop_random_true.json"))
    dataset.examples = dict(list(dataset.examples.items())[:N_CASES])

    with open(PRONTOQA_RAP_DIR.parent.parent / "CoT" / "prontoqa" / "data" / "example_next_steps.json") as f:
        init_prompt = json.load(f)

    base_model = HFModel(
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-v0.1",
        max_batch_size=2,
        max_new_tokens=512,
        quantized="nf4",
    )
    world_model = ProntoQAWorldModel(base_model=base_model)
    search_config = ProntoQAConfig(base_model=base_model, temperature=0.8, n_candidates=4)
    search_algo = MCTS(output_trace_in_each_iter=True, cum_reward=np.mean, depth_limit=6, n_iters=20)
    reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=search_algo)

    # Built only for sample_prompt() below - evaluator.evaluate() has side effects
    # (creates a logs/ directory unconditionally) that a hermetic test should avoid.
    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt["next_steps"],
        sample_prompt_type="rap",
        disable_log=True,
        output_extractor=rap_answer_extractor,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2]),
        disable_tqdm=True,
        dataset=dataset,
    )
    sampled_prompt = evaluator.sample_prompt(num_shot=4)

    outputs = []
    for example in dataset:
        algo_output = reasoner(example, prompt=sampled_prompt)
        outputs.append(rap_answer_extractor(algo_output))

    # Before PR #6's fix, this was 0/N every time - the search structurally could
    # never reach "Finish." regardless of iteration budget. At least one non-empty
    # output here means a terminal state was actually reached.
    assert any(o != "" for o in outputs), (
        f"no case reached a terminal state (all {len(outputs)} outputs empty) - "
        "this is the exact failure mode PR #6 fixed"
    )
