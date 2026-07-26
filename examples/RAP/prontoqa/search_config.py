import sys
from typing import Optional
import torch 

import prompts.finish
import prompts.valid_rap
import prompts.next_step
from dataset import ProntoQAExample
from reasoners import SearchConfig, LanguageModel
from world_model import ProntoQAState, ProntoQAAction, ProntoQAWorldModel


def format_examples(sampled_data):
    formatted_examples = ""
    for i, entry in enumerate(sampled_data, 1):
        facts = f"Facts {i}: {entry['Facts']}\n"
        query = f"Query {i}: {entry['Query']}\n"
        claims_and_next = ""

        for j, (claim, next_step) in enumerate(zip(entry['claims'], entry['next_steps']), 1):
            claims_and_next += f"Claim {i}.{j}: {claim}\nNext {i}.{j}: {next_step}\n"

        formatted_examples += facts + query + claims_and_next + "\n"

    return formatted_examples


def _state_chain(state: ProntoQAState) -> list:
    """Walk `last_state` back to the initial state, returning states oldest-first."""
    chain = []
    cur = state
    while cur is not None:
        chain.append(cur)
        cur = cur.last_state
    chain.reverse()
    return chain


def build_next_step_query(sampled_data, base_facts, query, state: ProntoQAState) -> str:
    # Mirrors format_examples()'s accumulated "Claim i.1...Next i.1...Claim i.j" shape
    # for the live query too. Previously this only ever rendered a bare "Claim N.1:
    # {state}" regardless of true search depth - the few-shot demos only ever show
    # "Finish." at the highest sub-step index, never at ".1", so a live query always
    # presented as ".1" could never pattern-match to it.
    problem_idx = len(sampled_data) + 1
    chain = _state_chain(state)

    input_prompt = format_examples(sampled_data)
    input_prompt += prompts.next_step.FACTS_FORMAT.format(problem_idx, ". ".join(base_facts))
    input_prompt += prompts.next_step.QUERY_FORMAT.format(problem_idx, query)
    for step_idx in range(len(chain) - 1):
        j = step_idx + 1
        input_prompt += prompts.next_step.CLAIM_FORMAT.format(problem_idx, j, chain[step_idx].body)
        input_prompt += f"Next {problem_idx}.{j}: {chain[step_idx + 1].last_action}\n"
    k = len(chain)
    input_prompt += prompts.next_step.CLAIM_FORMAT.format(problem_idx, k, chain[-1].body)
    input_prompt += prompts.next_step.NEXT_STEP_PREFIX.format(problem_idx, k)
    return input_prompt


class ProntoQAConfig(SearchConfig[ProntoQAState, ProntoQAAction,ProntoQAExample]):

    def __init__(self, base_model: LanguageModel, temperature=0.8, n_candidates=4):
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.example: ProntoQAExample = self.example

    def get_actions(self, state: ProntoQAState) -> list[ProntoQAAction]:

        *base_facts, init_state = self.example.test_example.question.split(". ")

        input_prompt = build_next_step_query(self.prompt, base_facts, self.example.test_example.query, state)

        # print(f"input_prompt: {input_prompt}")
        outputs = self.base_model.generate([input_prompt] * self.n_candidates, eos_token_id="\n", hide_input=True, temperature=self.temperature, do_sample=True).text
        outputs = [output.strip() for output in outputs]
        # deduplicate
        outputs = list(dict.fromkeys(outputs))

        return outputs

    # OLD fast reward code
    def fast_reward(
            self,
            state: ProntoQAState,
            action: ProntoQAAction,
    ) -> tuple[float, dict]:
        *base_facts, init_state = self.example.test_example.question.split(". ")
        input_prompt = ""
        # see world_model.py's step() for why this is startswith, not an exact match
        if action.strip().startswith("Finish."):
            input_prompt += prompts.finish.EXAMPLES
            input_prompt += prompts.finish.TARGET_FORMAT.format(self.example.test_example.query)
            input_prompt += prompts.finish.CLAIM_FORMAT.format(state)
            input_prompt += prompts.finish.OUTPUT_PREFIX
        else:
            input_prompt = prompts.valid_rap.TEMPLATE.replace("[[STATE]]", state.body)\
                .replace("[[ACTION]]", action)\
                .replace("[[QUERY]]", self.example.test_example.query)\
                .replace("[[FACTS]]", ". ".join(base_facts) + ".")

        output_logits = self.base_model.get_next_token_logits(
            input_prompt,
            candidates=["Yes", "No"]
        )
        print("output_logits: ", output_logits)
        # self_eval:float = torch.softmax(torch.tensor(output_logits[0]), dim=0)[0].item()
        self_eval:float = output_logits[0][0].item()

        # intuition reward

        *base_facts, init_state = self.example.test_example.question.split(". ")

        input_prompt = build_next_step_query(self.prompt, base_facts, self.example.test_example.query, state)
        outputs = input_prompt + " " + action
        intuition = self.base_model.get_loglikelihood(input_prompt, [outputs])[0]

        if action.strip().startswith("Finish."):
            print(f"S[{state}] Q[{self.example.test_example.query}] -> Self-eval[{self_eval}] Intuition[{intuition}]", flush=True)
        else:
            print(f"S[{state.last_state}] A[{action}] S'[{state}] -> Self-eval[{self_eval}] Intuition[{intuition}]", flush=True)

        return intuition + self_eval, {"self-eval": self_eval, "intuition": intuition}

    def reward(self,
               state: ProntoQAState,
               action: ProntoQAAction,
               **kwargs
               ) -> tuple[float, dict]:
        self_eval = kwargs["self-eval"]
        intuition = kwargs["intuition"]
        return intuition + self_eval, {"self-eval": self_eval, "intuition": intuition}
