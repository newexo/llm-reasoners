import pytest
import torch

from reasoners.lm.hf_model import HFModel


class FakeBatchEncoding(dict):
    def to(self, device):
        return self


class FakeGenerateOutput:
    def __init__(self, sequences, scores=None):
        self.sequences = sequences
        self.scores = scores


class FakeModel:
    """Records every .generate() call's batch size and generation_config, and
    returns dummy sequences - real content doesn't matter since FakeTokenizer's
    batch_decode() ignores `sequences` and instead reconstructs "prompt +
    continuation" from the prompts it was last called with."""

    def __init__(self):
        self.generate_calls = []

    def generate(self, input_ids=None, attention_mask=None, generation_config=None, output_scores=False, return_dict_in_generate=True):
        self.generate_calls.append({"batch": input_ids.shape[0], "generation_config": generation_config})
        batch = input_ids.shape[0]
        sequences = torch.zeros((batch, 1), dtype=torch.long)
        scores = [torch.zeros((batch, 5))] if output_scores else None
        return FakeGenerateOutput(sequences, scores)


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_token_id = 1
        self._last_prompts = None

    def __call__(self, prompts, return_tensors="pt", padding=True):
        self._last_prompts = list(prompts)
        input_ids = torch.zeros((len(prompts), 1), dtype=torch.long)
        return FakeBatchEncoding(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [p + " CONTINUATION" for p in self._last_prompts]


def _bare_hf_model(max_batch_size=8):
    hf_model = HFModel.__new__(HFModel)
    hf_model.tokenizer = FakeTokenizer()
    hf_model.model = FakeModel()
    hf_model.device = "cpu"
    hf_model.max_batch_size = max_batch_size
    hf_model.max_length = 2048
    hf_model.max_new_tokens = None
    hf_model._stop_string_token_ids_cache = {}
    return hf_model


def test_hide_input_strips_exactly_the_prompt_prefix():
    hf_model = _bare_hf_model()
    out = hf_model.generate(["Hello "], hide_input=True, do_sample=True, temperature=0.8)
    assert out.text == [" CONTINUATION"]


def test_hide_input_false_keeps_prompt_and_continuation():
    hf_model = _bare_hf_model()
    out = hf_model.generate(["Hello "], hide_input=False, do_sample=True, temperature=0.8)
    assert out.text == ["Hello  CONTINUATION"]


def test_inputs_are_batched_by_max_batch_size():
    hf_model = _bare_hf_model(max_batch_size=2)
    hf_model.generate(["a", "b", "c"], hide_input=False, do_sample=True, temperature=0.8)
    assert [call["batch"] for call in hf_model.model.generate_calls] == [2, 1]


def test_num_return_sequences_replicates_single_input():
    hf_model = _bare_hf_model()
    out = hf_model.generate(["a"], num_return_sequences=3, hide_input=False, do_sample=True, temperature=0.8)
    assert out.text == ["a CONTINUATION", "a CONTINUATION", "a CONTINUATION"]


def test_num_return_sequences_rejects_multiple_inputs():
    hf_model = _bare_hf_model()
    with pytest.raises(AssertionError):
        hf_model.generate(["a", "b"], num_return_sequences=2, hide_input=False, do_sample=True, temperature=0.8)


def test_eos_token_id_int_is_appended_alongside_tokenizer_eos():
    hf_model = _bare_hf_model()
    hf_model.generate(["a"], eos_token_id=99, hide_input=False, do_sample=True, temperature=0.8)
    config = hf_model.model.generate_calls[-1]["generation_config"]
    assert config.eos_token_id == [99, hf_model.tokenizer.eos_token_id]


def test_eos_token_id_unrecognized_type_warns_and_is_ignored():
    hf_model = _bare_hf_model()
    with pytest.warns(UserWarning, match="neither str nor int"):
        hf_model.generate(["a"], eos_token_id=[3.14], hide_input=False, do_sample=True, temperature=0.8)
    config = hf_model.model.generate_calls[-1]["generation_config"]
    assert config.eos_token_id == [hf_model.tokenizer.eos_token_id]


def test_do_sample_false_forces_greedy_params_and_warns():
    hf_model = _bare_hf_model()
    with pytest.warns(UserWarning, match="temperature=0.0 is equivalent to greedy search"):
        hf_model.generate(["a"], hide_input=False, do_sample=False)
    config = hf_model.model.generate_calls[-1]["generation_config"]
    assert config.do_sample is False
    assert config.temperature == 1.0
    assert config.top_k == 1


def test_output_log_probs_false_returns_none():
    hf_model = _bare_hf_model()
    out = hf_model.generate(["a"], hide_input=False, output_log_probs=False, do_sample=True, temperature=0.8)
    assert out.log_prob is None


def test_output_log_probs_true_returns_scores():
    hf_model = _bare_hf_model()
    out = hf_model.generate(["a"], hide_input=False, output_log_probs=True, do_sample=True, temperature=0.8)
    assert out.log_prob is not None
    assert len(out.log_prob) == 1
    assert out.log_prob[0].shape == (1, 5)
