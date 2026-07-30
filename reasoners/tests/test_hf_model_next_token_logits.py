import numpy as np
import pytest
import torch

from reasoners.lm.hf_model import HFModel


class FakeBatchEncoding(dict):
    """Minimal stand-in for a real tokenizer's BatchEncoding - just needs .to(device)."""

    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab  # text -> list of token ids, same shape as a real .encode() result

    def encode(self, text, add_special_tokens=False):
        return self._vocab[text]

    def __call__(self, prompts, return_tensors="pt", padding=True):
        input_ids = torch.zeros((len(prompts), 1), dtype=torch.long)
        return FakeBatchEncoding(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


class FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Ignores its inputs entirely and returns fixed logits - shape [batch, 1, vocab_size]."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **kwargs):
        return FakeModelOutput(self._logits)


def _bare_hf_model(tokenizer, model):
    hf_model = HFModel.__new__(HFModel)
    hf_model.tokenizer = tokenizer
    hf_model.model = model
    hf_model.device = "cpu"
    hf_model.max_batch_size = 8
    return hf_model


def test_get_next_token_logits_casts_bfloat16_to_float32_before_numpy():
    # Regression test for get_next_token_logits()'s `.float().cpu().numpy()` fix -
    # numpy has no bfloat16 dtype, so converting a bf16 tensor straight to numpy
    # (without the .float() cast) raises. bf16 is what quantized models actually
    # return in practice (e.g. nf4 quantization's bnb_4bit_compute_dtype).
    tokenizer = FakeTokenizer({"Yes": [0], "No": [1]})
    logits = torch.zeros((1, 1, 2), dtype=torch.bfloat16)
    logits[0, 0, 0] = 3.5
    logits[0, 0, 1] = 1.5
    hf_model = _bare_hf_model(tokenizer, FakeModel(logits))

    result = hf_model.get_next_token_logits("some prompt", ["Yes", "No"])

    assert len(result) == 1
    assert result[0].dtype == np.float32
    assert result[0][0] == pytest.approx(3.5)
    assert result[0][1] == pytest.approx(1.5)


def test_get_next_token_logits_selects_requested_candidates_only():
    # vocab has 4 tokens; candidates only ask for 2 of them - result must be
    # exactly those two logits, in the order requested, not all 4.
    tokenizer = FakeTokenizer({"A": [0], "B": [1], "C": [2], "D": [3]})
    logits = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]], dtype=torch.float32)
    hf_model = _bare_hf_model(tokenizer, FakeModel(logits))

    result = hf_model.get_next_token_logits("prompt", ["D", "A"])

    assert list(result[0]) == pytest.approx([40.0, 10.0])


def test_get_next_token_logits_multi_token_candidate_uses_second_token():
    # Current, if surprising, behavior: when a candidate encodes to more than one
    # token, get_next_token_logits() uses token[1] (not token[0]) as the id to
    # score against - pinning this so a future change to it is a deliberate one.
    tokenizer = FakeTokenizer({"multi": [7, 42]})
    logits = torch.zeros((1, 1, 50), dtype=torch.float32)
    logits[0, 0, 42] = 99.0
    hf_model = _bare_hf_model(tokenizer, FakeModel(logits))

    with pytest.warns(UserWarning, match="instead of 1"):
        result = hf_model.get_next_token_logits("prompt", ["multi"])

    assert result[0][0] == pytest.approx(99.0)
