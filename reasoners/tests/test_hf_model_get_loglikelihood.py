import pytest
import torch

from reasoners.lm.hf_model import HFModel

VOCAB = {"Yes": 0, "A": 1, "B": 2, "No": 3}


class FakeBatchEncoding(dict):
    def to(self, device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = -1

    def __call__(self, texts, return_tensors="pt", add_special_tokens=False, padding=True):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[VOCAB[word] for word in text.split()] for text in texts]
        max_len = max(len(row) for row in rows)
        padded = [row + [self.pad_token_id] * (max_len - len(row)) for row in rows]
        return FakeBatchEncoding(input_ids=torch.tensor(padded, dtype=torch.long))


class FakeModel:
    """Returns fixed logits at every position (only the first position's logits
    matter for these tests, since that's what predicts the sole post-prefix token
    in the non-padding cases)."""

    def __init__(self, first_position_logits):
        self.first_position_logits = first_position_logits

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        bsz, seq_len = input_ids.shape
        logits = torch.zeros((bsz, seq_len, len(self.first_position_logits)))
        logits[:, 0, :] = torch.tensor(self.first_position_logits)
        return type("Output", (), {"logits": logits})()


def _bare_hf_model(model):
    hf_model = HFModel.__new__(HFModel)
    hf_model.tokenizer = FakeTokenizer()
    hf_model.model = model
    hf_model.device = "cpu"
    hf_model.max_batch_size = 8
    return hf_model


def test_get_loglikelihood_matches_manually_computed_softmax_logprob():
    hf_model = _bare_hf_model(FakeModel(first_position_logits=[1.0, 2.0, 0.5, 0.0]))

    result = hf_model.get_loglikelihood("Yes", ["Yes A", "Yes B"])

    probs = torch.softmax(torch.tensor([1.0, 2.0, 0.5, 0.0]), dim=-1)
    assert result[0] == pytest.approx(torch.log(probs[VOCAB["A"]]).item())
    assert result[1] == pytest.approx(torch.log(probs[VOCAB["B"]]).item())


def test_get_loglikelihood_raises_if_content_does_not_start_with_prefix():
    hf_model = _bare_hf_model(FakeModel(first_position_logits=[1.0, 1.0, 1.0, 1.0]))

    with pytest.raises(AssertionError):
        hf_model.get_loglikelihood("Yes", ["No A"])


def test_get_loglikelihood_skips_padding_tokens():
    # uniform logits -> each of the 4 vocab entries gets log(0.25) as its logprob
    hf_model = _bare_hf_model(FakeModel(first_position_logits=[0.0, 0.0, 0.0, 0.0]))
    log_quarter = torch.log(torch.tensor(0.25)).item()

    result = hf_model.get_loglikelihood("Yes", ["Yes A B", "Yes A"])

    # "Yes A B": 2 real post-prefix tokens contribute
    assert result[0] == pytest.approx(2 * log_quarter)
    # "Yes A": padded to match "Yes A B"'s length, but the pad position is skipped
    assert result[1] == pytest.approx(log_quarter)


def test_get_loglikelihood_rejects_batch_larger_than_max_batch_size():
    hf_model = _bare_hf_model(FakeModel(first_position_logits=[1.0, 1.0, 1.0, 1.0]))
    hf_model.max_batch_size = 1

    with pytest.raises(AssertionError):
        hf_model.get_loglikelihood("Yes", ["Yes A", "Yes B"])
