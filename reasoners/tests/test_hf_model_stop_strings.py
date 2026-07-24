from typing import ClassVar

from reasoners.lm.hf_model import HFModel


class FakeTokenizer:
    """Vocab id -> decoded text. Token 3 simulates a BPE tokenizer merging punctuation with a
    trailing newline into one token (e.g. Qwen2 merges '.' + '\\n' into a single '.\\n' token) -
    the exact scenario that broke naive single-token stop-string handling."""

    _vocab: ClassVar[dict[int, str]] = {0: "a", 1: "b", 2: "\n", 3: ".\n"}

    def __len__(self):
        return len(self._vocab)

    def decode(self, ids):
        return "".join(self._vocab[i] for i in ids)

    def encode(self, s, add_special_tokens=False):
        for i, v in self._vocab.items():
            if v == s:
                return [i]
        raise ValueError(f"no exact single-token encoding for {s!r} in fake vocab")


def _bare_hf_model():
    # HFModel.__init__ loads a real model - bypass it, _stop_string_token_ids only needs
    # self.tokenizer and self._stop_string_token_ids_cache.
    model = HFModel.__new__(HFModel)
    model.tokenizer = FakeTokenizer()
    model._stop_string_token_ids_cache = {}
    return model


def test_stop_string_finds_every_vocab_token_ending_with_it():
    model = _bare_hf_model()
    ids = model._stop_string_token_ids("\n")
    # both the standalone '\n' (id 2) and the merged '.\n' (id 3) end with '\n'
    assert set(ids) == {2, 3}


def test_stop_string_result_is_cached():
    model = _bare_hf_model()
    first = model._stop_string_token_ids("\n")
    assert model._stop_string_token_ids_cache["\n"] is first
    second = model._stop_string_token_ids("\n")
    assert second is first


def test_stop_string_falls_back_to_standalone_encoding_if_no_vocab_token_matches():
    model = _bare_hf_model()
    assert model._stop_string_token_ids("a") == [0]
