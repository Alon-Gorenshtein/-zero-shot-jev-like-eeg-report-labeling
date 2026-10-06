from eegrl.labelers.headtail import head_tail, encode


def test_head_tail_keeps_short_and_trims_long():
    assert head_tail(list(range(10)), 20) == list(range(10))
    out = head_tail(list(range(1000)), 510, head=128)
    assert len(out) == 510 and out[:128] == list(range(128)) and out[128:] == list(range(1000 - 382, 1000))


class FakeTok:
    cls_token_id, sep_token_id, pad_token_id = 101, 102, 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[len(w) for w in t.split()] for t in texts]}


def test_encode_pair_budget_and_padding():
    enc = encode(FakeTok(), ["aa " * 600, "b"], max_len=512, pair="x yy", head=128)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    assert ids.shape == (2, 512)
    assert int(mask[0].sum()) == 512 and int(mask[1].sum()) == 1 + 1 + 1 + 2 + 1
    assert enc["truncated"] == [True, False]
