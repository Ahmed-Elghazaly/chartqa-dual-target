"""The array-closing decoder — the fix for the 26% of generations that never finish.

Every case here is one way the scanner could be wrong about *where the array is*, which
is the only thing it has to be right about. Being wrong early truncates good evidence;
being wrong late leaves the run-on in place and the record still scores zero.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from chartqa_dt.eval.decoding import ArrayScanner, CloseEvidenceArray, closing_token_ids

PREFIX = '{"answerable":true,"evidence":['
ITEM = '{"label":"%s","value":%d,"unit":null,"bbox":[1,2,3,4]}'


def scan(text: str, max_items: int = 8) -> ArrayScanner:
    s = ArrayScanner(max_items=max_items)
    s.feed(text)
    return s


# --- counting -------------------------------------------------------------------

def test_no_items_before_the_array_opens():
    s = scan('{"answerable":true,')
    assert not s.started and s.items == 0 and not s.must_close


def test_the_array_is_recognised_when_it_opens():
    assert scan(PREFIX).started


def test_one_complete_item_counts_once():
    assert scan(PREFIX + ITEM % ("a", 1)).items == 1


def test_a_half_finished_item_does_not_count():
    assert scan(PREFIX + '{"label":"a","value":1').items == 0


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 12, 24])
def test_n_items_count_as_n(n):
    body = ",".join(ITEM % (f"l{i}", i) for i in range(n))
    assert scan(PREFIX + body, max_items=10 ** 9).items == n


def test_must_close_fires_exactly_at_the_cap():
    for n in range(0, 6):
        body = ",".join(ITEM % (f"l{i}", i) for i in range(n))
        assert scan(PREFIX + body, max_items=3).must_close is (n >= 3)


def test_must_close_stays_false_before_the_array_starts():
    assert not scan('{"answerable":true,', max_items=0).must_close


# --- string safety: the thing a brace-counter gets wrong -------------------------

@pytest.mark.parametrize("label", [
    "a {weird} label", "brackets [x]", "close }", "open {",
    "array ]", 'a "quoted" thing', "back\\\\slash", "comma, brace }",
])
def test_structure_inside_a_label_is_not_structure(label):
    esc = label.replace("\\", "\\\\").replace('"', '\\"')
    text = PREFIX + ITEM % (esc, 1)
    s = scan(text)
    assert s.items == 1, f"{label!r} miscounted"
    assert s.depth == 0 and not s.in_string


def test_an_escaped_quote_does_not_end_the_string():
    s = scan(PREFIX + '{"label":"say \\" now","value":1}')
    assert s.items == 1 and not s.in_string


def test_an_escaped_backslash_before_a_quote_does_end_the_string():
    # "a\\" is a complete string: the backslash is escaped, the quote is real.
    s = scan(PREFIX + '{"label":"a\\\\","value":1}')
    assert s.items == 1 and not s.in_string


def test_a_bracket_inside_a_label_does_not_close_the_array():
    s = scan(PREFIX + '{"label":"]","value":1}')
    assert not s.closed and s.items == 1


def test_the_key_appearing_inside_a_string_does_not_start_the_array():
    s = scan('{"question":"what is \\"evidence\\" here","answerable":true')
    assert not s.started


# --- nesting --------------------------------------------------------------------

def test_a_nested_object_inside_an_item_counts_as_one_item():
    s = scan(PREFIX + '{"label":"a","extra":{"deep":{"deeper":1}},"value":1}')
    assert s.items == 1


def test_a_nested_array_inside_an_item_does_not_close_the_outer_one():
    s = scan(PREFIX + '{"label":"a","bbox":[1,2,3,4],"value":1}')
    assert not s.closed and s.items == 1


# --- closing --------------------------------------------------------------------

def test_the_array_closes_on_a_top_level_bracket():
    assert scan(PREFIX + ITEM % ("a", 1) + "]").closed


def test_once_closed_later_text_is_ignored():
    s = scan(PREFIX + ITEM % ("a", 1) + '],"plan":{"op":"lookup","args":["a"]}')
    assert s.closed and s.items == 1


def test_must_close_is_false_once_the_array_is_closed():
    s = scan(PREFIX + ",".join(ITEM % (f"l{i}", i) for i in range(9)) + "]", max_items=2)
    assert s.closed and not s.must_close


def test_must_close_never_fires_inside_an_unfinished_item():
    """Closing at depth 1 would emit `...},{"label":"x"]` — invalid JSON.

    Reachable whenever the cap is reached and the model has already opened the next
    item in the same step, and whenever a scanner is constructed against a lower cap
    than the text it is fed.
    """
    body = ",".join(ITEM % (f"l{i}", i) for i in range(8))
    s = scan(PREFIX + body + ',{"label":"ninth","value":9', max_items=8)
    assert s.items == 8 and s.depth == 1
    assert not s.must_close, "would have closed the array inside an open object"


def test_must_close_returns_once_the_extra_item_finishes():
    body = ",".join(ITEM % (f"l{i}", i) for i in range(8))
    s = scan(PREFIX + body + "," + ITEM % ("ninth", 9), max_items=8)
    assert s.depth == 0 and s.must_close


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_must_close_is_false_at_every_open_depth(depth):
    body = ",".join(ITEM % (f"l{i}", i) for i in range(8))
    s = scan(PREFIX + body + "," + '{"a":' * depth, max_items=8)
    assert s.depth == depth and not s.must_close


def test_must_close_never_fires_mid_string():
    body = ",".join(ITEM % (f"l{i}", i) for i in range(8))
    s = scan(PREFIX + body + ',{"label":"half')
    assert s.in_string and not s.must_close


# --- whitespace and formatting variants -----------------------------------------

@pytest.mark.parametrize("sep", ["", " ", "\n", "  \n\t"])
def test_whitespace_between_the_key_and_the_bracket(sep):
    s = scan('{"evidence"' + sep + ":" + sep + "[" + ITEM % ("a", 1))
    assert s.started and s.items == 1


def test_a_pretty_printed_record_counts_the_same_as_a_compact_one():
    compact = PREFIX + ",".join(ITEM % (f"l{i}", i) for i in range(3))
    pretty = json.dumps(json.loads(compact + "]}"), indent=2)
    assert scan(pretty, max_items=10 ** 9).items == 3


# --- streaming equals bulk ------------------------------------------------------

@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 13, 64])
def test_feeding_in_chunks_is_the_same_as_feeding_at_once(chunk):
    text = PREFIX + ",".join(ITEM % (f"l{i}", i) for i in range(6))
    bulk = scan(text, max_items=4)
    piece = ArrayScanner(max_items=4)
    for i in range(0, len(text), chunk):
        piece.feed(text[i:i + chunk])
    assert (piece.items, piece.depth, piece.closed, piece.must_close) == \
           (bulk.items, bulk.depth, bulk.closed, bulk.must_close)


def test_a_key_split_across_two_feeds_is_still_found():
    s = ArrayScanner()
    s.feed('{"answerable":true,"evid')
    s.feed('ence":[' + ITEM % ("a", 1))
    assert s.started and s.items == 1


def test_the_sliding_window_does_not_lose_a_late_key():
    s = ArrayScanner()
    s.feed('{"answerable":true,' + "x" * 500)
    s.feed('"evidence":[' + ITEM % ("a", 1))
    assert s.started and s.items == 1


# --- against the real generations ------------------------------------------------

GENERATIONS = pathlib.Path(
    "outputs/kaggle/repo/outputs/phase5/chartqa_val_structured.jsonl")


@pytest.mark.skipif(not GENERATIONS.exists(), reason="phase 5 generations not present")
def test_on_real_generations_the_scanner_agrees_with_json_where_json_can_read_it():
    rows = [json.loads(line) for line in
            GENERATIONS.read_text(encoding="utf-8").splitlines()]
    checked = 0
    for g in rows:
        try:
            rec = json.loads(g["raw"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(rec, dict) or not isinstance(rec.get("evidence"), list):
            continue
        s = scan(g["raw"], max_items=10 ** 9)
        assert s.items == len(rec["evidence"]), g["record_id"]
        assert s.closed
        checked += 1
    assert checked > 200, f"only {checked} valid records — the test is not exercising much"


@pytest.mark.skipif(not GENERATIONS.exists(), reason="phase 5 generations not present")
def test_every_truncated_record_would_have_been_closed_in_time():
    """The counterfactual behind `DECISIONS.md` 0114, pinned as a test."""
    rows = [json.loads(line) for line in
            GENERATIONS.read_text(encoding="utf-8").splitlines()]
    trunc = [g for g in rows if g.get("hit_token_cap")]
    assert len(trunc) > 100, "expected the truncated population to be substantial"
    reached = 0
    for g in trunc:
        s = ArrayScanner(max_items=8)
        for ch in g["raw"]:
            s._step(ch)
            if s.must_close:
                reached += 1
                break
    assert reached / len(trunc) > 0.99, f"only {reached}/{len(trunc)} reached the cap"


# --- the processor ---------------------------------------------------------------

class FakeTokenizer:
    """Enough of a byte-level tokenizer to drive the processor without loading one."""

    def __init__(self) -> None:
        self.pieces = ["]", "],", '],"', "]}", "{", '"', "a", "b", ",", ":"]
        self._vocab = {p: i for i, p in enumerate(self.pieces)}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_string(self, toks):
        return "".join(toks)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)


def test_closing_token_ids_finds_every_bracket_token():
    ids = closing_token_ids(FakeTokenizer())
    assert sorted(ids) == sorted([0, 1, 2, 3])


def test_closing_token_ids_respects_its_limit():
    assert len(closing_token_ids(FakeTokenizer(), limit=2)) == 2


def test_the_processor_refuses_a_vocabulary_with_no_closing_token():
    class NoBrackets(FakeTokenizer):
        def get_vocab(self):
            return {"a": 0, "b": 1}
    with pytest.raises(ValueError, match="begins with"):
        CloseEvidenceArray(NoBrackets(), prompt_len=0, max_items=2)


def _scores(rows: int, vocab: int):
    import numpy as np
    return np.zeros((rows, vocab), dtype=float)


def test_the_processor_leaves_scores_alone_before_the_cap():
    tk = FakeTokenizer()
    p = CloseEvidenceArray(tk, prompt_len=0, max_items=8, key="a")
    s = _scores(1, len(tk.pieces))
    out = p(([[6]]), s)          # "a" — nothing has started
    assert (out == 0).all() and p.forced == 0


def test_the_processor_masks_everything_but_a_closer_at_the_cap():
    import numpy as np
    tk = FakeTokenizer()
    # Drive the scanner directly, then hand the processor an already-capped row.
    p = CloseEvidenceArray(tk, prompt_len=0, max_items=1)
    p._scanners[0] = scan(PREFIX + ITEM % ("a", 1), max_items=1)
    p._decoded[0] = 0
    s = _scores(1, len(tk.pieces))
    out = p([[]], s)
    closers = set(closing_token_ids(tk))
    assert p.forced == 1
    for tid in range(len(tk.pieces)):
        if tid in closers:
            assert out[0][tid] == 0.0
        else:
            assert out[0][tid] == -np.inf


def test_rows_are_independent():
    import numpy as np
    tk = FakeTokenizer()
    p = CloseEvidenceArray(tk, prompt_len=0, max_items=1)
    p._scanners[0] = scan(PREFIX + ITEM % ("a", 1), max_items=1)   # capped
    p._scanners[1] = scan(PREFIX, max_items=1)                     # not capped
    p._decoded[0] = p._decoded[1] = 0
    out = p([[], []], _scores(2, len(tk.pieces)))
    assert (out[1] == 0).all(), "an unrelated row was masked"
    assert np.isneginf(out[0][4]), "the capped row was not masked"


def test_the_processor_only_decodes_the_newly_generated_tokens():
    tk = FakeTokenizer()
    seen = []
    original = tk.decode
    tk.decode = lambda ids, **kw: (seen.append(list(ids)), original(ids, **kw))[1]
    p = CloseEvidenceArray(tk, prompt_len=2, max_items=8)
    p([[9, 9, 6]], _scores(1, len(tk.pieces)))
    p([[9, 9, 6, 7]], _scores(1, len(tk.pieces)))
    assert seen == [[6], [7]], f"re-decoded the whole prefix: {seen}"
