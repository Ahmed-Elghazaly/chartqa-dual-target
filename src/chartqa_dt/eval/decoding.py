"""Making the model finish its record — a decode-time fix for a measured catastrophe.

**The measurement.** Over the 1,920 structured zero-shot generations in Phase 5:

| | |
|---|---:|
| hit the 900-token cap | **26.0%** |
| of those, failed to parse | **100%** |
| share of *all* parse failures caused by truncation | **80.1%** |
| truncated records that reached `"model_answer"` | **0 of 500** |

And the failure has exactly one shape. A truncated record emits a median of **24** evidence
items where a complete one emits 2; **72.4%** of them contain a byte-identical duplicate
item, against 2.6% of complete records; and **99.0%** of the characters in a truncated
record sit inside the evidence array. The model falls into a repetition loop enumerating
elements and never comes out.

`model_answer` is the last field of the schema, behind that array. So a run-on does not
cost us the grounding — it costs us **the whole record**, and every one of those 500 scores
zero.

**Why the prompt cannot fix it.** It already tried. `prompts.py` says *"NEVER more than
{max_evidence} items"*, *"Each label appears at most ONCE"*, and *"Do NOT keep listing"* —
that hardening and the 512→900 raise landed together on 2026-08-27, and this run is from
2026-08-29. The instructions are in the prompt being measured. A model in a repetition loop
is not reading them.

**The fix.** Not a bigger budget — a bigger budget buys longer garbage, which is what
512→900 already bought. Instead, close the array from the outside: once `MAX_EVIDENCE`
complete items have been emitted, mask every continuation except one starting with `]`.
The model is then past the array and completes `"plan"` and `"model_answer"` normally.

This adds nothing to the record and invents no field — it only *stops* an enumeration the
prompt already forbids, at exactly the bound the schema already declares. That keeps it on
the right side of non-negotiable rule 3 (`parsing.py`): we may drop, we may unwrap, we
never add.

**It applies to both arms or to neither.** The baseline is 48.70%, and ~26% of it is this
artifact. Fixing decoding only for the fine-tuned model would credit fine-tuning with
repairing a truncation bug — see `DECISIONS.md` 0114.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArrayScanner:
    """Tracks one JSON array of objects as characters arrive, and says when to close it.

    Deliberately *not* a JSON parser. It answers one question — how many complete objects
    has the evidence array produced so far — over a prefix that is not yet valid JSON, and
    it has to be right about strings: a label may legitimately contain a brace or a
    bracket, and counting those as structure would close the array early.
    """

    key: str = '"evidence"'
    max_items: int = 8

    started: bool = False
    closed: bool = False
    depth: int = 0
    items: int = 0
    #: Inside a string literal, where structural characters mean nothing.
    in_string: bool = False
    escaped: bool = False
    _buf: str = ""
    _seen: str = field(default="", repr=False)

    def feed(self, text: str) -> None:
        """Consume newly generated characters."""
        for ch in text:
            self._step(ch)

    def _step(self, ch: str) -> None:
        if self.closed:
            return
        if not self.started:
            self._buf += ch
            # Look for the key followed by its opening bracket, tolerating whitespace.
            i = self._buf.find(self.key)
            if i >= 0:
                rest = self._buf[i + len(self.key):]
                j = rest.find("[")
                if j >= 0 and rest[:j].strip() in {"", ":"}:
                    self.started = True
                    self._buf = ""
            elif len(self._buf) > 4 * len(self.key):
                self._buf = self._buf[-len(self.key):]  # keep a sliding window
            return

        if self.in_string:
            if self.escaped:
                self.escaped = False
            elif ch == "\\":
                self.escaped = True
            elif ch == '"':
                self.in_string = False
            return

        if ch == '"':
            self.in_string = True
        elif ch == "{":
            self.depth += 1
        elif ch == "}":
            self.depth -= 1
            if self.depth == 0:
                self.items += 1
        elif ch == "]" and self.depth == 0:
            self.closed = True

    @property
    def must_close(self) -> bool:
        """True when the next token has to be the one that ends the array."""
        return (self.started and not self.closed and not self.in_string
                and self.depth == 0 and self.items >= self.max_items)


def closing_token_ids(tokenizer, limit: int = 64) -> list[int]:
    """Every token whose text begins the array's closing bracket.

    A byte-level BPE has many: `]`, `],`, `],"`, `]}`. Any of them ends the array, and
    leaving the choice to the model keeps the continuation in-distribution instead of
    forcing a bare `]` it would never have picked.
    """
    out: list[int] = []
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        piece = tokenizer.convert_tokens_to_string([tok])
        if piece.startswith("]"):
            out.append(tid)
            if len(out) >= limit:
                break
    return sorted(out)


class CloseEvidenceArray:
    """A `LogitsProcessor` that ends the evidence array at `max_items`.

    One scanner per row of the batch, fed only the characters generated since the last
    step. Rows are independent: a run-on in one must not close another's array.

    Typing is deliberately loose — `transformers.LogitsProcessor` is an ABC with a single
    `__call__`, and duck-typing it keeps this module importable, and testable, without
    torch. `scores` only has to support boolean-mask assignment, which both torch and
    numpy do.
    """

    def __init__(self, tokenizer, prompt_len: int, *, max_items: int,
                 key: str = '"evidence"') -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.max_items = max_items
        self.key = key
        self.closers = closing_token_ids(tokenizer)
        if not self.closers:
            raise ValueError("no token in the vocabulary begins with ']'")
        self._scanners: dict[int, ArrayScanner] = {}
        self._decoded: dict[int, int] = {}
        #: How often the array was forced shut, so the rate is reported rather than hidden.
        self.forced = 0

    def _scanner(self, row: int) -> ArrayScanner:
        if row not in self._scanners:
            self._scanners[row] = ArrayScanner(key=self.key, max_items=self.max_items)
        return self._scanners[row]

    def __call__(self, input_ids, scores):
        for row in range(len(input_ids)):
            generated = list(input_ids[row])[self.prompt_len:]
            scanner = self._scanner(row)
            seen = self._decoded.get(row, 0)
            if len(generated) > seen:
                text = self.tokenizer.decode(generated[seen:],
                                             skip_special_tokens=True)
                scanner.feed(text)
                self._decoded[row] = len(generated)
            if scanner.must_close:
                self.forced += 1
                mask = [True] * len(scores[row])
                for tid in self.closers:
                    mask[tid] = False
                scores[row][mask] = float("-inf")
        return scores


__all__ = ["ArrayScanner", "CloseEvidenceArray", "closing_token_ids"]
