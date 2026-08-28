# Status

**Phase 5 in progress** (5.1 and 5.2 done, 5.3 running, 5.4 queued). **Phase 6 built and
tested ahead of its gate.** Phases 0–4 complete. Cost so far: **USD 0**.

## Where each phase stands

| phase | state | note |
|---|---|---|
| 0–3 | complete | data, mixtures, audit gate, mining |
| 4 | complete | metrics agree with both official evaluators on 11,690 real predictions |
| 5.1 prompt | complete | three prompts, sealed by hash |
| 5.2 variant | **complete** | **Instruct selected**, n=200 |
| 5.3 ChartQA zero-shot | running | full 1,920 validation split, ~6 h |
| 5.4 RefChartQA zero-shot | queued | 1,800 stratified rows |
| 5.5 pre-registration | drafted | generated from source; the seal guard rejects it while it says "TBD" |
| 6 | **built, untrained** | feed, collator, checkpointing, loop, validation, kill-and-resume verified |

**783 tests pass**; `ruff check src tests scripts` clean; preflight green.

## Phase 5.2 result — the first properly powered measurement

| | value |
|---|---:|
| relaxed accuracy | 50.0% |
| **round-trip agreement** | **69.0%** |
| plans that execute at all | 94.4% |
| schema-valid (after repair) | 46.5% |
| median latency | 11.4 s |

The n=24 probes had reported round-trip at 40–50%. At n=200 it is 69%, confirming
`DECISIONS.md` 0062: three prompt iterations were run on noise, and the probe could not
have detected any effect it was used to justify.

## What Phase 6's design pass found, before spending 10 GPU hours

Each of these would have produced a plausible-looking failure rather than an error.

1. **Training examples did not fit `max_seq_len`** (0064). The zero-shot prompt is 980
   tokens; with visual tokens and a target the example is 1,363–1,498 against a limit of
   1,024. Every example would have been silently truncated. Fixed with a 117-token training
   prompt — 389 tokens of headroom, no extra compute. Raising the limit was measured and
   rejected: ≥14.9 h against a 10 h gate.
2. **Targets did not reproduce their own answers** (0067). Four separate join defects; at
   worst **1 of 636** ChartQA records produced an executable target, and **100%** of
   RefChartQA targets failed the round-trip. Now 69% of planned ChartQA records, and every
   emitted target round-trips by construction.
3. **No end-of-turn token in the target.** A model trained that way is never taught to
   stop, and every generation runs to the token cap.
4. **Early stopping on AP is unsound** (0069). At an affordable slice the AP interval is
   ±8.7 points, which cannot detect "has not improved". Stopping moved to validation loss —
   free, low variance, and directly sensitive to the boxes because the target contains them.

## Open items

- **Deferred by Ahmed until the core result is in**: three training seeds (~30 h) and the
  RefChartQA scaling ladder (~30 h). Both measure or document a result rather than improve
  it.
- **The plan-rich mixture arm** is built and waiting: 3,331 compositional plans against the
  pre-registered arm's 1,665. Phase 6 trains both and reports the better (0066).
- **32.83 stays a Level C anchor** — cited, not reproducible by anyone (0052). The ChartQA
  reproduction of 79.1 is reachable but only at Phase 7, on the test split (0063).

## Next

5.3 and 5.4 finish → finalise and commit `PREREGISTRATION.md` → the seal opens → Phase 6
trains both stage-2 arms plus the direct-answer control.
