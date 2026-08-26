# Notes — cheap gates and expensive runs

## What this component is, in plain language

A checklist, a budget tracker, and a set of assertions that run in the first twenty seconds of every
GPU job. Together they answer one question before a long job commits to anything expensive: *is this
configuration actually the one I think it is?*

## Why it exists — what breaks without it

Free GPU time is rationed. Kaggle gives roughly thirty hours a week. The training run this project
is building toward needs six to ten of them. So a long job that turns out to be misconfigured does
not merely waste an afternoon — it consumes a fifth of the week's budget, and the budget does not
refill on demand.

That would be reason enough. The stronger reason is what the failures actually looked like.

Nine GPU sessions were spent before one produced a usable measurement. Here is the whole ledger:

| session | wall time | what went wrong |
|---|---|---|
| 1 | 30 s | code never attached — a mixed-case username in a metadata field |
| 2 | 30 s | generated kernel had a `SyntaxError` |
| 3 | 30 s | expected an archive, found it already extracted |
| 4 | 20 min | slow for reasons that turned out to be a wrong numeric format |
| 5 | 4 min | GPU assigned was unusable by the host's own PyTorch |
| 6 | 20 s | same GPU again — the request for a different one was ignored |
| 7 | 3 min | **worked** |
| 8 | 25 min | 100 steps succeeded, then two of my own bugs |

Sessions 1–3 and 6 cost under a minute *each*, because something cheap refused to continue.
Sessions 4 and 8 cost forty-five minutes between them, because nothing did.

That is the entire argument. The difference between a thirty-second failure and a twenty-five-minute
one is not the severity of the bug. It is whether anything checked.

## What surprised me

**Session 8 is the one worth studying.** It ran 100 optimizer steps successfully — memory inside
budget, loss falling from 2.879 to 0.968, adapters attached to both halves of the model, no NaN —
and *then* failed, on the checkpoint-resume test, with `KeyError: 'exp_avg'`.

The cause: the training loop builds a memory-efficient 8-bit optimizer, and the resume path rebuilt
an ordinary one. They store their internal state under different names, so loading one into the
other fails outright.

Here is the part I keep returning to. **I had already written that gap down.** It was in the
checklist, in a table titled "known gaps, carried forward deliberately", with the reasoning:

> *Resume test uses plain AdamW, not AdamW8bit … the implementation difference does not affect what
> is being verified.*

That sentence is confident, plausible, and wrong. The difference did not weaken the check; it
prevented the check from running at all. And verifying it would have taken two lines and one second
locally.

So the lesson is not "write down your risks". I had. It is:

> **Writing a risk down is not assessing it.** A gap accepted with a reason deserves the same
> scrutiny as any other claim — cheaply, before you rely on it.

The second bug in the same session had the same shape from a different angle. The smoke test sized
its synthetic charts from the pixel *budget*: `image_px = sqrt(max_pixels)`. At the 512 setting that
gives a 512-pixel chart, which is perfectly reasonable, so it worked. At the "native" setting it
gives `sqrt(16,777,216) = 4096` — a 4096×2867 chart, whose eleven thousand visual tokens overflow
the sequence limit and make the processor reject the batch.

The budget's job is to control *downscaling*. It should never have controlled the *source* size,
because in real training the source images are whatever the dataset contains. The bug was invisible
for as long as the chosen budget happened to have a plausible square root.

> **A test harness that does not exercise production shapes is not a rehearsal.**

## What I decided, and what I rejected

**Decided:** gates ordered by cost, cheapest first. Check the accelerator before downloading four
gigabytes. Check that the code dataset attached before installing anything. Check that adapters
reached both halves of the model before training a single step. Each gate sits immediately before
the expensive thing it protects.

**Rejected:** running the full job and reading the logs afterwards. Kaggle exposes no logs while a
kernel runs, so "afterwards" can be forty minutes later, and by then the session is spent whether it
told you anything or not.

**Decided:** a budget tracker that reads the run log. **Then immediately had to fix it** — the first
version summed every duration-shaped number in the file, including the quota itself, and reported
47.9 hours used against a 30-hour limit. Which is its own small lesson:

> **A tracker you cannot trust is worse than no tracker.** It teaches you to ignore it, and then the
> one time it is right, you ignore it then too.

**Decided:** not to re-run a completed measurement just to backfill a new diagnostic. Gradient-norm
recording was added after session 8 started. Capturing it would have cost another session, to
improve a number that no Phase 2 gate depends on. It matters for the training phase, where it will
be present from the first step. Spending a rationed resource to make a finished result tidier is
exactly the consumption the checklist exists to prevent.

## Which concept a reader must understand first

**The cost of a bug is not its severity. It is the time between committing and finding out.**

A syntax error is trivial. Found locally, it costs seconds. Shipped to a remote queue, it costs a
round trip. The bug did not change; the feedback loop did.

This reframes what testing is for in a project with rationed compute. The purpose is not
correctness in the abstract — it is *moving discovery earlier*. Every check in this project is
positioned by asking: what is the most expensive thing that happens after this point, and can I fail
before it?

That ordering principle produces some unintuitive placements. Checking which GPU you were given
seems like a detail; it goes first, because everything after it costs four gigabytes of download.
Verifying that a configuration request was actually honoured seems paranoid; it is cheap, and twice
now the answer was no.

The related idea, which took nine sessions to internalise properly: **the failures that cost most
are the ones that let you get far enough to believe things are working.** Sessions 1–3 failed
immediately and cost nothing. Session 8 got 100 steps in, produced real numbers, and looked like
success right up until it did not.

## Forward pointers

- The gate ordering here becomes mandatory for the training phase, where a run is six to ten hours
  rather than twenty-five minutes and the same discipline is worth an order of magnitude more.
- Gradient-norm checking is the newest gate and the one aimed at the nastiest remaining failure:
  numeric underflow, in which loss stays flat, nothing raises, and every other signal is green.
- The "acceptance is not compliance" note from the previous chapter is the same idea applied to
  configuration rather than to time: verify the effect, not the request.
