# Notes — one card, one budget

## What this component is, in plain language

Two lines of configuration and one assertion. The model is loaded onto **one** GPU explicitly rather
than letting the library decide, memory is measured across **every** GPU rather than the first, and a
run that ends up spread across two cards is treated as a failed measurement rather than a slow one.

## Why it exists — what breaks without it

Because the free GPU tier gave us two cards, and the loading library used both without being asked.

Hugging Face's `device_map="auto"` decides where each part of a model lives. On one GPU it puts
everything there. On two, it *splits the model between them* — some layers on the first card, some on
the second. That is a genuinely useful feature for models too large to fit on one card. Ours is not.
`auto` does not check whether splitting is necessary; it splits because it can.

Three things went wrong at once, and it took a while to see they were one thing:

**The crash.** Our training loop sends each batch to the device holding the first parameter. When a
later layer lives on the *other* card, the multiplication fails:

```
RuntimeError: Expected all tensors to be on the same device,
but got mat2 is on cuda:1, different from other tensors on cuda:0
```

At one image size the split happened to fall where this survived. At another it did not.

**The slowdown.** Every forward pass now copies activations between two cards. Measured cost: seconds
per step went from **8.664 to 13.128**, a 52% penalty, silently.

**The one that actually mattered.** Our memory measurement called
`torch.cuda.max_memory_reserved()`, which reports the *current* device — device zero. With the model
spread over two cards, we were measuring roughly half of it. The number was then checked against a
13.5 GB ceiling as though it meant something.

## What surprised me

**Two plausible measurements that straddled the gate.**

The project's ceiling is ten hours. Before pinning, the same configuration measured **7.22 hours** in
one session and **10.94 hours** in another. One passes. One fails. Both came from a run that
completed normally and printed a clean table.

If I had taken either at face value, I would have made a real decision — proceed, or drop down the
fallback ladder and change the backbone — on the strength of a coin toss. And the coin toss was
invisible: nothing in either run's output said *"by the way, this model was on two cards"*.

**The second surprise was in my own notes.** The single-device assumption was written down as a known
gap and cleared — twice. The reasoning:

> A 2B model in 4-bit uses 1.48 GB of a 15 GB card, so the model is not sharded and cannot become so
> at this size.

Read it again. It answers *"does this model need two cards?"* — no, obviously not. But the question
that mattered is *"will this library use two cards if they are there?"*, and the answer is yes,
unconditionally. I checked a true statement that was not the relevant one, and felt reassured.

## What I decided, and what I rejected

**Decided:** pin to one card. **Rejected:** making the training loop device-aware so sharding works.

The second option is more capable and is the wrong choice here. The project's entire compute budget —
the reference measurement it is calibrated against, the memory ceiling, the ten-hour projection — is
for a *single* card. A two-card run is a different experiment. Reporting its numbers against a
one-card budget would be an unmatched comparison, which is precisely what this project forbids itself
elsewhere when comparing to published results. The same standard has to apply internally.

**Decided:** measure memory across every visible device anyway, even though we now pin to one.
Because the pinning is a thing that could be changed or bypassed later, and the measurement should
not quietly become wrong when it is.

**Decided:** make sharding a *failure*, not a warning. A warning in a log that scrolls past during a
forty-minute remote run is not a control.

## Which concept a reader must understand first

**"Automatic" means the library makes a choice. It does not mean the choice is yours.**

`device_map="auto"` is convenience, and convenience is a decision someone else made about the common
case. It is exactly right for the situation it was designed for — a model too large for one card. It
is silently wrong for ours, and it never announces which situation it thinks it is in.

The general shape, which recurs constantly in this ecosystem: a default is tuned for the *typical*
user, and you are frequently not typical. `device_map`, `is_bf16_supported(including_emulation=True)`,
`truncation` behaviour in tokenisers, `max_pixels` in image processors — every one of these has a
sensible default that quietly does something specific. None of them are wrong. They just aren't
answering your question unless you check.

The second idea, and the harder one: **a measurement that completes is not a measurement you can
use.** Both runs finished. Both printed tidy numbers. Neither reported the one fact that made the
numbers incomparable. Before trusting a number, it is worth asking what would have to be true for it
to be meaningless — and then checking that specific thing, rather than checking that the run
succeeded.

## Forward pointers

- The same "measure the effect, not the request" discipline appears three other times in this project:
  quantisation skip patterns that matched nothing, an accelerator request that was ignored, and a code
  upload superseded by an older version. This is the fourth, and the first where the wrong answer was
  a *number* rather than a yes/no.
- The gradient-norm gate added just before this run paid for itself here in an unexpected way: it
  confirmed that gradients were healthy (`median 14.28`, zero dead steps, adapters in fp32), which
  ruled out a whole family of explanations for the slowdown and helped isolate the real one.
