# Notes — hardware contracts, and guards that don't guard

## What this component is, in plain language

Two small functions that look at which GPU is actually present and adjust two settings accordingly:
the numeric format the model computes in, and which attention implementation to use. If the
requested setting is not natively supported, they substitute a working one and **write down that
they did**.

## Why it exists — what breaks without it

Because "supported" and "fast" are different words, and the libraries only tell you about the first.

Modern GPUs support a numeric format called **bfloat16** — sixteen bits, arranged to trade precision
for a wide exponent range, which makes training stable. Every recent tutorial uses it. Our configs
requested it.

The free GPU this project runs on is a **Tesla T4**, and the T4 predates bfloat16 support in
hardware. It is *Turing*, "compute capability 7.5"; native bf16 arrives with *Ampere*, 8.0.

Here is the part that matters: **PyTorch does not refuse.** It emulates bf16 on a T4. Your code
runs. Your numbers are correct. It is just much slower.

Now consider where that lands. The entire purpose of the Phase 2 benchmark is to decide whether this
model fits in a ten-hour budget on free hardware. An emulated numeric format inflates
seconds-per-step. The honest-looking conclusion would have been *"this backbone is too slow for the
free tier"* — and the documented response to that conclusion is to drop down the fallback ladder to a
different model.

We would have changed the backbone of the project because of one wrong word in a config file, and
the evidence would have looked completely convincing.

## What surprised me

**The fix for this bug was itself broken by the same category of bug, and I only caught it because I
had started reading library source before trusting it.**

The natural way to write the check is to ask PyTorch:

```python
if want is torch.bfloat16 and not torch.cuda.is_bf16_supported():
    use float16 instead
```

That reads perfectly. It is inert. Here is the actual function in torch 2.13:

```python
def is_bf16_supported(including_emulation: bool = True):
    if torch.cuda.get_device_properties(device).major >= 8:
        return True
    if not including_emulation:
        return False
    return _check_bf16_tensor_supported(device)   # <- the emulation probe
```

The default is `including_emulation=True`. On a T4 it falls through to the emulation probe, finds
that emulated bf16 works, and returns **True**. So the guard never fires — on precisely the hardware
it was written for.

Two functions share one name here. `is_bf16_supported()` answers *"can this run?"*.
`is_bf16_supported(including_emulation=False)` answers *"can this run fast?"*. Only the second is
the question being asked, and the default gives you the first.

The correct check turned out to be the one PyTorch itself performs before it reaches the emulation
probe: is compute capability at least 8?

## What I decided, and what I rejected

**Decided:** test compute capability directly. **Rejected:** calling the library's own helper.
Usually deferring to the library is right — it knows more than you do about its own hardware
support. Here the helper's *default* encodes a different intent than mine, and one keyword away is
an answer that is silently the opposite.

**Decided:** keep `bfloat16` in the config files and substitute at load time.
**Rejected:** changing the configs to `float16`. The config should record what you *want*; on a
rented Ampere box bfloat16 is right and should be taken automatically. A config edited to work
around today's hardware is a config that silently underperforms on tomorrow's.

**Decided:** every substitution returns a note that is printed and stored in the run record.
**Rejected:** substituting quietly. A silent downgrade is indistinguishable from the config being
honoured, and the whole failure being prevented here is one of indistinguishability.

## Which concept a reader must understand first

**A guard that always passes is not a guard, and it looks exactly like one that works.**

Both versions of this check were three lines, read sensibly, passed review, and imported cleanly.
One of them did something. Testing "does the code run?" cannot separate them, because both run. The
only thing that separates them is asking: *under what conditions does this branch actually fire, and
have I ever seen it fire?*

That is why the test suite here contains a fake GPU that deliberately reproduces the trap — it
asserts that `is_bf16_supported()` returns `True` while
`is_bf16_supported(including_emulation=False)` returns `False`, and then asserts our function
downgrades anyway. If someone later "simplifies" it back to the library helper, that test fails
immediately. The test does not just check the behaviour; it *documents the wrong way and forbids it*.

The second, more general idea: **a function's default arguments are part of its contract.** Reading
`is_bf16_supported()` at a call site tells you almost nothing. You have to know what it defaults to,
because the default is making a decision on your behalf — and here that decision was the entire
question.

## Forward pointers

- float16 has a much narrower exponent range than bfloat16, so loss scaling matters more and NaN
  risk is higher. The smoke test already checks for non-finite loss and Phase 6's fallback ladder
  starts with a learning-rate reduction, so the guards exist — but if Stage 2 destabilises on a T4,
  this is the first note to re-read.
- The same "reading tells you what it says, running tells you what it does" discipline found the
  quantisation skip patterns that matched nothing, and the evaluator behaviour that reverses between
  one image and twenty.
