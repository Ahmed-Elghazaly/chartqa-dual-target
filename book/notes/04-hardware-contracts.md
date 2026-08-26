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

---

## Addendum — requests that are accepted but not honoured

Everything above is about *guards*: checks that answer a question adjacent to the one you meant.
There is a second, related failure that showed up immediately afterwards, and it has a different
remedy.

The free GPU turned out to matter more than expected. Kaggle offers two cards, and it handed us a
**Tesla P100**. That card is old enough (compute capability 6.0) that Kaggle's *own* PyTorch build
cannot use it at all:

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.
The current PyTorch install supports sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
```

The fail-fast check written a few hours earlier passed anyway, because it asked
`torch.cuda.is_available()` — and a GPU *was* available. It simply could not run anything. That is
the guard failure again, third instance.

So we started requesting a specific card. Kaggle's API has a field for it. We set it to
`"gpu_t4x2"`, which is what appears in the web interface's own URLs and reads entirely plausibly.

**The next run was assigned a P100 again.**

Not refused. *Ignored.* Here is the setter, in Kaggle's SDK:

```python
def machine_shape(self, machine_shape):
    if not isinstance(machine_shape, str):
        raise TypeError('machine_shape must be of type str')
    self._machine_shape = machine_shape
```

It validates that your request is a string. Nothing more. The three values it actually accepts are
documented four hundred lines away in a docstring — `NvidiaTeslaT4`, `NvidiaTeslaP100`,
`Tpu1VmV38` — and anything else is transmitted and discarded in silence.

From the client's side, a typo and a granted request look identical.

### Why this is a different bug from the others

A guard that asks the wrong question is fixed by testing the condition that actually causes the
failure. But nothing was being *tested* here. We made a request, it was accepted, and we assumed
acceptance meant compliance.

The remedy is correspondingly different, and it is two things rather than one:

1. **Verify the value against the receiver's own contract.** The three legal strings are now
   constants, and a test asserts each one literally appears in the installed SDK's source. If Kaggle
   renames them, the test fails instead of the next run landing on the wrong card.
2. **Verify the request took effect.** The kernel prints the device it actually received and refuses
   to continue if its architecture is absent from `torch.cuda.get_arch_list()`. Asking for a T4 and
   *checking you got one* are separate acts.

The second point is the general one. **Acceptance is not compliance.** Any time you configure
something across a boundary — an API field, an environment variable, a config file consumed by
another process, a flag passed to a library — the only evidence that it worked is an observation of
the effect. `llm_int8_skip_modules` was accepted and matched nothing. `machine_shape` was accepted
and changed nothing. Both were "configured correctly" in the sense that no error was raised.

The cost of checking is a printed line and an assertion. The cost of not checking, twice now, was a
session each.
