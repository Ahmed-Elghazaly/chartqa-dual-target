# Notes — Phase 2: the backbone, LoRA, and the smoke test

## What this component is, in plain language

Three things.

**A backend abstraction.** Two different libraries can load a vision-language model and attach
trainable adapters to it: plain Hugging Face (`transformers` + `peft` + `bitsandbytes`), and
Unsloth, which is faster and uses less memory. We wrote one interface with two implementations, so
the rest of the project never has to care which is in use.

**An assertion.** Before any training, we walk every parameter in the model, count how many are
trainable, and split that count into "belongs to the part that looks at the image" and "belongs to
the part that handles language". If either is zero, the run dies immediately.

**A smoke test.** Load the model, attach adapters, train for 100 steps on trivial made-up charts,
and record: how much GPU memory it peaked at, how long a step took, whether the loss went down and
stayed a real number, and whether killing it and restarting produces the same loss.

None of this learns anything useful. That is the point — it answers "can this run at all, here?"
before four weeks of work are staked on the answer.

## Why it exists — what breaks without it

**The backend abstraction exists because of a specific, verified gap.** Unsloth publishes vision
fine-tuning notebooks for Qwen3-VL 8B, Qwen2.5-VL 7B and Qwen3.5 2B/4B — and none for Qwen3-VL 2B,
which is the model this project is built on. Nobody has published that this model trains at this
size on this hardware. It probably does. But "probably" is not a foundation, and the cost of being
wrong in week four is the project.

**The assertion exists because of a bug that produces no symptoms.** Qwen3-VL's own fine-tuning
tooling has two open issues (#2016, #2079) where the flags that say "also train the vision part"
are silently ignored, because the model gets frozen before the adapters are attached. When that
happens, training runs. Loss falls. Checkpoints save. Evaluation produces believable numbers. The
only difference is that the half of the model that looks at pictures learned nothing — which would
silently delete the entire computer-vision contribution of a computer-vision project.

**The smoke test exists because free GPUs have hard limits.** A T4 has about 15 GB. The gates are
13.5 GB peak and 10 hours projected. Discovering you exceed either after starting the real run
wastes a session; discovering it after four sessions wastes the budget.

## What surprised me

**Three separate bugs, all caught before spending a minute of GPU time, all of which would have
passed silently.**

**1. I guessed a module name and was wrong.** To attach adapters you must name the model's internal
modules. I wrote what I expected the vision MLP to be called: `fc1` and `fc2`. It is actually
`linear_fc1` and `linear_fc2`.

The consequence is worse than it sounds. The vision *attention* modules (`qkv`, `attn.proj`) I had
right. So adapters would have attached to the vision tower's attention and **not** its MLP — and
the assertion would have passed, because it asks "are there trainable vision parameters?" and there
were. A partial, silent, plausible-looking failure.

The plan says, in as many words: *verify the names against the model you load, print them first,
don't assume.* I read that, and then assumed anyway. What caught it was building the real
architecture at toy size — 10 MB of random weights with genuine module names — and asserting that
**every declared target actually matches something**. A target that matches nothing is a silent
no-op.

**2. A name collision that would have adapted the wrong thing.** The vision tower has two modules
whose short name is `proj`: the attention output (a Linear), and the *patch embedding* (a Conv3d)
— the thing that cuts the image into squares in the first place. Adapter libraries match by name
suffix, so asking for `proj` would have quietly attached a trainable adapter to the patch
embedding, changing how the image is divided up at all. Specifying `attn.proj` excludes it.

**3. My smoke test was training on the wrong thing.** I wrote a docstring saying "only the answer
tokens carry loss" and then wrote code that masked out only padding and image tokens. Result: 23 of
45 positions supervised, when the answer is 2 tokens. The model was being trained to reproduce the
question as well as answer it.

For a smoke test that measures memory this barely matters. What matters is the shape of the error:
**it converges anyway.** The loss falls, the gates pass, the run looks perfect. I only found it
because I printed the supervised token count and it did not match the answer length.

The fix was less obvious than it looks. You cannot find the answer's start by counting text tokens,
because the image expands into a variable number of visual tokens first. You have to run the
processor on the prompt alone, with the same image, and use *that* length as the boundary.

## What I decided, and what I rejected

**Decided:** an unavailable backend raises an error naming the reason. **Rejected:** falling back to
the other backend automatically. A silent fallback would be friendlier and would destroy the
measurement — Phase 2's entire job is to record *which* backends work at this size, and a fallback
answers that question with "something worked", which is not an answer.

**Decided:** ship code to Kaggle as a private Kaggle *dataset*. **Rejected:** having the kernel
`git clone` the private GitHub repository. Cloning needs a GitHub token *inside* a remote service,
for a repo that also holds the project's history. The dataset route uses only the Kaggle credential
that already exists locally, and no token ever leaves this machine. A pleasant side effect: the
Phase 2 smoke test needs no secrets at all.

**Decided:** the resume check compares the *loss after resuming* against the same steps run without
interruption. **Rejected:** checking that the checkpoint files exist. Files existing proves a write
happened, not that the optimizer state, the learning-rate schedule and the RNG streams came back.
The plan puts it well: a resume that has never been tested does not work.

**Decided:** tolerance rather than equality on that comparison (`< 1e-2`). Four-bit matrix
multiplication and several CUDA kernels are not bit-deterministic, so demanding exactness would
fail for a reason that has nothing to do with whether resume works.

## Which concept a reader must understand first

**What LoRA actually is, and why "which modules" is the whole question.**

Fine-tuning normally means updating all of a model's weights. For 2 billion parameters that needs
far more memory than a free GPU has, mostly for the optimizer's bookkeeping rather than the weights
themselves.

LoRA freezes every original weight and inserts a pair of small matrices beside chosen ones. A
frozen 2048×2048 weight (4.2 million numbers) gets a companion pair of 2048×16 and 16×2048 (65
thousand numbers — about 1.5%). Only those train. Their product has the same shape as the original
weight, so at the end you can add it in, or keep it separate as a small "adapter" file.

Two consequences follow directly, and both drive everything in this phase:

1. **You choose which modules get adapters.** That choice is a list of *names*. Name something that
   does not exist and nothing happens — no error, no warning, just a module that never learns. This
   is why "I guessed `fc1`" is a real bug and not a typo.

2. **A vision-language model has two halves with different naming conventions.** Qwen's vision
   tower uses one fused `qkv` projection; its language model uses separate `q_proj`, `k_proj`,
   `v_proj`. Asking only for the language names attaches adapters only to the language model. The
   run looks completely normal.

So the question "did LoRA reach the vision tower?" is not paranoia. It is the single question that
determines whether a vision-language project is a vision-language project or an expensive language
project wearing a hat.

The second idea, which this phase demonstrates three times: **the errors that matter in this field
are the ones that converge.** A crash is cheap — you fix it in five minutes. A bug that lets loss
fall smoothly while optimising the wrong objective, or training half the model, costs weeks and is
only found by explicitly checking a thing you had no reason to doubt.

## Forward pointers

- The toy-scale-real-architecture trick (real config, shrunk layers, random weights) is broadly
  useful and reappears wherever a test needs genuine structure without genuine size.
- The 512-versus-native resolution measurement here feeds the pre-registration, and connects
  directly to the sub-token analysis from Phase 0: raising resolution is the only lever that moves
  targets out of the "smaller than one visual token" stratum.
- The prompt-masking problem returns in Phase 6 in a harder form, because the training target there
  is a long structured JSON record rather than a two-token number.
