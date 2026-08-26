# Notes — Phase 1: the package, the config system, the environment

## What this component is, in plain language

Before any model is trained, the project becomes an ordinary installable Python package with
command-line programs. You type `pip install -e .` and you get six commands — `cdt-data`,
`cdt-gen`, `cdt-mine`, `cdt-train`, `cdt-eval`, `cdt-report` — that work identically on a laptop,
a Kaggle kernel and a Colab VM.

Around that sit four small pieces of machinery:

- **`env.py`** decides where files go. It detects which machine it is on and hands back three
  directories: one for big downloads, one for model caches, one for run outputs. Nothing else in
  the codebase is allowed to know a path.
- **`config.py`** turns a YAML file plus command-line overrides into a typed object, and writes
  the fully resolved settings plus the git commit hash into the output directory at the start of
  every run.
- **`logging_utils.py`** writes every metric to a plain text file first and to Weights & Biases
  second.
- **`hub.py`** pushes results to a private Hugging Face repository.

## Why it exists — what breaks without it

The obvious way to do a deep-learning project is a notebook. It is also the way that guarantees
you cannot finish this one.

**Notebooks do not port.** The paths differ between Kaggle and Colab. **Notebooks run out of
order** — cell 7, then cell 3, then cell 7 again with a variable that only exists because of a
cell you have since edited — and the result is a number you cannot reproduce and cannot defend.
**Notebooks are not testable.** There is no way to assert that a function still behaves the way
it did last week.

The specific failure this protects against: you train for eight hours, get a good number, and
cannot say what produced it. That is not a bad result. It is *no* result.

Each piece has one failure it is designed to make impossible:

| Piece | The failure it prevents |
|---|---|
| Installable package + CLI | "It worked in my notebook" — code that cannot be run the same way twice |
| `env.py` | A hard-coded `/kaggle/working` that crashes on Colab, three weeks in |
| Typed config with **strict** key checking | A silently ignored `--train.leraning_rate` |
| `resolved_config.yaml` + git SHA + dirty flag | A result that cannot be traced to settings *or* to code |
| Always-on JSONL metric mirror | A session killed with the only record in a browser tab |
| Hub push on every save | A checkpoint that existed only on a VM that no longer exists |
| The rule-7 upload guard | A GPL-3.0 chart image pushed to a public repository |

## What surprised me

**A test caught a design flaw in the thing that was supposed to prevent flaws.**

`env.py` originally decided the Kaggle data directory by *probing the filesystem*: use
`/kaggle/temp` if that directory exists, otherwise `/kaggle/working`. On a real Kaggle kernel that
gives the right answer. On my laptop, where neither exists, it silently returned the wrong one —
and the test asserting the Kaggle policy failed.

I could have made the test pass by relaxing it. That would have been the wrong move: the test was
right and the code was wrong. The real problem was that a function stating *policy* ("bulky data
goes to /kaggle/temp because /kaggle/working is size-capped and counts as kernel output") was
also doing *discovery* ("does this directory exist?"). Mixing them makes the policy unassertable
anywhere except on the platform it describes — which is exactly where you cannot run tests.

The fix was to split them: `_root_candidates()` is pure and returns an ordered preference list;
`_first_writable()` walks that list and picks the first one that works. Now the policy is
testable from anywhere, and the resilience is a separate, also-testable concern.

That is a general lesson worth keeping: **when a test is awkward to write, the usual cause is
that the code has two jobs.**

## What I decided, and what I rejected

**Unknown config keys are a fatal error.** Rejected: the conventional "warn and continue".
Warnings scroll past. A typo'd hyperparameter that is silently dropped produces a run you believe
used your setting and did not — and *nothing downstream can detect it*, because the run completes
normally and reports a plausible number. Given the choice between a loud crash now and an
undetectable wrong result in three weeks, the crash is enormously cheaper.

**The GPU dependency versions are deliberately left unpinned.** `requirements.lock.txt` has an
empty GPU block with a comment explaining why. The plan says pin exact versions, and it is right —
but pinning `transformers==4.5x.y` from a laptop with no CUDA would be writing down a guess in the
format of a fact. The one thing that *is* recorded is the floor that was actually verified:
`transformers >= 4.57.0`, because Qwen3-VL's own config declares `4.57.0.dev0` and earlier
releases cannot load `model_type: "qwen3_vl"` at all. The rest gets filled in from the Phase 2
smoke test, measured on the machine that will run it.

**Vendored third-party code is excluded from the linter.** The official RefChartQA evaluator has
66 lint violations. Fixing them is exactly the wrong thing to do: its value is that it is
byte-identical to the one that produced the published number we are comparing against. A reformat
is a diff, and a diff you cannot rule out is a diff you have to defend.

## Which concept a reader must understand first

**Reproducibility is not a virtue you add at the end; it is a property of how the code is
structured from the first line.**

A beginner reads "record your hyperparameters" as advice about diligence. It is not. It is a
statement about *architecture*. You cannot reliably record settings that live scattered across
notebook cells, because there is no single moment at which the settings exist as one object. You
can trivially record them when a run begins by constructing one typed config object and dumping
it. The discipline follows from the structure, not from remembering.

The same holds for the git SHA and the dirty-tree flag. Recording "commit `a1b2c3d`" is worthless
if the working tree had uncommitted edits, because the code that ran is not the code at that
commit. So we record both, and we record the dirty flag *honestly* rather than refusing to run —
because a tool that refuses to run on a dirty tree just teaches people to commit noise.

## Forward pointers

- The strict-config idea recurs in the output schema: the model's JSON is validated against a
  schema with `additionalProperties: false`, for exactly the same reason.
- `env.py`'s split between policy and discovery is the same shape as the split between the
  official evaluator (policy: this is what the number means) and our own metrics (discovery:
  where is the error coming from).
- The rule-7 upload guard is the first appearance of a pattern used throughout: **encode the rule
  as an assertion in the code path, not as a sentence in a document.** The same pattern produces
  `assert_lora_on_both_sides` in Phase 2 and `test_no_test_split_leakage` in Phase 3.
