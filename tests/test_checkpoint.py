"""Checkpointing and early stopping — `PLAN.md` 6.3 and 6.6.

`PLAN.md` is blunt about why: *"a resume that has never been tested does not work."*
Phase 2 proved it twice — an optimizer rebuilt by the wrong factory raised
`KeyError: 'exp_avg'` after 100 successful steps, and a missing RNG state produced a
resume that did not crash and was simply different (`DECISIONS.md` 0026).
"""

from __future__ import annotations

import json

import pytest

from chartqa_dt.train.checkpoint import (
    RESUME_TOLERANCE,
    SAVE_EVERY_STEPS,
    EarlyStopping,
    TrainState,
    assert_resume_matched,
    load_checkpoint,
    resume_delta,
    save_checkpoint,
)

torch = pytest.importorskip("torch")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.saved_to = None

    def forward(self, x):
        return self.linear(x)

    def save_pretrained(self, path):
        self.saved_to = path
        torch.save(self.state_dict(), f"{path}/adapter.pt")


def factory(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)


def test_a_checkpoint_carries_everything_a_resume_needs(tmp_path):
    """`PLAN.md` 6.3 lists five things; all five must be on disk."""
    model = TinyModel()
    opt = factory(model)
    state = TrainState(step=100, epoch=1, stage="stage1",
                       feed={"position": 37, "epoch": 1, "n": 12000})
    save_checkpoint(tmp_path, model=model, optimizer=opt, state=state)

    for name in ("adapter.pt", "optimizer.pt", "rng_state.pt", "train_state.json"):
        assert (tmp_path / name).exists(), f"{name} missing from the checkpoint"
    on_disk = json.loads((tmp_path / "train_state.json").read_text())
    assert on_disk["feed"]["position"] == 37, "the dataloader position is required"


def test_a_partial_checkpoint_is_refused(tmp_path):
    """Resuming from half a checkpoint silently changes the run."""
    model = TinyModel()
    save_checkpoint(tmp_path, model=model, optimizer=factory(model), state=TrainState())
    (tmp_path / "rng_state.pt").unlink()
    with pytest.raises(FileNotFoundError, match="not a resumable checkpoint"):
        load_checkpoint(tmp_path, model=model, optimizer_factory=factory)


def test_the_training_state_round_trips(tmp_path):
    model = TinyModel()
    opt = factory(model)
    state = TrainState(step=250, epoch=2, stage="stage2",
                       feed={"position": 9, "epoch": 2, "n": 5},
                       best_metric=0.41, best_step=200, evals_without_improvement=1,
                       losses=[1.0, 0.5], grad_norms=[3.0])
    save_checkpoint(tmp_path, model=model, optimizer=opt, state=state)
    _, restored = load_checkpoint(tmp_path, model=model, optimizer_factory=factory)
    assert restored.step == 250 and restored.feed["position"] == 9
    assert restored.best_metric == pytest.approx(0.41) and restored.best_step == 200
    assert restored.evals_without_improvement == 1


def test_rng_state_is_restored_so_a_resume_is_equivalent(tmp_path):
    """Without this, Phase 2 measured a resume delta of 0.0438 (`DECISIONS.md` 0026)."""
    import random

    model = TinyModel()
    save_checkpoint(tmp_path, model=model, optimizer=factory(model), state=TrainState())
    expected = [random.random() for _ in range(3)]

    random.seed(999)                                   # disturb every generator
    torch.manual_seed(999)
    load_checkpoint(tmp_path, model=model, optimizer_factory=factory)
    assert [random.random() for _ in range(3)] == expected


def test_the_caller_supplies_the_optimizer_factory_so_it_cannot_diverge(tmp_path):
    """The Phase 2 failure: AdamW8bit's state keys are not torch.optim.AdamW's, so a
    checkpoint written by one cannot be stepped by the other — `KeyError: 'exp_avg'`,
    raised *after* 100 successful steps.

    `torch`'s `load_state_dict` is permissive and does not catch it at load time; the
    mismatch surfaces later, at `.step()`. The structural defence is therefore that
    `load_checkpoint` takes a **factory** rather than an optimizer, so exactly one place
    in the codebase decides which optimizer a run uses, on the first step and on resume.
    """
    model = TinyModel()
    original = factory(model)
    # Give the optimizer real state to preserve.
    model(torch.randn(2, 4)).sum().backward()
    original.step()
    save_checkpoint(tmp_path, model=model, optimizer=original, state=TrainState())

    restored, _ = load_checkpoint(tmp_path, model=model, optimizer_factory=factory)
    assert type(restored) is type(original)
    before = original.state_dict()["state"]
    after = restored.state_dict()["state"]
    assert set(before) == set(after) and before, "optimizer moments must survive"
    for key in before:
        assert torch.allclose(before[key]["exp_avg"], after[key]["exp_avg"])


# ------------------------------------------------------------------ resume equivalence


def test_a_matching_resume_passes_and_a_diverging_one_fails():
    assert resume_delta(1.2345, 1.2350) == pytest.approx(0.0005, abs=1e-9)
    assert assert_resume_matched(1.2345, 1.2350) < RESUME_TOLERANCE
    with pytest.raises(AssertionError, match="RNG state"):
        assert_resume_matched(1.2345, 1.2783)          # the 0.0438 Phase 2 measured


def test_the_tolerance_sits_between_the_two_measured_regimes():
    """0.0014–0.0053 with the RNG state, 0.0438 without. The threshold must separate."""
    assert 0.0053 < RESUME_TOLERANCE < 0.0438


# ---------------------------------------------------------------------- early stopping


def test_early_stopping_fires_after_the_pre_registered_patience():
    stopper = EarlyStopping(patience=2)
    state = TrainState()
    assert stopper.update(state, 0.30, step=100) is False
    assert stopper.update(state, 0.40, step=200) is False          # improved
    assert stopper.update(state, 0.39, step=300) is False          # 1 without
    assert stopper.update(state, 0.38, step=400) is True           # 2 without -> stop
    assert state.best_metric == pytest.approx(0.40)
    assert state.best_step == 200, "the reported checkpoint is the last that IMPROVED"


def test_an_improvement_resets_the_counter():
    stopper = EarlyStopping(patience=2)
    state = TrainState()
    stopper.update(state, 0.30, 100)
    stopper.update(state, 0.29, 200)
    assert state.evals_without_improvement == 1
    stopper.update(state, 0.35, 300)
    assert state.evals_without_improvement == 0 and state.best_step == 300


def test_a_negligible_gain_does_not_count_as_improvement():
    stopper = EarlyStopping(patience=1, min_delta=0.01)
    state = TrainState()
    stopper.update(state, 0.400, 100)
    assert stopper.update(state, 0.405, 200) is True, "under min_delta is not improvement"


def test_the_save_interval_is_the_one_the_plan_specifies():
    assert SAVE_EVERY_STEPS == 100
