"""Kill-and-resume, verified rather than assumed — `PLAN.md` 6.3.

    **Test the resume path by deliberately killing a run** — a resume that has never been
    tested does not work.

Phase 2 proved the point twice: an optimizer rebuilt by the wrong factory raised
`KeyError: 'exp_avg'` after 100 successful steps, and a missing RNG state produced a
resume that did not crash and was simply *different* (`DECISIONS.md` 0026).

This runs the real loop, the real checkpoint writer and the real loader against a tiny
model, so the plumbing is exercised without a GPU. The GPU version is the same code with a
real backbone.
"""

from __future__ import annotations

import pytest

from chartqa_dt.train.checkpoint import (
    RESUME_TOLERANCE,
    TrainState,
    assert_resume_matched,
    load_checkpoint,
)
from chartqa_dt.train.loop import TrainConfig, train

torch = pytest.importorskip("torch")


class NoisyStub(torch.nn.Module):
    """Its loss depends on the RNG, so a resume with the wrong RNG state diverges.

    That is the whole point: a model whose loss ignored randomness would pass a resume
    test that a real model fails.
    """

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2, 2))

    def forward(self, **batch):
        noise = torch.rand(())                      # consumes the global RNG
        return type("Out", (), {"loss": self.w.sum() * 1e-3 + noise})()

    def save_pretrained(self, path):
        torch.save(self.state_dict(), f"{path}/adapter.pt")

    def train(self, mode=True):
        return self


class Holder:
    def __init__(self):
        self.model = NoisyStub()
        self.processor = object()


class CountingFeed:
    def __init__(self):
        self.epoch = 0
        self.position = 0

    def batches(self, batch_size):
        while True:
            self.position += 1
            yield [object()] * batch_size

    def state_dict(self):
        return {"position": self.position, "epoch": self.epoch, "n": 100}


def _patch(monkeypatch):
    monkeypatch.setattr("chartqa_dt.train.loop.build_batch",
                        lambda *a, **k: {"input_ids": torch.ones(1, 2, dtype=torch.long),
                                         "_supervised_positions": 2})
    monkeypatch.setattr("chartqa_dt.modeling.lora_assert.assert_lora_on_both_sides",
                        lambda *a, **k: None)
    monkeypatch.setattr("chartqa_dt.train.smoke.build_optimizer",
                        lambda m, lr: torch.optim.SGD(list(m.parameters()), lr=lr))


def _factory(model):
    return torch.optim.SGD(list(model.parameters()), lr=1e-4)


def test_a_killed_run_resumes_to_the_same_loss(monkeypatch, tmp_path):
    """The measurement `PLAN.md` 6.3 asks for, end to end through the real code."""
    _patch(monkeypatch)

    # 1. An uninterrupted run of 6 steps.
    torch.manual_seed(0)
    cfg = TrainConfig(steps=6, grad_accum=1, save_every=3, eval_every=0,
                      out_dir=tmp_path / "whole")
    whole = train(Holder(), CountingFeed(), cfg)
    uninterrupted = whole.logs[3].loss                 # the 4th step

    # 2. The same run "killed" after 3 steps, then resumed from the checkpoint.
    torch.manual_seed(0)
    killed_cfg = TrainConfig(steps=3, grad_accum=1, save_every=3, eval_every=0,
                             out_dir=tmp_path / "killed")
    train(Holder(), CountingFeed(), killed_cfg)

    holder = Holder()
    optimizer, state = load_checkpoint(tmp_path / "killed" / "stage1-step3",
                                       model=holder.model, optimizer_factory=_factory)
    feed = CountingFeed()
    feed.position = state.feed["position"]
    resume_cfg = TrainConfig(steps=4, grad_accum=1, save_every=0, eval_every=0,
                             out_dir=tmp_path / "resumed")
    resumed = train(holder, feed, resume_cfg, state=state, optimizer=optimizer)

    delta = assert_resume_matched(uninterrupted, resumed.logs[0].loss)
    assert delta < RESUME_TOLERANCE


def test_a_resume_without_the_rng_state_diverges(monkeypatch, tmp_path):
    """The negative control. Without this the test above could pass vacuously.

    Phase 2 measured 0.0438 with the RNG state missing; the tolerance is 0.02.
    """
    _patch(monkeypatch)
    torch.manual_seed(0)
    cfg = TrainConfig(steps=6, grad_accum=1, save_every=3, eval_every=0,
                      out_dir=tmp_path / "whole")
    whole = train(Holder(), CountingFeed(), cfg)
    uninterrupted = whole.logs[3].loss

    torch.manual_seed(0)
    train(Holder(), CountingFeed(),
          TrainConfig(steps=3, grad_accum=1, save_every=3, eval_every=0,
                      out_dir=tmp_path / "killed"))

    holder = Holder()
    optimizer, state = load_checkpoint(tmp_path / "killed" / "stage1-step3",
                                       model=holder.model, optimizer_factory=_factory)
    torch.manual_seed(12345)                          # simulate a lost RNG state
    resumed = train(holder, CountingFeed(),
                    TrainConfig(steps=4, grad_accum=1, save_every=0, eval_every=0,
                                out_dir=tmp_path / "resumed"),
                    state=state, optimizer=optimizer)
    with pytest.raises(AssertionError, match="RNG state"):
        assert_resume_matched(uninterrupted, resumed.logs[0].loss)


def test_the_feed_position_survives_the_round_trip(monkeypatch, tmp_path):
    """A resume that restarts the epoch never reaches the last examples."""
    _patch(monkeypatch)
    feed = CountingFeed()
    train(Holder(), feed,
          TrainConfig(steps=5, grad_accum=1, save_every=5, eval_every=0,
                      out_dir=tmp_path))
    _, state = load_checkpoint(tmp_path / "stage1-step5", model=Holder().model,
                               optimizer_factory=_factory)
    assert state.feed["position"] == 5
    assert state.step == 5
    assert isinstance(state, TrainState)
