"""The Phase 6 training loop — `PLAN.md` 6.1, 6.2, 6.5, 6.6.

Run against a tiny stand-in model, because what needs guarding is the *control flow*:
which learning rate a stage uses, whether checkpoints land on the right steps, whether
early stopping fires when it should, and whether a dead gradient is visible. None of that
needs a vision-language model, and all of it is expensive to discover on a GPU.
"""

from __future__ import annotations

import pytest

from chartqa_dt.train.checkpoint import TrainState
from chartqa_dt.train.loop import STAGE2_FALLBACK_LR, STAGE_LR, TrainConfig, train

torch = pytest.importorskip("torch")


class Stub(torch.nn.Module):
    """A model whose loss falls predictably, with LoRA-shaped parameter names."""

    def __init__(self):
        super().__init__()
        self.visual_qkv_lora_A = torch.nn.Parameter(torch.randn(2, 2))
        self.language_q_proj_lora_B = torch.nn.Parameter(torch.randn(2, 2))
        self.calls = 0

    def forward(self, **batch):
        self.calls += 1
        loss = (self.visual_qkv_lora_A.sum() + self.language_q_proj_lora_B.sum()) * 0.0 \
            + torch.tensor(1.0 / self.calls, requires_grad=True) \
            + self.visual_qkv_lora_A.sum() * 1e-6
        return type("Out", (), {"loss": loss})()

    def save_pretrained(self, path):
        torch.save(self.state_dict(), f"{path}/adapter.pt")

    def train(self, mode=True):
        return self


class StubProcessorHolder:
    def __init__(self):
        self.model = Stub()
        self.processor = object()


def fake_batch(*args, **kwargs):
    return {"input_ids": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.ones(1, 4, dtype=torch.long),
            "_supervised_positions": 4}


class FakeFeed:
    def __init__(self):
        self.epoch = 0

    def batches(self, batch_size):
        while True:
            yield [object()] * batch_size

    def state_dict(self):
        return {"position": 7, "epoch": self.epoch, "n": 100}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr("chartqa_dt.train.loop.build_batch", fake_batch)
    monkeypatch.setattr("chartqa_dt.modeling.lora_assert.assert_lora_on_both_sides",
                        lambda *a, **k: None)
    monkeypatch.setattr("chartqa_dt.train.smoke.build_optimizer",
                        lambda m, lr: torch.optim.SGD(
                            [p for p in m.parameters() if p.requires_grad], lr=lr))
    return None


def test_the_stage_learning_rates_are_the_ones_the_plan_specifies():
    """`PLAN.md` 6.1 and 6.2, and the documented fallback."""
    assert STAGE_LR == {"stage1": 1e-4, "stage2": 5e-5}
    assert STAGE2_FALLBACK_LR == 2e-5
    assert TrainConfig(stage="stage1").learning_rate == 1e-4
    assert TrainConfig(stage="stage2").learning_rate == 5e-5
    assert TrainConfig(stage="stage2", lr=1e-6).learning_rate == 1e-6


def test_a_short_run_logs_every_step(patched, tmp_path):
    cfg = TrainConfig(steps=5, grad_accum=2, save_every=0, eval_every=0, out_dir=tmp_path)
    result = train(StubProcessorHolder(), FakeFeed(), cfg)
    assert len(result.logs) == 5
    assert result.state.step == 5
    for log in result.logs:
        assert log.seconds >= 0 and log.supervised == 8, "grad_accum batches per step"
    assert "loss" in result.summary()


def test_checkpoints_land_on_the_configured_interval(patched, tmp_path):
    cfg = TrainConfig(steps=6, save_every=2, eval_every=0, out_dir=tmp_path,
                      stage="stage1")
    train(StubProcessorHolder(), FakeFeed(), cfg)
    saved = sorted(p.name for p in tmp_path.iterdir())
    assert "stage1-step2" in saved and "stage1-step4" in saved
    assert "stage1-final" in saved
    import json
    state = json.loads((tmp_path / "stage1-step2" / "train_state.json").read_text())
    assert state["feed"]["position"] == 7, "the dataloader position is checkpointed"


def test_early_stopping_ends_the_run(patched, tmp_path):
    """`PLAN.md` 6.6, with the patience fixed in advance."""
    scores = iter([0.5, 0.4, 0.3, 0.2, 0.1])
    cfg = TrainConfig(steps=100, save_every=0, eval_every=1, patience=2,
                      out_dir=tmp_path)
    result = train(StubProcessorHolder(), FakeFeed(), cfg,
                   evaluate=lambda model, step: next(scores))
    assert result.stopped_early
    assert result.state.step == 3, "best at step 1, then two evals without improvement"
    assert result.state.best_step == 1


def test_a_run_that_keeps_improving_is_not_stopped(patched, tmp_path):
    scores = iter([0.1, 0.2, 0.3, 0.4, 0.5])
    cfg = TrainConfig(steps=5, save_every=0, eval_every=1, patience=2, out_dir=tmp_path)
    result = train(StubProcessorHolder(), FakeFeed(), cfg,
                   evaluate=lambda model, step: next(scores))
    assert not result.stopped_early and result.state.step == 5


def test_dead_gradients_are_reported_rather_than_hidden(patched, tmp_path):
    """float16 without a scaler underflows to exactly zero; the loss curve looks fine
    (`DECISIONS.md` 0017)."""
    cfg = TrainConfig(steps=3, save_every=0, eval_every=0, out_dir=tmp_path)
    result = train(StubProcessorHolder(), FakeFeed(), cfg)
    for log in result.logs:
        log.grad_norm = 0.0
    assert "float16 underflow" in result.summary()


def test_lora_is_asserted_on_both_sides_before_any_compute(monkeypatch, tmp_path):
    """A Phase 6 acceptance criterion, and it must run before compute is spent.

    A model with LoRA on only the language side trains happily, shows a healthy loss
    curve, and answers a different research question than the one being asked. The
    assertion is the guard; what this test pins is that it fires *first*.
    """
    monkeypatch.setattr("chartqa_dt.train.loop.build_batch", fake_batch)
    monkeypatch.setattr("chartqa_dt.train.smoke.build_optimizer",
                        lambda m, lr: torch.optim.SGD(list(m.parameters()), lr=lr))

    def refuse(model):
        raise AssertionError("LoRA is missing on the vision side")

    monkeypatch.setattr("chartqa_dt.modeling.lora_assert.assert_lora_on_both_sides",
                        refuse)
    holder = StubProcessorHolder()
    cfg = TrainConfig(steps=1, save_every=0, eval_every=0, out_dir=tmp_path)
    with pytest.raises(AssertionError, match="vision side"):
        train(holder, FakeFeed(), cfg)
    assert holder.model.calls == 0, "it must fire before the first forward pass"


def test_resuming_continues_from_the_supplied_state(patched, tmp_path):
    cfg = TrainConfig(steps=4, save_every=0, eval_every=0, out_dir=tmp_path)
    state = TrainState(step=2, stage="stage1")
    result = train(StubProcessorHolder(), FakeFeed(), cfg, state=state)
    assert len(result.logs) == 2, "only the remaining steps are run"
    assert result.state.step == 4
