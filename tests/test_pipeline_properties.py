"""End-to-end properties nothing else asserts — `DECISIONS.md` 0136.

Two of them, both stated requirements that had no test:

* **Reproducibility.** `Prompt.md` lists it among the things every change must not damage,
  and `PREREGISTRATION.md` depends on it, but nothing checked that building records twice
  gives the same records. Unit tests seed *functions*; this seeds the *pipeline*.
* **The coordinate contract.** A box crosses four representations between the annotation
  and the score — pixels, 0–1000 normalised, the clamp for the official evaluator, and the
  emitted target. Each step is tested; the composition was not.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

CACHE = pathlib.Path.home() / ".cache/chartqa_dt/data/refchartqa_train.jsonl"


def _signature(records) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r.record_id}|{r.plan}|{r.evidence}|{len(r.elements or [])}".encode())
    return h.hexdigest()


def _archive_present() -> bool:
    import sys

    sys.path.insert(0, ".")
    from scripts.build_mixtures import archive_path

    return pathlib.Path(archive_path()).exists()


needs_archive = pytest.mark.skipif(not _archive_present(), reason="ChartQA archive absent")
needs_cache = pytest.mark.skipif(not CACHE.exists(), reason="RefChartQA cache absent")


# --- reproducibility --------------------------------------------------------------------

@needs_archive
def test_building_chartqa_records_twice_gives_the_same_records():
    import sys

    sys.path.insert(0, ".")
    from scripts.build_mixtures import archive_path, chartqa_records

    from chartqa_dt.data.chartqa import ArchiveReader

    reader = ArchiveReader(archive_path())
    first = _signature(chartqa_records(reader, limit=300, seed=0))
    second = _signature(chartqa_records(reader, limit=300, seed=0))
    assert first == second, "the same seed produced different records"


@needs_archive
def test_a_different_seed_gives_different_records():
    """Otherwise the seed is decorative and every 'seeded sample' is the same sample."""
    import sys

    sys.path.insert(0, ".")
    from scripts.build_mixtures import archive_path, chartqa_records

    from chartqa_dt.data.chartqa import ArchiveReader

    reader = ArchiveReader(archive_path())
    assert (_signature(chartqa_records(reader, limit=300, seed=0))
            != _signature(chartqa_records(reader, limit=300, seed=1)))


@needs_cache
def test_reading_the_refchartqa_cache_twice_gives_the_same_records():
    """This path has no seed at all, so any difference would be genuine nondeterminism —
    dict ordering, set iteration, or a mutable default carried between calls."""
    import sys

    sys.path.insert(0, ".")
    from scripts.build_mixtures import refchartqa_records

    assert (_signature(refchartqa_records(cap=600, cache=CACHE))
            == _signature(refchartqa_records(cap=600, cache=CACHE)))


# --- the coordinate contract, composed ---------------------------------------------------

@pytest.mark.parametrize("box,size", [
    ({"x": 0, "y": 0, "w": 10, "h": 10}, (100, 100)),
    ({"x": 50, "y": 25, "w": 25, "h": 50}, (200, 100)),
    ({"x": 1, "y": 1, "w": 798, "h": 555}, (800, 557)),
    ({"x": 399, "y": 0, "w": 1, "h": 479}, (400, 479)),
])
def test_a_box_survives_the_whole_journey_in_order(box, size):
    """pixels → 0–1000 → clamp for the official evaluator, never inverted or out of range."""
    from chartqa_dt.data.refchartqa import xywh_to_norm1000
    from chartqa_dt.eval.official_format import clamp_for_official_evaluator

    w, h = size
    x1, y1, x2, y2 = xywh_to_norm1000(box, w, h)
    assert 0.0 <= x1 <= x2 <= 1000.0
    assert 0.0 <= y1 <= y2 <= 1000.0
    cx1, cy1, cx2, cy2 = clamp_for_official_evaluator((x1, y1, x2, y2))
    assert cx1 <= cx2 and cy1 <= cy2
    assert all(0 <= v <= 999 for v in (cx1, cy1, cx2, cy2)), (
        "the official evaluator silently discards a coordinate at exactly 1000")


def test_the_clamp_never_widens_a_box():
    from chartqa_dt.eval.official_format import clamp_for_official_evaluator

    x1, y1, x2, y2 = clamp_for_official_evaluator((10.4, 20.6, 900.5, 950.5))
    assert x1 >= 10 and y1 >= 20 and x2 <= 901 and y2 <= 951


def test_a_box_at_the_far_edge_does_not_clamp_to_zero_area():
    """A mark flush against the right edge is real; clamping it away would delete it."""
    from chartqa_dt.data.refchartqa import xywh_to_norm1000
    from chartqa_dt.eval.official_format import clamp_for_official_evaluator

    norm = xywh_to_norm1000({"x": 390, "y": 0, "w": 10, "h": 100}, 400, 100)
    x1, y1, x2, y2 = clamp_for_official_evaluator(tuple(norm))
    assert x2 > x1 and y2 > y1


@needs_cache
def test_every_emitted_target_box_is_inside_the_official_range():
    """The end of the journey: whatever a target emits must be scoreable."""
    import json
    import sys

    sys.path.insert(0, ".")
    from scripts.build_mixtures import refchartqa_records

    from chartqa_dt.train.targets import TargetError, build_target

    checked = 0
    for record in refchartqa_records(cap=1200, cache=CACHE):
        try:
            target = json.loads(build_target(record))
        except TargetError:
            continue
        for item in target["evidence"]:
            x1, y1, x2, y2 = item["bbox"]
            assert 0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999, (
                f"{record.record_id} emitted {item['bbox']}, which the official "
                f"evaluator cannot score")
            checked += 1
    assert checked > 200, f"only {checked} boxes checked"


@needs_cache
def test_no_plan_anywhere_folds_over_fewer_than_two_elements():
    """A degenerate fold verifies by construction — `max([x])` is `x` — so it looks like a
    success everywhere except in the target itself. 97.6% of the first PoT conversion's
    folds were this (0133); the mined plans have never had one (0137). Both held here.
    """
    import sys

    sys.path.insert(0, ".")
    from scripts.build_mixtures import refchartqa_records

    from chartqa_dt.plans.executor import FOLD_OPS

    offenders = []
    for record in refchartqa_records(cap=8000, cache=CACHE):
        plan = record.plan or {}
        if plan.get("op") not in FOLD_OPS or plan.get("args"):
            continue
        n = len([e for e in (record.elements or []) if e.get("label") is not None])
        if n < 2:
            offenders.append((record.record_id,
                              (record.meta or {}).get("plan_provenance", "mined"),
                              plan["op"], n))
    assert not offenders, (
        f"{len(offenders)} plans fold over fewer than two elements, e.g. {offenders[:3]}")
