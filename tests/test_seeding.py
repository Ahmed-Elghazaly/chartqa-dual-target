from __future__ import annotations

import random

from chartqa_dt.seeding import load_rng_state, rng_state, set_seed


def test_same_seed_gives_same_stream():
    set_seed(7)
    a = [random.random() for _ in range(5)]
    set_seed(7)
    assert a == [random.random() for _ in range(5)]


def test_different_seed_gives_different_stream():
    set_seed(7)
    a = [random.random() for _ in range(5)]
    set_seed(8)
    assert a != [random.random() for _ in range(5)]


def test_numpy_is_seeded():
    np = __import__("numpy")
    set_seed(11)
    a = np.random.rand(4)
    set_seed(11)
    assert (a == np.random.rand(4)).all()


def test_pythonhashseed_is_recorded():
    import os
    set_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_report_is_honest_about_what_was_seeded():
    r = set_seed(1)
    assert r.seed == 1 and r.python is True
    assert r.numpy is True                 # numpy is a core dependency
    text = r.describe()
    assert "seed=1" in text and "numpy=True" in text


def test_rng_state_roundtrip_restores_the_stream():
    set_seed(3)
    state = rng_state()
    first = [random.random() for _ in range(4)]
    load_rng_state(state)
    assert first == [random.random() for _ in range(4)]


def test_rng_state_captures_numpy():
    np = __import__("numpy")
    set_seed(5)
    state = rng_state()
    assert "numpy" in state
    a = np.random.rand(3)
    load_rng_state(state)
    assert (a == np.random.rand(3)).all()


def test_cublas_workspace_is_set_for_determinism():
    import os
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
