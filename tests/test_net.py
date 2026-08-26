"""TLS trust store repair.

A python.org Python on macOS ships no CA store, so raw `urllib` raises
CERTIFICATE_VERIFY_FAILED. That surfaces as a connection error and reads exactly
like a rejected credential -- it sent the Kaggle credential diagnosis in the wrong
direction, and then recurred in a data-loading script hours later.

`requests` bundles certifi and is unaffected, which is precisely what makes the
failure look intermittent and environment-specific rather than systematic.
"""

from __future__ import annotations

import ssl

import pytest

from chartqa_dt.net import ensure_ca_bundle, get_bytes, get_json


def test_default_context_can_verify_after_import():
    """Importing chartqa_dt.net must leave the process able to verify TLS."""
    ctx = ssl.create_default_context()
    assert ctx.cert_store_stats().get("x509_ca", 0) > 0, (
        "the default SSL context has no CA certificates loaded; "
        "chartqa_dt.net should have installed certifi's bundle"
    )


def test_ensure_is_idempotent():
    first = ensure_ca_bundle()
    second = ensure_ca_bundle()
    # Once repaired, a second call finds a working store and does nothing.
    assert second is None or second == first


def test_env_vars_point_at_a_real_bundle():
    import os
    from pathlib import Path

    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        value = os.environ.get(var)
        if value:
            assert Path(value).is_file(), f"{var} points at {value!r}, which does not exist"


@pytest.mark.network
def test_urllib_works_after_repair():
    """The concrete symptom: urllib over HTTPS, which failed before the fix."""
    import json
    import urllib.request

    url = ("https://datasets-server.huggingface.co/rows?dataset=omoured/RefChartQA"
           "&config=default&split=validation&offset=0&length=1")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read())
    assert data["rows"][0]["row"]["id"].startswith("RefChartQA_")


@pytest.mark.network
def test_get_json_helper():
    data = get_json(
        "https://datasets-server.huggingface.co/size?dataset=omoured/RefChartQA"
    )
    splits = {s["split"]: s["num_rows"] for s in data["size"]["splits"]}
    assert splits["test"] == 11690, "RefChartQA test split size changed"


@pytest.mark.network
def test_get_bytes_helper():
    blob = get_bytes("https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/config.json")
    assert b"qwen3_vl" in blob
