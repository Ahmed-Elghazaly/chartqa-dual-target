"""Network helpers, and a fix for a TLS trap that costs an hour every time.

A python.org Python on macOS ships **no CA trust store**. Raw ``urllib`` then
raises::

    ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
    unable to get local issuer certificate

which surfaces as a connection error and reads exactly like a rejected
credential. During setup that sent the Kaggle diagnosis in the wrong direction
for some time, and it recurred later in a data-loading script.

``requests`` bundles its own CA store via ``certifi`` and is unaffected — which is
why it silently works where ``urllib`` does not, and why the failure looks
intermittent and environment-specific rather than systematic.

Importing this module repairs the default SSL context for the whole process, so
``urllib``, ``http.client`` and anything built on them work too. It is a no-op
where the system store is already usable.
"""

from __future__ import annotations

import os
import ssl
from typing import Any


def ensure_ca_bundle() -> str | None:
    """Point Python's default TLS verification at a working CA bundle.

    Returns the bundle path if one had to be installed, else None.
    Safe to call repeatedly.
    """
    try:
        ctx = ssl.create_default_context()
        # A store with zero loaded certificates cannot verify anything.
        if ctx.cert_store_stats().get("x509_ca", 0) > 0:
            return None
    except Exception:  # noqa: BLE001
        pass

    try:
        import certifi
    except ImportError:
        return None

    bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)

    original = ssl.create_default_context

    def _patched(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        kwargs.setdefault("cafile", bundle)
        return original(*args, **kwargs)

    ssl.create_default_context = _patched  # type: ignore[assignment]
    ssl._create_default_https_context = _patched  # type: ignore[attr-defined]
    return bundle


def get_json(url: str, *, timeout: int = 90, headers: dict[str, str] | None = None) -> Any:
    """GET a URL and parse JSON, using requests so TLS is never the problem."""
    import requests

    resp = requests.get(url, timeout=timeout, headers=headers or {})
    resp.raise_for_status()
    return resp.json()


def get_bytes(url: str, *, timeout: int = 90, headers: dict[str, str] | None = None) -> bytes:
    import requests

    resp = requests.get(url, timeout=timeout, headers=headers or {})
    resp.raise_for_status()
    return resp.content


# Repair TLS on import: any module that imports this gets a working default.
_INSTALLED = ensure_ca_bundle()
