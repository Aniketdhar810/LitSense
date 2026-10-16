"""LitSense serving package: recommendations, explanations and chat over a prebuilt bundle.

Stateless: the backend builds a ``UserState`` for every call and persists whatever comes back.
It never imports torch or the ``ml`` package.

Real so far (Week 5): ``load_store``, ``download_bundle``, ``health``, ``recommend``, ``catalog``.
Every other function of CLAUDE.md s15.1 is still served by ``litsense_core.mock`` until its week
(see docs/API_CONTRACT.md).
"""

__api_version__ = "0.1.0"

from litsense_core.recommend import catalog, download_bundle, health, recommend  # noqa: E402
from litsense_core.store import load_store  # noqa: E402

__all__ = ["__api_version__", "catalog", "download_bundle", "health", "load_store", "recommend"]
