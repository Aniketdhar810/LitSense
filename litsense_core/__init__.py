"""LitSense serving package: recommendations, explanations and chat over a prebuilt bundle.

Stateless: the backend builds a ``UserState`` for every call and persists whatever comes back.
It never imports torch or the ``ml`` package. Until the artifact bundle exists, the public API
of CLAUDE.md s15.1 is served by ``litsense_core.mock`` (see docs/API_CONTRACT.md).
"""

__api_version__ = "0.1.0"
