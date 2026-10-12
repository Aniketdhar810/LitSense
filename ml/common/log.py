"""Logging setup shared by stages and scripts."""

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``litsense`` logger, attaching one stderr handler on first use."""
    root = logging.getLogger("litsense")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return root.getChild(name)
