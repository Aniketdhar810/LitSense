"""Exceptions raised by the offline pipeline."""


class ConfigError(ValueError):
    """ml/config.yaml is missing a key or holds an invalid value."""


class StageError(RuntimeError):
    """A stage is unknown, misdeclared, missing an input, or did not write its outputs."""
