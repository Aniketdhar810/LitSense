"""Exceptions raised by the offline pipeline."""


class ConfigError(ValueError):
    """ml/config.yaml is missing a key or holds an invalid value."""


class StageError(RuntimeError):
    """A stage is unknown, misdeclared, missing an input, or did not write its outputs."""


class DataError(ValueError):
    """Raw data broke an expectation the pipeline relies on (schema, units, integrity)."""


class StreamError(RuntimeError):
    """A streamed source could not be read completely (HTTP error, truncation, retries spent)."""


class LeakageError(ValueError):
    """An evaluation input is newer than its cutoff, or comes from another cutoff's artifact set."""


class DegenerateScoresError(ValueError):
    """A scorer returned a row whose finite scores all tie (the zero-user-vector bug, s2.2 #7)."""
