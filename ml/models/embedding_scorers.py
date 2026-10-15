"""Scorers over stored user and item embeddings (LightGCN now, the projection hybrid next).

Numpy only, so evaluation never imports torch. A stored scorer is not refit: ``fit`` checks that
the artifact belongs to the evaluation's fit data (shapes, and which items have fit interactions),
and the cutoff comes from the artifact itself, so the evaluator's cutoff check catches an artifact
from another artifact set (s2.2 #5, s3.2).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.eval.data import FitData
from ml.eval.errors import LeakageError


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length; all-zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


@dataclass
class EmbeddingArtifact:
    """What a CF training stage stores: embeddings, item degrees and its metadata."""

    user: np.ndarray  # (n_users, dim) float32
    item: np.ndarray  # (n_items, dim) float32
    degree: np.ndarray  # (n_items,) fit interactions per item
    meta: dict[str, Any]  # cutoff, model, config, training history

    @classmethod
    def load(cls, folder: Path, user: str, item: str) -> "EmbeddingArtifact":
        """Read ``<folder>/{user,item}`` plus ``item_degree.npy`` and ``meta.json``."""
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(
            np.load(folder / user).astype(np.float32),
            np.load(folder / item).astype(np.float32),
            np.load(folder / "item_degree.npy"),
            meta,
        )


class EmbeddingScorer:
    """score = user . item, or user . item / |item| with ``normalize_items`` (the L2-normalized
    LightGCN of s9.4). Items with no fit interactions score -inf (s8.1)."""

    def __init__(self, name: str, artifact: EmbeddingArtifact, normalize_items: bool) -> None:
        self.name = name
        self.artifact = artifact
        self.user = artifact.user
        self.item = l2_normalize_rows(artifact.item) if normalize_items else artifact.item
        self.cold = artifact.degree == 0
        self.cutoff: int | None = int(artifact.meta["cutoff"])

    def fit(self, data: FitData) -> None:
        """Check the stored model was trained on exactly this fit data; nothing is learned."""
        if self.user.shape[0] != data.n_users or self.item.shape[0] != data.n_items:
            raise LeakageError(f"{self.name}: artifact shapes do not match the id map")
        if self.cutoff != data.cutoff:
            raise LeakageError(f"{self.name}: artifact cutoff {self.cutoff} != {data.cutoff}")
        counts = np.bincount(data.interactions["i"], minlength=data.n_items)
        if not np.array_equal(counts, self.artifact.degree):
            raise LeakageError(f"{self.name}: artifact item degrees differ from the fit data")

    def score(self, users: np.ndarray) -> np.ndarray:
        scores = (self.user[users] @ self.item.T).astype(np.float32)
        scores[:, self.cold] = -np.inf
        return scores
