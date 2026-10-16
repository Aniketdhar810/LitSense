"""The serving bundle and the process-wide ``ArtifactStore`` (CLAUDE.md s4.6, s12.1).

The bundle format is defined here, next to its reader; ``ml.serve.export`` imports these names to
write it, so writer and reader cannot drift. Large matrices are memory-mapped and cast to float32
only per slice; small arrays are loaded once. The store is read-only after load (thread-safe).
"""

import gzip
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from litsense_core.vector_index import VectorIndex

MANIFEST = "manifest.json"
CATALOG = "catalog.npz"
CATALOG_TEXT = "catalog_text.json.gz"
CORPUS_USERS = "corpus_users.npz"
ITEM_NN = "item_nn.npy"
TRENDING_TOP = "trending_top.npy"
CATALOG_ARRAYS = (  # catalog.npz keys, one value per item
    "year", "pages", "series_pos", "series_idx", "primary_genre", "genre_mask", "author_idx",
    "bayes_rating", "rating_count", "rating_std", "trending", "is_longtail", "n_reviews",
)  # fmt: skip
TEXT_FIELDS = (  # catalog_text.json.gz per-item lists
    "canonical_id", "title", "author", "genres", "primary_genre", "description", "image",
    "isbn13", "series_name",
)  # fmt: skip

_STORE: "ArtifactStore | None" = None


class BundleError(RuntimeError):
    """The bundle is missing files or its contents disagree with its manifest."""


@dataclass
class ArtifactStore:
    """Everything ``recommend`` reads, loaded once per process."""

    bundle_dir: Path
    manifest: dict[str, Any]
    catalog: dict[str, np.ndarray]
    text: dict[str, list[Any]]
    lgcn_item: np.ndarray  # (n_items, d) float16, memory-mapped
    lgcn_item_norm: np.ndarray
    content: np.ndarray  # (n_items, 384) float32, L2-normalized
    item_nn: np.ndarray  # (n_items, 50) int32
    trending_top: np.ndarray  # (k,) int32, best first
    corpus_ids: list[str]
    corpus_vectors: np.ndarray  # (n_corpus, d) float32
    corpus_history: dict[str, np.ndarray]  # offsets, items, stars, ts (CSR over corpus users)
    index: VectorIndex
    by_author: dict[int, np.ndarray] = field(default_factory=dict)  # author_idx -> items
    by_series: dict[int, np.ndarray] = field(default_factory=dict)  # series_idx -> items by pos
    ranker: Any = None  # LightGBM booster (Week 9); None means the blend ranker

    @property
    def version(self) -> str:
        return str(self.manifest["bundle_version"])

    @property
    def cutoff(self) -> int:
        return int(self.manifest["cutoff"])

    @property
    def n_items(self) -> int:
        return int(self.manifest["n_items"])

    @property
    def config(self) -> dict[str, Any]:
        return self.manifest["config"]

    @property
    def genre_labels(self) -> list[str]:
        return list(self.manifest["genre_labels"])

    def corpus_user(self, user_id: str) -> int | None:
        """Row of a demo user in ``corpus_vectors``, or None."""
        return self._corpus_rows.get(user_id)

    def __post_init__(self) -> None:
        self._corpus_rows = {u: r for r, u in enumerate(self.corpus_ids)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_store(bundle_dir: str | Path, verify_hashes: bool | None = None) -> ArtifactStore:
    """Load the bundle at ``bundle_dir`` (once per process; later calls with the same directory
    return the same store). Fails loudly when files are missing or shapes disagree with the
    manifest (s12.1)."""
    global _STORE
    bundle = Path(bundle_dir)
    if _STORE is not None and _STORE.bundle_dir == bundle:
        return _STORE
    path = bundle / MANIFEST
    if not path.exists():
        raise BundleError(f"no {MANIFEST} in {bundle}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verify = (
        manifest["config"]["serving"]["verify_hashes_on_load"]
        if verify_hashes is None
        else verify_hashes
    )
    files = manifest["files"]
    for name, entry in files.items():
        if not (bundle / name).exists():
            raise BundleError(f"bundle file missing: {name}")
        if verify and sha256_file(bundle / name) != entry["sha256"]:
            raise BundleError(f"sha256 mismatch: {name}")
    n = int(manifest["n_items"])

    def matrix(name: str) -> np.ndarray:
        arr = np.load(bundle / name, mmap_mode="r")
        if list(arr.shape) != files[name]["shape"] or arr.shape[0] != n:
            raise BundleError(f"{name} has shape {arr.shape}, manifest says {files[name]['shape']}")
        return arr

    with np.load(bundle / CATALOG) as npz:
        catalog = {key: npz[key] for key in CATALOG_ARRAYS}
    for key, arr in catalog.items():
        if len(arr) != n:
            raise BundleError(f"catalog {key} has {len(arr)} rows for {n} items")
    text = read_text(bundle / CATALOG_TEXT)
    for key in TEXT_FIELDS:
        if len(text[key]) != n:
            raise BundleError(f"catalog_text {key} has {len(text[key])} rows for {n} items")
    content = np.asarray(matrix("content_item.f16.npy"), dtype=np.float32)
    with np.load(bundle / CORPUS_USERS) as npz:
        corpus_ids = [str(u) for u in npz["user_ids"]]
        corpus_vectors = npz["vectors"].astype(np.float32)
        history = {k: npz[k] for k in ("offsets", "items", "stars", "ts")}
    store = ArtifactStore(
        bundle_dir=bundle,
        manifest=manifest,
        catalog=catalog,
        text=text,
        lgcn_item=matrix("lgcn_item.f16.npy"),
        lgcn_item_norm=matrix("lgcn_item_norm.f16.npy"),
        content=content,
        item_nn=matrix(ITEM_NN),
        trending_top=np.load(bundle / TRENDING_TOP),
        corpus_ids=corpus_ids,
        corpus_vectors=corpus_vectors,
        corpus_history=history,
        index=VectorIndex(content),
    )
    store.by_author = _group(catalog["author_idx"], order=-catalog["rating_count"])
    store.by_series = _group(catalog["series_idx"], order=catalog["series_pos"])
    _STORE = store
    return store


def get_store() -> ArtifactStore:
    """The loaded store; ``load_store`` must have been called."""
    if _STORE is None:
        raise RuntimeError("call litsense_core.load_store(bundle_dir) first")
    return _STORE


def _group(keys: np.ndarray, order: np.ndarray) -> dict[int, np.ndarray]:
    """key -> item ids sorted by ``order`` (stable), skipping key -1."""
    items = np.lexsort((np.arange(len(keys)), order))
    out: dict[int, list[int]] = {}
    for i in items.tolist():
        k = int(keys[i])
        if k >= 0:
            out.setdefault(k, []).append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


def rss_mb() -> float | None:
    """Resident memory of this process in MB (Linux ``/proc``; None elsewhere)."""
    status = Path(f"/proc/{os.getpid()}/status")
    if sys.platform.startswith("linux") and status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    return None
