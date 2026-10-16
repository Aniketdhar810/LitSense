"""A small synthetic serving bundle written by the real bundle writer (``ml.serve.export``)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from ml.common.config import DEFAULT_CONFIG_PATH
from ml.serve.export import RANKING_OFFLINE_ONLY, write_bundle

N_ITEMS = 60
GENRES = ["Fantasy", "Mystery", "Romance", "Science Fiction", "Thriller"]
SERIES = (0, 1, 2)  # items 0-2: volumes 1-3 of one series, all Fantasy, by author 0
NO_REVIEWS = 59  # too few reviews for the main feed
BASE_TS = 1_600_000_000_000


def bundle_inputs() -> dict:
    """Catalogue, text and matrices for N_ITEMS books: item i has genre i % 5 and author i % 12."""
    rng = np.random.default_rng(0)
    n = N_ITEMS
    genre = np.arange(n) % len(GENRES)
    content = rng.normal(size=(n, 16)).astype(np.float32)
    content[:3] = content[0] + 0.1 * rng.normal(size=(3, 16))  # the series reads alike
    content /= np.linalg.norm(content, axis=1, keepdims=True)
    sims = content @ content.T
    np.fill_diagonal(sims, -np.inf)
    item_nn = np.argsort(-sims, axis=1)[:, :10]
    lgcn = rng.normal(size=(n, 8)).astype(np.float32)
    count = rng.integers(1, 200, n)
    series_idx = np.full(n, -1, np.int32)
    series_pos = np.full(n, np.nan, np.float32)
    series_idx[list(SERIES)] = 0
    series_pos[list(SERIES)] = [1.0, 2.0, 3.0]
    author = (np.arange(n) % 12).astype(np.int32)
    author[list(SERIES)] = 0
    genre[list(SERIES)] = 0
    n_reviews = np.full(n, 10, np.int32)
    n_reviews[NO_REVIEWS] = 0
    catalog = {
        "year": (1990 + np.arange(n) % 30).astype(np.float32),
        "pages": (150 + 10 * np.arange(n)).astype(np.float32),
        "series_pos": series_pos,
        "series_idx": series_idx,
        "primary_genre": genre.astype(np.int16),
        "genre_mask": (np.uint64(1) << genre.astype(np.uint64)),
        "author_idx": author,
        "bayes_rating": rng.uniform(3, 5, n).astype(np.float32),
        "rating_count": count.astype(np.int32),
        "rating_std": rng.uniform(0, 1, n).astype(np.float32),
        "trending": count.astype(np.float32),
        "is_longtail": count < np.quantile(count, 0.8),
        "n_reviews": n_reviews,
    }
    text = {
        "canonical_id": [f"am_{i:03d}" for i in range(n)],
        "title": [f"Book {i}" for i in range(n)],
        "author": [f"Author {a}" for a in author],
        "genres": [[GENRES[g]] for g in genre],
        "primary_genre": [GENRES[g] for g in genre],
        "description": ["A premise." for _ in range(n)],
        "image": [None] * n,
        "isbn13": [None] * n,
        "series_name": ["The Saga" if s >= 0 else None for s in series_idx],
        "author_keys": [f"author {a}" for a in range(12)],
        "series_ids": ["am_saga"],
        "genre_labels": GENRES,
    }
    users = rng.normal(size=(2, 8)).astype(np.float32)
    corpus = {
        "user_ids": np.array(["DEMO1", "DEMO2"]),
        "vectors": users,
        "offsets": np.array([0, 2, 3]),
        "items": np.array([5, 6, 7], np.int32),
        "stars": np.array([5.0, 4.0, 5.0], np.float32),
        "ts": np.array([BASE_TS, BASE_TS + 1, BASE_TS + 2]),
    }
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    ranking = {k: v for k, v in cfg["ranking"].items() if k not in RANKING_OFFLINE_ONLY}
    ranking["candidates_total"] = 40
    config = {"ranking": ranking, "serving": cfg["serving"], "aspects": cfg["aspects"],
              "llm": cfg["llm"]}  # fmt: skip
    trending = np.argsort(-count, kind="stable")
    return {
        "catalog": catalog,
        "text": text,
        "matrices": {
            "lgcn_item": lgcn,
            "lgcn_item_norm": lgcn / np.linalg.norm(lgcn, axis=1)[:, None],
            "content_item": content,
            "item_nn": item_nn,
            "trending_top": trending[:30],
        },  # fmt: skip
        "corpus": corpus,
        "meta": {
            "data_version": "test",
            "cutoff": BASE_TS + 10,
            "genre_labels": GENRES,
            "config": config,
        },  # fmt: skip
    }


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bundle") / "serve_bundle"
    write_bundle(out, **bundle_inputs())
    return out
