"""Stage serve.export (CLAUDE.md s12.7): the serving bundle that ``litsense_core.load_store`` reads.

Collects the ``serve`` artifact set (LightGCN fit on every interaction), the content embeddings
and neighbours, and catalogue statistics computed at the serve cutoff; casts embeddings to float16;
writes ``manifest.json`` (versions, cutoff, sha256, shapes, the config slice the core reads) and a
tarball. File names and fields come from ``litsense_core.store`` so writer and reader agree. The
bundle's process RSS is measured by loading it in a subprocess (s18.1).

Not in this v0 bundle yet (later weeks): aspects and descriptors, review centroids, evidence
snippets, the projection matrix, ONNX encoder, spoiler model and LightGBM ranker. Author, series
and genre indices are derived at load from ``catalog.npz`` instead of shipped as JSON (D47).
"""

import gzip
import json
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from litsense_core import __api_version__
from litsense_core.store import (
    CATALOG,
    CATALOG_ARRAYS,
    CATALOG_TEXT,
    CORPUS_USERS,
    ITEM_NN,
    MANIFEST,
    TEXT_FIELDS,
    TRENDING_TOP,
    sha256_file,
)
from ml.common.config import REPO_ROOT
from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.eval.data import longtail_items, serve_fit_data
from ml.models.baselines import PopularityScorer
from ml.models.embedding_scorers import EmbeddingArtifact, check_artifact, l2_normalize_rows

LIGHTGCN = "serve/lightgcn"
CONFIG_SLICE = ("ranking", "serving", "aspects", "llm")
RANKING_OFFLINE_ONLY = ("lgbm", "negatives_per_positive", "ranker_user_split")
MAX_GENRES = 64  # genre bitmask is uint64


def _index(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Per-row int32 index into the sorted unique non-null values (-1 for null)."""
    labels = sorted({str(v) for v in values.dropna()})
    pos = {v: n for n, v in enumerate(labels)}
    idx = np.array([pos[str(v)] if pd.notna(v) else -1 for v in values], dtype=np.int32)
    return idx, labels


def _float(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(np.float32)


def _text(value: Any) -> str | None:
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def catalog_tables(
    books: pd.DataFrame,
    interactions: pd.DataFrame,
    n_users: int,
    popularity: Mapping[str, Any],
    longtail_fraction: float,
    description_chars: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
    """(catalog arrays, catalog text, cutoff) at the serve cutoff, from every interaction."""
    n = len(books)
    if list(books["i"]) != list(range(n)):
        raise ValueError("books.parquet rows must be items 0..n-1 in order")
    fit = serve_fit_data(interactions, n_users, n)
    pop = PopularityScorer(popularity)
    pop.fit(fit)
    rows = fit.interactions
    count = np.bincount(rows["i"], minlength=n)
    sq = np.bincount(rows["i"], weights=rows["rating"].to_numpy(np.float64) ** 2, minlength=n)
    mean = np.bincount(rows["i"], weights=rows["rating"], minlength=n) / np.maximum(count, 1)
    std = np.sqrt(np.maximum(sq / np.maximum(count, 1) - mean**2, 0))
    genre_lists = [list(g) if g is not None else [] for g in books["genres"]]
    labels = sorted({g for gs in genre_lists for g in gs} | set(books["primary_genre"].dropna()))
    if len(labels) > MAX_GENRES:
        raise ValueError(f"{len(labels)} genre labels do not fit a uint64 bitmask")
    bit = {g: np.uint64(1) << np.uint64(k) for k, g in enumerate(labels)}
    mask = np.array([sum((bit[g] for g in gs), np.uint64(0)) for gs in genre_lists], np.uint64)
    author_idx, authors = _index(books["author_key"])
    series_idx, series = _index(books["series_id"])
    has_text = interactions.loc[interactions["has_text"], "i"]
    catalog = {
        "year": _float(books["year"]),
        "pages": _float(books["pages"]),
        "series_pos": _float(books["series_pos"]),
        "series_idx": series_idx,
        "primary_genre": np.array([labels.index(g) for g in books["primary_genre"]], np.int16),
        "genre_mask": mask,
        "author_idx": author_idx,
        "bayes_rating": pop.bayes_rating.astype(np.float32),
        "rating_count": count.astype(np.int32),
        "rating_std": np.where(count > 0, std, np.nan).astype(np.float32),
        "trending": pop.trending.astype(np.float32),
        "is_longtail": longtail_items(count, longtail_fraction),
        "n_reviews": np.bincount(has_text, minlength=n).astype(np.int32),
    }
    blurbs = []
    for f, d in zip(books["features"], books["description"], strict=True):
        text = _text(f) or _text(d) or ""
        blurbs.append(text[:description_chars])
    text = {
        "canonical_id": books["canonical_id"].astype(str).tolist(),
        "title": books["title"].astype(str).tolist(),
        "author": [_text(a) for a in books["author"]],
        "genres": genre_lists,
        "primary_genre": books["primary_genre"].astype(str).tolist(),
        "description": blurbs,
        "image": [_text(v) for v in books["image"]],
        "isbn13": [_text(v) for v in books["isbn13"]],
        "series_name": [_text(v) for v in books["series_name"]],
        "author_keys": authors,
        "series_ids": series,
        "genre_labels": labels,
    }
    return catalog, text, fit.cutoff


def corpus_users(
    interactions: pd.DataFrame, user_ids: list[str], vectors: np.ndarray, n: int
) -> dict[str, np.ndarray]:
    """The ``n`` most active users: ids, exact LightGCN vectors and rating histories (CSR)."""
    counts = interactions.groupby("u").size().sort_values(ascending=False, kind="stable")
    chosen = np.sort(counts.index[:n].to_numpy())
    rows = interactions[interactions["u"].isin(chosen)].sort_values(["u", "ts", "i"])
    sizes = rows.groupby("u").size().reindex(chosen).to_numpy()
    return {
        "user_ids": np.array([user_ids[u] for u in chosen]),
        "vectors": vectors[chosen].astype(np.float32),
        "offsets": np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        "items": rows["i"].to_numpy(np.int32),
        "stars": rows["rating"].to_numpy(np.float32),
        "ts": rows["ts"].to_numpy(np.int64),
    }


def write_bundle(
    out: Path,
    catalog: Mapping[str, np.ndarray],
    text: Mapping[str, Any],
    matrices: Mapping[str, np.ndarray],
    corpus: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Write every bundle file into ``out`` and return the manifest (written last).

    ``matrices`` holds ``lgcn_item``, ``lgcn_item_norm``, ``content_item`` (cast to float16),
    ``item_nn`` and ``trending_top``. ``meta`` holds ``data_version``, ``cutoff``,
    ``genre_labels`` and ``config``."""
    out.mkdir(parents=True, exist_ok=True)
    n = len(catalog["year"])
    missing = set(CATALOG_ARRAYS) ^ set(catalog)
    if missing or set(TEXT_FIELDS) - set(text):
        raise ValueError(f"catalog fields do not match the bundle format: {sorted(missing)}")
    np.savez(out / CATALOG, **{k: np.asarray(catalog[k]) for k in CATALOG_ARRAYS})
    with gzip.open(out / CATALOG_TEXT, "wt", encoding="utf-8") as f:
        json.dump(dict(text), f, ensure_ascii=False)
    for name in ("lgcn_item", "lgcn_item_norm", "content_item"):
        np.save(out / f"{name}.f16.npy", np.asarray(matrices[name], dtype=np.float16))
    np.save(out / ITEM_NN, np.asarray(matrices["item_nn"], dtype=np.int32))
    np.save(out / TRENDING_TOP, np.asarray(matrices["trending_top"], dtype=np.int32))
    np.savez(out / CORPUS_USERS, **corpus)
    files: dict[str, Any] = {}
    for path in sorted(out.iterdir()):
        if path.name == MANIFEST:
            continue
        entry: dict[str, Any] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            entry |= {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        files[path.name] = entry
    digest = sha256("".join(e["sha256"] for e in files.values()).encode()).hexdigest()[:10]
    manifest = {
        "bundle_version": f"{meta['data_version']}-{digest}",
        "api_version": __api_version__,
        "data_version": meta["data_version"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cutoff": int(meta["cutoff"]),
        "n_items": n,
        "dims": {
            "cf": int(matrices["lgcn_item"].shape[1]),
            "content": int(matrices["content_item"].shape[1]),
        },  # fmt: skip
        "genre_labels": list(meta["genre_labels"]),
        "components": {"ranker": False, "aspects": False, "encoder": False, "spoiler": False},
        "config": meta["config"],
        "files": files,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def measure_rss(bundle: Path) -> float | None:
    """RSS (MB) of a fresh interpreter after ``load_store`` (Linux only; None elsewhere)."""
    code = (
        "import json, litsense_core.store as s; "
        f"s.load_store(r'{bundle}'); print(json.dumps(s.rss_mb()))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


@stage(
    "serve.export",
    inputs=[
        "data:books.parquet",
        "data:interactions.parquet",
        "data:id_map.json",
        "artifacts:content/content_item.npy",
        "artifacts:content/item_nn.npy",
        *[
            f"artifacts:{LIGHTGCN}/{n}"
            for n in ("lgcn_user.npy", "lgcn_item.npy", "item_degree.npy", "meta.json")
        ],
    ],  # fmt: skip
    outputs=[
        "artifacts:serve_bundle/",
        "artifacts:serve_bundle.tar.gz",
        "results:serve/bundle_report.json",
    ],  # fmt: skip
    config_keys=[
        *CONFIG_SLICE,
        "models.popularity",
        "data.longtail_fraction",
        "project.data_version",
    ],  # fmt: skip
)
def build_serve_export(ctx: StageContext) -> None:
    """Assemble, write, tar and measure the v0 serving bundle."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    books = pd.read_parquet(ctx.path("data:books.parquet")).sort_values("i")
    interactions = pd.read_parquet(
        ctx.path("data:interactions.parquet"), columns=["u", "i", "rating", "ts", "has_text"]
    )
    serving = ctx.param("serving")
    catalog, text, cutoff = catalog_tables(
        books, interactions, ids.n_users, ctx.param("models.popularity"),
        float(ctx.param("data.longtail_fraction")), int(serving["description_chars"]),
    )  # fmt: skip
    ctx.cutoff = cutoff
    lightgcn = EmbeddingArtifact.load(
        ctx.path(f"artifacts:{LIGHTGCN}/meta.json").parent, "lgcn_user.npy", "lgcn_item.npy"
    )
    check_artifact(
        "serve lightgcn", lightgcn, serve_fit_data(interactions, ids.n_users, ids.n_items)
    )
    content = l2_normalize_rows(np.load(ctx.path("artifacts:content/content_item.npy")))
    item_nn = np.load(ctx.path("artifacts:content/item_nn.npy"))
    rated = np.flatnonzero(catalog["rating_count"] > 0)
    trending = rated[np.argsort(-catalog["trending"][rated], kind="stable")]
    ranking = {k: v for k, v in ctx.param("ranking").items() if k not in RANKING_OFFLINE_ONLY}
    config = {"ranking": ranking, "serving": serving, "aspects": ctx.param("aspects"),
              "llm": ctx.param("llm")}  # fmt: skip
    out = ctx.out("artifacts:serve_bundle/")
    manifest = write_bundle(
        out,
        catalog,
        text,
        {
            "lgcn_item": lightgcn.item,
            "lgcn_item_norm": l2_normalize_rows(lightgcn.item),
            "content_item": content,
            "item_nn": item_nn,
            "trending_top": trending[: int(serving["trending_top"])],
        },  # fmt: skip
        corpus_users(interactions, ids.users, lightgcn.user, int(serving["corpus_users"])),
        {
            "data_version": ctx.param("project.data_version"),
            "cutoff": cutoff,
            "genre_labels": text["genre_labels"],
            "config": config,
        },  # fmt: skip
    )
    tar_path = ctx.out("artifacts:serve_bundle.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out, arcname="serve_bundle")
    rss = measure_rss(out)
    budget = float(serving["max_rss_mb"])
    report = {
        "bundle_version": manifest["bundle_version"],
        "cutoff": cutoff,
        "n_items": manifest["n_items"],
        "bundle_bytes": sum(e["bytes"] for e in manifest["files"].values()),
        "tar_bytes": tar_path.stat().st_size,
        "rss_mb_after_load": rss,
        "max_rss_mb": budget,
        "within_budget": None if rss is None else rss <= budget,
        "files": {
            k: {f: v[f] for f in ("bytes", "shape") if f in v} for k, v in manifest["files"].items()
        },  # fmt: skip
    }
    path = ctx.out("results:serve/bundle_report.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ctx.log.info("bundle %s: %.1f MB, RSS after load %s MB", manifest["bundle_version"],
                 report["bundle_bytes"] / 1e6, rss)  # fmt: skip
    if rss is not None and rss > budget:
        ctx.log.warning("RSS %.0f MB is above serving.max_rss_mb %.0f", rss, budget)
    ctx.count("items", manifest["n_items"])
    ctx.notes["bundle_version"] = manifest["bundle_version"]
    ctx.notes["rss_mb"] = rss
