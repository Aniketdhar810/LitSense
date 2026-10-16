"""The main feed (CLAUDE.md s12.6) plus ``catalog``, ``health`` and ``download_bundle``.

recommend: profile -> candidates -> filters -> blend score (the LightGBM ranker plugs in at Week 9)
-> MMR, genre cap, exploration slot -> evidence and template reason per item. The next volume of a
series the reader is in goes to ``continue_series``, never the main feed.
"""

import tarfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from litsense_core import __api_version__
from litsense_core.candidates import filter_mask, generate, next_in_series
from litsense_core.postprocess import postprocess
from litsense_core.profile import Profile, build_profile
from litsense_core.rank import blend
from litsense_core.reasons import build_evidence, template_reason
from litsense_core.schemas import CatalogBook, Filters, RecItem, RecResponse, UserState
from litsense_core.store import ArtifactStore, get_store, rss_mb

DOWNLOAD_TIMEOUT_S = 120.0


def _item(
    store: ArtifactStore,
    profile: Profile,
    book_id: int,
    rank: int,
    score: float,
    sources: list[str],
    signals: dict[str, Any],
) -> RecItem:
    evidence = build_evidence(profile, book_id, store, sources, signals)
    return RecItem(
        book_id=book_id,
        rank=rank,
        score=round(float(score), 6),
        sources=sources,
        reason=template_reason(evidence, profile, store),
        evidence=evidence,
    )


def recommend(
    state: UserState,
    k: int = 10,
    filters: Filters | None = None,
    exclude: list[int] | None = None,
    use_ranker: bool = True,
) -> RecResponse:
    """Top-k main feed for ``state``. Raises ValueError only for invalid input (s15.3)."""
    started = time.perf_counter()
    if k < 1:
        raise ValueError("k must be >= 1")
    store = get_store()
    excluded = set(exclude or [])
    if any(i < 0 or i >= store.n_items for i in excluded):
        raise ValueError("exclude holds an unknown book_id")
    profile = build_profile(state, store)
    cands = generate(profile, store)
    keep = filter_mask(cands, profile, store, filters, excluded)
    ids = cands.ids[keep]
    # the LightGBM ranker joins in Week 9 (use_ranker); until then the blend is the ranker and
    # degraded stays False (s16.2: ranker missing -> blend, not degraded)
    scores = blend(cands, profile, store.config["ranking"]["blend"])[keep]
    top = postprocess(ids, scores, store, cands.lists["longtail"], k) if len(ids) else []
    by_id = dict(zip(ids.tolist(), scores.tolist(), strict=True))
    items = [
        _item(store, profile, i, rank, by_id[i], cands.sources_of(i), _signals(cands, i))
        for rank, i in enumerate(top, 1)
    ]
    return RecResponse(
        items=items,
        continue_series=continue_series(profile, store),
        model_version=store.version,
        profile_version=state.profile_version,
        bucket=profile.bucket,
        degraded=False,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def _signals(cands: Any, item: int) -> dict[str, Any]:
    return {"cf_rank": cands.rank_in("cf", item), "semantic_rank": cands.rank_in("semantic", item)}


def continue_series(profile: Profile, store: ArtifactStore) -> list[RecItem]:
    """The next unseen volume of every series the reader has started (s12.4)."""
    nxt = sorted(next_in_series(profile, store).values())
    return [_item(store, profile, i, r, 0.0, ["series"], {}) for r, i in enumerate(nxt, 1)]


def catalog() -> list[CatalogBook]:
    """Every catalogue book, for seeding the backend's books table (proposed in D12)."""
    store = get_store()
    t, c = store.text, store.catalog
    out = []
    for i in range(store.n_items):
        s = int(c["series_idx"][i])
        year, pages, pos = c["year"][i], c["pages"][i], c["series_pos"][i]
        out.append(
            CatalogBook(
                book_id=i,
                canonical_id=t["canonical_id"][i],
                title=t["title"][i],
                author=t["author"][i],
                genres=list(t["genres"][i]),
                primary_genre=t["primary_genre"][i],
                year=None if np.isnan(year) else int(year),
                pages=None if np.isnan(pages) else int(pages),
                description=t["description"][i],
                image=t["image"][i],
                isbn13=t["isbn13"][i],
                series_id=t["series_ids"][s] if s >= 0 else None,
                series_name=t["series_name"][i],
                series_pos=None if np.isnan(pos) else float(pos),
            )
        )
    return out


def health() -> dict[str, Any]:
    """Bundle version, cutoff, data version, RSS and component availability (s12.1)."""
    store = get_store()
    components = store.manifest.get("components", {})
    return {
        "status": "ok",
        "mock": False,
        "api_version": __api_version__,
        "bundle_version": store.version,
        "data_version": store.manifest["data_version"],
        "cutoff": store.cutoff,
        "rss_mb": rss_mb(),
        "ranker": store.ranker is not None,
        "llm": False,
        "n_books": store.n_items,
        "components": components,
    }


def download_bundle(url: str, dest: str) -> str:
    """Download a bundle tarball (e.g. a GitHub Release asset) and unpack it under ``dest``;
    returns the bundle directory to pass to ``load_store``."""
    target = Path(dest)
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "serve_bundle.tar.gz"
    with httpx.stream("GET", url, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_S) as r:
        r.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    return str(target / "serve_bundle")
