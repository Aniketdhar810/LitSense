"""Candidate sources, union and filters (CLAUDE.md s13.1-13.2).

Week-5 sources: ``cf`` (LightGCN), ``semantic`` (content index), ``item_item`` (precomputed
content neighbours), ``popularity`` (trending), ``author`` (favourite authors) and ``longtail``.
The ``aspect`` source returns nothing until the bundle has item aspects (Week 7). Until the bundle
carries the projection, ``longtail`` ranks tail items by normalized LightGCN when a CF vector exists
and by content otherwise (D47).

Every source skips the reader's seen items; quotas come from ``ranking.quotas[bucket]``; the union
is built round-robin over sources in quota order and capped at ``ranking.candidates_total``.
"""

from dataclasses import dataclass

import numpy as np

from litsense_core.profile import Profile
from litsense_core.schemas import Filters
from litsense_core.store import ArtifactStore

SOURCES = ("cf", "semantic", "item_item", "popularity", "author", "aspect", "longtail")
ANCHORS = 3  # liked items feeding item_item (s13.1)
PER_ANCHOR = 15


@dataclass
class Candidates:
    """Candidate ids in union order, and each source's ranked list."""

    ids: np.ndarray
    lists: dict[str, list[int]]

    def has(self, source: str, item: int) -> bool:
        return item in self._sets[source]

    def sources_of(self, item: int) -> list[str]:
        return [s for s in SOURCES if item in self._sets[s]]

    def rank_in(self, source: str, item: int) -> int | None:
        return self._ranks[source].get(item)

    def __post_init__(self) -> None:
        self._sets = {s: set(v) for s, v in self.lists.items()}
        self._ranks = {s: {i: r for r, i in enumerate(v)} for s, v in self.lists.items()}


def _top(scores: np.ndarray, k: int, banned: np.ndarray) -> list[int]:
    scores = np.where(banned, -np.inf, scores)
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top], kind="stable")].tolist()


def genre_bits(genres: set[int]) -> np.uint64:
    return np.uint64(sum(1 << g for g in genres))


def source_lists(
    profile: Profile, store: ArtifactStore, quotas: dict[str, int], anchors: list[int] | None = None
) -> dict[str, list[int]]:
    """Each source's ranked items (seen items never included)."""
    cat = store.catalog
    n = store.n_items
    seen = np.zeros(n, dtype=bool)
    seen[list(profile.seen)] = True
    no_cf = seen | (cat["rating_count"] == 0)  # LightGCN cannot score items without interactions
    lists: dict[str, list[int]] = {s: [] for s in SOURCES}
    if profile.cf_vec is not None and quotas.get("cf", 0):
        scores = np.asarray(store.lgcn_item, dtype=np.float32) @ profile.cf_vec
        lists["cf"] = _top(scores, quotas["cf"], no_cf)
    if profile.semantic_vec is not None and quotas.get("semantic", 0):
        ids, _ = store.index.search(profile.semantic_vec, quotas["semantic"], exclude=profile.seen)
        lists["semantic"] = ids.tolist()
    if quotas.get("item_item", 0):
        rows = [store.item_nn[a][:PER_ANCHOR] for a in (anchors or profile.liked)[:ANCHORS]]
        merged = [int(i) for group in zip(*rows, strict=False) for i in group] if rows else []
        lists["item_item"] = [i for i in dict.fromkeys(merged) if not seen[i]][
            : quotas["item_item"]
        ]
    if quotas.get("popularity", 0):
        trending = [int(i) for i in store.trending_top if not seen[i]]
        wanted = genre_bits(profile.preferred_genres)
        if wanted:
            preferred = [i for i in trending if cat["genre_mask"][i] & wanted]
            trending = preferred + [i for i in trending if i not in set(preferred)]
        lists["popularity"] = trending[: quotas["popularity"]]
    if profile.favorite_authors and quotas.get("author", 0):
        ranked = sorted(profile.favorite_authors.items(), key=lambda kv: -kv[1])
        items = [int(i) for a, _ in ranked for i in store.by_author.get(a, []) if not seen[i]]
        lists["author"] = items[: quotas["author"]]
    if quotas.get("longtail", 0):
        tail = cat["is_longtail"].astype(bool)
        if profile.cf_vec is not None:
            scores = np.asarray(store.lgcn_item_norm, dtype=np.float32) @ profile.cf_vec
            lists["longtail"] = _top(scores, quotas["longtail"], no_cf | ~tail)
        elif profile.semantic_vec is not None:
            ids, _ = store.index.search(
                profile.semantic_vec, quotas["longtail"], exclude=profile.seen, allow=tail
            )
            lists["longtail"] = ids.tolist()
    return lists


def union(lists: dict[str, list[int]], quotas: dict[str, int], total: int) -> np.ndarray:
    """Round-robin over sources in quota order (largest first), deduplicated, capped at total."""
    order = sorted((s for s in SOURCES if lists.get(s)), key=lambda s: -quotas.get(s, 0))
    seen: dict[int, None] = {}
    for depth in range(max((len(lists[s]) for s in order), default=0)):
        for s in order:
            if depth < len(lists[s]):
                seen.setdefault(lists[s][depth], None)
        if len(seen) >= total:
            break
    return np.array(list(seen)[:total], dtype=np.int64)


def generate(
    profile: Profile, store: ArtifactStore, anchors: list[int] | None = None
) -> Candidates:
    """Candidates for ``profile`` with its bucket's quotas."""
    ranking = store.config["ranking"]
    quotas = {k: int(v) for k, v in ranking["quotas"][profile.bucket].items()}
    lists = source_lists(profile, store, quotas, anchors)
    return Candidates(union(lists, quotas, int(ranking["candidates_total"])), lists)


def next_in_series(profile: Profile, store: ArtifactStore) -> dict[int, int]:
    """series_idx -> the unseen next volume (position progress + 1) when the catalogue has it."""
    out = {}
    for s, done in profile.series_progress.items():
        items = store.by_series.get(s, np.empty(0, np.int64))
        pos = store.catalog["series_pos"][items]
        nxt = items[(pos > done) & (pos <= done + 1)]
        nxt = [int(i) for i in nxt if int(i) not in profile.seen]
        if nxt:
            out[s] = nxt[0]
    return out


def filter_mask(
    cands: Candidates,
    profile: Profile,
    store: ArtifactStore,
    filters: Filters | None,
    exclude: set[int],
) -> np.ndarray:
    """Boolean mask over ``cands.ids``: True = may appear in the main feed (s13.2). The series
    guard admits a volume p > 1 only when volume p - 1 is read; the next volume itself goes to
    ``continue_series``, never the main feed. Unknown year or pages pass year/page filters."""
    cat = store.catalog
    ids = cands.ids
    keep = ~np.isin(ids, list(profile.seen | exclude))
    f = filters or Filters()
    year, pages = cat["year"][ids], cat["pages"][ids]
    for value, lo, hi in ((year, f.min_year, f.max_year), (pages, f.min_pages, f.max_pages)):
        if lo is not None:
            keep &= np.isnan(value) | (value >= lo)
        if hi is not None:
            keep &= np.isnan(value) | (value <= hi)
    labels = {g: k for k, g in enumerate(store.genre_labels)}
    masks = cat["genre_mask"][ids]
    include = genre_bits({labels[g] for g in f.genres_include if g in labels})
    if f.genres_include:
        keep &= (masks & include) > 0
    blocked = genre_bits(
        {labels[g] for g in f.genres_exclude if g in labels} | profile.muted_genres
    )
    keep &= (masks & blocked) == 0
    min_reviews = int(store.config["ranking"]["min_reviews_main_feed"])
    longtail = np.array([cands.has("longtail", i) for i in ids.tolist()], dtype=bool)
    keep &= (cat["n_reviews"][ids] >= min_reviews) | longtail
    series, pos = cat["series_idx"][ids], cat["series_pos"][ids]
    in_series = (series >= 0) & ~np.isnan(pos) & (pos > 1)
    done = np.array([profile.series_progress.get(int(s), 0.0) for s in series])
    keep &= ~in_series | (done >= pos - 1)
    nxt = set(next_in_series(profile, store).values())
    keep &= ~np.isin(ids, list(nxt))
    return keep
