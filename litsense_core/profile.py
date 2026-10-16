"""Online reader profile, recomputed from ``UserState`` on every call (CLAUDE.md s11.8, s12.2).

MVP scope (Week 5): rating weights and time decay, the CF vector (exact for demo accounts, pseudo
from liked items otherwise), semantic vectors, favourite authors, genre affinities, series progress
and the seen set. Aspect vectors and descriptors join when the bundle carries item aspects (Week 7).

"Now" for time decay is the reader's newest event, so the same state always gives the same profile.
"""

from dataclasses import dataclass, field

import numpy as np

from litsense_core.schemas import UserState
from litsense_core.store import ArtifactStore

DAY_MS = 86_400_000
RECENT = 10  # interactions in the "recent" variants (s11.8)
TOP_AUTHORS = 20
DISLIKED_BELOW = -0.3
LIST_WEIGHT = 0.5  # reading-list adds count half a rating (s12.2)
SEEN_STATUSES = ("read", "dropped")


@dataclass
class Profile:
    """What candidate generation, ranking and reasons read about one reader."""

    n: float  # ratings + 0.5 x reading-list adds
    bucket: str
    seen: set[int]
    liked: list[int]  # rated >= 4 or picked, newest first
    stars: dict[int, float]
    cf_vec: np.ndarray | None
    cf_available: float  # 0 none, 0.5 pseudo, 1 exact (feature 4)
    semantic_vec: np.ndarray | None
    semantic_recent: np.ndarray | None
    favorite_authors: dict[int, float]  # author_idx -> affinity in [0, 1]
    genre_affinity: np.ndarray  # (n_genres,) in [-1, 1]
    preferred_genres: set[int]
    disliked_genres: set[int]
    muted_genres: set[int]
    series_progress: dict[int, float]  # series_idx -> highest position read
    mean_rating: float | None
    picks: list[int] = field(default_factory=list)


def bucket_of(n: float, buckets: dict[str, list[float]]) -> str:
    """The last bucket whose lower bound is <= n (D11)."""
    name = "cold"
    for key, (lo, _) in sorted(buckets.items(), key=lambda kv: kv[1][0]):
        if lo <= n:
            name = key
    return name


def _normalize(v: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(v))
    return None if norm == 0 else (v / norm).astype(np.float32)


def check_ids(state: UserState, n_items: int) -> None:
    """Raise ValueError for any book id outside the catalogue (s15.3: invalid input)."""
    ids = [r.book_id for r in state.ratings] + [e.book_id for e in state.reading_list]
    ids += state.dismissed + state.viewed + state.onboarding_picks
    bad = [i for i in ids if i >= n_items]
    if bad:
        raise ValueError(f"unknown book_id(s): {sorted(set(bad))[:5]}")


def build_profile(state: UserState, store: ArtifactStore) -> Profile:
    """The reader's profile from ``state`` and the bundle."""
    check_ids(state, store.n_items)
    cfg = store.config
    weights = {float(k): float(v) for k, v in cfg["aspects"]["rating_weights"].items()}
    half_life = float(cfg["aspects"]["user_half_life_days"])
    cat = store.catalog
    events = [e.ts for e in state.ratings] + [e.ts for e in state.reading_list]
    now = max(events) if events else 0

    def decay(ts: int) -> float:
        return 0.5 ** (max(now - ts, 0) / DAY_MS / half_life)

    # (item, signed weight r_w * d, |r_w| * d, ts) for every signal about a book
    signals: list[tuple[int, float, float, int]] = []
    stars: dict[int, float] = {}
    for r in sorted(state.ratings, key=lambda r: r.ts):
        rw, d = weights.get(float(round(r.stars)), 0.0), decay(r.ts)
        signals.append((r.book_id, rw * d, abs(rw) * d, r.ts))
        stars[r.book_id] = r.stars
    listed = [e for e in state.reading_list if e.status != "dropped"]
    for e in listed:
        if e.book_id not in stars:
            signals.append((e.book_id, LIST_WEIGHT * decay(e.ts), LIST_WEIGHT * decay(e.ts), e.ts))
    for b in state.onboarding_picks:
        if b not in stars:
            signals.append((b, 1.0, 1.0, now))
    positive = [(i, w, ts) for i, w, _, ts in signals if w > 0]
    liked = [i for i, _, ts in sorted(positive, key=lambda s: -s[2])]
    liked = list(dict.fromkeys(liked))

    n = len(state.ratings) + LIST_WEIGHT * len(state.reading_list)
    bucket = bucket_of(n, cfg["ranking"]["buckets"])

    cf_vec, cf_available = None, 0.0
    row = store.corpus_user(state.corpus_user_id) if state.corpus_user_id else None
    if row is not None:
        cf_vec, cf_available = store.corpus_vectors[row], 1.0
    elif positive:
        items = np.array([i for i, _, _ in positive])
        w = np.array([w for _, w, _ in positive], dtype=np.float32)
        cf_vec = (w @ store.lgcn_item[items].astype(np.float32)) / w.sum()
        cf_available = 0.5

    semantic_vec = semantic_recent = None
    if positive:
        items = np.array([i for i, _, _ in positive])
        w = np.array([w for _, w, _ in positive], dtype=np.float32)
        semantic_vec = _normalize(w @ store.content[items])
        recent = np.argsort([-ts for _, _, ts in positive], kind="stable")[:RECENT]
        semantic_recent = _normalize(w[recent] @ store.content[items[recent]])

    n_genres = len(store.genre_labels)
    bits = np.uint64(1) << np.arange(n_genres, dtype=np.uint64)
    num, den = np.zeros(n_genres), np.zeros(n_genres)
    authors: dict[int, float] = {}
    progress: dict[int, float] = {}
    for i, w, aw, _ in signals:
        has = (cat["genre_mask"][i] & bits) > 0
        num[has] += w
        den[has] += aw
        a = int(cat["author_idx"][i])
        if a >= 0 and (i in liked):
            authors[a] = authors.get(a, 0.0) + w
        s, pos = int(cat["series_idx"][i]), float(cat["series_pos"][i])
        if s >= 0 and not np.isnan(pos) and w > 0:
            progress[s] = max(progress.get(s, 0.0), pos)
    for e in state.reading_list:  # read and dropped books advance the series too
        s, pos = int(cat["series_idx"][e.book_id]), float(cat["series_pos"][e.book_id])
        if s >= 0 and not np.isnan(pos) and e.status in SEEN_STATUSES:
            progress[s] = max(progress.get(s, 0.0), pos)
    affinity = np.divide(num, den, out=np.zeros(n_genres), where=den > 0)
    labels = {g: k for k, g in enumerate(store.genre_labels)}
    onboarding = state.onboarding
    if onboarding is not None:
        for g in onboarding.liked_genres:
            if g in labels:
                affinity[labels[g]] = max(affinity[labels[g]], 1.0)
        for g in onboarding.disliked_genres:
            if g in labels:
                affinity[labels[g]] = min(affinity[labels[g]], -1.0)
    muted = {labels[g] for g in state.overrides.muted_genres if g in labels}
    top = sorted(authors.items(), key=lambda kv: -kv[1])[:TOP_AUTHORS]
    peak = max((v for _, v in top), default=0.0)
    favorites = {a: v / peak for a, v in top if v > 0} if peak > 0 else {}

    seen = set(stars) | set(state.dismissed) | set(state.onboarding_picks)
    seen |= {e.book_id for e in state.reading_list if e.status in SEEN_STATUSES}
    return Profile(
        n=n,
        bucket=bucket,
        seen=seen,
        liked=liked,
        stars=stars,
        cf_vec=cf_vec,
        cf_available=cf_available,
        semantic_vec=semantic_vec,
        semantic_recent=semantic_recent,
        favorite_authors=favorites,
        genre_affinity=affinity.astype(np.float32),
        preferred_genres={int(k) for k in np.flatnonzero(affinity > 0)},
        disliked_genres={int(k) for k in np.flatnonzero(affinity < DISLIKED_BELOW)} | muted,
        muted_genres=muted,
        series_progress=progress,
        mean_rating=float(np.mean(list(stars.values()))) if stars else None,
        picks=list(state.onboarding_picks),
    )
