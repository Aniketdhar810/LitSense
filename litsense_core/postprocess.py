"""Post-processing of ranked candidates (CLAUDE.md s13.6): MMR, genre cap, exploration slot.

The series guard runs earlier, in ``candidates.filter_mask``; overrides (muted genres) remove items
before ranking, and pinned aspects need item aspects (Week 7).
"""

import numpy as np

from litsense_core.store import ArtifactStore

WINDOW = 10  # genre cap window (s13.6)


def mmr(
    ids: np.ndarray, scores: np.ndarray, content: np.ndarray, lam: float, pool: int
) -> list[int]:
    """Maximal marginal relevance over the top ``pool``: argmax of lam * s - (1 - lam) * max
    similarity to the items already picked, s = min-max-normalized score. The rest keep score
    order after the pool."""
    order = np.argsort(-scores, kind="stable")
    head, tail = order[:pool], order[pool:]
    s = scores[head]
    span = float(s.max() - s.min()) if len(s) else 0.0
    s = (s - s.min()) / span if span > 0 else np.zeros_like(s)
    vecs = content[ids[head]]
    picked: list[int] = []
    max_sim = np.full(len(head), -np.inf)
    left = np.ones(len(head), dtype=bool)
    for _ in range(len(head)):
        penalty = np.where(np.isfinite(max_sim), max_sim, 0.0)
        value = np.where(left, lam * s - (1 - lam) * penalty, -np.inf)
        k = int(np.argmax(value))
        picked.append(k)
        left[k] = False
        max_sim = np.maximum(max_sim, vecs @ vecs[k])
    return [int(ids[head[k]]) for k in picked] + [int(ids[t]) for t in tail]


def genre_cap(items: list[int], genre: np.ndarray, cap: int) -> list[int]:
    """At most ``cap`` items of one primary genre in any 10 consecutive positions; items that do
    not fit yet move later (and are appended when nothing else fits)."""
    out: list[int] = []
    waiting = list(items)
    while waiting:
        for k, item in enumerate(waiting):
            window = out[-(WINDOW - 1) :]
            if sum(genre[j] == genre[item] for j in window) < cap:
                out.append(waiting.pop(k))
                break
        else:
            out.append(waiting.pop(0))
    return out


def explore(items: list[int], longtail: list[int], slot: int, k: int) -> list[int]:
    """Put the best long-tail item not already in the top ``k`` at 1-based position ``slot``."""
    if slot > k or slot > len(items) + 1:
        return items
    top = set(items[:k])
    pick = next((i for i in longtail if i not in top), None)
    if pick is None:
        return items
    rest = [i for i in items if i != pick]
    return rest[: slot - 1] + [pick] + rest[slot - 1 :]


def postprocess(
    ids: np.ndarray, scores: np.ndarray, store: ArtifactStore, longtail: list[int], k: int
) -> list[int]:
    """Final top-k order: MMR, genre cap, exploration slot."""
    params = store.config["ranking"]["postprocess"]
    ranked = mmr(ids, scores, store.content, float(params["mmr_lambda"]), int(params["mmr_pool"]))
    ranked = genre_cap(ranked, store.catalog["primary_genre"], int(params["genre_cap"]))
    ranked = explore(
        ranked, [i for i in longtail if i in set(ranked)], int(params["explore_slot"]), k
    )
    return ranked[:k]
