"""Blend ranker (CLAUDE.md s12.3): the Week-5 MVP ranker and the permanent fallback.

Per-source rank score r = 1 - rank / len(list) (0 when absent); with alpha(n) = n / (n + alpha_k):
score = alpha(n) r_cf + (1 - alpha(n)) r_semantic + w_pop r_popularity + w_aspect aspect_match.
``aspect_match`` is 0 until the bundle carries aspects (Week 7).
"""

import numpy as np

from litsense_core.candidates import Candidates
from litsense_core.profile import Profile


def rank_scores(cands: Candidates, source: str) -> np.ndarray:
    """r = 1 - rank / len(list) for each candidate, 0 when the source did not return it."""
    size = len(cands.lists.get(source, []))
    out = np.zeros(len(cands.ids), dtype=np.float32)
    for k, item in enumerate(cands.ids.tolist()):
        rank = cands.rank_in(source, item)
        if rank is not None:
            out[k] = 1.0 - rank / size
    return out


def blend(
    cands: Candidates,
    profile: Profile,
    params: dict[str, float],
    aspect_match: np.ndarray | None = None,
) -> np.ndarray:
    """Blend score per candidate (s12.3)."""
    alpha = profile.n / (profile.n + float(params["alpha_k"]))
    score = alpha * rank_scores(cands, "cf") + (1 - alpha) * rank_scores(cands, "semantic")
    score += float(params["w_pop"]) * rank_scores(cands, "popularity")
    if aspect_match is not None:
        score += float(params["w_aspect"]) * np.nan_to_num(aspect_match)
    return score.astype(np.float32)
