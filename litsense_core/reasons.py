"""Evidence and template reasons (CLAUDE.md s12.5). Every number in a reason comes from evidence.

Templates, first match wins:
1. favourite author -> "Because you loved {author}'s {liked_title}"
2. aspect praise (needs item aspects, Week 7)
3. content similarity >= 0.6 to a liked book -> "Similar to {liked_title}, which you rated {stars}★"
   (a liked book without stars, e.g. a pick: "Similar to {liked_title}, which you liked")
4. onboarding theme (needs the ONNX encoder, Week 6)
5. long-tail source -> "A lesser-known pick that fits your taste"
6. fallback -> "Popular with {genre} readers right now" ("Popular with readers right now" for a
   reader the core knows nothing about, s12.4)
"""

import numpy as np

from litsense_core.profile import Profile
from litsense_core.schemas import Evidence, LikedItem
from litsense_core.store import ArtifactStore

SIMILAR_MIN = 0.6
SIMILAR_SHOWN = 3


def similar_liked(profile: Profile, store: ArtifactStore, book_id: int) -> list[tuple[int, float]]:
    """The reader's liked books most similar to ``book_id`` by content, best first."""
    if not profile.liked:
        return []
    liked = np.array(profile.liked)
    sims = store.content[liked] @ store.content[book_id]
    order = np.argsort(-sims, kind="stable")
    return [(int(liked[k]), float(sims[k])) for k in order]


def build_evidence(
    profile: Profile, book_id: int, store: ArtifactStore, sources: list[str], signals: dict
) -> Evidence:
    """Facts only: the book, what the reader liked that relates to it, and the ranking signals."""
    text = store.text
    author_idx = int(store.catalog["author_idx"][book_id])
    favorite = text["author"][book_id] if author_idx in profile.favorite_authors else None
    similar = [
        LikedItem(book_id=i, title=text["title"][i], stars=profile.stars.get(i, 0.0)).model_dump()
        | {"similarity": round(s, 3)}
        for i, s in similar_liked(profile, store, book_id)[:SIMILAR_SHOWN]
    ]
    return Evidence(
        book={
            "book_id": book_id,
            "title": text["title"][book_id],
            "author": text["author"][book_id],
            "genres": text["genres"][book_id],
        },  # fmt: skip
        user_evidence={
            "top_positive_aspects": [],
            "top_negative_aspects": [],
            "similar_liked_items": similar,
            "favorite_author": favorite,
        },
        item_evidence={
            "aspect_profile": {},
            "mention_rates": {},
            "pos_rates": {},
            "snippet": None,
        },  # fmt: skip
        signals={"sources": sources, "bucket": profile.bucket, "aspect_match": None, **signals},
    )


def template_reason(evidence: Evidence, profile: Profile, store: ArtifactStore) -> str:
    """The first template whose condition holds (module docstring)."""
    book = evidence.book
    author = evidence.user_evidence["favorite_author"]
    if author:
        idx = int(store.catalog["author_idx"][book["book_id"]])
        mine = [i for i in profile.liked if int(store.catalog["author_idx"][i]) == idx]
        if mine:
            return f"Because you loved {author}'s {store.text['title'][mine[0]]}"
    similar = evidence.user_evidence["similar_liked_items"]
    if similar and similar[0]["similarity"] >= SIMILAR_MIN:
        top = similar[0]
        if top["stars"]:
            return f"Similar to {top['title']}, which you rated {top['stars']:g}★"
        return f"Similar to {top['title']}, which you liked"
    if "longtail" in evidence.signals["sources"]:
        return "A lesser-known pick that fits your taste"
    if not profile.liked and not profile.preferred_genres:
        return "Popular with readers right now"
    return f"Popular with {store.text['primary_genre'][book['book_id']]} readers right now"
