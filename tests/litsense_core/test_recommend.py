"""The real recommender over a synthetic bundle (CLAUDE.md s16.2 Serving)."""

import json
from pathlib import Path

import numpy as np
import pytest

import litsense_core
from litsense_core.candidates import union
from litsense_core.postprocess import explore, genre_cap
from litsense_core.profile import bucket_of
from litsense_core.schemas import Filters, ListEvent, Overrides, RatingEvent, UserState
from litsense_core.store import BundleError, load_store
from litsense_core.vector_index import VectorIndex
from tests.litsense_core.conftest import BASE_TS, NO_REVIEWS

TS = BASE_TS


@pytest.fixture(scope="module")
def store(bundle_dir: Path):
    return load_store(bundle_dir)


def rated(*pairs: tuple[int, float]) -> UserState:
    return UserState(ratings=[RatingEvent(book_id=b, stars=s, ts=TS + k) for k, (b, s) in
                              enumerate(pairs)])  # fmt: skip


def test_store_loads_and_checks_shapes(store, bundle_dir: Path, tmp_path: Path) -> None:
    assert store.n_items == 60 and store.content.dtype == np.float32
    assert store.by_series[0].tolist() == [0, 1, 2]
    health = litsense_core.health()
    assert health["bundle_version"].startswith("test-") and health["ranker"] is False
    broken = tmp_path / "broken"
    broken.mkdir()
    for f in bundle_dir.iterdir():
        (broken / f.name).write_bytes(f.read_bytes())
    manifest = json.loads((broken / "manifest.json").read_text())
    manifest["files"]["lgcn_item.f16.npy"]["shape"] = [61, 8]
    (broken / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="shape"):
        load_store(broken)
    load_store(bundle_dir)  # restore the module singleton for later tests


def test_recommend_valid_k_items_never_seen_or_dismissed(store) -> None:
    state = rated((10, 5.0), (11, 4.0), (12, 2.0))
    state.dismissed = [15]
    state.reading_list = [ListEvent(book_id=20, status="read", ts=TS + 9)]
    resp = litsense_core.recommend(state, k=8)
    ids = [i.book_id for i in resp.items]
    assert len(ids) == 8 and len(set(ids)) == 8
    assert not {10, 11, 12, 15, 20} & set(ids)
    assert [i.rank for i in resp.items] == list(range(1, 9))
    assert resp.bucket == "warming" and not resp.degraded
    assert all(item.reason and item.evidence.book["book_id"] == item.book_id for item in resp.items)
    assert NO_REVIEWS not in ids or "longtail" in resp.items[ids.index(NO_REVIEWS)].sources


def test_filters_exclude_and_muted_genres(store) -> None:
    state = rated((10, 5.0), (11, 5.0))
    state.overrides = Overrides(muted_genres=["Romance"])
    f = Filters(min_year=2000, genres_exclude=["Mystery"])
    resp = litsense_core.recommend(state, k=10, filters=f, exclude=[4, 9])
    for item in resp.items:
        i = item.book_id
        assert store.catalog["year"][i] >= 2000 and i not in (4, 9)
        assert store.text["primary_genre"][i] not in ("Romance", "Mystery")
    only = litsense_core.recommend(state, k=5, filters=Filters(genres_include=["Thriller"]))
    assert {store.text["primary_genre"][i.book_id] for i in only.items} == {"Thriller"}


def test_series_guard_and_continue_series(store) -> None:
    state = rated((0, 5.0), (30, 4.0))  # read volume 1 of the saga
    for _ in range(3):
        resp = litsense_core.recommend(state, k=20)
        ids = [i.book_id for i in resp.items]
        assert 2 not in ids and 1 not in ids  # never volume 3; volume 2 only in continue_series
    assert [i.book_id for i in resp.continue_series] == [1]
    state.reading_list = [ListEvent(book_id=1, status="read", ts=TS + 5)]
    assert [i.book_id for i in litsense_core.recommend(state).continue_series] == [2]


def test_cold_paths(store) -> None:
    nothing = litsense_core.recommend(UserState(), k=5)
    assert nothing.bucket == "cold" and len(nothing.items) == 5
    assert {i.reason for i in nothing.items} == {"Popular with readers right now"}
    top = [int(i) for i in store.trending_top if store.catalog["n_reviews"][i] >= 3][:5]
    assert sorted(i.book_id for i in nothing.items) == sorted(top)
    picks = litsense_core.recommend(UserState(onboarding_picks=[0, 25]), k=5)
    assert picks.bucket == "cold" and len(picks.items) == 5
    assert not {0, 25} & {i.book_id for i in picks.items}
    assert any(i.reason.startswith(("Similar to", "Because you loved")) for i in picks.items)


def test_corpus_user_gets_exact_cf_and_invalid_ids_raise(store) -> None:
    from litsense_core.profile import build_profile

    state = UserState(corpus_user_id="DEMO1", ratings=rated((5, 5.0)).ratings)
    prof = build_profile(state, store)
    assert prof.cf_available == 1.0 and np.allclose(prof.cf_vec, store.corpus_vectors[0])
    assert build_profile(rated((5, 5.0)), store).cf_available == 0.5
    assert build_profile(UserState(), store).cf_vec is None
    with pytest.raises(ValueError, match="unknown book_id"):
        litsense_core.recommend(rated((999, 5.0)))
    with pytest.raises(ValueError):
        litsense_core.recommend(UserState(), exclude=[60])


def test_author_reason_and_catalog(store) -> None:
    state = rated((12, 5.0), (24, 5.0))  # author 0 wrote 12, 24, 36, 48 (and the saga)
    resp = litsense_core.recommend(state, k=40)  # every candidate: the blend ignores the source
    by_author = [i for i in resp.items if store.catalog["author_idx"][i.book_id] == 0]
    assert by_author and all(i.reason.startswith("Because you loved Author 0's") for i in by_author)
    books = litsense_core.catalog()
    assert len(books) == 60 and books[1].series_pos == 2.0 and books[1].series_id == "am_saga"


def test_helpers() -> None:
    assert bucket_of(0, {"cold": [0, 0], "warming": [1, 4], "blending": [5, 19]}) == "cold"
    assert bucket_of(4.5, {"cold": [0, 0], "warming": [1, 4], "blending": [5, 19]}) == "warming"
    lists = {"cf": [1, 2, 3], "semantic": [3, 4], "popularity": [9]}
    assert union(lists, {"cf": 2, "semantic": 5, "popularity": 1}, 4).tolist() == [3, 1, 9, 4]
    genre = np.array([0, 0, 0, 1, 1])
    assert genre_cap([0, 1, 2, 3, 4], genre, 2) == [0, 1, 3, 4, 2]
    assert explore([1, 2, 3, 4], [9, 2], slot=2, k=4) == [1, 9, 2, 3, 4]
    index = VectorIndex(np.eye(4, dtype=np.float32), use_faiss=False)
    ids, _ = index.search(np.array([1, 0.5, 0, 0]), 2, exclude={0})
    assert ids.tolist() == [1, 2] or ids.tolist()[0] == 1
    faiss_index = VectorIndex(np.eye(4, dtype=np.float32))
    assert faiss_index.search(np.array([0, 1, 0.5, 0]), 1, exclude={1})[0].tolist() == [2]
