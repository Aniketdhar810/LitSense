# Progress

## 2026-10-12: Phase 0 (Week 1). Repo skeleton, stage framework, API contract, mock

### Built
- **Skeleton and tooling:** `pyproject.toml` (setuptools; dependencies read from the requirements files; ruff and
  pytest config), `requirements-core.txt` / `requirements-ml.txt` (exact pins, D1–D3), `Makefile`, `.gitignore`,
  `README.md`, GitHub Actions CI (`.github/workflows/ci.yml`: Python 3.11 and 3.12, ruff, `pytest -m "not slow"`).
  Empty `ml/` subpackages for later phases.
- **`ml/config.yaml`:** §5 verbatim except quoted regexes (D4) and `project.local_raw_root` (D5).
- **`ml/common/`:** `config.py` (load, dotted lookup, structural validation), `seed.py`
  (`set_global_seed`), `log.py`, `errors.py`, `manifest.py` (hashes, file descriptions, fingerprints, atomic JSON,
  git SHA), `stage.py` (registry, `@stage`, `StageContext`, fingerprints, skip/rerun, staging plus atomic moves,
  manifests, work dirs for checkpoints, dependency ordering, status).
- **`ml/stages.py`:** CLI with `list` and `run <stage|all> [--sample N] [--force] [--until X]`. No stages
  are registered yet; Phase 1 adds them to `STAGE_MODULES`.
- **`litsense_core/`:** `schemas.py` (all §15.2 models; deviations in D10), `mock.py` (every §15.1 function over
  a synthetic 200-book catalogue, plus the proposed `catalog()`), `__api_version__ = "0.1.0"`.
- **`docs/API_CONTRACT.md`:** functions, schema rules, chat protocol, behaviour rules, budgets, open questions.
- **`tests/fixtures/tiny/make_tiny.py`:** deterministic raw dataset in Amazon and Goodreads formats (50 users,
  80 books, 600 interactions over 3 years) with `truth.json` for every planted case (D13).

### Tests run
- `make install-ci lint test` (via `mingw32-make`, Windows, Python 3.12 venv with the CI package set):
  ruff clean, **100 passed, 1 skipped**. The skipped test is torch seeding (no torch in the venv); checked by hand
  with the system Python and torch 2.8.0 (same seed gives the same draws, a different seed differs).
- Test files: `tests/ml/common/test_stage.py` (22), `test_config.py` (11), `test_manifest.py` (5),
  `test_seed.py` (3), `tests/litsense_core/test_mock.py` (44), `test_schemas.py` (5), `test_imports.py` (3,
  including the no-torch subprocess check), `tests/fixtures/tiny/test_make_tiny.py` (8).
- Dependency check: the full core+ml pin set resolves with binary wheels only for manylinux cp311 and cp312
  (`pip install --dry-run --only-binary=:all:`); the core set installs on Windows cp312.
- Packaging: `pip wheel .` builds `litsense-0.1.0` with the pinned core dependencies and an `ml` extra.

### Week 1 gate
- [x] Phase 0 deliverables built: skeleton, pyproject, requirements, config, stage framework and manifests,
  seed utils, schemas, mock, API contract, tiny fixture generator, Makefile, CI workflow
- [x] Tests for every new module pass locally; lint clean
- [x] Decisions logged (D1–D14 in `docs/decisions.md`)
- [ ] CI green on GitHub: the workflow is written and its exact commands pass locally, but the branch has not been
  pushed (waiting for the user's go-ahead)
- [ ] Mock serves the teammate's skeleton app: teammate's step, using `docs/API_CONTRACT.md` §1
- [ ] User reviewed the gate
- Not applicable yet: `--sample`/Colab stage runs, manifests and registry entries, results and figures (no data
  stages exist until Phase 1).

### Open issues
1. Needs user OK: stage fingerprint uses each stage's import closure instead of the repo-wide git SHA (D6);
   proposed `catalog()` contract addition and the mock's other protocol choices (D12); keeping Colab's preinstalled
   torch (D2).
2. For the teammate: exact core pins (pydantic 2.13.5, httpx 0.28.1, numpy 2.4.6) must fit the backend; Python
   3.11; half stars; whether onboarding picks are also stored as ratings (API_CONTRACT §7).
3. Deferred, by design: the tiny serving bundle (Phase 4, with `serve.export`); `make sample` (Phase 1, with
   `data.download`); `dev/dev_server.py` (optional, Phase 4).
4. Verify in Phase 1 against real files: Goodreads values typed as strings, Amazon timestamps in milliseconds, the
   §6 field lists, and that `fasttext-wheel` predicts correctly under numpy 2.

### Next (Phase 1, Week 2)
`data.download`, `data.meta` with `category_counts.csv`, then stop for the user to choose the genre slice;
`data.goodreads`.

## 2026-10-12: Raw sample data and format inspection

### Built
- `local/raw_sample/` (gitignored): first 200,000 lines of the 5-core ratings, 50,000 of `meta_Books`, 50,000
  of the reviews and 5,000 of the Goodreads spoiler file, streamed with `curl | gunzip | head` (no full
  downloads) and stored gzipped under the original names.
- `ml/config.yaml`: `sources.goodreads_base` (UCSD mirror; the Drive links are dead).
- `docs/decisions.md`: D15 (Goodreads source), D16 (sample files), F1–F12 (format findings).

### Tests run
- `pytest tests/ml/common/test_config.py` and the raw-file-name test after the config change: pass.

### Open issues
1. F1: `meta_Books` includes Kindle editions (about 21% of 5-core interactions in the sample) and audiobooks;
   the user decides before `data.canonical` (Week 3) whether to keep them as editions of their works.
2. The Week-2 plan (download stages, framework fixes, `data.meta`, `data.goodreads`) still awaits approval.

## 2026-10-13: Week 2 (Phase 1) up to the genre-slice gate

### Built
- **Stage framework:** `url:<source>` inputs streamed from their URL; `ctx.source()` returns the URL, or the
  local head in `local/raw_sample/` for `--sample` runs (D18).
- **`ml/data/stream.py`:** line streaming with on-the-fly gunzip, Range-resume after dropped connections,
  truncation and size checks, `PartWriter` output checkpoints.
- **`ml/data/parsing.py`:** HTML cleanup, ISBN classification and conversion, years, pages, authors,
  formats and edition kinds, images, Goodreads series suffixes, fast field reads from raw JSON lines.
- **Stages:** `data.download` (5-core CSV, validated, kept on Drive), `data.meta` (streams `meta_Books`,
  writes `books_raw.parquet`, `category_counts.csv`, `genre_slice_proposal.json`, then STOP),
  `data.goodreads` (off by default; when on, keeps only Goodreads rows matching `books_raw` ISBNs).
- **Config:** `data.use_goodreads_graph: false`, `data.stream`, `data.genre_slice_rule`,
  `project.local_raw_root: ./local/raw_sample`; `gdown` removed.
- **Colab:** `make install-colab` (no torch), `notebooks/00_colab_setup.ipynb`, `notebooks/01_data.ipynb`.
- **Fixture:** Kindle, audiobook and other F-cases planted (D23).
- **Decisions:** D17 (Kindle/audio as editions, user), D18–D23.

### Tests run
- `ruff check .`, `ruff format --check .`: clean (notebooks included).
- `pytest -m "not slow"`: **170 passed, 1 skipped** (torch seeding; no torch in the venv). New:
  `tests/ml/data/` (stream 10, parsing 36, download 5, meta 8, goodreads 2, real-sample 3) and the `url:`
  framework tests (2).
- Local sample runs (real data, laptop):
  - `run all --until data.meta --sample 50000`: 37 s. `data.download` fetched and validated the full
    5-core (9,488,297 ratings, 776,370 users, 495,063 items). `data.meta` read the 50,000-line local head:
    7,707 lines matched the 5-core, 155 dropped by the title blocklist, 417 by `min_pages`, 7,135 kept.
    Proposal on this head: Literature & Fiction, Mystery Thriller & Suspense, Teen & Young Adult (not
    meaningful on a head; the full Colab run decides).
  - `run data.goodreads --sample 50000` with the flag off: 0.1 s, no network, empty tables with schema.
  - Smoke test with the flag on (separate output folder): 77 of the first 50,000 Goodreads books matched
    the 7,135-book head, with series names and work titles joined, in 16 s.

### Gate (Week 2): status
- [x] `data.download`, `data.meta` (with `category_counts.csv`), `data.goodreads` built and tested
- [x] `--sample` runs pass locally
- [x] Decisions logged
- [ ] Full run on Colab (`notebooks/01_data.ipynb`)
- [ ] User chooses `data.genre_slices`; recorded in `ml/config.yaml` and `docs/decisions.md`

### Follow-up (same day)
- `books_raw.parquet` and checkpoint parts written with zstd: 24.5 MB to 17.1 MB on the sample, same
  content (D24). `features` checked against `description` and kept: not a duplicate (D24).
- Branch rebased onto `main` after PR #2 merged.

### Open issues
1. After the slice choice: optionally enable `data.use_goodreads_graph` and run `data.goodreads` on Colab.

## 2026-10-13: First Colab run and fixes

### What happened
- `data.download` ran on Colab in 23 s with the same counts as locally. `data.meta` was interrupted by the
  user while streaming (no progress output; Colab's input box looked like a prompt). Colab runs Python 3.13,
  where the `fasttext-wheel` pin had no wheel, so the pinned install could not complete.

### Fixed (D25)
- Progress lines every minute for long streams; notebook guidance about Colab's input box; `< /dev/null`.
- `fasttext-wheel` replaced by `fasttext-predict==0.9.2.4` (cp311-cp313 wheels, no numpy dependency).
- CI matrix gains Python 3.13.

### Tests run
- `pytest -m "not slow"`: 171 passed, 1 skipped (new: `tests/ml/common/test_log.py`).
- Pin resolution with binary wheels only: cp311, cp312, cp313 all resolve (104 packages).

### Next
- Rerun `notebooks/01_data.ipynb` on Colab. Because `data.meta`'s code changed, its fingerprint changed
  and the interrupted run's checkpoints are discarded: the stage starts the stream from the beginning.

