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

## 2026-10-13: Full Colab run, genre slices, notebook check

### Full Colab run of `notebooks/01_data.ipynb` (branch `main` at `04c3419`)
- The notebook cloned `main`, which lacked the fix `7b3fc9e`, so the pinned install failed again on
  `fasttext-wheel` and the stages ran on Colab's preinstalled pandas 2.2.3 / pyarrow 23.0.1 / numpy 2.1.3
  (Python 3.13.15). The stages do not depend on those versions; the outputs are valid.
- `data.download`: 23.9 s; 9,488,297 ratings, 776,370 users, 495,063 items.
- `data.meta`: 446 s; 4,448,181 lines read; 495,063 matched; dropped 13,408 (title blocklist) and
  27,647 (`min_pages`); 454,008 books in `books_raw`; no duplicate parent_asins. Category table in D26.
- `data.goodreads`: flag off, empty tables, 0.1 s.

### Built
- Fresh branch `claude/litsense-week2-colab-fixes` from `main` with the Colab fix cherry-picked (D27).
- `tests/test_notebooks.py`: CI fails on notebooks with outputs (D27).
- `data.genre_slices` set (D26); `require_genre_slices` test updated.
- `python -m ml.data.meta --find TITLE ...`: title lookup in `books_raw` (category, edition kind, 5-core
  interactions per edition).

### Tests run
- `ruff check .`, `ruff format --check .`: clean. `pytest -m "not slow"`: 175 passed, 1 skipped (torch).

### Gate (Week 2)
- [x] `data.download`, `data.meta`, `data.goodreads` built, tested, run in full on Colab
- [x] Genre slices chosen by the user and recorded (config, D26)
- [x] Title lookups run on Colab and reviewed (2026-10-13, see below): **Week 2 gate closed**

## 2026-10-13: Week 2 gate closed; Week 3 part 1 (`data.canonical`)

### Week 2 gate: title lookups (`python -m ml.data.meta --find`, full Colab run)
- All eight titles are in `books_raw`. Under rule A (D28), every one has a kept work.
- Editions of one work sit in different categories: Project Hail Mary print in Literature & Fiction, audio
  in SF & Fantasy; Mistborn book 1 and the box set in SF & Fantasy, books 2–6 in Literature & Fiction; The
  Midnight Library has editions in SF & Fantasy, Literature & Fiction, no category and Textbooks.
- The most-rated edition is often Kindle: Gone Girl 8,184 interactions, The Silent Patient 2,217, Circe 743.
- The lookup rows became a regression test of the expected work groupings
  (`test_real_titles_from_the_gate_lookup`).
- User answers (Week 3 plan approved): A = merge all editions, keep a work if any edition is in a slice;
  B = results come back through Google Drive for desktop (or a zip of `results/v0/`), only `results/v0/` and
  the registry entry are copied into the repo; C = keep `lid.176.ftz` in `data/`.

### Built
- `ml/data/titles.py`: matching key, display title, box-set and volume detection, series parsing, author key.
- `ml/data/genres.py`: 26 genre labels plus fallback (D29).
- `ml/data/canonical.py`: stage `data.canonical` (D28). Outputs `asin_to_canonical.parquet`,
  `books_canonical.parquet`, `results/v0/canonical/canonical_report.json` and `merge_audit.csv`; CLI
  `python -m ml.data.canonical` prints the report summary.
- Stage `data.lid` (D30); config `canonical.subtitle_noise_patterns`, `canonical.subset_guard_threshold`,
  more edition-noise terms; `notebooks/01_data.ipynb` Week 3 cells.

### Tests run
- `ruff check .`, `ruff format --check .`: clean. `pytest -m "not slow"`: 225 passed, 1 skipped (torch).
  New: titles 33, genres 3, canonical 12 (§16.2 cases, the subset trap, other authors, Kindle page-source
  ISBN, Goodreads works, the gate-lookup titles, the tiny fixture end to end with Goodreads off and on, a
  seeded audit), `data.lid` 2.
- Local sample `run all --until data.canonical --sample 50000`: download 41.6 s, lid 1.2 s (the real model
  passed the magic-number check), meta 15.5 s, goodreads 0.1 s, canonical 2.2 s. On the 7,135-edition
  head: 7,108 works (27 with 2+ editions, all `title` merges), 36 box sets, 2,612 kept works with 69,687
  interactions. All 12 kept multi-edition works in the sample audit are correct on inspection; the head is
  too small to measure precision (the first 50K meta lines rarely hold two editions of a work).
- Scale check (laptop, sample head copied 64 times = 456,640 editions with heavy duplication): prepare
  38 s, merge 103 s, records 39 s. That run produced 3.7M near-miss pairs, so near misses are now kept in a
  bounded reservoir per reason.
- Bug found and fixed: pandas stored missing volume numbers as NaN, and NaN != NaN made the volume guard
  block every merge in frames that mixed known and unknown volumes. Volumes now stay `None`, with a
  regression test.

### Gate (Week 3, part 1): status
- [x] `data.canonical` built and tested; `--sample` run passes locally
- [x] Decisions logged (D28–D30)
- [ ] Full run on Colab (`notebooks/01_data.ipynb`, Week 3 cells)
- [ ] User reviews `results/v0/canonical/merge_audit.csv`; merge precision recorded in D28
- Then: `data.reviews`, `data.clean`, `data.kcore_split`, `data.eda`, registry hash.


## 2026-10-14: `data.canonical` full Colab run and merge audit (first pass)

### Full Colab run (main after PR #6; output pasted by the user)
- Upstream stages were up to date; `data.canonical` took 193.9 s.
- 454,008 editions became 383,360 works (50,383 with 2 or more editions); 2,948 box sets.
- Merge edges: isbn 705, page_source 2,370, title 67,573.
- Kept 131,081 works, 170,864 editions, 4,497,594 interactions. Rule A rescued 4,071 editions and
  111,155 interactions.
- Kept works by primary slice:
  - Literature & Fiction: 77,539 works, 2,202,974 interactions.
  - Mystery, Thriller & Suspense: 42,862 works, 2,079,937 interactions.
  - Science Fiction & Fantasy: 10,680 works, 214,683 interactions.
- Near misses by reason: below_threshold 38,325, subset_guard 35,644, volume 8,784, year 3,448.

### Audit (334 rows)
- My first pass, for the user to confirm: 97/100 merged works correct; 44/50 not-merged pairs correct.
- Wrong merges:
  - G018: shared-ISBN data error (fixed, D31).
  - G054: sequel at `token_sort` exactly 90.
  - G004: a two-in-one "Animal Farm: 1984" edition (debatable).
- All 6 wrong non-merges were blocked by the subset guard (same work, subtitle on one side only).

### Built
- D31: author guard on shared-ISBN merges, plus a regression test. `pytest -m "not slow"`: 226 passed,
  1 skipped (torch).
- `context.md` handoff notes carried over (their commit had missed PR #6).

### Next
- The user returns the edited verdicts; rerun `data.canonical` on Colab; record precision in D28/D31; then
  `data.reviews`.

## 2026-10-14: Subtitle rule (D32), audit recorded

- `title_head` (titles.py), `subtitle_unions` plus blurb readers (canonical.py), `assign_works` (shared
  glue), a `subtitle_merge` audit section and a `subtitle_rule` report block. Config:
  `canonical.subtitle_description_threshold: 70`, `subtitle_description_chars: 500`.
- Tests: Cold Hand in Mine merges; Mistborn vs Mistborn: The Final Empire with different blurbs does not;
  guards for other author, empty description, volume and box set; audit section; `title_head`.
  `pytest -m "not slow"`: 234 passed, 1 skipped (torch).
- Sample run (50K head): subtitle rule: 6 candidates, 2 merged, 3 dissimilar blurbs, 1 blocked by the
  volume or work guard.
- Run-1 audit verdicts (model-drafted, spot-checked by the user) recorded in D31 and committed as
  `results/v0/canonical/merge_audit_run1_reviewed.csv`.
- Next: merge PR #7 once CI is green; the user reruns `data.canonical` on Colab (one cell), and the
  `subtitle_merge` section is reviewed the same way; then `data.reviews`.

## 2026-10-14: Run 2 of `data.canonical`, number guard (D33)

- Colab run 2 (after PR #7): 454,008 editions became 381,949 works; merge edges isbn 513, page_source
  2,370, subtitle 1,603, title 67,573; kept 130,555 works and 4,499,855 interactions; 225 s.
- Subtitle audit: 30/30 correct (model-drafted). Found NYPD Red 2/3/4/6 merged into one work by the title
  rule; fixed with D33 plus a test. `pytest -m "not slow"`: 235 passed, 1 skipped.
- Next: run 3 on Colab; check that the largest kept works are clean; then `data.reviews`.

## 2026-10-14: `data.canonical` run 3 accepted; `data.reviews` built

- Run 3 (D33 number guard): 384,579 works, 130,960 kept, 4,499,455 interactions; largest works clean.
- `ml/data/reviews.py` (stage `data.reviews`, D34) with tests: text hash, the seconds-timestamp guard, the
  two-step dedup, and the tiny fixture end to end. `pytest -m "not slow"`: 239 passed, 1 skipped.
- Local sample: 194 reviews matched (the sample heads of meta and reviews barely overlap, D16).
- Next: `data.clean`, `data.kcore_split`, `data.eda`; then one Colab run of the remaining Week 3 stages.

## 2026-10-14: `data.clean`, `data.kcore_split`, `data.eda`

- Built `ml/data/clean.py` (D35), `ml/nlp/aspect_seeds.py`, `ml/data/kcore.py` + `ml/data/split.py`
  (stage `data.kcore_split`, D36) and `ml/data/eda.py` (stage `data.eda`). `01_data.ipynb` gets one
  cell for all of them.
- Tests: cleaning rules (with a stub language ID), k-core invariants, sizing reproducibility, split
  chronology, one interaction per work with `has_text`, the tiny fixture end to end through `data.eda`,
  and `id_map.json` stable across reruns. `pytest -m "not slow"`: 254 passed, 1 skipped.
- Local sample: `data.clean` on the 194 sample reviews kept 183 as `text_ok`; 11 were too short.
- Next: ONE Colab run of `run all --until data.eda`; then record sizing f, split stats and the slice
  mix, and close the Week 3 gate (registry hash).

## 2026-10-14: Colab `data.reviews` failure, checkpoint fix (D37)

- A first Colab run streamed all reviews and stopped; the rerun failed because Drive had lost
  `part-00089.parquet` while keeping `progress.json`. `PartWriter` now validates parts, rolls back to the
  last good one, and marks finished streams. Tests: lost part, old-format checkpoint, finished stream.
  `pytest -m "not slow"`: 257 passed, 1 skipped.

## 2026-10-14: Week 3 gate

- Colab: every stage through `data.eda` complete. `results/v0/` (canonical audit and report, EDA
  summary, attrition table and figures) is committed; `models/registry.json` holds the v0 file sha256s,
  fingerprint and sizing.
- The notebook's data cell now ends with `drive.flush_and_unmount()` (D38).

### Gate (Week 3)
- [x] All Phase 1 stages run with `--sample` locally and in full on Colab; manifests complete
- [x] Registry updated (`models/registry.json`, v0)
- [x] Tests pass; CI green
- [x] Results committed (`results/v0/canonical`, `results/v0/eda`)
- [x] Decisions logged (D28-D38); merge audit recorded (model-drafted, spot-checked by the user)
- [ ] User reviews the gate, including the SF & Fantasy share after sizing (`summary.json` `slice_mix`)

## 2026-10-14: Week 4 harness and baselines built

- `ml/eval/` (data, metrics, evaluator, strata, bootstrap, report, lock), `ml/models/baselines.py`
  (random, popularity, ItemKNN, ALS, content-only), `ml/nlp/embed.py` + `ml/nlp/content.py` (stage
  `nlp.content`), stage `eval.baselines_val`, `notebooks/02_eval_baselines.ipynb`. Decisions D39 (v1
  open items) and D40.
- Tests: hand-computed metrics, masking, degenerate scores, cutoff mismatch, bootstrap, strata, test
  lock, baseline shapes and cold items, content text builder (no review text), the tiny fixture
  through `eval.baselines_val`. `pytest -m "not slow"`: 273 passed, 1 skipped.
- Next: one Colab run (`run all --until eval.baselines_val`); commit `results/v0/val` and
  `results/v0/content`; the Week 4 gate.

## 2026-10-15: Week 4 gate (validation baselines)

- Colab: `data.clean` and `data.kcore_split` rebuilt the output Drive had lost (D42). `data.eda` stayed
  up to date, so the rebuilt v0 files have the same sha256 as `models/registry.json`. `nlp.content` and
  `eval.baselines_val` complete.
- Committed `results/v0/val/`:
  - `table_main.csv`, `table_strata.csv`, `table_longtail.csv`, `bootstrap.csv`;
  - per-model JSON and per-user rows;
  - two figures.
  Also `results/v0/content/` (anisotropy check, neighbour samples).
- Read from those files: ItemKNN, content-only and ALS all beat popularity on NDCG@10 and Recall@20
  (paired CIs exclude zero). Content-only is the only baseline that scores cold items and the strongest
  on long-tail truth. Popularity recommends no long-tail items. Mean pairwise content cosine is below
  the anisotropy warning. Exact values are in `results/v0/val/` and `results/v0/content/`.

### Gate (Week 4)
- [x] Metric unit tests pass (hand-computed case, masks, degenerate scores, bootstrap, lock)
- [x] Random and popularity (and ItemKNN, ALS, content-only) evaluated end to end on val, with strata and CIs
- [x] Results committed; decisions logged (D39-D42); CI green
- [ ] User reviews the gate

## 2026-10-15: Handoff (state at the end of Week 4)

- Weeks 0-4 done; PRs #1-#18 merged; decisions D1-D42. Week 4 gate criteria met (user review pending).
- Strongest baseline on val: ItemKNN (`results/v0/val/table_main.csv`).
- Open v1 items (D39): slice-aware sizing, `kcore.item` 5 vs 10, catalogue size (10-20K target), and a
  serving-catalogue proposal.
- `notebooks/make_notebooks.py` (the notebook generator) is now in the repo; `context.md` and CLAUDE.md
  §0.5 are updated. Next: Week 5 (LightGCN, projection hybrid and controls, MVP `recommend`, bundle v0).


## 2026-10-15: Week 4 gate approved; Week 5 part 1 (LightGCN)

- The user approved the Week 4 gate.
- Built `ml/models/lightgcn.py` (stage `models.lightgcn_val`), `ml/models/embedding_scorers.py`
  (`lightgcn_raw`, `lightgcn_norm`), and renamed the evaluation stage to `eval.val`, which now also
  evaluates LightGCN and compares every model with both LightGCN variants (D43). CI installs CPU
  torch; `02_eval_baselines.ipynb` runs `--until eval.val` on a GPU runtime.
- Tests (`tests/ml/models/test_lightgcn.py`): the sampler never returns a positive, normalized
  adjacency with zero rows for isolated nodes, the inner split, LightGCN beats random on training
  NDCG, seeded training and early stopping, -inf for zero-degree items in both scorers, artifact
  leakage checks; the tiny fixture runs through `eval.val`. `pytest -m "not slow"`: 280 passed,
  2 skipped (local raw samples absent in this worktree).
- Next: one Colab run (`run all --until eval.val`, GPU); then the Week 5 gate check (LightGCN vs
  ALS on val), the projection hybrid and its controls.

## 2026-10-16: LightGCN on Colab (Week 5, part 1 results)

- Colab T4 run of `run all --until eval.val` (main after PR #20): data stages and `nlp.content` up to
  date; `models.lightgcn_val` 234 s (best inner epoch 275); `eval.val` 23 s, 2,578 evaluated users,
  4,190 items.
- Committed `results/v0/val/` (now with `lightgcn_raw` and `lightgcn_norm`, paired bootstraps
  against both) and `results/v0/training/lightgcn_val.csv`.
- From those files: LightGCN raw beats ALS on NDCG@10 (paired 95% CI excludes zero; Recall@20 CI
  crosses zero) and ties ItemKNN; normalized LightGCN trades all-user accuracy for long-tail accuracy
  and coverage. Details and reasoning in D44.

### Gate (Week 5): status
- [x] LightGCN raw and normalized on val with strata and CIs; LightGCN >= ALS (NDCG@10)
- [ ] Projection hybrid, projection-shuffled and static-fusion controls on val
- [ ] MVP blend `recommend` on real artifacts; bundle export v0
- [ ] User reviews the gate

## 2026-10-16: Projection hybrid and controls built (Week 5, part 2)

- Built `ml/models/projection.py` (stage `models.projection_val`: projection + shuffled control),
  `StaticFusionScorer`, `pca` and `check_artifact` in `ml/models/embedding_scorers.py`; `eval.val`
  now also evaluates `projection`, `projection_shuffled` and `static_fusion_{0.5,0.7,0.9}` (D45).
- Tests (`tests/ml/models/test_projection.py`): same normalization in training and scoring, cold
  items represented by W c only, raw vs normalized items, seeding, loss decreases, cold items scored,
  static fusion (alpha 1 ranks like normalized LightGCN; cold items scored by content; cutoff check);
  the tiny fixture runs through `eval.val` with every model. `pytest -m "not slow"`: 284 passed,
  2 skipped (local raw samples absent).
- Next: one Colab run (`run all --until eval.val`, T4); then the Week 5 projection rows.

## 2026-10-16: Projection results (Week 5, part 2)

- Colab T4 run (main after PR #23): `models.projection_val` 75 s, `eval.val` 35 s; LightGCN reran with
  identical metrics. Committed `results/v0/val/` (projection, shuffled control, static fusion rows) and
  the refreshed `results/v0/training/lightgcn_val.csv`.
- From those files: the projection does not differ from its shuffled control, and is below raw
  LightGCN on all users; static fusion (alpha 0.7) is the strongest fusion row; no fusion recovers
  cold items. Reported as a null result in D46.

### Gate (Week 5): status
- [x] Val results with strata and CIs for random, popularity, ItemKNN, ALS, content-only, LightGCN raw
  and normalized, projection, projection-shuffled, static fusion
- [x] LightGCN >= ALS on val (NDCG@10, D44)
- [ ] MVP blend `recommend` on real artifacts; bundle export v0; recommendations visible in the browser
- [ ] User reviews the gate

## 2026-10-17: MVP recommend and bundle v0 (Week 5, part 3)

- The user kept the projection as is (D46) and asked for the MVP and bundle.
- Built: stage `models.lightgcn_serve`; `ml/serve/export.py` (stage `serve.export`: bundle, tarball,
  `results/<v>/serve/bundle_report.json` with size and RSS); `litsense_core` `store`, `vector_index`,
  `profile`, `candidates`, `rank` (blend), `postprocess` (MMR, genre cap, exploration slot),
  `reasons`, `recommend` (with `continue_series`, `catalog`, `health`, `download_bundle`);
  `dev/dev_server.py` (standard library viewer); API contract section 1.1 (D47).
- Tests: `tests/litsense_core/test_recommend.py` over a synthetic bundle written by the real writer
  (store shape checks, k unique unseen items, filters and muted genres, series guard and
  `continue_series`, cold paths, exact vs pseudo CF, invalid ids, author reason, catalog, helpers);
  the tiny fixture through `serve.export` and `recommend`. Dev server smoke-tested on the synthetic
  bundle. `pytest -m "not slow"`: 293 passed, 2 skipped.
- Next: one Colab run (`run all --until serve.export`), then measure the real bundle (size, RSS),
  view it with the dev server, and close the Week 5 gate.
