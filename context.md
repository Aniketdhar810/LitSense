# LitSense: session handoff (as of 2026-10-13)

Read this, then `CLAUDE.md` (the brief; §0.5 lists what has changed), `docs/progress.md` (session log) and
`docs/decisions.md` (D1–D30, F1–F12). Repo: https://github.com/Aniketdhar810/LitSense (owner `Aniketdhar810`).

## Where the project stands

- **Phase 0 (Week 1): done.** PR #1 merged. Stage framework (`ml/common/stage.py`, CLI `python -m ml.stages`),
  config (`ml/config.yaml`), `litsense_core/schemas.py` + `litsense_core/mock.py` (full §15.1 API over a
  synthetic 200-book catalogue), `docs/API_CONTRACT.md`, tiny fixture (`tests/fixtures/tiny/make_tiny.py`), CI.
- **Phase 1, Week 2: done except one gate item.** PRs #2, #3, #4 merged. Stages `data.download`, `data.meta`,
  `data.goodreads` (registered in `ml/stages.py`). Full Colab run: 9,488,297 ratings / 776,370 users /
  495,063 items; `data.meta` 446 s, 454,008 books in `books_raw.parquet` (13,408 blocklist + 27,647
  `min_pages` drops). Genre slices chosen and recorded (D26).
- **Week 2 gate: closed** (title lookups reviewed, see `docs/progress.md`).
- **Week 3 part 1: `data.canonical` built (D28–D30): PR #6, branch `claude/litsense-week3-canonical`
  (commit 5d681f7), open, awaiting the user's merge and CI.** Stages now: download, lid, meta, goodreads,
  canonical (`ml/stages.py`). New modules: `ml/data/titles.py`, `genres.py`, `canonical.py`;
  `python -m ml.data.canonical [--sample N]` prints the report. Plan approved with:
  A = merge all editions into works, keep a work if any edition is in a slice; B = results return via
  Google Drive for desktop (or a zip of `results/v0/`), copy only `results/v0/` and the registry entry into
  the repo, never data files; C = `data.lid` keeps `lid.176.ftz` in `data/`.
- **Waiting on the user:** run the Week 3 cells of `notebooks/01_data.ipynb` on Colab (data.download and
  data.meta rerun once, ~8 min, because their code fingerprints changed), then review
  `results/v0/canonical/merge_audit.csv` (fill `verdict` Y/N). Record merge precision in D28. **Do not run
  `data.reviews` before that.** Thresholds (fuzzy 90, subset guard 90, year gap 3) need the user's OK
  after the audit (§0.3).
- **Then (Week 3 part 2):** `data.reviews`, `data.clean` (needs `ml/nlp/aspect_seeds.py` lexicons),
  `data.kcore_split`, `data.eda` (genre mix at every step: slices → works → k-core → sizing), registry hash.

## Latest (2026-10-14)

- Working mode (user, 2026-10-14): act autonomously. Commit, push, open PRs and merge my own PRs once CI
  is green; update context/progress/decisions myself. Stop only for Colab runs (give ONE cell plus the exact
  output to send back) or real decisions. Keep messages short.
- `data.canonical` is done (run 3 accepted; D31-D33). `data.reviews` is built (D34). Next: build
  `data.clean`, `data.kcore_split` and `data.eda`, then ONE Colab cell running `run all --until data.eda`.

## Week 3 implementation notes (for the next session)

- `canonical.py`: `prepare` (keys) → `merge` (union-find `Works`; `isbn`, `page_source`, `goodreads`,
  `title` rules; `NearMisses` = seeded reservoir of 2,000 per reason) → `canonical_records` (list-based,
  fast) → `fill_texts` (batched read of description/features) → report + audit (`merge_audit.csv`,
  utf-8-sig, `verdict` column for the user).
- Pitfall fixed: keep nullable per-edition values (volume, year, pages) as Python `None` in object
  columns; pandas NaN broke `volume_conflict` (NaN != NaN).
- Known trade-off to look for in the audit: the volume guard blocks a title merge between
  "Title (Series, Book 2)" and a marker-less "Title" (reason `volume`).
- Genres: ~35% of in-slice sample interactions are under "Literature & Fiction > Genre Fiction" with no
  deeper level → "Fiction (general)" (D29).
- Notebook generator lives only in the session scratchpad (`make_notebooks.py`); if lost, edit
  `notebooks/01_data.ipynb` JSON directly and keep cells output-free (CI test).
- Laptop scale test: ~457K synthetic editions take ~3 min for prepare+merge+records.

## Data facts that matter (measured; see F1–F12)

- 5-core CSV: `user_id,parent_asin,rating,timestamp`, ms timestamps, grouped by user. meta_Books 4.94 GB,
  reviews 6.22 GB, Goodreads books 2.08 GB / series 28 MB / works 75 MB / spoilers 620 MB (all strings).
- meta_Books contains Kindle (B0… ids, no ISBN fields, `Print length`, sometimes `Page numbers source ISBN`)
  and Audible records; Kindle editions are 70.6% of Mystery and 30.9% of Literature & Fiction interactions.
- Category interactions (full run): Literature & Fiction 2,143,678; Mystery, Thriller & Suspense 2,064,657;
  Teen & YA 383,164; SF & Fantasy 202,160 (2.3%, small); Romance 122,661.
- `features` is usually the publisher blurb (spoiler-safe premise text); `description` mostly editorial
  reviews and author bios. Both kept (D24). `books_raw.parquet` is zstd (~1.2 GB on Drive).

## How to work here

- **Local:** Windows, worktree under `.claude/worktrees/`; venv `.venv` (Python 3.12; no torch).
  `.venv/Scripts/python.exe -m pytest -m "not slow"` (175+ pass), `.venv/Scripts/ruff.exe check .` and
  `format --check .`. `make` is `mingw32-make` (in `/c/MinGW/bin`). `gh` is at
  `/c/Program Files/GitHub CLI/gh.exe` (not on PATH).
- **Sample runs:** `python -m ml.stages run all --until data.meta --sample 50000` (reads `local/raw_sample/`,
  writes `./local/`; both gitignored). `local/raw_sample/` holds heads of the 5-core (200K lines), meta (50K),
  reviews (50K) and Goodreads spoilers (5K).
- **Colab (Python 3.13, user's account):** `notebooks/01_data.ipynb` mounts Drive, clones `BRANCH`
  (default `main`; only merged code reaches Colab), `make install-colab`, runs stages with `< /dev/null`.
  Outputs: `MyDrive/litsense/{data,manifests,work,results}/v0/`. Colab shows an input box under running
  `!` cells: tell the user to ignore it. Long streams print progress lines every minute.
- **Git workflow:** new branch from `origin/main` per piece of work; PRs against `main`; end commits with the
  `Co-Authored-By` line. CodeRabbit reviews PRs; Auto-fix may relay comments. Never commit notebook outputs.
- **Shell quirk:** heredocs in the Bash tool sometimes mangle `\n`, `\r` or apostrophes; write files with the
  Write/Edit tools when content has escapes or quotes.
- **Rules in force:** never fabricate numbers (copy from run output or `results/`); ask before changing the
  evaluation protocol, core dependencies, thresholds after audits, or touching the test split (§0.3).
