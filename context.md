# LitSense: session handoff (as of 2026-10-13)

Read this, then `CLAUDE.md` (the brief; §0.5 lists what has changed), `docs/progress.md` (session log) and
`docs/decisions.md` (D1–D27, F1–F12). Repo: https://github.com/Aniketdhar810/LitSense (owner `Aniketdhar810`).

## Where the project stands

- **Phase 0 (Week 1): done.** PR #1 merged. Stage framework (`ml/common/stage.py`, CLI `python -m ml.stages`),
  config (`ml/config.yaml`), `litsense_core/schemas.py` + `litsense_core/mock.py` (full §15.1 API over a
  synthetic 200-book catalogue), `docs/API_CONTRACT.md`, tiny fixture (`tests/fixtures/tiny/make_tiny.py`), CI.
- **Phase 1, Week 2: done except one gate item.** PRs #2, #3, #4 merged. Stages `data.download`, `data.meta`,
  `data.goodreads` (registered in `ml/stages.py`). Full Colab run: 9,488,297 ratings / 776,370 users /
  495,063 items; `data.meta` 446 s, 454,008 books in `books_raw.parquet` (13,408 blocklist + 27,647
  `min_pages` drops). Genre slices chosen and recorded (D26).
- **Open gate item:** the user runs, on Colab with `main` freshly cloned:
  `python -m ml.data.meta --find "Project Hail Mary" "The Martian" "Mistborn" "Circe" "Gone Girl" "The Silent Patient" "The Midnight Library" "Where the Crawdads Sing"`
  and pastes the output. Review which category each title landed in and its interactions (concern:
  Amazon's `categories[1]` is not always the genre, e.g. The Handmaid's Tale → Politics & Social Sciences).
  Then tick the gate in `docs/progress.md`.
- **Next: Week 3 plan (shown to the user, NOT yet approved).** `data.canonical`, `data.reviews`, `data.clean`,
  `data.kcore_split`, `data.eda` (EDA must report the genre mix at every step: slices → works → k-core →
  sizing, to show how much SF & Fantasy survives). Waiting on user answers:
  - A. Merge all 454,008 books into works first, then keep a work if any edition is in a slice (instead of
    §7.4's slice-before-merge, which would lose Kindle editions filed under another category)?
  - B. How results reach the repo: the user downloads `MyDrive/litsense/results/v0/` and gives the path; Claude
    commits it (no Colab "Save in GitHub").
  - C. `data.download` also keeps `lid.176.ftz` (0.9 MB) in `data/`.
  Do not start Week 3 before approval.

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
