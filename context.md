# LitSense: session handoff (as of 2026-10-15)

Read this, then `CLAUDE.md` (the brief; §0.5 lists what overrides it), `docs/progress.md` (session log) and
`docs/decisions.md` (D1–D42, F1–F12). Repo: https://github.com/Aniketdhar810/LitSense (owner `Aniketdhar810`).

## Where the project stands

- **Weeks 0–4 are done; every PR is merged (#1–#18).** `main` holds everything.
- **Week 1 (Phase 0):** stage framework (`ml/common/stage.py`, CLI `python -m ml.stages`), config,
  `litsense_core/schemas.py` + `mock.py`, `docs/API_CONTRACT.md`, tiny fixture, CI.
- **Weeks 2–3 (Phase 1, data):** stages `data.download`, `data.lid`, `data.meta`, `data.goodreads` (off by
  default), `data.canonical`, `data.reviews`, `data.clean`, `data.kcore_split`, `data.eda`. The v0 dataset
  is frozen; hashes, fingerprint and sizing fraction are in `models/registry.json`, stats in
  `results/v0/eda/` and `results/v0/canonical/`. The merge audit is recorded as model-drafted verdicts,
  spot-checked by the user (D31–D33).
- **Week 4 (Phase 2, harness and baselines):**
  - `ml/eval/`: data, metrics, evaluator, strata, bootstrap, report, lock, errors.
  - `ml/models/baselines.py`: random, popularity, ItemKNN, ALS, content-only.
  - `ml/nlp/embed.py` and `ml/nlp/content.py`.
  - Stages `nlp.content` and `eval.val` (formerly `eval.baselines_val`). Results are in `results/v0/val/` (tables with 95% CIs,
    strata, long tail, paired bootstrap vs popularity, figures).
- **Strongest baseline so far: ItemKNN** on NDCG@10 and Recall@20 (see `results/v0/val/table_main.csv`).
  Content-only is strongest on long-tail and cold-item truth; popularity recommends no long-tail items.
- **Week 4 gate:** approved by the user (2026-10-15).
- **Week 5 in progress:** LightGCN done and run on Colab (D43, D44): it beats ALS on val NDCG@10 and ties
  ItemKNN. Projection hybrid + controls built (D45), waiting for a Colab run; then the MVP `recommend`
  and bundle v0.
- **Next step: Week 5** (§17):
  - LightGCN, raw and normalized (§9.4), with artifacts and manifests in `artifacts/v0/eval_val/`;
  - the projection hybrid with shuffled and static-fusion controls (§9.5);
  - the MVP blend `recommend` on real artifacts (§12), and bundle export v0.
  The gate: LightGCN >= ALS on val, or a logged investigation.

## Open items for the v1 rebuild (D39; resolve before Week 11)

1. **Slice-aware sizing:** sample users so each genre slice keeps roughly its pre-sizing share (SF & Fantasy
   shrinks sharply in v0, see `results/v0/eda/summary.json` -> `slice_mix`).
2. **Book threshold:** compare `kcore.item` 5 vs 10. For each, report catalogue size, long-tail share and
   evaluable test users; the user chooses.
3. **Catalogue size:** v0 has 4,190 books; target roughly 10–20K in v1. Report what each sizing option gives.
4. **Serving catalogue vs training set:** propose, do not build, serving a larger catalogue (e.g. all
   in-slice works with >= N reviews) scored by content and review features, with its cost.

## Working mode (user, 2026-10-14)

- Act autonomously: write code and tests, commit, push, open PRs, merge your own PRs once CI is green
  (`gh pr merge --merge`), and update context/progress/decisions yourself.
- Stop only for a Colab run or a real decision. For Colab, give ONE cell to paste and say exactly what to send
  back. Keep messages short and simple.
- Audit verdicts drafted by Claude are labelled "model-drafted, spot-checked by the user".

## Colab recipe (every run)

```bash
from google.colab import drive; drive.mount("/content/drive")
%cd /content
!rm -rf LitSense && git clone -q --depth 1 https://github.com/Aniketdhar810/LitSense.git
%cd /content/LitSense
!make install-colab > /dev/null && python -m ml.stages run all --until <STAGE> < /dev/null > /content/drive/MyDrive/litsense/run.log 2>&1; grep -vE "MB/s|lines," /content/drive/MyDrive/litsense/run.log | tail -40
import shutil; shutil.make_archive("/content/drive/MyDrive/litsense/results_v0", "zip", "/content/drive/MyDrive/litsense/results/v0")
drive.flush_and_unmount(); print("saved to Drive")
```

- Always end with `drive.flush_and_unmount()`: Drive lost outputs twice when sessions ended (D37, D38).
- The user downloads `results_v0.zip` into `C:\Users\KIIT0001\Downloads`; unzip it into `results/v0/` and
  commit. Never commit data files.
- Stages are skipped when up to date: the fingerprint covers config, inputs and code import closure; output
  sizes must match the manifest (D42). **Do not edit modules imported by finished stages** (e.g.
  `ml/common/errors.py`, `ml/data/stream.py`) unless a rerun is intended; put new shared code in new
  modules (D41). A data rerun streams 6 GB of reviews again (about an hour).

## How to work locally

- Windows worktree under `.claude/worktrees/`; venv `.venv` (Python 3.12, no torch; scipy, implicit,
  matplotlib installed). `.venv/Scripts/python.exe -m pytest -m "not slow"` (275 passed, 1 skipped);
  `.venv/Scripts/ruff.exe check .` and `format --check .`. `gh` is `/c/Program Files/GitHub CLI/gh.exe`.
- New branch from `origin/main` per piece of work; end commits with the `Co-Authored-By` line. CodeRabbit
  reviews PRs; address real findings in a follow-up PR.
- Notebooks are generated by `notebooks/make_notebooks.py` (`python notebooks/make_notebooks.py notebooks`)
  and must stay output-free (CI test).
- Heredocs in the Bash tool mangle `\n` and some escapes; use the Edit/Write tools for code with escapes.
- Tests that need fastText or MiniLM inject stubs (`clean.load_language_id`, `embed.load_encoder`).
- Shared test fixtures `tiny_cfg`/`file_server` live in `tests/ml/data/conftest.py` (re-exported for eval).

## Data facts that matter (measured; see F1–F12 and `results/v0/`)

- 5-core: 9,488,297 ratings, 776,370 users, 495,063 items; timestamps in ms. meta_Books contains Kindle and
  audio editions (merged as editions, D17). `features` is usually the publisher blurb; `description` mostly
  editorial reviews (D24).
- Canonical run 3: 454,008 editions -> 384,579 works; 130,960 kept works in the three slices.
- v0: 109,454 interactions, 10,987 users, 4,190 books; 103,140 interactions with usable review text.
- fastText batch `predict` is broken in fasttext-predict 0.9.2.4: score one text at a time (D35).
