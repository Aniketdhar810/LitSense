# Decisions

Each entry: date, decision, reason, alternatives considered. Items marked **(needs user OK)** are defaults
taken to keep Phase 0 moving; they are easy to reverse.

## 2026-10-12: Phase 0

### D1. One set of pins for Python 3.11 and 3.12
- **Decision:** pin `numpy==2.4.6` and `scipy==1.17.1`; CI tests Python 3.11 and 3.12.
- **Reason:** CLAUDE.md fixes Python 3.11 (the backend runs the core), while Colab runs 3.12. numpy 2.5 and
  scipy 1.18 require Python ≥ 3.12, so the newest versions both interpreters share are 2.4.x and 1.17.x.
  Verified with `pip install --dry-run --only-binary=:all:` for manylinux cp311 and cp312 (all 109 packages
  resolve), and an actual install on Windows cp312 for the core set.
- **Alternatives:** move everything to 3.12 (would contradict the brief and the backend's Python); separate pin
  files per interpreter (two sets of truth).

### D2. Version pins: conservative where code exists, provisional elsewhere
- **Decision:** pins are the latest releases at least two weeks old that resolve together, except
  `pandas==2.3.3` (last 2.x) instead of 3.0. Pins for packages not imported until later phases (torch,
  sentence-transformers 6.x, transformers 5.x, spaCy, implicit, onnx, fastText) are provisional and get
  re-verified in the phase that first imports them. `optimum` is left out until Phase 3 picks the ONNX export
  path.
- **Reason:** pandas 3.0 changes defaults (copy-on-write, string dtype) and Colab images tend to ship 2.x; the
  rest has no code yet, so a pin cannot be tested.
- **Alternatives:** pandas 3.0.6 (newer but riskier); leaving later packages unpinned (violates rule 0.2.10).
- **Open:** pinning `torch` will make Colab download a multi-GB CUDA wheel if its preinstalled torch differs.
  Proposal for the Phase 1 Colab setup notebook: install `requirements-ml.txt` without torch, keep Colab's torch
  and record its version in manifests. **(needs user OK)**

### D3. fastText from `fasttext-wheel`
- **Decision:** pin `fasttext-wheel==0.9.2`.
- **Reason:** the official `fasttext` package ships no wheels (needs a C++ build) and its `predict()` breaks under
  numpy 2 (`np.array(..., copy=False)`). `fasttext-wheel` has cp311/cp312 wheels for Linux and Windows; Phase 1
  will call the low-level predict (or confirm the wrapper works) when `data.clean` is built.
- **Alternatives:** `fasttext-numpy2-wheel` (Linux only, so no local sample runs on Windows); a different
  language-ID library (contradicts CLAUDE.md §6.3).

### D4. Quoted regexes in `ml/config.yaml`
- **Decision:** the six list entries containing `?` (three `shipping_patterns`, three `boxset_patterns`) are
  single-quoted. Values are unchanged.
- **Reason:** the §5 YAML does not parse: PyYAML ends a plain scalar inside a flow sequence at `?`. Single quotes
  keep backslashes literal. A test checks the patterns still match their intended text.
- **Alternatives:** block-style lists (bigger diff from the brief).

### D5. New config key `project.local_raw_root: ./local/raw`
- **Decision:** `--sample` runs read raw files from `project.local_raw_root` and write under
  `project.local_root`; full runs use `raw_root` and `drive_root`. Relative paths resolve from the repo root.
- **Reason:** §7.0 says `--sample` writes under `local_root` but names no local raw location; `/content/raw` does
  not exist on a laptop. How files get there (download only the first N lines, or `make sample`) is decided in
  Phase 1 with `data.download`.
- **Alternatives:** CLI flags for every root (more surface); deriving it from `local_root` (implicit).

### D6. Stage fingerprint uses each stage's own code, not the repo-wide git SHA **(needs user OK)**
- **Decision:** the code part of a fingerprint is the sha256 of the stage's module plus, transitively, every
  `ml`/`litsense_core` module it imports (found by parsing imports), excluding framework plumbing
  (`ml.common.stage`, `ml.common.manifest`, `ml.common.log`, `ml.stages`). Line endings are normalized so Windows
  and Colab agree. The git SHA (with `-dirty`) is still recorded in every manifest for provenance.
- **Reason:** §7.0 puts the git SHA of `ml/` + `litsense_core/` into every fingerprint, so any commit (even to the
  evaluator or the LLM client) would invalidate every stage and rerun the multi-hour 29.5M-line review stream.
  The import closure reruns exactly the stages whose code changed, with no manual declarations to forget.
- **Alternatives:** the spec's git SHA (safe but reruns everything); manually declared code dependencies (easy to
  forget).

### D7. Input identity is the producer's recorded output hash
- **Decision:** an input produced by another stage is identified by the sha256 that stage's manifest recorded
  for that file; inputs no stage produces (raw files for now) by size + mtime.
- **Reason:** content addressing means a rerun that writes identical bytes does not cascade downstream, while
  changed output always does. Using the producer's fingerprint (the spec's wording) would also cascade on
  no-op reruns. Raw files on Colab's ephemeral disk get a new mtime every session; once `data.download` (Phase 1)
  declares them as outputs, their identity becomes the recorded sha256 and session restarts stop invalidating
  downstream stages.
- **Alternatives:** producer fingerprints (more reruns); hashing raw files on every run (minutes per GB).

### D8. Staging directory instead of `<path>.tmp`; manifest location
- **Decision:** `ctx.out(spec)` returns a path in a per-stage staging directory on the same filesystem as the
  destination; after the stage returns, outputs are moved into place with `os.replace` and the manifest is written
  last to `<root>/manifests/<data_version>/<stage>.json`. A started run deletes the stage's old manifest first; a
  failed run leaves none.
- **Reason:** same atomicity as `<path>.tmp`, but file names keep their extensions (`np.save("x.npy.tmp")` would
  write `x.npy.tmp.npy`). Several stages write into the same `data/<version>/` folder, so manifests are per stage.
- **Alternatives:** `.tmp` suffix (breaks `np.save` and some writers); one manifest per output directory (clashes).

### D9. Framework files and declared access
- **Decision:** the stage framework lives in `ml/common/stage.py` and config loading/validation in
  `ml/common/config.py` (both new files beside the §4.3 layout); `ml/stages.py` is only the CLI. `ctx.path`,
  `ctx.out` and `ctx.param` raise if a stage touches an input, output or config key it did not declare.
- **Reason:** stage modules import the decorator while the CLI imports the stage modules, so they cannot share one
  file without circular imports. Enforcing declarations guarantees the fingerprint covers everything a stage
  reads.
- **Alternatives:** framework inside `ml/stages.py` (circular imports; `python -m` double-import of the registry).

### D10. Schema deviations from §15.2 (identical JSON)
- **Decision:** `HomeRails.because_you_liked: list[BecauseYouLiked]`, `Clarification.options:
  list[ClarificationOption]`, `SessionState.history: list[ChatMessage]` instead of `list[dict]`. Added
  validation: ms timestamps in [1e11, 1e13], stars in [1, 5], `book_id ≥ 0`; `Bucket` and `Intent` literals;
  `default_factory` for mutable defaults; `ASPECT_LABELS` (aspect → UI label, §11.2 table). `Evidence` keeps its
  four `dict` fields until Phase 4 fixes their contents.
- **Reason:** typed models produce better OpenAPI types for the frontend at no JSON cost; validation catches
  backend unit bugs at the boundary. Tests check `ASPECT_LABELS` and `Bucket` against the config.
- **Alternatives:** the literal §15.2 types.

### D11. Bucket membership with fractional `n`
- **Decision:** a reader is in the last bucket whose lower bound is ≤ `n` (cold 0, warming 1, blending 5, warm 20).
- **Reason:** `n` = ratings + 0.5 × list adds can fall between `[1, 4]` and `[5, 19]` (for example 4.5).
- **Alternatives:** `floor(n)` (same results for these bounds, less explicit).

### D12. Mock behaviour choices **(needs user OK where marked)**
- **Decision:** (a) onboarding picks count as seen and are never recommended back; (b) `metrics_report()`
  returns `null` metrics plus `"is_mock": true`, not zeros; (c) chip removal = backend edits the session and
  sends an empty message; a picked clarification option is sent back as the message; (d) the mock catalogue is
  synthetic (invented titles and authors, no real books) and generated from a seeded `random.Random`;
  (e) **proposed** `catalog() -> list[CatalogBook]` so the backend can seed its `books` table. **(needs user OK)**
- **Reason:** (a) picks are books the reader says they enjoyed; (b) rule 0.2.4, zeros would read as results;
  (c) §14.7 says the backend removes a modifier and the core recomputes, but names no call; (d) real titles with
  invented descriptions would be misleading; (e) nothing in §15.1 hands the backend the catalogue.
- **Alternatives:** (e) a CSV export inside the bundle.

### D13. Tiny fixture scope
- **Decision:** Phase 0's `make_tiny.py` generates the raw dataset (5-core CSV, `meta_Books`, reviews, four
  Goodreads files) plus `truth.json` with every planted case. The tiny serving bundle (§16.1, second half) is
  added in Phase 4.
- **Reason:** the bundle's format is defined by `serve.export` (Phase 4) and its feature list by `features.py`
  (Phase 5). Writing it now means guessing both and maintaining the guess twice.
- **Assumption to verify in Phase 1:** Goodreads values are strings (`"num_pages": "352"`) and Amazon timestamps
  are milliseconds, as CLAUDE.md §6 says from memory.

### D14. CI scope
- **Decision:** CI installs `requirements-core.txt` plus the subset of `requirements-ml.txt` the fast tests need
  (`CI_ML_PACKAGES` in the Makefile, currently pandas, pyarrow, pytest, ruff), runs `ruff check`,
  `ruff format --check` and `pytest -m "not slow"` on Python 3.11 and 3.12. Markdown is excluded from ruff (it
  would reformat code blocks in CLAUDE.md).
- **Reason:** the full ML stack (torch with CUDA from PyPI) would add gigabytes per run for tests that do not
  import it yet. The subset grows when fast tests need more (the LightGCN tiny test will add CPU torch).
- **Alternatives:** a third requirements file for CI (another source of pins).

## 2026-10-12: Raw data inspection (`local/raw_sample/`)

Numbers below are measured on the local samples (first N lines of each file), not on full files; they show
what the parsers must handle, not final statistics. Full-file numbers come from the Colab runs.

### D15. Goodreads files come from the UCSD mirror
- **Decision:** new config key `sources.goodreads_base:
  https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads`; a file's URL is `<base>/<file name>`.
- **Reason:** every Google Drive link on the dataset page (and the Drive file pages) returns 404. The mirror
  serves all files over plain HTTPS with Range support: books 2.08 GB, series 28 MB, works 75 MB, spoilers
  620 MB. `gdown` is therefore unnecessary; removing it from `requirements-ml.txt` is left for the Week-2
  download work.
- **Alternatives:** asking the user to upload the files to Drive by hand (CLAUDE.md §6.2 fallback).

### D16. Local sample files
- **Decision:** `local/raw_sample/` (gitignored by the existing `local/` rule) holds the first lines of four
  files, streamed with `curl | gunzip | head` and re-gzipped with `gzip -n` under the original file names so
  they can stand in for raw files: `Books.csv.gz` (200,000 lines = header + 199,999 ratings),
  `meta_Books.jsonl.gz` (50,000), `Books.jsonl.gz` (50,000 reviews), `goodreads_reviews_spoiler.json.gz` (5,000).
- **Finding that shapes `--sample` runs:** the files are not mutually consistent. The 5-core CSV and the
  review file are both grouped by user in the same user order, so 60.2% of sampled review pairs are in the
  ratings sample. The meta file is ordered differently: only 3.6% of sampled meta items appear in the ratings
  sample (15.4% appear in the full 5-core). A useful local `data.meta` run therefore needs the full 5-core CSV
  (0.14 GB) as its item list, not its first N lines.

### Data format findings (F1–F12)
- **F1. `meta_Books` contains Kindle editions and audiobooks.** `main_category` is "Buy a Kindle" for 6.6% of
  meta records and "Audible Audiobooks" for 2.5%. Among sampled meta items that are in the 5-core, Kindle
  editions carry **21.1% of the 5-core interactions** (audiobooks 0.4%). This contradicts PRD §17.1 ("Kindle
  e-books live in a separate file"). Kindle records have `parent_asin` `B0…` (100%), no `ISBN 10`/`ISBN 13`
  (0%), page counts under `Print length` (98%), `Publication date` instead of a dated `Publisher`, and
  sometimes `Page numbers source ISBN` (17%), which points at the print edition. Consequences for
  `data.canonical`: Kindle/print merges are common and mostly heuristic (no ISBN for the Goodreads join);
  `FORMAT_KEYS` needs a Kindle/Audible format; `Page numbers source ISBN` can link a Kindle edition to its
  print ISBN. **Needs a user decision before Week 3:** keep Kindle and audiobook editions (merged into their
  works) or drop them.
- **F2. `main_category` is not always "Books":** also "" (82), null (3), "Office Products", "Toys & Games",
  "Musical Instruments". `categories` is empty for 8.9% of records; `categories[0]` can be
  "Audible Books & Originals". Genre slicing on `categories[1]` must tolerate both.
- **F3. `details["ISBN 10"]` sometimes holds a 13-digit value** (1,511 of 40,138; also a few 4/5/12-digit
  values). `norm_isbn` must validate length and route 13-digit values to ISBN-13. `ISBN 13` has a dash in
  84.7% of records.
- **F4. Author sources disagree.** `author` is null for 28.5% of records (`store` is present for 95% of
  those). One `author` dict has no `name` key. Where both exist, the `author.name` surname is missing from
  `store` 5.3% of the time (for example an editor credited in `store`). `store` lists several contributors
  for 25% of records ("Name (Author), Name (Illustrator)…"), sometimes with a trailing "Format: Kindle
  Edition", and has no "(Role)" for 203 records. Parse `author.name` first, then the first "(Author)" in
  `store`.
- **F5. `description` mixes sections.** 51.4% of records have an empty `description` (`features` is
  non-empty for 90.9%, so it is the main fallback). Non-empty descriptions often start with section labels:
  "Review" (19% of records), "About the Author" (18%), "From Publishers Weekly", "From the Back Cover". The
  §10.1 content text builder should drop author bios and editorial-review sections or prefer the blurb.
- **F6. `images` entries usually have only `large` and `variant`** (97%); `thumb`/`hi_res` appear in 30
  records. No dict-of-lists form seen.
- **F7. No JSON-string encodings seen.** `details` is always a dict and `author` a dict or null in this
  sample; the §7.2 `_maybe_json` handling stays as cheap insurance.
- **F8. Dates.** The `Publisher` "(Month D, YYYY)" pattern gives a year for 92.0% of records; Kindle records
  use `Publication date` (6.8%); `subtitle` often ends in a date ("Paperback – January 1, 2004"), a useful
  third fallback. `subtitle` is null for 4.3% and is sometimes just "Kindle Edition" or "1st Edition".
- **F9. 5-core CSV layout.** Ratings are floats ("5.0"), timestamps are milliseconds (sample range
  1997-08-14 to 2023-03-20; full file 1996-08-17 to 2023-09-12). Rows are grouped by user and time-sorted
  within a user, but users are not in alphabetical order. The first 200K lines are 10,210 complete user
  histories (5 to 3,134 ratings each).
- **F10. Reviews.** Timestamps are milliseconds. `images` entries are dicts of `small_image_url`,
  `medium_image_url`, … (unused). `asin` differs from `parent_asin` in 0.3% of reviews. The raw file has
  duplicate (user, parent_asin) pairs (11 in 50K) that the 5-core file does not. 37.9% of texts contain HTML
  tags (mostly `<br />`) and 6.0% HTML entities; 6.8% are shorter than 20 characters.
- **F11. Goodreads spoiler file.** `rating` is an int 0–5 where 0 means "no rating" (1.9% of reviews);
  `timestamp` is a date string ("2017-08-30"), not milliseconds; `review_sentences` is `[[int, str], …]` and
  `has_spoiler == any(label)` holds. Spoiler sentences are rare (1.9% of sentences, 5.9% of reviews), which
  supports the balanced sampling in §11.7. The file is grouped by user (52 users in the first 5,000 reviews),
  so its head is not a representative sample.
- **F12. Goodreads book records** (checked while planning, first 2,000 lines): every scalar is a string,
  empty string for missing; `series` is a list of id strings; real series suffixes include "(Pluto's Snitch
  #2)" (no comma) and "(Sun Wolf and Starhawk, #1-2)" (a range, correctly left unparsed by the §7.3 regex).
  The tiny fixture's assumptions (D13) hold.
