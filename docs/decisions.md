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

## 2026-10-13: Week 2 (Phase 1 data: download, meta, Goodreads)

### D17. Kindle and audiobook records are editions of the same work (user decision on F1)
- **Decision:** keep Kindle and audiobook records from `meta_Books` and treat them as editions of the same
  work. `data.meta` keeps and tags them (`edition_kind`: print, kindle, audio, unknown; Kindle records also
  carry `page_source_isbn13`, the print ISBN from "Page numbers source ISBN"). `data.canonical` (Week 3)
  merges them with print editions by ISBN when one is present (including the page-source ISBN), otherwise
  by the title+author heuristic with the box-set and volume guards of CLAUDE.md §7.4.
- **Reason:** user decision (2026-10-13). In the 50K-line sample Kindle editions carry 21% of 5-core
  interactions overall and 68.5% of the "Mystery, Thriller & Suspense" interactions; dropping them would
  remove most of some genres.
- **Alternatives:** drop Kindle and audio records (PRD §17.1 assumed they were not in this file).

### D18. Large sources are streamed from their URLs, never stored (user rule)
- **Decision:** `meta_Books` (and in Week 3 the reviews; later the Goodreads spoiler file) are read with
  `ml/data/stream.py`: bytes are gunzipped in memory and filtered line by line. The stage framework has a
  `url:<source>` input type; `ctx.source()` returns the URL, or with `--sample` the same-named local head in
  `project.local_raw_root` (now `./local/raw_sample`) when it exists, else the URL read for N lines. The
  input's fingerprint identity is the URL (plus the local head's size and mtime in sample runs).
- **Resilience:** a dropped connection resumes with an HTTP Range request from the exact compressed byte
  reached, feeding the same decompressor (tested with a server that hangs up mid-file). A crashed process
  cannot resume a gzip stream, so `data.meta` checkpoints its output every `data.stream.checkpoint_rows`
  kept records into its work dir on Drive (`PartWriter`); a rerun re-streams from byte 0 but skips the
  lines already processed without parsing them, and counters resume from the checkpoint (tested: output
  and counts identical to an uninterrupted run). Every stream must end with a complete gzip trailer and
  deliver exactly its Content-Length.
- **Alternatives:** download then parse (needs about 11 GB of Colab disk per session for meta + reviews and
  re-downloads after every restart).

### D19. `data.download` keeps only the 5-core ratings, always whole
- **Decision:** `data.download` fetches the 5-core CSV into `data/<v>/ratings_5core.csv.gz` on Drive (not
  ephemeral raw disk), byte for byte, while checking the header, stars in {1..5} and millisecond
  timestamps (`DataError` otherwise). It reads the whole file even with `--sample` (user OK), because every
  filter keys on it and its first lines barely overlap the meta head (D16). fastText's model moves to Week 3
  with `data.clean`; the Goodreads files are streamed by `data.goodreads`.
- **Reason:** 0.14 GB, used by every later stage; keeping it on Drive avoids re-downloads after restarts.

### D20. The Goodreads book graph is optional (user rule)
- **Decision:** `data.use_goodreads_graph: false` by default. Off, `data.goodreads` makes no network request
  and writes `gr_isbn_index.parquet` and `gr_series.parquet` with zero rows and their final schema, so
  `data.canonical` has one code path. On, it streams the books, series and works files and keeps only
  books whose ISBN-13 (Goodreads ISBN-13, or its ISBN-10 converted) matches a `books_raw` ISBN: print
  ISBN-10/13 or a Kindle edition's page-source ISBN (user requirement). The user enables it after
  `data.meta` has run on Colab.
- **Consequence:** this departs from CLAUDE.md §1.5 (Goodreads ISBN join first, heuristics as fallback)
  while the flag is off: series ids and positions then come only from Amazon title patterns (Week 3), so the
  series guard and "continue series" cover fewer books.
- **Spoiler file:** the Goodreads spoiler reviews (Phase 3) are streamed independently of this flag.

### D21. Torch on Colab: use the preinstalled build (user decision; closes D2's open item)
- **Decision:** never install torch on Colab. `make install-colab` installs `requirements-core.txt` and
  `requirements-ml.txt` minus the `torch==` line; `notebooks/00_colab_setup.ipynb` calls it and prints the
  torch version actually used. `requirements-ml.txt` keeps its torch pin for laptops and CI.

### D22. `data.meta` details
- **Prefilter:** `parent_asin` is read from the raw line (`json_str_field`) and only 5-core lines are
  JSON-parsed. On the real 50K-line head this parses 7,707 of 50,000 lines; `parent_asin` extraction agreed
  with full JSON parsing on every meta and review line.
- **Parsing rules from F1–F8:** author = `author.name`, else the first "(Author)" in `store` with any
  "Format: …" suffix removed, else its first credit; ISBNs are classified by shape, so a 13-digit "ISBN 10"
  becomes the ISBN-13; pages come from any "N pages" value (Kindle `Print length` included); year from the
  Publisher date, then `Publication date`, then the subtitle; the binding is a `details` key or, for about
  a tenth of print books, the subtitle's first part ("Paperback – January 1, 1994"). With these rules the
  real head has 1.0% `unknown` editions (10.4% before the subtitle fallback).
- **Memory:** `books_raw.parquet` is written from the checkpoint parts one part at a time (duplicates
  dropped across parts), and the category counts read back only four columns. On the sample it takes
  about 2.4 KB per book on disk with zstd (D24), about 1.2 GB for the full 5-core, too much to hold as one
  pandas table on Colab's ~12 GB.
- **Columns beyond §7.2:** `subtitle`, `main_category`, `features` (kept apart from `description`, which
  often holds author bios and editorial reviews, F5), `page_source_isbn13`, `format`, `edition_kind`,
  `n_interactions`. Explicit pyarrow schemas fix column types even when a sample has only nulls.
- **Outputs for the decision:** `category_counts.csv` adds Kindle and audio interaction columns;
  `genre_slice_proposal.json` applies the §7.2 rule with its constants in config
  (`data.genre_slice_rule: {min_slices: 2, max_slices: 3, target_multiple: 4}`).
- **Stop:** the stage ends with a STOP message and `python -m ml.data.meta` reprints the report.
  `require_genre_slices(cfg)` raises while `data.genre_slices` is empty; Week-3 stages call it first.
- **Dependencies:** `gdown` removed from `requirements-ml.txt` (D15); `requests` added to the CI package
  subset because `ml.data.stream` imports it.

### D23. Tiny fixture additions
- **Decision:** the fixture now plants a Kindle edition of a print book (page-source ISBN), a Kindle
  edition whose print edition exists only in Goodreads (proves the page-source match), an audiobook, a
  13-digit "ISBN 10", an author dict without `name`, a `store` whose author is not the first credit, and a
  record with an empty title. Totals are unchanged (50 users, 80 books, 600 ratings).

### D24. zstd for `books_raw.parquet`; `features` stays (user request, 2026-10-13)
- **Decision:** `books_raw.parquet` and `data.meta`'s temporary checkpoint parts are written with zstd. On
  the 50K-line sample `books_raw` shrinks from 24.5 MB (snappy, the default) to 17.1 MB (70%) with
  identical content; about 1.2 GB instead of 1.7 GB at full scale.
- **`features` checked, kept:** the user asked to drop `features` if it just duplicates `description`. On
  the sample it does not: the two are never identical; `features` is contained in `description` for 1.5%
  of books, both exist and differ for 76.4%, and `features` is the only text for 20.2% (`description`
  empty). Spot checks show `features` is usually the publisher's blurb (back-cover copy) while
  `description` mostly holds editorial reviews ("Review", "From Publishers Weekly") and author bios (F5).
  The blurb is the spoiler-safe premise text Phase 3 needs, so the column stays unchanged.
- **Alternatives:** drop `features` (loses the only text for one book in five); blank it where it is
  contained in `description` (1.5% of rows; would push Phase 3 to the noisier review/bio text).

## 2026-10-13: First Colab run of `notebooks/01_data.ipynb`

What happened: `data.download` completed on Colab in 23 s (9,488,297 ratings, 776,370 users, 495,063
items, the same counts as locally). `data.meta` was streaming normally (the traceback shows it inside
`parse_meta`) when the cell was stopped: the stage printed nothing for minutes, and Colab's input box
under the running `!` command looked like a prompt. The second traceback (`PosixPath ... '_str'`) was a
second Ctrl+C landing in the staging cleanup; no manifest was written, so nothing downstream trusts the
partial run. The run also showed that Colab now uses **Python 3.13**.

### D25. Colab feedback and Python 3.13
- **Progress output:** long streams (`data.meta`, `data.goodreads`) log a status line about once a minute:
  GB done of the total, percent, MB/s, minutes left, lines and 5-core matches so far (`Progress` in
  `ml/common/log.py`, which is framework plumbing and so outside stage fingerprints).
- **Notebooks:** both now say that the empty box under a running `!` cell is Colab's input box, not a
  question; run cells read stdin from `/dev/null` so nothing typed there reaches the process.
- **fastText:** `fasttext-wheel==0.9.2` has no Python 3.13 wheel, so `make install-colab` could not
  install the pin set on Colab (the stages then ran on Colab's preinstalled pandas/pyarrow/requests).
  Replaced by `fasttext-predict==0.9.2.4`: the prediction-only fastText build (all that language ID in
  `data.clean` needs), same `fasttext.load_model(...).predict(...)` API, wheels for cp311-cp313 on Linux
  and Windows, and no numpy dependency, which also removes the numpy-2 `predict()` problem of D3. Checked
  with the real `lid.176.ftz` model (English, Spanish and German sentences identified correctly).
- **Pins:** the full core+ml set now resolves with binary wheels for cp311, cp312 and cp313 (104
  packages each); CI tests 3.11, 3.12 and 3.13.
- **Alternatives:** compiling `fasttext` from source on every Colab session; pinning Colab to an older
  runtime (not offered for free accounts).

## 2026-10-13: Genre slices and repository hygiene

### D26. Genre slices: Literature & Fiction; Mystery, Thriller & Suspense; Science Fiction & Fantasy
- **Decision (user):** `data.genre_slices: [Literature & Fiction, 'Mystery, Thriller & Suspense',
  Science Fiction & Fantasy]` (values of `categories[1]`).
- **Evidence (full Colab run of `data.meta`, 2026-10-13, 446 s):** 4,448,181 meta lines; all 495,063
  5-core items found; 13,408 dropped by the title blocklist and 27,647 by `min_pages`; 454,008 books in
  `books_raw`. 5-core interactions per allowlisted category: Literature & Fiction 2,143,678 (24.6%, 98,546
  books, 30.9% Kindle); Mystery, Thriller & Suspense 2,064,657 (23.7%, 56,335 books, 70.6% Kindle); Teen &
  Young Adult 383,164 (4.4%); Science Fiction & Fantasy 202,160 (2.3%, 12,939 books); Romance 122,661 (1.4%).
- **Reason:** the §7.2 rule proposed the first two (4,208,335 interactions, threshold 1,000,000). The user
  added Science Fiction & Fantasy (total 4,410,495) because the PRD's demo and persona examples are mostly
  SF and fantasy; without the slice those books would not exist in the catalogue.
- **To watch in Week 3:** SF & Fantasy is small (2.3%); the EDA reports the genre mix after k-core and
  sizing to show how much of it survives. Amazon's `categories[1]` is not always the literary genre (in the
  sample, "The Handmaid's Tale" is under Politics & Social Sciences and Harry Potter under Children's Books),
  so some well-known fiction falls outside the slices; the title lookups at the gate check this.
- **Alternatives:** the two proposed slices only; adding Teen & Young Adult (larger than SF & Fantasy but
  further from the demo examples).

### D27. Notebooks are committed without outputs
- **Decision:** `tests/test_notebooks.py` fails CI if any notebook in `notebooks/` has cell outputs or
  execution counts (no new dependency; checked against the Colab-saved notebook, which it rejects). The user
  will not use Colab's "Save in GitHub" again.
- **Why:** Colab's "Save in GitHub" committed three executed copies of `01_data.ipynb` (with outputs) to
  `claude/litsense-week2-data`. Outputs belong on Drive and in `results/`, not in git (rule 0.2.11).
- **Branch cleanup:** the Colab fix `7b3fc9e` missed PR #3's merge; it was cherry-picked onto a fresh branch
  from `main` (`claude/litsense-week2-colab-fixes`) without the three Colab commits, and
  `claude/litsense-week2-data` was deleted once nothing on it was needed.
- **Alternatives:** an `nbstripout` pre-commit hook (a new dependency, and hooks do not run when Colab
  commits directly).


## 2026-10-13: Week 3, part 1 (`data.canonical`)

### D28. Edition canonicalization rules (deviations from §7.4; thresholds await the merge audit)
- **Merge all editions first, then slice (user, rule A):** the 454,008 `books_raw` editions are merged into
  works, and a work is kept if *any* edition's `categories[1]` is a genre slice. §7.4 filtered by slice
  first. Why: Amazon files editions of one work under different categories. In the Week 2 lookup, the
  Project Hail Mary audiobook sits in SF & Fantasy and the print editions in Literature & Fiction; Mistborn
  book 1 and the box set sit in SF & Fantasy and books 2–6 in Literature & Fiction. Slicing first would
  split works and lose their Kindle ratings. `asin_to_canonical.in_slice_edition` and the report's
  `rescued_*` counts show how much rule A adds.
- **Merge rules, in order:** `isbn` (same ISBN-13, ISBN-10s converted); `page_source` (a Kindle edition's
  "Page numbers source ISBN" equals a print edition's ISBN, F1/D17, unless both have authors and they
  differ); `goodreads` (same work id, only with `use_goodreads_graph`); `title` (below). Union-find; two
  components are never joined if they hold different Goodreads works or different known volume numbers.
- **Title rule:** block by (first token of the normalized title, author surname); merge when
  `token_set_ratio >= fuzzy_threshold` (90, §5) **and** `token_sort_ratio >= subset_guard_threshold` (90,
  new key) **and** the same author key **and** the volume guard passes **and**, only for non-identical
  normalized titles, the year gap is at most `max_year_gap` (3).
  - *Subset guard:* `token_set_ratio` is 100 whenever one title's words are a subset of the other's:
    "martian" vs "martian chronicles", "alex cross" vs "alex cross s trial".
  - *Year gap only for non-identical titles:* reissues of the same title can be decades apart, and Kindle
    publication years differ from print.
  - *Volume guard:* conflict when both volumes are known and differ, or when one is unknown and the other is
    2 or more ("Mistborn" may be book 1, never book 3). Known cost: a Kindle title carrying "(Series, Book 2)"
    does not merge **by title** with a print edition whose title has no marker (it still merges by ISBN or
    page-source ISBN). The audit's `not_merged` rows with reason `volume` show how often this happens.
  - *Author required:* editions without an author never merge by title.
- **Matching key (`ml/data/titles.py`):** NFKC, lowercase, every bracketed segment dropped (series info is
  read first), a subtitle after ':' dropped when it is format noise, marketing or a volume marker
  ("A Novel", "TikTok sensation", "Book Three of Mistborn"), trailing format words and dates dropped
  ("Paperback 18 Feb 2021"), punctuation stripped, leading article dropped. New config:
  `canonical.subtitle_noise_patterns`, plus edition-noise terms (US/UK/international/revised/new edition,
  reissue, movie/media tie-in, MTI).
- **Box sets:** excluded from every rule, shared ISBNs included, and dropped (`drop_boxsets: true`).
- **Canonical record (§7.4 step 7):** title = the most-rated edition's title with format noise removed
  (series bracket kept); author = most common author key (ties go to the most-rated edition); description and
  `features` = the longest among editions (D24); year = min; pages = median; image and ISBN-13 from the
  most-rated edition that has one; also `slices`, `primary_slice` (first slice among the editions, most-rated
  first), `edition_kinds`, `n_interactions`. Series: Goodreads when enabled, else parsed from titles, with
  `series_id = "am_<author key>:<series key>"`. Ids: `gr_<work_id>` or `am_<min parent_asin>`.
- **Audit (§7.4 step 9):** `merge_audit.csv` has 100 random kept works with 2 or more editions (up to 15
  edition rows each, scored against the most-rated edition) and 50 title pairs that were not merged
  (`token_set >= 80`), spread over the reasons `below_threshold`, `subset_guard`, `volume`, `year` and
  `goodreads_work`, drawn from a seeded reservoir of 2,000 pairs per reason (all pairs are counted in the
  report). The reviewer fills `verdict` (Y = right decision, N = wrong). The file is UTF-8 with a BOM so
  Excel opens it correctly. Both audit files are under `results:canonical/`, so they reach the repo with
  `results/v0/` (user rule B: only `results/v0/` and the registry entry are copied from Drive, never data
  files).
- **Open (§0.3):** merge precision from the user's review is recorded here before `data.reviews` runs; the
  thresholds above are not accepted until then.
- **Alternatives:** slice first (§7.4 as written); `token_set_ratio` alone; `fuzz.WRatio`; no volume guard
  for one-sided volumes.

### D29. Genre labels (`ml/data/genres.py`, §7.4 step 8)
- 26 ordered keyword labels plus the fallback "Fiction (general)". A path's labels come from its **deepest**
  element that matches anything, so "Fantasy" wins over the compound top level "Science Fiction & Fantasy",
  which would match both. Audience labels (Young Adult, Children's) are added from any level. A work's
  genres are the union over its editions, most-rated first; `primary_genre` is the first non-audience label.
- **Limitation measured on the 50,000-line sample head:** 24,817 of the 69,963 in-slice interactions sit
  under "Literature & Fiction > Genre Fiction" with no deeper level, so those works get the fallback label.
  Genres are filters and ranker features, not the slice rule. If the full-run EDA shows the fallback
  dominating, Goodreads shelves (with `use_goodreads_graph`) are the obvious refinement.
- **Alternatives:** labels from the top level only (too coarse); every matching label at every level (SF &
  Fantasy books would all get both labels).

### D30. `data.lid` keeps `lid.176.ftz` in `data/` (user decision C)
- A separate stage, `data.lid` (in `ml/data/download.py`), fetches fastText's language-ID model (0.9 MB,
  always whole, even in `--sample` runs) for `data.clean` and checks the fastText magic number, so an HTML
  error page cannot pass as a model.
- **Side effect on Colab:** this change and the new config validation (`ml/common/config.py`) are in the
  code fingerprints of `data.download` and `data.meta`, so both rerun once (about 8 minutes, mostly the
  `meta_Books` stream); `require_genre_slices` now takes the slice list.

## 2026-10-14: Merge audit, first pass

### D31. Shared-ISBN merges need the same author (user decision after the audit)
- **Finding:** in the full-run `merge_audit.csv`, group G018 joined a crafts book (*Log Cabin
  Variations*, author "Annies") to *The Long Ships* because Amazon lists both under one ISBN. The `isbn`
  rule had no author check.
- **Decision:** an `isbn` merge now applies the same rule as `page_source`: the authors must match unless
  one is missing. No threshold changed. Regression test `test_shared_isbn_needs_the_same_author`.
- **Not changed (user's choice):** the subset guard stays at 90, so G054 (a sequel merged at
  `token_sort` exactly 90) stays a known false merge.
- **Audit of run 1 (verdicts model-drafted, spot-checked by the user; the user will spot-check
  again later):** 97/100 merged works correct (wrong: G018 shared-ISBN data error, fixed here; G054 sequel
  at the subset-guard boundary; G004 one two-in-one "Animal Farm: 1984" edition) and 44/50 not-merged
  pairs correct (all 6 misses were same works blocked by the subset guard, which led to D32). File:
  `results/v0/canonical/merge_audit_run1_reviewed.csv` (verdict + note columns).

### D32. "Title" merges with "Title: subtitle" when the blurbs agree (user decision)
- **Rule (`subtitle`, runs after every other rule):** an edition whose title has no subtitle joins an
  edition whose title is that text plus ": subtitle" when all of these hold:
  - the bare title's matching key equals the key of the other's text before the colon (`title_head`);
  - the author key is the same (never missing);
  - the volume guard passes and the two works hold no different Goodreads works;
  - neither is a box set or omnibus;
  - the blurbs are similar: `features`, else `description` (D24), first
    `canonical.subtitle_description_chars` = 500 characters, lowercased, punctuation collapsed; rapidfuzz
    `token_sort_ratio` >= `canonical.subtitle_description_threshold` = 70.
  An empty blurb on either side means no merge.
- **Why:** 6 of the 50 audited non-merges were the same work with a subtitle on one side ("Cold Hand in
  Mine" vs "Cold hand in mine: Strange stories"). The subset guard blocks these by design, and the blurb
  check stops a bare series name joining book 1 ("Mistborn" vs "Mistborn: The Final Empire").
- **Threshold evidence (50K-line sample head; small, rough):** 27 same-work edition pairs and 534
  different-work pairs by the same author. At 70, 41% of same-work pairs and 1.9% of different-work pairs
  pass. At 60: 44% / 3.9%. 70 favours precision; the title-head condition is the main filter.
- **Cost control:** at most 3 description comparisons per pair of works (most-rated editions first);
  blurbs are read from `books_raw` only for candidate editions.
- **Audit:** `merge_audit.csv` gains a `subtitle_merge` section: 30 random kept works joined only by
  this rule (each subtitle edge joins two works the other rules kept apart), showing the two joining
  editions with `merge_rule = subtitle` and the description similarity in `reason`. Pairs rejected for
  dissimilar blurbs are near misses with reason `subtitle_description`. `canonical_report.json` has a
  `subtitle_rule` block of counts by outcome.
- **Alternatives:** TF-IDF cosine (needs scikit-learn in the stage; no clear gain at this sample size);
  `token_set_ratio` (similar separation, more lenient on subsets); merging without a blurb check (would
  join series names to book 1).

## 2026-10-14: Run 2 of `data.canonical` (D31 + D32)

### D33. Different numbers in fuzzy-matched titles block a title merge
- **Finding (run 2):** one kept work held *NYPD Red 2, 3, 4* and *6* (17 editions, 2,649 interactions).
  "nypd red 2" vs "nypd red 3" scores 90 on both fuzzy ratios, and `volume_number` does not read a
  bare trailing number. The editions were chained by the `title` rule; the subtitle rule then added one
  more NYPD Red 6 edition. The bug existed in run 1 too; the random merged-work audit did not sample it.
- **Decision:** for non-identical normalized titles, the `title` rule refuses a merge when the titles'
  number tokens differ (digits and multi-letter roman numerals, `number_tokens`). The near-miss reason is
  `number`. Test `test_numbered_titles_never_merge`. Identical titles are unaffected.
- **Effect on the 50K sample:** one title merge removed: two editions of an out-of-slice programming book
  ("... iPhone SDK" vs "... iPhone 3.0 SDK"), which arguably are different editions anyway.
- **Alternatives:** read a trailing number as a volume (would break titles like "Catch-22" and "1984");
  raise the fuzzy thresholds (would lose real edition merges).

### Run-2 audit (subtitle section; verdicts model-drafted, spot-checked by the user)
- `subtitle_merge`: 30/30 sampled works correct (the same book with a marketing, genre or series
  subtitle). Run-2 totals (from the user's Colab output): 1,603 subtitle merges from 10,850 candidates;
  5,375 rejected for dissimilar blurbs, 555 for missing blurbs, 2,734 by the volume or work guard;
  648 kept works joined.
- The merged-work and not-merged sections of run 2 were not re-reviewed: the rules behind them changed
  only by D31 (fewer ISBN merges: 705 to 513) and now D33.

## 2026-10-14: `data.canonical` closed; `data.reviews`

- **Run 3 of `data.canonical` (after D33):** 454,008 editions became 384,579 works (50,127 with 2 or more
  editions); kept 130,960 works and 4,499,455 interactions; the NYPD Red chain is gone from the largest
  works. Near misses: number 6,852, year 1,496 (some pairs moved from year to number). Edition merging is
  accepted with the audits recorded in D31-D33 (model-drafted verdicts, spot-checked by the user).

### D34. `data.reviews` details
- **Join:** a review is kept when its `(user_id, parent_asin)` is a 5-core rating of an edition whose work
  is `kept`. Both ids are read from the raw line before JSON parsing; join on `parent_asin`, never `asin`
  (F10). Timestamps must be milliseconds or the stage fails.
- **Dedup order (§7.5 step 4):**
  1. The latest review per `(user_id, canonical_id)`. This covers both the same review on two editions
     and two editions that `data.canonical` merged.
  2. Then one per `(user_id, text_hash)`. `text_hash` is a 16-hex sha1 of the text with tags, case and
     punctuation removed. Empty texts are never text duplicates.
- **No-review ratings** are not rows here. `data.kcore_split` builds interactions from the 5-core file and
  sets `has_text` from this table; the stage logs `pairs_without_review`.
- **Memory:** the pair set holds the kept works' 5-core pairs as strings (about 4.5M on the full run).
  Parts are checkpointed every `data.stream.checkpoint_rows` matches, and dedup reads only the key
  columns before rewriting the parts.

### D35. `data.clean` details
- **Rules, in order:** clean the text (tags, entities, NFKC, whitespace; texts over `max_review_chars`
  are cut at the last sentence end); then `too_short` (< `min_review_chars`); then `non_english`;
  then `shipping` (a `shipping_patterns` match and no aspect-lexicon term from `ml/nlp/aspect_seeds.py`).
  Every review keeps its row; `text_ok` and `text_drop_reason` only gate NLP use.
- **Language ID:** fastText `lid.176` on the first 500 characters flags `non_english` only when the
  other language has probability >= `data.lid_min_confidence` (0.5, new key). Short English texts are
  often misread: "Great read!!" -> en 0.12, "meh" -> de 0.44, "Gut" -> sr 0.48, while "Muy bueno el
  libro, lo recomiendo" -> es 0.95.
- **fastText bug:** `fasttext-predict` 0.9.2.4 mis-unpacks batch `predict(list)` results (a 3-text
  list raises; a 2-text list returns labels in place of probabilities). Texts are scored one at a
  time; the loader is injectable, so CI tests use a stub because fastText is not installed there.
- **`ml/nlp/aspect_seeds.py`** (taxonomy, lexicons, prototypes, descriptor poles from §11.2) is added
  now because the shipping rule needs the lexicon.

### D36. `data.kcore_split` details
- **One interaction per (user, work):** the latest 5-core rating when a reader rated two editions of
  one work (§7.7 assumed this would not happen). `has_text` = that reader's review of the work survived
  `data.clean` with `text_ok`.
- **Sizing:** every user draws one seeded uniform number; a fraction f keeps users whose draw is
  below f, then the k-core reruns. f is found by bisection (up to 40 steps) until the size is within
  `target x (1 +/- size_tolerance)`; otherwise the stage fails and reports the closest size. f is in the
  manifest notes and in `splits.json` (`sizing`); it is copied here after the full run.
- **`splits.json`** also carries the attrition counts and the slice and genre mix after the k-core and
  after sizing (the user's "how much SF & Fantasy survives" request). `data.eda` adds the works-kept
  step from `canonical_report.json`.
- **Ids:** `id_map.json` = sorted user ids and sorted canonical ids, written only here; a rerun gives
  identical bytes (tested).
- **Framework:** `StageContext.upstream_notes(stage)` reads an upstream stage's manifest counts (for
  the EDA attrition table); the EDA declares one output of each such stage as an input.
- **CI** installs matplotlib (`CI_ML_PACKAGES`) for the EDA test.

### D37. Checkpoints survive a part lost by Google Drive
- **Finding (Colab, 2026-10-14):** a first `data.reviews` run streamed the whole file and stopped
  before finishing. On rerun, `progress.json` named 90 parts but `part-00089.parquet` was missing; the
  Drive FUSE mount had kept the later progress file and lost the earlier part.
- **Decision:** `PartWriter` records each part's line offset and counters (`history`). On resume it
  checks every recorded part and rolls back to the last one that is present and readable. Old
  checkpoints without history restart from line 0. `close()` marks the stream `finished`, so a
  rerun after a post-processing failure skips the re-stream. `data.reviews` frees its pair set
  before the dedup.
- **Cost of this fix on Colab:** `ml/data/stream.py` is in the code fingerprints of `data.meta` and
  `data.reviews`, so both rerun once, and the reviews stream starts over.

## 2026-10-14: v0 frozen (Week 3 run on Colab)

### D38. v0 sizing, and Drive flushing
- **Sizing:** the user fraction recorded by `data.kcore_split` is in `models/registry.json`
  (`datasets.v0.sizing`), and the v0 numbers are in `results/v0/eda/summary.json` and `attrition.csv`.
  Sampling users and then re-applying the 10-per-item core leaves a small catalogue (see `summary.json`);
  this is fine for development, and v1 sizing is revisited before Week 11.
- **Drive flushing:** after a run, Drive's FUSE mount can still be uploading when the session ends (a
  second runtime briefly saw a 0-byte `reviews_clean.parquet` and no later manifests). The data notebook
  now ends with `drive.flush_and_unmount()`, and one-off cells do the same.
- **Open (user decision):** the slice mix shifts away from SF & Fantasy at the k-core and sizing steps
  (`summary.json` -> `slice_mix`).

### D39. Open items for the v1 rebuild (user decision, 2026-10-14; resolve before Week 11)
v0 stays as built, and Week 4 starts on it. Before v1:
1. **Slice-aware sizing:** sample users so each genre slice keeps roughly its pre-sizing share of
   interactions.
2. **Book threshold:** compare `kcore.item` 5 vs 10. For each, report catalogue size, long-tail share
   and number of evaluable test users; the user chooses (the long tail is the thesis).
3. **Catalogue size:** target roughly 10-20K books in v1; report what each sizing option gives.
4. **Serving catalogue vs training set:** propose (do not build) serving a larger catalogue, e.g. all
   in-slice works with >= N reviews, scored by content and review features for books outside
   training, with its cost (bundle size, RSS, latency, evaluation caveats).

## 2026-10-14: Week 4 (evaluation harness and baselines)

### D40. Harness and baseline choices
- **Scorer cutoff:** each scorer records the cutoff of the `FitData` it was fit on. `evaluate` raises
  `LeakageError` when it differs from the mode's cutoff, and `FitData` refuses rows at or after its
  cutoff. Stages write the cutoff into their manifest (§3.2).
- **Degenerate scores:** a row whose finite scores all tie, or that has fewer than two finite scores,
  raises `DegenerateScoresError`; -inf outside a candidate set is allowed (§8.3).
- **Cold items:** collaborative scorers (popularity, ItemKNN, ALS) give items with no fit interactions
  -inf (§8.1); content-only scores every item.
- **Baselines are refit, not stored:** `eval.baselines_val` fits random, popularity, ItemKNN, ALS and
  content-only inside the stage (seconds to minutes each). Model artifacts with manifests start with
  LightGCN in Week 5, which serving needs.
- **Content text:** title, author, genre labels, then the blurb (`features`, else `description`)
  truncated by tokens to the 256-piece budget. Reviews and LLM text never go in.
- **Content-only profile:** the rating-weighted mean of content vectors of the user's fit items rated
  >= `relevance_min_rating` (all fit items if none), L2-normalized (§9.2).
- **Reports:** `table_main.csv` (means with 95% bootstrap CIs, beyond-accuracy metrics, catalogue size
  as the coverage denominator), `table_strata.csv`, `table_longtail.csv` (long-tail and cold-item
  truth), and `bootstrap.csv` (each model vs popularity, paired, all users and long tail), plus two
  figures. The admin `metrics.json` (§15.6) follows once LightGCN results exist.
- **Test lock:** `ml/eval/lock.py` implements §8.8 now (tested); the `eval.test` stage is Week 13.
- **ALS:** `implicit` 0.7.3 on the (users x items) matrix with confidence alpha x binary; output
  shapes are asserted, and BLAS runs single-threaded during the fit.
- **CI** installs scipy and implicit.

### D41. Edits to shared modules rerun every stage that imports them
- Adding `LeakageError` and `DegenerateScoresError` to `ml/common/errors.py` would have changed the code
  fingerprint of every data stage (all import it) and rerun the whole data pipeline on Colab, including
  the review stream. They live in `ml/eval/errors.py` instead, and `ml/common/errors.py` is back to the
  bytes Colab ran. Rule: keep finished stages' import closures stable; put new shared code in new modules.

### D42. "Up to date" now also checks output sizes
- **Finding:** on Colab, `reviews_clean.parquet` stayed on Drive as a 0-byte file while its manifest
  said complete. The next stage that read it failed (`Parquet file size is 0 bytes`).
- **Decision:** a stage is current only if every output exists with the byte size its manifest
  recorded (directories: existence). Sizes are cheap; full hashes are not. A truncated output makes the
  stage stale, so `run all` rebuilds it. `ml/common/stage.py` is framework code, outside stage
  fingerprints, so this change reruns nothing by itself.

## 2026-10-15: Week 5, LightGCN

### D43. LightGCN training, artifacts and evaluation stage
- **Stage `models.lightgcn_val`** (`ml/models/lightgcn.py`) trains on train (cutoff t1) and writes
  `artifacts/<v>/eval_val/lightgcn/`: `lgcn_user.npy`, `lgcn_item.npy`, `lgcn_item_norm.npy`,
  `item_degree.npy` and `meta.json` (cutoff, mode, params, best epoch, device, torch version), plus
  `results/<v>/training/lightgcn_val.csv` (loss and inner NDCG@20 per evaluation). The test-mode set
  (train + val, t2) reuses `build_lightgcn` in Week 13.
- **Training choices not fixed by s9.4:** one epoch = every fit positive once in shuffled batches;
  full-graph propagation every step; the L2 term counts the negatives' ego embeddings divided by
  `n_negs`; the inner split holds out max(1, floor(0.1 n)) of the latest rows of users with >= 6 fit
  interactions and is scored by the shared `build_eval_data` + `evaluate` (same masks and relevance
  rule as val). The refit trains from scratch on the whole fit window, same seed, for the best epoch
  count. GPU sparse products are not bit-reproducible; CPU runs are (tested).
- **Negative sampler:** vectorized redraws, then the rare survivors are drawn from the user's
  complement, so a positive is never returned (s16.2).
- **Scoring:** `ml/models/embedding_scorers.py` (numpy only, so evaluation never imports torch) gives
  `lightgcn_raw` (u . i) and `lightgcn_norm` (u . i/|i|); items with no fit interactions score -inf.
  A stored scorer's `fit` refuses an artifact whose cutoff, shapes or item degrees differ from the
  evaluation's fit data.
- **Stage rename:** `eval.baselines_val` became `eval.val` (`ml/eval/validation.py`, the name in
  s7.0). It refits the baselines, loads LightGCN, copies `meta.json` into each LightGCN result file,
  and writes paired bootstraps against popularity, `lightgcn_raw` and `lightgcn_norm` (s2.2 #1). The
  baseline numbers are recomputed with the same seed and code.
- **CI** installs CPU-only torch (the `requirements-ml.txt` pin) from PyTorch's CPU index; the PyPI
  wheel bundles CUDA.
- **Speed:** about 6 s per epoch on a laptop CPU at v0 size (synthetic benchmark), so Colab runs use a
  T4 GPU runtime.
- **Hyperparameters:** the s5 values for the first run; if the best epoch is the last one allowed,
  training was still improving and `lr`/`max_epochs` are tuned on val (logged here).

### D44. First LightGCN run on v0 (Colab T4, 2026-10-16): gate check and settings kept
- **Run:** `models.lightgcn_val` took 234 s on a T4 (inner search 300 epochs, best epoch 275, then the
  refit); 379 items have no train interactions. Inner NDCG@20 rose quickly to epoch ~200 and changed
  little over the last 60 epochs (`results/v0/training/lightgcn_val.csv`), so the s5 values
  (lr 1e-3, `max_epochs` 300) are kept; no tuning yet.
- **Week 5 gate item "LightGCN >= ALS on val": met.** In `results/v0/val/bootstrap.csv`, ALS minus
  `lightgcn_raw` on NDCG@10 has a 95% CI below zero; on Recall@20 the CI crosses zero. ItemKNN and
  `lightgcn_raw` are not distinguishable on either metric (CIs cross zero).
- **Normalization (s2.2 #1) shows up as expected:** `lightgcn_norm` is below `lightgcn_raw` on all
  users but above it on long-tail truth, with far higher coverage and tail share
  (`table_main.csv`, `table_longtail.csv`). Long-tail claims must compare against both.
- **Baselines reproduced:** the refit baselines match Week 4; ALS differs only in the 7th decimal
  (multithreaded BLAS float order).
- **Colab environment noise (harmless):** pip warns about google-colab's own pins, and `implicit`
  disables its GPU module (CPU ALS is what we use).

## 2026-10-16: Week 5, projection hybrid and controls

### D45. Projection hybrid, shuffled control and static fusion
- **Stage `models.projection_val`** (`ml/models/projection.py`) trains W (dim x 384, no bias,
  Xavier-uniform init, seeded) by BPR with LightGCN users and items frozen, and writes
  `artifacts/<v>/eval_val/projection/` and `.../projection_shuffled/` (`W.npy`, `proj_item.npy`,
  the LightGCN `user.npy` it scores with, `item_degree.npy`, `meta.json` with the normalize flag and
  per-epoch BPR). Training and scoring use the same function: u . e'/|e'| with `normalize_items`
  (scoring re-normalizes the stored, already normalized rows, which changes nothing).
- **New config key `models.projection.n_negs: 5`** (s9.5 says negatives come from
  `sample_negatives` but names no count; 5 matches LightGCN). Each step draws a batch of fit
  positive pairs uniformly with replacement.
- **Cold items:** e_gcn = 0 for items without train interactions, so their vector is W c only.
  They are never positives in the fit window but can be drawn as negatives. Both projection scorers
  rank cold items (`EmbeddingScorer(score_cold=True)`); LightGCN scorers still give them -inf.
- **Shuffled control** permutes the content rows with the stage RNG and is otherwise identical; the
  stage refuses `shuffled_control: false`, since s9.5 requires the control.
- **Static fusion (interpretation of s9.5):** score = alpha * cos(u, e_gcn) + (1 - alpha) *
  cos(p_u, PCA(c)), where PCA reduces content to the LightGCN dim (centered SVD, numpy) and p_u is the
  content-only profile (s9.2) in that space. It is one dot product of concatenated vectors
  [alpha * u/|u|, (1 - alpha) * p_u] . [e_gcn/|e_gcn|, PCA(c)/|PCA(c)|]. Summing
  normalize(e_gcn) + normalize(W_pca c) into one 64-d vector and scoring it with the LightGCN user
  vector (the literal reading) would mix two unaligned spaces, so the content half would be noise.
  Built inside `eval.val` (no training), one row per `models.static_fusion_alphas`; the best alpha is
  read from val only.
- **Speed:** about 6 s per 100 steps on a laptop CPU (10,000 steps per variant), so Colab runs on
  a T4.

### D46. Projection and controls on v0 val (Colab T4, 2026-10-16): a null result, reported as such
- **Run:** `models.projection_val` 75 s (both variants, 10,000 steps each); `eval.val` 35 s.
  `models.lightgcn_val` also reran (its import closure includes `embedding_scorers.py`, changed in
  D45) and reproduced the same best epoch and metrics.
- **Reading (from `results/v0/val/table_main.csv`, `table_longtail.csv`, `bootstrap.csv`):**
  - The projection hybrid is below raw LightGCN on all users (paired CI below zero) and not
    distinguishable from normalized LightGCN; on long-tail truth it is above raw LightGCN but below
    normalized LightGCN.
  - **Projection vs its shuffled control: no difference** on any headline metric. On v0 the
    projection's gain over raw LightGCN on the long tail is a normalization effect, not a semantic
    one (exactly the s2.2 #1 confound).
  - Static fusion (best alpha 0.7 on val) is the strongest fusion row: it beats normalized LightGCN
    on all users and on the long tail (paired CIs above zero), ties raw LightGCN on all users and
    beats it on the long tail.
  - **No fusion row recovers cold items** (cold-item NDCG@10 is zero): cold items are never positives
    in training, and in static fusion their score has only the content half, so warm items outrank
    them. Content-only remains the only model that ranks cold items.
- **Kept as is:** no tuning of the projection on val yet. It stays the long-tail candidate source
  and the `proj_hybrid_score` feature (s13), where the ranker decides how much it helps; static
  fusion stays a control. Possible later val-only experiments (not scheduled): higher projection
  `lr`/`epochs`, W initialized at zero.

## 2026-10-17: Week 5, MVP recommender and bundle v0

### D47. Serve artifacts, bundle v0 and the MVP `recommend`
- **Serve set:** stage `models.lightgcn_serve` trains LightGCN on every interaction (cutoff = newest
  ts + 1 ms, `ml.eval.data.serve_fit_data`) with the same early-stopping recipe as val. Catalogue
  statistics (Bayesian rating, counts, rating std, trending, long tail, reviewed-interaction counts) are
  computed at that cutoff inside `serve.export`.
- **Bundle format lives in `litsense_core/store.py`** and `ml/serve/export.py` imports its names, so
  writer and reader cannot drift. v0 files: `catalog.npz`, `catalog_text.json.gz`, `lgcn_item.f16.npy`,
  `lgcn_item_norm.f16.npy`, `content_item.f16.npy`, `item_nn.npy`, `trending_top.npy`,
  `corpus_users.npz` (the `serving.corpus_users` most active readers: ids, exact LightGCN vectors and
  rating histories for demo accounts), `manifest.json` (version = data version + hash of the file
  hashes, cutoff, shapes, sha256, the config slice the core reads, `components` flags).
  Author, series and genre indices are derived at load from `catalog.npz` instead of shipped as JSON
  (s4.6 lists them as files; derivation is cheap and avoids a second copy). Genres are a uint64
  bitmask over the sorted label set.
- **Not in v0 (later weeks, flagged in `components`):** aspects/descriptors, review centroids,
  snippets, projection, ONNX encoder, spoiler model, LightGBM ranker.
- **MVP choices beyond s12:**
  - "Now" for time decay is the reader's newest event (deterministic profiles).
  - Seen = rated + read/dropped + dismissed + onboarding picks (D12); sources skip seen items so they
    do not waste quota.
  - Until the projection is bundled, the `longtail` source ranks tail items by normalized LightGCN
    when a CF vector exists (D46: normalization carries the long-tail gain), else by content.
  - The `aspect` source is empty until aspects exist.
  - Filters: unknown year or pages pass year/page filters; `min_reviews_main_feed` uses reviewed
    interactions and spares `longtail` items (s13.2).
  - Template 3 for a liked book without stars (a pick or reading-list add) says "which you liked".
  - The blend (s12.3) only scores `cf`, `semantic` and `popularity` ranks, so items found only by
    `item_item`, `author` or `longtail` tie at the bottom of the blend; the exploration slot and the
    Week-9 ranker are what lift them. Kept as specified.
- **Config:** new keys `serving.corpus_users: 300`, `serving.trending_top: 200`,
  `serving.description_chars: 1500`.
- **Dev server without FastAPI:** `dev/dev_server.py` uses the standard library (`http.server`), so it
  adds no dependency; it is never deployed.
- **RSS:** `serve.export` loads the bundle in a fresh interpreter and records RSS in
  `results/<v>/serve/bundle_report.json` (Linux `/proc`; null on Windows).
