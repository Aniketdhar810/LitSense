# LitSense core API contract (`litsense_core`)

| | |
| --- | --- |
| API version | `litsense_core.__api_version__ = "0.1.0"` |
| Status | Draft. Schemas freeze in Week 6; after that every change bumps the version and gets a changelog entry below. |
| Source of truth | `litsense_core/schemas.py` (Pydantic v2). This document explains it; if they disagree, the code wins and this file gets fixed. |
| Owners | ML/GenAI: core package. Backend: everything that calls it. |

This is the boundary between the ML package and the FastAPI backend (CLAUDE.md §15). The backend builds a
`UserState` from its database on every call, calls a core function, renders the result and persists anything
returned (for example a chat `SessionState`). The core is stateless and never touches the database.

## 1. Using it now: the mock

Until the artifact bundle ships (Week 5), `litsense_core.mock` serves every function below with the same
signatures and schemas, over a deterministic synthetic catalogue of 200 books (`book_id` 0–199).

```python
from litsense_core import mock as core          # Week 1-4
# from litsense_core import <same names>        # Week 5+: same signatures, real models

feed = core.recommend(core_state, k=10)          # -> RecResponse
```

What the mock does and does not do:

- Feed: popularity plus genre and author affinity. Rating a book visibly changes the feed. Seen, dismissed,
  `exclude`, filters, muted genres and the series guard all behave as specified, so backend tests written
  against the mock stay valid.
- Reasons: the §12.5 templates. Explanations, chat and onboarding always report `method: "template"` /
  `"embed_only"`; the mock never calls an LLM.
- Chat: a few regular expressions ("like X but shorter", "more by that author", "why the second one?",
  "something else", "cozier", "nothing older than 2010"). Good enough to build the chat UI and session
  handling.
- Titles, authors, review snippets, aspect bars and review summaries are invented. Summaries and snippets start
  with "Mock". `metrics_report()` returns the admin JSON shape with every metric `null`.
- `load_store` and `download_bundle` do nothing (no network, no files); `health()` reports `"mock": true`.

## 2. Public functions (CLAUDE.md §15.1)

`book_id` everywhere is the integer item index from `id_map.json`; the backend stores it as
`books.model_item_idx`.

| Function | Returns | Notes | Budget (§15.4: CPU, warm, one thread) |
| --- | --- | --- | --- |
| `load_store(bundle_dir: str)` | `ArtifactStore` | Call once at boot (process-wide singleton). | ≤ 20 s, RSS ≤ 350 MB |
| `download_bundle(url: str, dest: str)` | `str` (bundle dir) | Fetches the GitHub Release bundle at boot. | n/a |
| `health()` | `dict` | Bundle version, cutoff, data version, `rss_mb`, `ranker`, `llm`. | not set |
| `recommend(state, k=10, filters=None, exclude=None, use_ranker=True)` | `RecResponse` | Main feed. Never calls the LLM. | p95 ≤ 80 ms (ranker path, 200 candidates) |
| `home_rails(state)` | `HomeRails` | Every home rail in one call. | p95 ≤ 250 ms |
| `similar_books(book_id, k=10, state=None)` | `list[RecItem]` | With a state: hides seen books and personalizes order. | not set |
| `more_by_author(book_id, k=10, state=None)` | `list[RecItem]` | Series order first. | not set |
| `semantic_search(query, k=20, filters=None)` | `list[SearchHit]` | `matched` says whether title, author or meaning matched. | p95 ≤ 60 ms |
| `resolve_entities(query)` | `EntityMatch` | For author-mode search; `ambiguous` returns `options`. | not set |
| `onboarding_grid(n=60)` | `list[int]` | Popular, well-rated, genre-diverse books for the cover picker. | not set |
| `parse_onboarding(text, picks)` | `OnboardingResult` | Show `display_chips` back so the reader can correct them. | not set |
| `explain(state, book_id)` | `Explanation` | LLM text when available and valid, else the template. | template ≤ 10 ms; LLM ≤ 4 s typical, 8 s hard timeout |
| `chat_turn(state, session, message)` | `ChatReply` | See §4. | ≤ 6 s typical (two LLM calls) |
| `book_insights(book_id)` | `BookInsights` | Aspect bars, review summary, spoiler-safe snippets. | not set |
| `analyze_user_review(text)` | `ReviewAnalysis` | Store the result and send it back in `UserState.review_analyses`. | not set |
| `metrics_report()` | `dict` | Admin page JSON (§6). | not set |
| `catalog()` **(proposed)** | `list[CatalogBook]` | Not in CLAUDE.md §15.1; see §7. | not set |

## 3. Key schema rules

The full models are in `litsense_core/schemas.py`. Rules that matter to the backend:

- **Timestamps are epoch milliseconds** (`RatingEvent.ts`, `ListEvent.ts`, `ReviewAnalysis.ts`). Values outside
  `1e11–1e13` are rejected, which catches seconds sent by mistake.
- **Stars are 1–5** (half stars allowed). `book_id` is ≥ 0. An unknown `book_id` anywhere in the state raises
  `ValueError`.
- **`UserState.viewed`**: recent detail-page views, newest first, at most 50. **`onboarding_picks`**: books the
  reader said they enjoyed; feeds treat them as already read and never recommend them back.
- **Seen books** (never in feeds): rated ∪ reading list with status `read` or `dropped` ∪ `dismissed` ∪
  onboarding picks. `want_to_read` and `currently_reading` books can still appear.
- **Interaction bucket** (`RecResponse.bucket`): `n` = ratings + 0.5 × reading-list entries. The reader is in the
  last bucket whose lower bound is ≤ `n`: `cold` 0, `warming` 1, `blending` 5, `warm` 20. So `n = 4.5` is
  `warming`.
- **`RecItem.reason`** is one sentence, safe to show as is. **`Evidence`** holds the facts behind it; its dicts
  contain plain JSON values. `Evidence.allowed_titles()` lists the only titles generated text may mention.
- **Titles in chat**: render titles and covers from `book_id` using your own `books` table. Chat `reason`
  strings never contain the book's own title.
- **`HomeRails.because_you_liked`** is a list of `BecauseYouLiked{anchor_book_id, items}` (up to 2).
  `hidden_gems` stays empty until `n ≥ 5`; `more_by_authors` stays empty unless a favourite author has 2+ books;
  `continue_exploring` needs `viewed`; `from_reading_list` needs `want_to_read` entries.
- **Series guard**: a later volume never appears in a feed until the reader has read the one before it. The exact
  next volume goes to `continue_series` (in `RecResponse` and `HomeRails`), never the main feed.

## 4. Chat protocol

1. First turn: `chat_turn(state, None, message)`. The core creates a `SessionState` with a new `session_id`.
2. Persist `reply.session` exactly as returned (it is JSON; `SessionState.model_validate_json` reads it back)
   and pass it to the next turn. `history` keeps the last 6 messages, each cut to 300 characters.
3. Render `reply.intro`, then each `reply.items[i]` (title and cover from `book_id`, plus `reason`), then
   `reply.followup`. `reply.intent` says what the core understood.
4. **Chips**: `reply.chips` lists the active modifiers (`key`, `label`). To remove one, delete the matching field
   from the session and call `chat_turn(state, session, "")`. An empty message re-runs the session as it stands.
   Key formats: `anchor:book:<id>`, `anchor:author`, `descriptor:<name>`, `aspect:<name>`,
   `filter:<field>` and `filter:genres_include:<genre>` / `filter:genres_exclude:<genre>`.
5. **Clarification**: when `reply.clarification` is set, `items` is empty. Show the question and the options; when
   the reader picks one, send its `value` as the next message.
6. "Why …?" questions return `intent: "explain_recommendation"` with the explained book as the only item and leave
   `session.last_shown` unchanged.

## 5. Behaviour rules (CLAUDE.md §15.3)

- **No raises for model trouble.** If the ranker, the LLM or FAISS fails, the core returns a valid but degraded
  result (`degraded: true`) and logs it. `ValueError` means invalid input only.
- **Thread safety**: the store is read-only after load; the LLM client locks its cache and budget.
- **Caching** is the backend's job, keyed by `(user_id, profile_version)`; the core echoes `profile_version`.
- **LLM off**: with `llm.enabled: false` or `state.overrides.disabled_ai`, every function still works and reports
  `method: "template"`.
- **Privacy**: `user_id` and `corpus_user_id` never leave the core; nothing identifying is sent to the LLM.
- **Logging**: the backend may pass an `on_event` callback (arrives with the LLM client, Week 10) to receive LLM
  events and timings.

## 6. Admin metrics JSON (`metrics_report()`)

Shape from CLAUDE.md §15.6: `data_version`, `generated_at`, `bundle_version`, `dataset{users, items,
interactions, attrition, splits}`, `models[{id, name, mode, metrics{ndcg@10, recall@20, precision@10, mrr@10,
coverage@10}, ci95}]`, `ablation[{id, vs, metric, diff, ci95}]`, `strata[]`, `longtail[]`, `coldstart[]`,
`genai{intent_field_accuracy, rejection_rates, audit}`. Real values come only from `results/` files produced by
the pipeline. The mock adds `"is_mock": true` and leaves every metric `null`. Show `null` as "—", never as 0.

## 7. Open questions for the ML owner and backend owner

1. **`catalog()` (proposed addition).** Nothing in §15.1 gives the backend the catalogue it must seed into
   `books` (title, author, genres, year, pages, description, image, series, `book_id`). The mock offers
   `catalog() -> list[CatalogBook]`; the real core would read the same from the bundle. Accept, or agree on another
   export (for example a CSV in the bundle)?
2. **Onboarding picks as read books.** The mock never recommends a picked book back. Should the backend also store
   picks as ratings, or only as `onboarding_picks`?
3. **Half stars.** The schema accepts any value in 1–5. If the UI only allows whole stars, the backend should round
   before sending.
4. **Pinned versions.** The core pins `pydantic==2.13.5`, `httpx==0.28.1` and `numpy==2.4.6` (see
   `requirements-core.txt`). It runs inside the backend process, so the backend must accept these versions.
5. **Python 3.11** is the serving target (CI tests 3.11 and 3.12).

## 8. Changelog

| Version | Date | Change |
| --- | --- | --- |
| 0.1.0 | 2026-10-12 | First contract and mock. Differences from CLAUDE.md §15.2, all with identical JSON: `because_you_liked` items are `BecauseYouLiked`, clarification options are `ClarificationOption{label, value}`, history entries are `ChatMessage{role, text}`. Added input validation (ms timestamps, 1–5 stars, `book_id ≥ 0`), `Bucket`/`Intent` literals, `ASPECT_LABELS`, and the proposed `CatalogBook`/`catalog()`. |
