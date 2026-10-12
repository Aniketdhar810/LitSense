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
