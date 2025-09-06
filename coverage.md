# Test coverage

This repository measures unit test coverage via `pytest-cov`. A GitHub Actions workflow runs on every push and PR, generates a coverage report, enforces a minimum threshold, and uploads the `coverage.xml` as a build artifact.

What the workflow does
- Installs dev dependencies (including `pytest-cov`).
- Runs `pytest --cov=app` with:
  - `--cov-report=term-missing` to show uncovered lines inline.
  - `--cov-report=xml` to produce `coverage.xml`.
  - `--cov-fail-under=80` to fail the job if total coverage drops below 80%.
- Uploads `coverage.xml` as a build artifact.
- Optionally uploads coverage to Codecov if `CODECOV_TOKEN` is configured (Settings → Secrets and variables → Actions).

Configuration files
- `.coveragerc` sets:
  - `branch = True` for branch coverage.
  - `source = app` to focus metrics on first-party code.
  - Omits tests, virtualenvs, and site-packages from coverage.
- You can tune the threshold by editing `--cov-fail-under` in `.github/workflows/coverage.yml`.

Run locally
```bash
# From repo root
pip install -r requirements-dev.txt  # ensure pytest + pytest-cov installed
pytest --cov=app --cov-report=term-missing
```

Tips
- If adding large UI or integration modules, consider marking those tests with `@pytest.mark.slow` and keep the coverage gate focused on core logic.
- For per-file or per-folder tuning, use `[report] fail_under` in `.coveragerc` or set `--cov=app/submodule1 --cov=app/submodule2` in CI.
