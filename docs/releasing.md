# Releasing

How a `batlab` release is cut, and what CI does and does not do on its own.

## The short version

```bash
# 1. Bump the version in the three places that state it (see below)
# 2. Add a CHANGELOG entry
# 3. Commit, tag, push the tag
git tag v0.3.0
git push origin v0.3.0
```

The `Release` workflow (`.github/workflows/release.yml`) then builds the sdist
and wheel, runs `twine check`, **fails if the tag, `pyproject.toml` and the
wheel's own metadata disagree**, attaches the artifacts to a GitHub release, and
publishes to PyPI when `PYPI_API_TOKEN` is configured.

## The name situation, and why it matters

| | Value |
|---|---|
| PyPI distribution | **`battery-lab`** |
| Import package | **`batlab`** |
| PyPI's `batlab` | Lexcelon's unrelated Batlab V1.0 hardware library (v0.5.8) — taken since long before this project |

`battery-lab` returned `404` on `https://pypi.org/pypi/battery-lab/json` when this
was checked (2026-09-19), i.e. it is unclaimed. **An unclaimed name on PyPI
belongs to whoever uploads first**, so the first release claims it as a side
effect. Until then, verify the pair is still free before tagging:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/battery-lab/json   # want 404
```

`tests/test_public_api.py` pins the naming pair in the packaging metadata, because
renaming the distribution back to `batlab` would silently collide.

## Version consistency

Three files state the version and all three must agree — the release workflow
checks the first two against the tag:

1. `pyproject.toml` → `[project] version`
2. `batlab/__init__.py` → `__version__` (asserted equal to `pyproject` by
   `tests/test_public_api.py`)
3. `CITATION.cff` → `version` / `date-released` (not tagged-checkable, so update
   it in the same commit)

## Local dry run

Worth doing before pushing a tag, since it catches the same class of error
without burning a workflow run:

```bash
python -m pip install --upgrade build twine
python -m build                      # dist/battery_lab-<version>.tar.gz + .whl
twine check dist/*

python -m venv /tmp/wheel-check
/tmp/wheel-check/bin/pip install dist/*.whl
/tmp/wheel-check/bin/python -m batlab version
/tmp/wheel-check/bin/python -m batlab cite
```

The `wheel` job in `.github/workflows/ci.yml` runs exactly that (plus the
quickstart snippet) on every push, across Linux/macOS/Windows × Python 3.10–3.13,
so a broken build is normally caught well before a tag exists.

## Publishing credentials

The workflow publishes only when `PYPI_API_TOKEN` exists as a repository secret.
Without it the run still builds, checks and attaches the artifacts to the GitHub
release, and prints a warning saying nothing was published — a release that
silently skipped PyPI would be worse than one that says so.

Preferred, once the project is on PyPI: switch `pypa/gh-action-pypi-publish` to
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC, no stored
token) and delete the secret.

## Zenodo DOI

`CITATION.cff` carries a versioned DOI for the concept, which resolves to the
latest archived release. After tagging, archive the release on Zenodo, update
`CITATION.cff`'s `version`/`date-released` and the DOI description if the concept
recording changed, and commit that as a follow-up — a citation file that points at
the wrong release is a citation that misattributes the work.

## What CI does not do

- **It does not bump versions for you.** A release workflow that writes version
  numbers is a workflow that can publish a version nobody chose.
- **It does not decide that a number changed.** A metric movement is recorded
  through `scripts/update_metric_baselines.py` with a reason, reviewed in the
  diff, and gated by the accuracy regression gate — see
  [API stability](api_stability.md#what-this-policy-does-not-cover-the-numbers).
- **It does not touch the demo app's deployment.** Streamlit Cloud and the REST
  container are deployed from `master`, not from tags.
