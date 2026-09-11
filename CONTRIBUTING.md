# Contributing

Issues and pull requests are welcome. Please follow the [code of conduct](CODE_OF_CONDUCT.md),
and report security problems privately as described in [SECURITY.md](SECURITY.md).

## Set up

```bash
git clone https://github.com/OussamaMesbah/sperner.git && cd sperner
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"
```

## Before you open a pull request

```bash
ruff format .
ruff check .
pytest
```

CI runs the same checks on Python 3.10 and 3.13, checks that the package installs and
imports without optional dependencies, and builds it. Add a test for new behaviour, and an
entry under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for anything users will notice.

A few rules keep the results trustworthy:

- Decisions in the walk and the division use integers and fractions, never floating point.
- A labeling that breaks the Sperner condition raises an error; nothing is repaired
  silently.
- The package has no runtime dependencies. NumPy and SciPy are for the benchmark's
  baseline only.
- A change to the algorithms comes with a new benchmark run
  (`python -m benchmarks.run --out benchmarks/results.md`), and README.md shows its numbers.

## Dependencies

The web app on Streamlit Community Cloud installs the exact versions in
`requirements.txt`. After changing the `app` extra in `pyproject.toml`, regenerate it:

```bash
uv pip compile pyproject.toml --extra app --python-version 3.12 -o requirements.txt
```

Dependabot proposes updates for it and for the GitHub Actions once a month.

## Releases

1. Set the new version in `pyproject.toml`, `sperner/__init__.py` and `CITATION.cff` (a
   test checks that they agree), add `date-released` to `CITATION.cff`, and replace
   "(unreleased)" in the CHANGELOG.md heading with the date.
2. Merge the pull request into `main` once CI passes.
3. Tag `main` and push the tag:

   ```bash
   git switch main && git pull
   git tag -a v0.3.0 -m "sperner 0.3.0"
   git push origin v0.3.0
   ```

4. The release workflow (`.github/workflows/publish.yml`) runs the tests, checks that the
   tag matches the version, builds the package and creates the GitHub release with the
   changelog section as its notes. It publishes to PyPI only when the repository variable
   `PUBLISH_TO_PYPI` is `true`.
