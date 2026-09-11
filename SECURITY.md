# Security policy

## Supported versions

Fixes go into the latest release only.

| Version | Supported |
|---|---|
| 0.3.x | yes |
| older | no |

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately instead:
on the repository's **Security** tab, choose **Report a vulnerability**
([direct link](https://github.com/OussamaMesbah/sperner/security/advisories/new)). Describe
what an attacker could do, how to reproduce it and which version is affected. The
maintainer aims to reply within a week and will credit you in the release notes unless
you prefer otherwise.

## Scope

In scope: the `sperner` package, the web app in `streamlit_app.py`, the benchmark and the
GitHub Actions workflows.

Out of scope: whether a split suits your situation. What a result guarantees, and under
which assumptions, is stated in [docs/THEORY.md](docs/THEORY.md).

## How the project handles data and secrets

The package has no runtime dependencies, makes no network requests and stores nothing.
The web app keeps answers in the memory of the current session only and writes nothing to
disk; "Save the answers" hands them to the user as a file. Neither needs credentials. The
workflows run with a read-only `GITHUB_TOKEN` unless a job states otherwise (creating a
release needs write access to contents), and publishing to PyPI uses trusted publishing,
so no PyPI token is stored anywhere. Secret scanning with push protection is enabled for
the repository, and Dependabot keeps the pinned actions and the app's dependencies up to
date.
