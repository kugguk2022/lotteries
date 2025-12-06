# Contributing

Thanks for helping! This project is research-focused; please keep experiments isolated and documented.

## Getting started

- Fork + clone, create a virtualenv, then `pip install -e ".[dev]"`.
- Run `make test` before sending a PR (Ruff + pytest, no network).

## Good first experiments

- Add documentation examples (e.g., small notebooks showing new metrics).
- Tighten parsers when upstream HTML changes (Totoloto/EuroDreams).
- Extend tests around caching or schema edge cases.
- Contribute another lottery fetcher with a clear schema + tests.

## Guidelines

- Prefer pure-Python, offline tests; avoid hitting remote endpoints in CI.
- Document inputs/outputs for any new scripts; lab work should live in `labs/` or a dedicated subfolder.
- Keep public API changes minimal (`euromillions` exports are the stable surface); mark anything experimental in docstrings/README notes.
- Open a GitHub Discussion if you want to propose a larger research direction before coding.

## Releases

Tag milestones (e.g., `v0.1` for stable EuroMillions fetch + schema + scoring) so downstream users can pin versions and cite stable URIs.
