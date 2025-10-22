# Repository Guidelines

## Project Structure & Module Organization
Adopt the `src` layout: core modules live in `src/snn_py/` and mirror the runtime domains (`neurons`, `layers`, `learning`, `datasets`, `io`). Keep exploratory notebooks in `notebooks/` with names like `2403-spike-timing.ipynb` so they sort chronologically. Store reusable figures or weight dumps in `assets/` and large artefacts outside of the repo. Tests should shadow the code tree inside `tests/`, using `tests/helpers/` for shared fixtures and fakes. Example scripts belong in `examples/` and must run as standalone entry points.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates the local environment.
- `pip install -e .[dev]` installs the package plus linters, type checkers, and pytest.
- `pytest` runs the full suite; add `-k name` to target a subset during iteration.
- `python -m snn_py.cli --help` sanity-checks the CLI wiring before publishing changes.

## Coding Style & Naming Conventions
Format code with `black` (line length 88) and group imports with `isort --profile black`. Run `ruff check .` to enforce lint rules; fix new warnings before opening a PR. Use type hints everywhere except trivial data containers, and prefer `Protocol`/`TypedDict` when interactions cross modules. Modules and packages stay lowercase with underscores (`learning_rules.py`), classes use CapWords, and public functions use descriptive verbs (`train_snn`, `encode_spike_train`). Keep module-level constants uppercase and document non-obvious tensors with Google-style docstrings.

## Testing Guidelines
Write tests with `pytest`, naming files `test_<module>.py` so discovery stays predictable. Co-locate fixture factories in `tests/helpers/factories.py` and share reusable spike traces through parametrized fixtures. Aim for ≥85 % coverage via `pytest --cov=snn_py --cov-report=term-missing`; justify any exclusions in the PR. When adding stochastic behaviour, freeze seeds (`np.random.seed`, `torch.manual_seed`) and add deterministic regression tests that assert on firing-rate envelopes rather than raw spikes.

## Commit & Pull Request Guidelines
Use imperative, concise commit subjects (`feat: add STDP learning rule`). Group related changes to keep diffs reviewable and include a focused body when context is non-obvious. PRs must describe the change, link related issues, and note any follow-up work. Attach benchmark numbers or plots for performance-sensitive updates and include screenshots or rich diffs for documentation changes. Request at least one review, respond to feedback with follow-up commits (not force-pushes), and wait for CI to pass before merging.
