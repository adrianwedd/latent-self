# Contributing to Latent Self

Thank you for considering contributing! We welcome pull requests and issues.

## Development Setup
1. Install Python 3.11 and create a virtual environment.
2. Install the project dependencies and optional development tools.
3. Run `python -m py_compile latent_self.py ui/*.py` before committing.
4. Execute tests with `pytest`.

## Dependency Installation
Activate your virtual environment and install all Python requirements:

```bash
pip install -r requirements.txt
```

For linting, type checking, documentation, and packaging you may also install
the optional development tools used in CI:

```bash
pip install mypy ruff mkdocs mkdocs-material pyinstaller
```

## Running Tests
To execute the unit tests run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install mypy ruff mkdocs mkdocs-material pyinstaller  # optional
pytest
```

The tests use the sample configuration files in the `data/` directory and the
latent direction archive `models/latent_directions.npz.txt`. Ensure these files
are present before running the suite. No manual
environment variables are required; the test suite sets `PASS` internally when
testing MQTT authentication.

## Code Style
- Follow [PEP8](https://peps.python.org/pep-0008/) and type-hint all public APIs.
- Use Google-style docstrings.

## Pull Request Checklist
- [ ] Tests pass (`pytest`)
- [ ] `python -m py_compile latent_self.py ui/*.py`
- [ ] Updated documentation or comments
- [ ] Updated `tasks.yml` statuses

## Issue Labels
- `bug`: something isn't working
- `enhancement`: new feature or request
- `documentation`: docs improvements
- `maintenance`: refactoring or cleanup

## Release Process
For maintainers: follow [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) when preparing a new version.
