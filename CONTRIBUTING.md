# Contributing to Latent Self

Thank you for considering contributing! We welcome pull requests and issues.

## Development Setup
1. Install Python 3.11 and create a virtual environment.
2. `pip install -r requirements.txt`
3. Run `python -m py_compile latent_self.py ui/*.py` before committing.
4. Execute tests with `pytest`.

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
