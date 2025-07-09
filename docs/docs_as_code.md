# Docs-as-Code Workflow

This project stores all documentation in version control alongside the code.
Documentation lives in the `docs/` directory and is built with [MkDocs](https://www.mkdocs.org/).
A GitHub Actions workflow builds and deploys the site whenever changes are merged to `main`.

## Writing Documentation

1. Edit or add Markdown files under `docs/`.
2. Run `mkdocs serve` locally to preview changes.
3. Open a pull request with your documentation updates.

## Continuous Integration

The `.github/workflows/docs.yml` workflow installs MkDocs, builds the docs for the
main application and the demo portal, checks links with `lychee`, and publishes
all generated sites to GitHub Pages when the `main` branch is updated.
