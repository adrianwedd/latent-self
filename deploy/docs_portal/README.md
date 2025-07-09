# Docs Portal Proof-of-Concept

This directory contains a minimal MkDocs configuration serving as a centralized documentation portal.

## Rationale

Backstage was evaluated as a potential solution but was deemed too heavyweight for the current scope. MkDocs with the Material theme provides a lightweight alternative with built-in search and simple deployment.

## Usage

Install the requirements and launch the site:

```bash
pip install mkdocs mkdocs-material
mkdocs serve -f mkdocs.yml
```

The portal indexes all Markdown files under the repository `docs/` directory, providing full-text search across the documentation.
