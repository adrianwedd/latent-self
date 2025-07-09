# Docs Portal Proof-of-Concept

This directory contains a minimal MkDocs configuration serving as a centralized documentation portal.

## Rationale

Backstage was evaluated as a potential solution but was deemed too heavyweight for the current scope. MkDocs with the Material theme provides a lightweight alternative with built-in search and simple deployment.

## Usage

Install the requirements and launch the site. The portal uses the
`mkdocs-monorepo-plugin` to merge documentation from several
repositories into a single searchable site:

```bash
pip install mkdocs mkdocs-material mkdocs-monorepo-plugin
mkdocs serve -f mkdocs.yml
```
The portal indexes Markdown files from this repository as well as
`demo_portal/docs`, showcasing how multiple codebases can be combined
into one portal with full-text search.
