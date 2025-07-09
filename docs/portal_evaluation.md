# Documentation Portal Options

This document outlines three approaches considered for centralising the project's documentation.

## Backstage TechDocs

* **Pros**: Rich plugin ecosystem; integrates well with infrastructure at scale.
* **Cons**: Requires a persistent service and additional maintenance; heavy for a small project.

## MkDocs Material

* **Pros**: Lightweight static site generator; Material theme provides built‑in search and good UX; easy to host anywhere.
* **Cons**: Limited plugin ecosystem compared to Backstage; search is static and not analytics-driven.

## Custom Portal

* **Pros**: Maximum flexibility to integrate with bespoke tooling and UI.
* **Cons**: Higher initial effort; search and navigation must be built from scratch.

### Recommendation

For the current scope, **MkDocs Material** strikes the best balance of simplicity and features. A proof‑of‑concept exists under `deploy/docs`.
The portal uses the `mkdocs-monorepo-plugin` so documentation from
multiple repositories can be indexed in one searchable site.
