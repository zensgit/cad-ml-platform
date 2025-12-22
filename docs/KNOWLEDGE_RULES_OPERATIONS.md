#!/usr/bin/env markdown
# Knowledge Rules Operations

This guide documents versioning and hot-reload operations for dynamic knowledge rules.

## Storage

Rules are stored as JSON files under `data/knowledge/` (per category). The in-memory
cache is built from these files on service start, and can be reloaded on demand.

## Hot Reload (API)

Requires `maintenance` API key.

```
POST /api/v1/maintenance/knowledge/reload
```

Response includes previous/current version and a `changed` flag.

## Status (API)

```
GET /api/v1/maintenance/knowledge/status
```

Returns:
- current version
- total rule count
- counts by category

## Hot Reload (CLI)

```
python3 scripts/reload_knowledge.py
```

This reloads rules from disk and prints the version change.

## Recommended Workflow

1. Update rule JSON under `data/knowledge/`
2. Call the reload endpoint (or CLI script)
3. Verify with `/knowledge/status`
