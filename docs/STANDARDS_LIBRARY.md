#!/usr/bin/env markdown
# 2D Standards Library (DXF)

This library provides a deterministic, versioned set of 2D standard parts to
seed cold-start recognition and rule learning.

## Generate / Refresh

```
python3 scripts/generate_dxf_standards.py
```

Outputs:
- DXF files: `data/standards_dxf`
- Manifest: `data/standards_dxf/MANIFEST.json`

## Manifest Contract

`MANIFEST.json` is written by the generator and includes:
- `version`: UTC timestamp when the set was generated.
- `generator`: script path used to generate.
- `config`: sizes used for bolts/washers/flanges.
- `file_count` and `files`: inventory of the generated set.

Treat the manifest as the authoritative version fingerprint for the library.

## Versioning Guidelines

- Store the generated DXFs and `MANIFEST.json` together.
- Do not mutate DXFs in-place; regenerate with the script to ensure consistency.
- When updating sizes, change the generator inputs and re-run the script.

## Downstream Usage

- Parsing is handled by `src/adapters/factory.py` (DXF adapter).
- Rule learning script consumes this library:
  - `scripts/learn_from_standards.py`
  - Output rules are stored in `data/knowledge/geometry_rules.json`

## Recommended Workflow

1. Generate standards: `python3 scripts/generate_dxf_standards.py`
2. Learn rules: `python3 scripts/learn_from_standards.py`
3. Review and commit updated `data/knowledge/geometry_rules.json`
