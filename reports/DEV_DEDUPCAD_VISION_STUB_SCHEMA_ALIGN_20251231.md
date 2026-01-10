# DedupCAD Vision Stub Schema Alignment (2025-12-31)

## Scope

- Align the local dedupcad-vision stub response shape with contract expectations.

## Changes

- Expanded `level_stats` in `scripts/dedupcad_vision_stub.py` to include per-level objects
  with `passed`, `filtered`, and `time_ms` fields.

## Notes

- Required to run `make test-dedupcad-vision` when a real dedupcad-vision instance is
  unavailable.
