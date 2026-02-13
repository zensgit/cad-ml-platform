# DEV_TITLEBLOCK_INSERT_VIRTUAL_ENTITIES_SUPPORT_20260213

## Summary

Improved DXF titleblock extraction by teaching `TitleBlockExtractor` to read static
TEXT/MTEXT/DIMENSION content inside `INSERT` blocks via ezdxf `virtual_entities()`.
This increases robustness when title blocks are implemented as reusable blocks (common in CAD).

## Problem

`src/ml/titleblock_extractor.py` previously extracted:
- modelspace TEXT/MTEXT/DIMENSION/ATTRIB
- INSERT ATTRIB values (via `insert.attribs`)

But many drawings store titleblock content as static TEXT/MTEXT inside the block definition
instead of ATTRIB values, which meant titleblock signals could be missed (especially when
filenames are masked/unreliable).

## Changes

- `src/ml/titleblock_extractor.py`
  - For `INSERT` entities, after processing `insert.attribs`, also scans
    `insert.virtual_entities()` (best-effort) and records text-like entities:
    - TEXT / MTEXT / DIMENSION / ATTRIB
  - Recorded texts follow the same region gating (bottom-right) and parsing logic as
    modelspace entities.

- `tests/unit/test_titleblock_extractor.py`
  - Added `test_titleblock_extraction_from_block_text`:
    - builds a block with TEXT titleblock lines
    - inserts it into modelspace
    - asserts extracted `part_name` and `material` are detected.

## Validation

Executed:

```bash
.venv/bin/python -m pytest tests/unit/test_titleblock_extractor.py -v
.venv/bin/python -m pytest tests/unit/test_golden_dxf_hybrid_manifest.py -v
make validate-core-fast
```

Results:

- `tests/unit/test_titleblock_extractor.py`: `5 passed`
- `tests/unit/test_golden_dxf_hybrid_manifest.py`: `1 passed`
- `make validate-core-fast`: passed

## Risk Assessment

- Low risk:
  - New logic is best-effort and guarded; failures fall back to previous behavior.
  - Only inspects text-like virtual entities, not geometry.
  - Covered by a new unit regression test.

