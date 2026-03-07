# Knowledge Extraction Enhancements Validation 2026-03-07

## Summary

Extended `build_knowledge_summary()` so the main analysis chain can expose
more user-visible engineering knowledge from drawing text and filename/title
signals without introducing a new model.

Newly recognized signal types:

- `GB/T 1804-*` general tolerance designations
- surface finish requirements from `Ra` values
- surface finish requirements from `N1-N12` grades
- material annotations such as `材料: 304`

This improves the product-facing value of the existing `knowledge_checks`,
`violations`, and `standards_candidates` outputs.

## Files

- `src/core/knowledge/analysis_summary.py`
- `tests/unit/test_knowledge_analysis_summary.py`
- `tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py`

## Implemented behavior

### General tolerance

Added direct parsing of Chinese-equivalent general tolerance designations:

- `GB/T 1804-M`

Output behavior:

- `knowledge_checks[].category = general_tolerance`
- `standards_candidates[].type = general_tolerance`
- includes `equivalent_designation = ISO 2768-M`

### Surface finish

Added parsing for:

- `Ra 3.2`
- `N8`

Output behavior:

- `knowledge_checks[].category = surface_finish`
- `standards_candidates[].type = surface_finish`
- reports both the parsed `ra_um` and normalized `grade`

### Material

Reused the existing materials classifier rather than adding a second material
matching path.

Handled title-block and filename-like fragments such as:

- `材料: 304`
- `材料304`
- `材料: 铝合金 6061`

Output behavior:

- `knowledge_checks[].category = material`
- `standards_candidates[].type = material`
- includes normalized material grade, e.g. `304 -> S30408`

## Validation

Static checks:

```bash
python3 -m py_compile \
  src/core/knowledge/analysis_summary.py \
  tests/unit/test_knowledge_analysis_summary.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py

flake8 \
  src/core/knowledge/analysis_summary.py \
  tests/unit/test_knowledge_analysis_summary.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  --max-line-length=100
```

Targeted tests:

```bash
pytest -q \
  tests/unit/test_knowledge_analysis_summary.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py
```

Observed result:

- `2 passed`

## Verified output examples

From the targeted unit and integration inputs:

- `GB/T 1804-M` is emitted as a `general_tolerance` candidate
- `Ra 3.2` normalizes to `N8`
- `N8` is emitted as a surface-finish candidate
- `304` normalizes to material grade `S30408`

## Notes

- This change intentionally stays within the existing text-driven knowledge
  path; it does not depend on OCR provider changes or new model weights.
- Material extraction uses a conservative prefix-based matcher so strings like
  `材料304 Ra3.2` do not accidentally get classified by unrelated aliases.
