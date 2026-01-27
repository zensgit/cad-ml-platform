# DEV_DXF_TITLEBLOCK_SYNONYM_PATCH_20260125

## Objective
Close remaining title-block label gaps by extending the synonym map for shaft parts.

## Changes
- `data/knowledge/label_synonyms_template.json`
  - Added `小减速机轴` as a synonym for label `轴类`.
  - Added `小轴承座(盖)` as a synonym for label `短轴承座(盖)`.

## Rationale
- Two DXF samples reported title-block part name `小减速机轴`, which was not mapped to any existing label.
- One DXF sample reported title-block part name `小轴承座(盖)`, aligning with `短轴承座(盖)` in filenames.
- Mapping to `轴类` aligns with the existing taxonomy without introducing a new class.

## Notes
- These two files have filename labels of `调节螺栓`; the title-block label now maps to `轴类` for coverage tracking.
