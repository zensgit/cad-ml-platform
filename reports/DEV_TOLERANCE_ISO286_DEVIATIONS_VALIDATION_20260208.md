# DEV_TOLERANCE_ISO286_DEVIATIONS_VALIDATION_20260208

## Goal
Make the ISO 286 / GB/T 1800 deviations table (`data/knowledge/iso286_deviations.json`) reproducible and self-validating:

- Regenerable from the GB/T 1800.2-2020 PDF via script
- Deterministic validation (structure + invariants + spot-checks)
- Assistant output uses correct labels (`H7`, `g6`) and shows limit deviations

## Implementation

### 1) Extractor Normalization
Updated `scripts/extract_iso286_deviations.py` to normalize deviation ordering and table consistency:

- Normalize two-value cells to `(lower, upper)` using `min/max` to handle PDF table ordering differences.
- Post-process aggregated `holes` / `shafts` tables:
  - sort rows by `size_upper`
  - de-dupe duplicate `size_upper`
  - apply a light sign-consistency filter for a few well-known symbol families:
    - shafts: `a..g` must remain non-positive, `h` must have `upper==0`
    - holes: `A..G` must remain non-negative, `H` must have `lower==0`
  - record any drops in `warnings` in the output JSON

### 2) Validation Script
Added `scripts/validate_iso286_deviations.py`:

- Validates JSON structure and invariants:
  - `holes`/`shafts` are objects
  - rows are `[[size_upper, lower, upper], ...]`
  - `size_upper` strictly increasing
  - `lower <= upper`
- Optional deterministic spot-checks (through the public tolerance helpers):
  - `H7 @ 25mm == (0, 21) um`
  - `g6 @ 10mm == (-14, -5) um`
  - `H7/g6 @ 25mm` deviation sanity

### 3) Assistant Formatting
Improved tolerance context formatting so ISO 286 limit-deviation results are rendered as:

- `【极限偏差: H7】`
- `EI/ES` (hole) or `ei/es` (shaft)
- derived tolerance band

Files:
- `src/core/assistant/context_assembler.py`

### 4) Query Diameter Extraction Fix
Adjusted diameter extraction to prefer explicit `mm/毫米` units to avoid capturing grade digits from strings like `IT7` / `H7`.

Files:
- `src/core/assistant/query_analyzer.py`

## Verification

### Scripts
Ran:

```bash
python3 scripts/extract_iso286_deviations.py --pdf "/Users/huazhou/Downloads/GB-T 1800.2-2020 产品几何技术规范（GPS） 线性尺寸公差ISO代号体系 第2部分：标准公差带代号和孔、轴的极限偏差表在线预览.pdf" --out data/knowledge/iso286_deviations.json
python3 scripts/validate_iso286_deviations.py --spot-check
```

Result: `OK` (holes=203, shafts=129).

### Tests
Ran:

```bash
.venv/bin/pytest -q \
  tests/integration/test_tolerance_api.py \
  tests/unit/test_tolerance_limit_deviations.py \
  tests/unit/assistant/test_assistant.py::TestQueryAnalyzer::test_analyze_tolerance_lookup \
  tests/unit/assistant/test_assistant.py::TestQueryAnalyzer::test_analyze_limit_deviation_query \
  tests/unit/assistant/test_assistant.py::TestKnowledgeRetriever::test_retrieve_limit_deviations_h7_25mm \
  tests/unit/assistant/test_assistant.py::TestContextAssembler::test_assemble_formats_limit_deviations_label
```

Result: `20 passed`.

## Notes / Caveats
- The extractor keeps raw `tables` metadata for manual review. Runtime lookup uses the aggregated `holes` / `shafts` sections.
- The post-processor intentionally drops a small number of inconsistent rows for a limited set of symbol families; any drops are recorded in `warnings` in the JSON.

