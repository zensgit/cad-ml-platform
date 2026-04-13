# A1: Split `src/core/materials/classifier.py` into Modules

## What was done and why

The monolithic `classifier.py` (15,763 LOC) was split into 8 focused sub-modules
plus a backward-compatible re-export wrapper. The original file contained enums,
dataclasses, three large lookup tables (~12k lines of data), and 20+ functions
spanning classification, search, equivalence, cost, compatibility, and export
concerns -- all in a single file.

Splitting improves maintainability, code navigation, and enables independent
change tracking per concern area.

**This is a pure refactoring**: no function signatures, behavior, or public APIs
were changed.

## Module dependency diagram

```
__init__.py  (re-exports all public symbols)
    |
classifier.py  (thin backward-compat wrapper, re-exports everything)
    |
    +-- data_models.py     [enums, dataclasses, MATERIAL_DATABASE,
    |                        MATERIAL_MATCH_PATTERNS, MATERIAL_EQUIVALENCE]
    |
    +-- classify.py        [classify_material_detailed, classify_material_simple,
    |       |                search_materials, PINYIN_MAP, _calculate_similarity]
    |       +-- depends on: data_models
    |
    +-- properties.py      [get_material_info, search_by_properties]
    |       +-- depends on: data_models, classify
    |
    +-- processing.py      [get_process_recommendations, get_material_recommendations,
    |       |                get_alternative_materials, list_applications,
    |       |                APPLICATION_MAP, MATERIAL_ALTERNATIVES]
    |       +-- depends on: data_models, classify
    |
    +-- equivalence.py     [get_material_equivalence, find_equivalent_material,
    |       |                list_material_standards]
    |       +-- depends on: data_models, classify
    |
    +-- cost.py            [get_material_cost, compare_material_costs,
    |       |                search_by_cost, get_cost_tier_info,
    |       |                MATERIAL_COST_DATA, COST_TIER_DESCRIPTIONS]
    |       +-- depends on: data_models, classify
    |
    +-- compatibility.py   [check_weld_compatibility, check_galvanic_corrosion,
    |       |                check_heat_treatment_compatibility, check_full_compatibility,
    |       |                WELD_COMPATIBILITY, GALVANIC_SERIES, GALVANIC_RISK_THRESHOLDS]
    |       +-- depends on: data_models, classify
    |
    +-- export.py          [export_materials_csv, export_equivalence_csv]
            +-- depends on: data_models
```

## List of all re-exports

All symbols listed below are available from both `src.core.materials` and
`src.core.materials.classifier`:

| Symbol | Source module |
|---|---|
| `MaterialCategory` | data_models |
| `MaterialSubCategory` | data_models |
| `MaterialGroup` | data_models |
| `MaterialProperties` | data_models |
| `ProcessRecommendation` | data_models |
| `MaterialInfo` | data_models |
| `MATERIAL_DATABASE` | data_models |
| `MATERIAL_MATCH_PATTERNS` | data_models |
| `MATERIAL_EQUIVALENCE` | data_models |
| `classify_material_detailed` | classify |
| `classify_material_simple` | classify |
| `search_materials` | classify |
| `PINYIN_MAP` | classify |
| `_calculate_similarity` | classify |
| `get_material_info` | properties |
| `search_by_properties` | properties |
| `get_process_recommendations` | processing |
| `get_material_recommendations` | processing |
| `get_alternative_materials` | processing |
| `list_applications` | processing |
| `APPLICATION_MAP` | processing |
| `MATERIAL_ALTERNATIVES` | processing |
| `get_material_equivalence` | equivalence |
| `find_equivalent_material` | equivalence |
| `list_material_standards` | equivalence |
| `get_material_cost` | cost |
| `compare_material_costs` | cost |
| `search_by_cost` | cost |
| `get_cost_tier_info` | cost |
| `MATERIAL_COST_DATA` | cost |
| `COST_TIER_DESCRIPTIONS` | cost |
| `check_weld_compatibility` | compatibility |
| `check_galvanic_corrosion` | compatibility |
| `check_heat_treatment_compatibility` | compatibility |
| `check_full_compatibility` | compatibility |
| `WELD_COMPATIBILITY` | compatibility |
| `GALVANIC_SERIES` | compatibility |
| `GALVANIC_RISK_THRESHOLDS` | compatibility |
| `export_materials_csv` | export |
| `export_equivalence_csv` | export |

## Test verification results

```
$ python3 -m pytest tests/unit/test_material_classifier.py -x -q --timeout=30
1473 passed in 13.78s
```

All 1473 material classifier tests pass without modification.

## Before/after file sizes

### Before
| File | Lines |
|---|---|
| `classifier.py` | 15,763 |
| `__init__.py` | 61 |
| **Total** | **15,824** |

### After
| File | Lines | Contents |
|---|---|---|
| `data_models.py` | 13,701 | Enums, dataclasses, databases |
| `cost.py` | 693 | Cost data & functions |
| `compatibility.py` | 435 | Compatibility checks |
| `processing.py` | 425 | Recommendations |
| `classify.py` | 281 | Classification & search |
| `export.py` | 130 | CSV export |
| `properties.py` | 116 | Property queries |
| `__init__.py` | 103 | Re-exports |
| `classifier.py` | 82 | Backward-compat wrapper |
| `equivalence.py` | 81 | Equivalence table lookups |
| **Total** | **16,047** | |

The slight increase (~220 lines) is due to module headers, imports, and the
backward-compatible wrapper in `classifier.py`.
