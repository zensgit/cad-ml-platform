# B2: Data Collection Toolchain Design

## 1. Data Scarcity Analysis

The 25-class taxonomy v2 requires approximately 30 samples per class minimum for viable
supervised training. Current inventory of ~8,876 DXF files is large but concentrated:
- ~55% in the "other" catch-all class
- Many rare classes (锥体, 弹簧, 阀门) have fewer than 10 samples
- Estimated 300-400 additional samples needed to reach minimum viability

### Strategy
1. **Relabel** existing "other" files that actually belong to a taxonomy v2 class
2. **Augment** existing files with geometric transformations (3-5x multiplier)
3. **Collect** targeted samples for persistently underrepresented classes
4. **Synthesize** via `scripts/synthesize_dxf_v2.py` for geometry-driven classes

## 2. Annotation Workflow

### Tool: `scripts/label_annotation_tool.py`

Interactive CLI for labeling DXF files against taxonomy v2.

**Flow**:
1. Scan input directory for DXF files
2. For each file:
   - Extract Chinese part name from filename (regex)
   - Match against `label_synonyms_template.json`
   - Map synonym to taxonomy v2 class
   - Present suggestion to annotator
3. Annotator accepts, overrides, skips, or quits
4. Results appended to CSV (supports resume)

**Usage**:
```bash
# Dry run — preview extractions without writing
python scripts/label_annotation_tool.py \
  --input-dir data/training_merged_v2/ --dry-run

# Interactive annotation
python scripts/label_annotation_tool.py \
  --input-dir data/training_merged_v2/

# Auto-accept all suggestions (batch mode)
python scripts/label_annotation_tool.py \
  --input-dir data/training_merged_v2/ --non-interactive
```

**Output**: `data/annotations/manifest_annotated.csv`

| Column | Description |
|--------|-------------|
| file_path | Absolute path to DXF file |
| filename | Basename |
| extracted_name | Chinese part name from filename |
| synonym_label | Matched label from synonyms table |
| taxonomy_v2_class | Mapped taxonomy v2 class |
| annotator_label | Final label (user-confirmed or overridden) |
| timestamp | ISO 8601 annotation time |

## 3. Augmentation Strategy

### Tool: `scripts/augment_dxf_data.py`

Applies geometric augmentations to DXF files via ezdxf:

| Augmentation | Range | Rationale |
|-------------|-------|-----------|
| Rotation | 0/90/180/270 degrees | CAD drawings viewed at different orientations |
| Scale | 0.8x - 1.2x | Size-invariant feature learning |
| Mirror | Horizontal / Vertical | Symmetry-invariant features |
| Entity dropout | 5-10% | Robustness to incomplete drawings |

**Usage**:
```bash
# Dry run — preview augmentations
python scripts/augment_dxf_data.py \
  --input-dir data/training_merged_v2/ --dry-run

# Generate 5 copies per file
python scripts/augment_dxf_data.py \
  --input-dir data/training_merged_v2/ --copies 5

# Custom output directory
python scripts/augment_dxf_data.py \
  --input-dir data/training_merged_v2/ \
  --output-dir data/augmented/ --copies 3
```

**Design notes**:
- `ezdxf` import is deferred: `--dry-run` works without ezdxf installed
- Deterministic augmentations via seed for reproducibility
- Output filenames encode augmentation parameters: `{stem}_aug{idx}_r{angle}_s{scale}.dxf`

## 4. Manifest Schema

### Tool: `scripts/build_unified_manifest.py`

Merges all data sources into a single training manifest.

**Scanned directories**:
- `data/training*` (all versions)
- `data/standards_dxf/`
- `data/synthetic_v2/`
- `data/augmented/`

**Merge logic**:
1. Scan all directories for DXF files
2. For each file: extract part name, match synonyms, map to taxonomy v2
3. Override with manual annotations from `data/annotations/*.csv`
4. Exclude noise/educational labels
5. Report class distribution and gap analysis

**Usage**:
```bash
# Dry run with gap analysis
python scripts/build_unified_manifest.py --dry-run

# Build manifest with custom target
python scripts/build_unified_manifest.py --target-per-class 50
```

**Output**: `data/manifests/unified_manifest_v2.csv`

| Column | Description |
|--------|-------------|
| file_path | Absolute path |
| filename | Basename |
| extracted_name | Chinese part name |
| synonym_label | Matched synonym label |
| taxonomy_v2_class | Final class (25-class) |
| source | "annotation" or "auto" |
| split | Train/val/test (assigned later) |
| timestamp | Build time |

## 5. Data Directories

```
data/
  annotations/          # Manual annotation CSVs
    manifest_annotated.csv
  augmented/            # Augmented DXF files
  manifests/            # Unified manifests
    unified_manifest_v2.csv
```

## 6. Testing

Unit tests in `tests/unit/test_annotation_tool.py` cover:
- Filename extraction for various naming conventions
- Synonym matching (exact, partial, no match)
- Taxonomy v2 mapping correctness
- CSV resume logic
- Augmentation parameter determinism and ranges
- DXF file iteration with recursive/non-recursive modes
