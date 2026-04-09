# Sequence Reference Migration

Date: 2026-03-06

## Scope

This note captures the first migration pass from external CAD-sequence and
CAD-graph references into this repository. The goal is not to copy whole
training stacks. The goal is to isolate reusable representations, sequence
parsing rules, and lightweight model contracts that fit the current
`history_sequence` and future `brep-graph` roadmap.

## Repositories Reviewed

| Repository | License | Primary Use Here |
| --- | --- | --- |
| `UV-Net` | MIT | B-Rep face-adjacency graph and UV-grid representation |
| `AAGNet` | MIT | richer face/edge attributed adjacency extraction |
| `SketchGraphs` | MIT | sketch/history sequence representation ideas |
| `DeepCAD` | MIT | command-plus-parameter sequence schema |
| `HPSketch` | MIT | `.h5` vector dataset format and history inputs |
| `ezdxf` | MIT | DXF text, title block, and entity extraction |
| `CadQuery` | Apache-2.0 | synthetic CAD sample generation |

## Concrete Migration Targets

### History Sequence

| Reference | Borrowed Idea | Local Target |
| --- | --- | --- |
| `HPSketch` `.h5` vectors | `vec` command sequence as a stable history input | `src/ml/history_sequence_tools.py` |
| `SketchGraphs` sequence ops | sequence-centric CAD state transitions | `src/ml/history_sequence_tools.py` |
| `DeepCAD` command sequence encoding | lightweight sequence encoder/classifier split | `src/ml/train/sequence_encoder.py` |

### Future 3D B-Rep Graph

| Reference | Borrowed Idea | Local Target |
| --- | --- | --- |
| `UV-Net/process/solid_to_graph.py` | face adjacency graph construction | `src/core/geometry/engine.py` |
| `AAGNet/dataset/AAGExtractor.py` | richer face/edge attributes, topology checks | `src/core/geometry/engine.py` |
| `UV-Net/datasets/util.py` | UV-grid normalization and augmentation | `src/ml/augmentation/graph.py` |

## What Was Implemented In This Pass

### Added

- `src/ml/history_sequence_tools.py`
  - load `.h5` vectors
  - extract command tokens
  - truncate and summarize sequences
  - discover `.h5` files
  - build stable label maps
- `src/ml/train/sequence_encoder.py`
  - `SequenceCommandEncoder`
  - `SequenceCommandClassifier`
- `src/ml/train/hpsketch_dataset.py`
  - manifest-driven or directory-driven dataset loading
  - padded batch collation
  - label-map bootstrapping

### Updated

- `src/ml/history_sequence_classifier.py`
  - reuses shared `.h5` token loader
  - exposes `sequence_summary` in prediction payloads
  - keeps prototype/model fallback behavior

## Validation Added

- `tests/unit/test_history_sequence_tools.py`
- `tests/unit/test_sequence_encoder.py`
- `tests/unit/test_hpsketch_dataset.py`
- `tests/unit/test_history_sequence_classifier.py`

## What We Did Not Migrate

- `UV-Net` / `AAGNet` training stacks: too tightly coupled to DGL/OCC-specific
  environments for a first-pass import.
- `SketchGraphs` Onshape and CUDA-specific pieces: not compatible with the
  current repository runtime.
- `DeepCAD` generative autoencoder stack: wrong objective for the current
  history classification path.

## Next Steps

1. Add parameter-aware history features instead of command-only tokens.
2. Teach `HistorySequenceClassifier` to accept multiple `vec` schemas and
   optional token remapping.
3. Introduce a dedicated `extract_brep_graph()` path in
   `src/core/geometry/engine.py` using the `UV-Net` and `AAGNet` references.
4. Wire the improved history branch into `HybridClassifier` behind a feature
   flag and shadow evaluation.
