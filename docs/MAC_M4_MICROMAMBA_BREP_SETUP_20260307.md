# macOS M4 Micromamba B-Rep Setup 2026-03-07

## Summary

Added a micromamba-based setup path for macOS ARM64 machines that do not already
have Conda installed.

This is intended to unblock local `pythonocc-core` setup for STEP/B-Rep smoke
validation and related 3D feature extraction tasks.

New script:

- `scripts/setup_mac_m4_micromamba.sh`

## Why this was needed

The existing Apple Silicon setup script:

- `scripts/setup_mac_m4.sh`

assumes `conda` is already installed on the machine.

The current workstation state that triggered this gap:

- `conda`: missing
- `mamba`: missing
- `micromamba`: missing
- Docker daemon: unavailable

That combination blocks both:

1. local Conda-based `pythonocc-core` setup
2. docker-based linux/amd64 B-Rep validation

## What the new script does

`scripts/setup_mac_m4_micromamba.sh`:

1. downloads `micromamba` into `${HOME}/.local/bin` if absent
2. creates or updates a local environment under `${HOME}/.micromamba`
3. installs:
   - `pythonocc-core`
   - `ezdxf`
   - `trimesh`
   - `h5py`
   - `pip`
4. optionally installs:
   - `requirements.txt`
   - `pytorch` / `torchvision`
   - PyTorch Geometric extras
5. can optionally run:
   - `scripts/validate_online_example_ai_inputs.py`

## Usage

Minimal B-Rep environment setup:

```bash
bash scripts/setup_mac_m4_micromamba.sh
```

Run online example smoke after setup:

```bash
bash scripts/setup_mac_m4_micromamba.sh --run-smoke
```

Include PyTorch and PyG:

```bash
bash scripts/setup_mac_m4_micromamba.sh --with-pytorch --with-pyg
```

Skip installing `requirements.txt`:

```bash
bash scripts/setup_mac_m4_micromamba.sh --skip-project-requirements
```

## Environment Variables

Supported overrides:

- `ENV_NAME`
- `PYTHON_VERSION`
- `MAMBA_ROOT_PREFIX`
- `MAMBA_BIN_DIR`
- `INSTALL_PROJECT_REQUIREMENTS`
- `INSTALL_PYTORCH`
- `INSTALL_PYG`
- `RUN_ONLINE_SMOKE`
- `H5_FILE`
- `STEP_FILE`
- `SMOKE_OUTPUT`

## Validation

Static validation:

```bash
bash -n scripts/setup_mac_m4_micromamba.sh
```

Recommended runtime verification after environment creation:

```bash
~/.local/bin/micromamba run -r ~/.micromamba -n cad-ml-brep-m4 \
  python -c 'from src.core.geometry.engine import HAS_OCC; print(HAS_OCC)'
```

Optional online example smoke:

```bash
~/.local/bin/micromamba run -r ~/.micromamba -n cad-ml-brep-m4 \
  python scripts/validate_online_example_ai_inputs.py \
  --step-file /private/tmp/cad-ai-example-data-20260307/foxtrot/examples/cube_hole.step \
  --output reports/experiments/20260307/online_example_ai_inputs_validation_micromamba.json
```

### Runtime validation completed on this machine

Executed:

```bash
INSTALL_PROJECT_REQUIREMENTS=0 RUN_ONLINE_SMOKE=1 \
  bash scripts/setup_mac_m4_micromamba.sh
```

Environment result:

- `micromamba` installed at `~/.local/bin/micromamba`
- environment created at `~/.micromamba/envs/cad-ml-brep-m4`
- `pythonocc-core 7.9.3` installed successfully on macOS ARM64
- `HAS_OCC` transitioned from unavailable to available inside the new environment

Generated report:

- `reports/experiments/20260307/online_example_ai_inputs_validation_micromamba.json`

Observed smoke results:

- `.h5` validation: `status=ok`
- `.step` validation: `status=ok`
- `shape_loaded=true`
- `brep_features.valid_3d=true`
- `brep_features.faces=7`
- `brep_features.surface_types.plane=6`
- `brep_features.surface_types.cylinder=1`
- `brep_graph.graph_schema_version=v2`
- `brep_graph.node_count=7`
- `brep_graph.edge_count=28`

Interpretation:

- the local Apple Silicon path is now sufficient for real STEP/B-Rep smoke
  validation without Docker
- the online `cube_hole.step` sample is parsed correctly through the current
  `GeometryEngine` implementation
- the existing `extract_brep_graph()` pipeline is producing the expected `v2`
  graph metadata on this machine

### Directory-level STEP validation also completed

After environment bootstrap, the same machine successfully ran:

```bash
~/.local/bin/micromamba run -r ~/.micromamba -n cad-ml-brep-m4 \
  python scripts/eval_brep_step_dir.py \
  --step-dir /private/tmp/cad-ai-example-data-20260307/foxtrot/examples \
  --output-dir reports/experiments/20260307/brep_step_dir_eval_foxtrot
```

Observed result:

- `sample_size=3`
- `status_counts.ok=3`
- `valid_3d_count=3`
- `graph_schema_version_counts.v2=3`

Related validation note:

- `docs/BREP_STEP_DIR_EVAL_VALIDATION_20260307.md`

## Current limitations

- The current machine still lacks an active Docker daemon, so docker-based
  validation scripts remain unavailable until Docker is restored.
- PyTorch Geometric installation on macOS ARM64 can still depend on upstream
  wheel availability; the script keeps it optional for that reason.
