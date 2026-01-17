#!/usr/bin/env markdown
# UV-Net Graph Schema Validation

## Goal
Record graph schema metadata in UV-Net checkpoints and validate inference inputs
against the expected schema when available.

## Behavior
- `UVNetGraphModel` stores optional `node_schema` and `edge_schema` in its
  checkpoint config.
- `UVNetEncoder.encode()` compares input schemas to the checkpoint config when
  both are present.
- Schema mismatches return a zero embedding and emit an error log.

## Notes
Schema checks are best-effort and only enforced when both the model config and
input provide schema definitions.
