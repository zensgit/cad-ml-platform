#!/usr/bin/env markdown
# UV-Net Encoder Dimension Guard

## Goal
Detect node feature dimensionality mismatches early in inference and return a
safe embedding instead of raising hard errors.

## Behavior
- Verify the node feature tensor is 2D.
- Validate `x.shape[1]` matches the model's `node_input_dim`.
- Return a zero embedding (matching the model embedding size) if validation
  fails.

## Notes
The guard is only applied when a UV-Net model is loaded. Mock mode remains
unchanged.
