# DEV_GNN_DEPENDENCY_CHECK_20260117

## Summary
Checked the local Python runtime for PyTorch and PyTorch Geometric availability to
select the UV-Net backend path.

## Steps
- Ran: `python3 -c "import torch; import torch_geometric"` (inline script to print versions).

## Results
- System Python did not have `torch` or `torch_geometric` installed.
- Fallback path remains available via the pure PyTorch GCN implementation.

## Notes
- Tests and model runs should use a dedicated virtualenv (e.g. `.venv-graph`) where
  `torch` is installed.
- `torch-geometric` remains optional; the runtime will use pure PyTorch if missing.
