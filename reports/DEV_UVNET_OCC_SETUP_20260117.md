# DEV_UVNET_OCC_SETUP_20260117

## Summary
Attempted to provision pythonocc-core locally to enable STEP-based UV-Net graph dry-runs.

## Steps
- Ran: `source .venv-graph/bin/activate && pip install pythonocc-core`.
- Re-ran: `source .venv-graph/bin/activate && python3 scripts/train_uvnet_graph_dryrun.py --data-dir data/abc_subset`.
- Downloaded micromamba to `.tools/bin/micromamba` and ran `./.tools/bin/micromamba --version`.
- Ran a docker-based micromamba setup:
  `docker run --rm --platform linux/amd64 -v "$PWD":/workspace -w /workspace mambaorg/micromamba:1.5.8 bash -lc "micromamba create -y -n uvnet -c conda-forge python=3.10 pythonocc-core pytorch -v && micromamba run -n uvnet python scripts/train_uvnet_graph_dryrun.py --data-dir data/abc_sample"`.
  A second attempt set `MAMBA_DEFAULT_REPODATA_FN=repodata.json` but stalled and was stopped.
  A third attempt added low download concurrency and cache:
  `docker run --rm --platform linux/amd64 -v "$PWD":/workspace -w /workspace -v "$PWD/.mamba-cache":/root/.cache/mamba -e MAMBA_DEFAULT_REPODATA_FN=repodata.json -e MAMBA_FETCH_THREADS=1 -e MAMBA_DOWNLOAD_THREADS=1 mambaorg/micromamba:1.5.8 bash -lc "micromamba create -y -n uvnet -c conda-forge python=3.10 pythonocc-core pytorch -v && micromamba run -n uvnet python scripts/train_uvnet_graph_dryrun.py --data-dir data/abc_sample"` (stalled).

## Results
- `pip install pythonocc-core` failed: no matching distribution for this Python environment.
- Dry-run skipped with the expected message because OCC is unavailable.
- Local micromamba binary was killed on launch (exit code 137).
- Docker micromamba attempt failed due to a partial `repodata.json.zst` download; retry stalled without progress.

## Notes
- A conda-based environment may be required for pythonocc-core on macOS.
- The docker-based attempt may need a more stable network or cached repodata mirror to complete.
