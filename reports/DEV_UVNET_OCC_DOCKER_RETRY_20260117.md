# DEV_UVNET_OCC_DOCKER_RETRY_20260117

## Summary
Retried pythonocc-core provisioning in Docker with low download concurrency and
local cache mounted; install still stalled during repodata download.

## Steps
- Ran: `docker run --rm --platform linux/amd64 -v "$PWD":/workspace -w /workspace -v "$PWD/.mamba-cache":/root/.cache/mamba \
  -e MAMBA_DEFAULT_REPODATA_FN=repodata.json -e MAMBA_FETCH_THREADS=1 -e MAMBA_DOWNLOAD_THREADS=1 \
  mambaorg/micromamba:1.5.8 bash -lc "micromamba create -y -n uvnet -c conda-forge python=3.10 pythonocc-core pytorch -v && micromamba run -n uvnet python scripts/train_uvnet_graph_dryrun.py --data-dir data/abc_sample"`.

## Results
- The container stalled at repodata download with no progress for several minutes.
- The container was stopped manually to avoid indefinite hang.

## Notes
- Suggest running the dry-run in a Linux/CI environment with stable access to
  conda-forge or a cached mirror.
