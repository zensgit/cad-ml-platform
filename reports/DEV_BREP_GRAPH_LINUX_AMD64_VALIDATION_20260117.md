# DEV_BREP_GRAPH_LINUX_AMD64_VALIDATION_20260117

## Summary
Attempted to run the B-Rep graph extraction integration test inside a linux/amd64 micromamba
container with pythonocc-core; the conda-forge repodata download timed out.

## Environment
- Host: macOS (arm64)
- Docker image: mambaorg/micromamba:1.5.8 (linux/amd64)
- Cache volume: cadml-micromamba-cache
- MAMBA_NO_REPODATA_ZST=1

## Steps
- Started a linux/amd64 micromamba container with the repo mounted at `/work`.
- Ran `micromamba create -y -n cadml -c conda-forge python=3.10 pythonocc-core pytest`.
- Intended to run `pytest tests/integration/test_brep_graph_extraction.py -v` after env creation.

## Results
- micromamba failed to download `conda-forge/noarch/repodata.json.zst` due to a timeout:
  "Operation too slow. Less than 30 bytes/sec transferred the last 60 seconds".
- Environment was not created; test did not run.

## Next Steps
- Retry on a stable linux/amd64 host or with a more reliable network path.
- Consider mirroring conda-forge or using a closer channel mirror for repodata downloads.
