# Model Weights Policy

This repository keeps model weights out of Git history by default. Local
checkpoints are expected to live under `models/` and are ignored via
`.gitignore` (e.g. `models/*.pt`).

## Recommended workflow
- Store weights locally in `models/`.
- Document required filenames in README or release notes.
- Keep training artifacts in `reports/` or external storage.

## Optional: Git LFS
If you need to version weights in-repo, enable Git LFS and track the
weight patterns:

```bash
git lfs install
git lfs track "models/*.pt"
```

Then commit `.gitattributes` and the weights. Ensure CI/build runners have
Git LFS enabled before pulling.
