# FAISS Installation (2025-12-31)

## Scope

- Install FAISS CPU build in the local virtual environment to validate the FAISS backend path.

## Steps

- Installed `faiss-cpu` and pinned `numpy` back to 1.26.4 for compatibility with existing deps.

## Notes

- `faiss-cpu` pulled in `numpy>=1.25,<3`. Downgraded to `numpy==1.26.4` to satisfy scipy/pandas/matplotlib.
