# Dedup2D Secure Callback E2E Smoke (2026-01-01)

## Scope

- Run the Dedup2D secure callback smoke test against the local API + worker stack.

## Command

- `DEDUPCAD_VISION_URL=http://host.docker.internal:8100 make dedup2d-secure-smoke`

## Results

- OK: secure callback smoke test succeeded.

## Notes

- DedupCAD Vision was reached via the host service on port 8100.
- Docker compose warned about deprecated `version` fields and an orphan container.
