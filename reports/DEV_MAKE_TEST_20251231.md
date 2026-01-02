# Full test run (2025-12-31)

## Command
- `make test`

## Results
- **Pass**: 3965
- **Skipped**: 25
- **Duration**: 147.20s
- **Coverage**: 71% total (`htmlcov/` generated)

## Notes
- Skips are expected optional/performance coverage (e.g., psutil-dependent tests).
- Coverage is below the 80% target globally; no coverage gate was enforced by `make test`.
