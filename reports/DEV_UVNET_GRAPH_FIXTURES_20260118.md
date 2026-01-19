# DEV_UVNET_GRAPH_FIXTURES_20260118

## Summary
- Added two STEP fixtures to broaden UV-Net graph dry-run coverage.
- Updated the dry-run workflow to copy all STEP/STP fixtures into `data/abc_sample`.

## Fixtures
- `tests/fixtures/eight_cyl.stp`
  - Source: https://raw.githubusercontent.com/tpaviot/pythonocc-core/master/test/test_io/eight_cyl.stp
- `tests/fixtures/as1_oc_214.stp`
  - Source: https://raw.githubusercontent.com/tpaviot/pythonocc-core/master/test/test_io/as1-oc-214.stp

## Validation
- Confirmed ISO-10303 headers for both fixtures:
  - `ISO-10303-21;` + `HEADER;`
- File sizes:
  - `eight_cyl.stp`: 63,663 bytes
  - `as1_oc_214.stp`: 433,606 bytes
- Runtime parsing with `pythonocc-core` was not executed locally due to the missing OCC runtime; rerun the
  UV-Net graph dry-run workflow to exercise the new fixtures.
