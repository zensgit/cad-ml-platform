# DEV_WEEK7_UNIGNORED_TESTS_20251223

## Scope
- Un-ignore pytest addopts and validate full test suite behavior.
- Fix failures in phase14/18/19/22/24 vision suites and OCR-only integration flow.

## Changes Applied
- Added `TOP_SECRET` to security governance data classification and kept `HIGHEST` as alias.
- Prevented phase22 exports in `src/core/vision/__init__.py` from overriding phase19/14 enums and factories (P22 aliases).
- Implemented Phase14 access control API expectations:
  - Added `ANALYZE` permission and `IMAGE` resource type.
  - Added `Policy` dataclass + `PolicyEngine` with add/get/list.
  - Expanded `RoleManager` for user-role assignment, permission checks, and user-role resolution.
  - Added `AccessController` convenience APIs (`create_user`, `check_access`, enhanced `register_resource`).
  - Implemented `AccessControlVisionProvider` gating with `create_access_control_provider` args.
- Ensured OCR-only flow returns minimal description when `include_description=False`.
- Unified transformation enums across phase18/24:
  - Expanded feature-store `TransformationType` to include lifecycle operations.
  - Reused feature-store `TransformationType` inside data lifecycle.

## Tests
- Command: `.venv/bin/python -m pytest tests -v --override-ini addopts=`
- Result: `3934 passed, 42 skipped, 5 warnings`.

## Warnings Observed
- Unknown pytest marks: `performance`, `slow`.
- Pydantic v2 deprecation warning for `__fields__`.

## Files Updated
- `src/core/vision/security_governance.py`
- `src/core/vision/__init__.py`
- `src/core/vision/access_control.py`
- `src/core/vision/manager.py`
- `src/core/vision/feature_store.py`
- `src/core/vision/data_lifecycle.py`
