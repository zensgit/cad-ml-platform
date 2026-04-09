# Assistant Security Event Filter Validation

## Scope

Branch: `feat/assistant-security-boundary`

Commit:
- `971aa11` `fix: tighten assistant security event time filters`

## What Changed

- Changed `SecurityAuditor.get_events(start_time=...)` behavior:
  - `start_time=0.0` is now treated as an active filter
  - boundary semantics changed from `>=` to `>`
- Added explicit tests for zero-time and boundary-time behavior.

## Key Files

- `src/core/assistant/security.py`
- `tests/unit/assistant/test_assistant_security.py`

## Validation

Commands:

```bash
python3 -m pytest -q tests/unit/assistant/test_assistant_security.py

flake8 \
  src/core/assistant/security.py \
  tests/unit/assistant/test_assistant_security.py \
  --max-line-length=100

python3 -m py_compile \
  src/core/assistant/security.py \
  tests/unit/assistant/test_assistant_security.py
```

Results:

- `63 passed`
- `flake8` passed
- `py_compile` passed

## Risks

- This is a behavior change, not a pure refactor. Callers relying on inclusive boundary semantics will observe stricter filtering.
