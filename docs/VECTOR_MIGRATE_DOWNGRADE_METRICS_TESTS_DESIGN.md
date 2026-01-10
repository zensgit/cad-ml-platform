# Vector Migrate Downgrade Metrics Tests Design

## Overview
Extend vector migration metrics tests to verify the downgraded status counter
increments when a v4 vector is migrated to a lower version.

## Updates
- Added downgrade metrics assertion with a safe skip when Prometheus collectors
  are unavailable.
- Seeded a v4 vector with expected dimensions for deterministic downgrade.

## Files
- `tests/unit/test_vector_migrate_metrics.py`
