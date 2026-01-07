# Maintenance Endpoint Error Context Design

## Overview
Standardize maintenance endpoint error responses with explicit operation and
resource identifiers, and ensure cleanup analysis errors return structured
responses.

## Updates
- Added `operation` and `resource_id` context fields to maintenance errors.
- Added structured error handling for analysis result cleanup failures.
- Extended maintenance endpoint tests to validate error context fields.

## Files
- `src/api/v1/maintenance.py`
- `tests/unit/test_maintenance_endpoint_coverage.py`
