# Vision CAD Feature Benchmark Compare Design

## Overview
Add an optional JSON comparison mode to the CAD feature benchmark script so
baseline deltas are captured for tuning regressions.

## CLI
- `--compare-json`: path to a prior benchmark JSON result.

## Output
- Prints per-combo summary deltas and per-sample deltas to stdout.
- When `--output-json` is supplied, embeds a `comparison` block in the JSON
  payload alongside `results`.

## Tests
- Command-level validation using a baseline JSON in reports.
