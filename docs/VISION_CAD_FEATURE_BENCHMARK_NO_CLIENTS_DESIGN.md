# Vision CAD Feature Benchmark No-Clients Design

## Overview
Add a CLI switch to skip initializing external vision clients during the CAD
feature benchmark. This keeps benchmark output clean and avoids dependency
warnings.

## CLI
- `--no-clients`: disable external client initialization in `VisionAnalyzer`.

## Notes
- The benchmark uses heuristic extraction only, so external clients are not
  required.
