## Format Validation Fail Runbook

### Summary
Alert or high count in `format_validation_fail_total` indicates incoming CAD files failing deep format validation under strict mode.

### Metrics
- `format_validation_fail_total{format,reason}` failure breakdown.
- `signature_validation_fail_total{format}` signature-level mismatches.
- `strict_mode_enabled` gauge (1 when strict mode active).

### Common Reasons
| Reason | Meaning | Action |
|--------|---------|--------|
| missing_step_header | STEP header token absent | Verify file not truncated; request re-export as AP214 / AP203. |
| missing_step_HEADER_section | STEP HEADER section missing | Re-export with full header metadata. |
| stl_too_small | STL file below minimal size | Check exporter; ensure facets not zero. |
| iges_section_markers_missing | IGES missing required section markers | Ensure IGES export not partial; avoid proprietary compression. |
| dxf_section_missing | DXF lacks SECTION token | Confirm text DXF vs binary DWG; convert DWG properly. |

### Diagnosis Steps
1. Inspect sample failing file with a hex/text viewer.
2. Compare against a known-good sample (`examples/`).
3. If signature fails but deep passes, possible exporter variation; adjust heuristics.
4. If deep fails for majority of one format, consider disabling strict mode temporarily (`FORMAT_STRICT_MODE=0`).

### Mitigation
- Communicate required export settings to upstream data providers.
- Allow fallback by disabling strict mode while updating heuristics.
- Add new reason mapping or relax specific checks for verified variants.

### Escalation
- If persistent and impacts ingestion SLA, open incident and attach failing samples.
- Update README validation matrix after rule changes.

### Owners
Primary: CAD Platform Ops
Secondary: Data Engineering

