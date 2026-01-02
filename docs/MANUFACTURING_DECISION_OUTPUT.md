#!/usr/bin/env markdown
# Manufacturing Decision Output

When quality, process recommendation, or cost estimation is available, the
analysis response includes a `manufacturing_decision` object under `results`.

Example:

```json
{
  "manufacturing_decision": {
    "feasibility": "high",
    "risks": [],
    "process": {"process": "cnc_milling", "method": "3_axis"},
    "cost_estimate": {"total_unit_cost": 12.34, "currency": "USD"},
    "cost_range": {"low": 11.11, "high": 13.57},
    "currency": "USD"
  }
}
```

Notes:
- `feasibility` and `risks` are derived from DFM output.
- `process` is the primary recommendation (or legacy process fields).
- `cost_range` is a simple +/-10% band around `total_unit_cost`.
