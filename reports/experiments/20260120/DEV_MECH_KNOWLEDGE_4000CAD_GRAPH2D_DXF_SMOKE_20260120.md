# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_DXF_SMOKE_20260120

## Summary
Exercised the Graph2D-enabled analyze API path against a real DXF sample and
re-ran the DXF fusion integration test in the graph2d environment.

## Environment
- Python: `./.venv-graph/bin/python`
- `GRAPH2D_ENABLED=true`
- `GRAPH2D_MODEL_PATH=models/graph2d_merged_latest.pth`
- DXF sample: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf/J0224071-11捕集器组件v1.dxf`

## API Smoke (TestClient)
Command:
```
GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_merged_latest.pth \
  ./.venv-graph/bin/python - <<'PY'
import io
import json
import os
from fastapi.testclient import TestClient

from src.main import app

path = "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf/J0224071-11捕集器组件v1.dxf"

client = TestClient(app)
with open(path, "rb") as handle:
    payload = handle.read()
options = {"extract_features": True, "classify_parts": True}
resp = client.post(
    "/api/v1/analyze/",
    files={"file": (os.path.basename(path), io.BytesIO(payload), "application/dxf")},
    data={"options": json.dumps(options)},
    headers={"x-api-key": os.getenv("API_KEY", "test")},
)
print("status", resp.status_code)
body = resp.json()
classification = body.get("results", {}).get("classification", {})
summary = {
    "part_type": classification.get("part_type"),
    "confidence": classification.get("confidence"),
    "rule_version": classification.get("rule_version"),
    "confidence_source": classification.get("confidence_source"),
    "graph2d_prediction": classification.get("graph2d_prediction"),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
```

Result:
```
status 200
{
  "part_type": "complex_assembly",
  "confidence": 0.55,
  "rule_version": "v1",
  "confidence_source": "rules",
  "graph2d_prediction": {
    "label": "机械制图",
    "confidence": 0.07665518671274185,
    "status": "ok"
  }
}
```

## Validation
Command:
```
./.venv-graph/bin/python -m pytest tests/integration/test_analyze_dxf_fusion.py -v
```

Result:
- PASSED (1 test)
- Warnings: ezdxf queryparser deprecations (`addParseAction`, `oneOf`, etc.)

## Notes
- Installed `fastapi==0.121.0`, `uvicorn[standard]==0.24.0`, and
  `python-multipart` into `.venv-graph` to run the TestClient-based API smoke.
