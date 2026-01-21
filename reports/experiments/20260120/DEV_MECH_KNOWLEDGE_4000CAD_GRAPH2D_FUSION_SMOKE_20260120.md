# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_FUSION_SMOKE_20260120

## Summary
Validated the Graph2D + FusionAnalyzer path on a real DXF sample and re-ran the
DXF fusion integration test after updating rules and the merged checkpoint.

## Environment
- `GRAPH2D_ENABLED=true`
- `GRAPH2D_MODEL_PATH=models/graph2d_merged_latest.pth`
- `GRAPH2D_FUSION_ENABLED=true`
- `FUSION_ANALYZER_ENABLED=true`

## API Smoke (TestClient)
Command:
```
GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_merged_latest.pth \
GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true \
  ./.venv-graph/bin/python - <<'PY'
import io
import json
import os
from fastapi.testclient import TestClient
from src.main import app

path = "/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf/1-1泵盖图.dxf"

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
    "fusion_decision": classification.get("fusion_decision"),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
```

Result:
```
status 200
{
  "part_type": "泵",
  "confidence": 0.9,
  "rule_version": "L2-Fusion-v1",
  "confidence_source": "fusion",
  "graph2d_prediction": {
    "label": "机械制图",
    "confidence": 0.06944217532873154,
    "status": "ok"
  },
  "fusion_decision": {
    "primary_label": "Standard_Part",
    "confidence": 0.5,
    "source": "rule_based",
    "reasons": [
      "Default fallback: No specific features detected"
    ],
    "rule_hits": [
      "RULE_DEFAULT"
    ],
    "ai_raw_score": 0.06944217532873154,
    "consistency_check": "none",
    "consistency_notes": null,
    "schema_version": "v1.0",
    "feature_vector_id": null
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
