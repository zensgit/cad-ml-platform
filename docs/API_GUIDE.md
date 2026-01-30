# CAD ML Platform - API Guide

## Overview

CAD ML Platform provides enterprise-grade APIs for CAD file analysis, OCR processing, and machine learning inference.

**Base URL**: `https://api.example.com/api/v2`

**Authentication**: All endpoints require an API key via header:
```
X-API-Key: your-api-key
```

## Quick Start

### 1. Health Check

```bash
curl -X GET "https://api.example.com/health" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "model": "healthy"
  }
}
```

### 2. OCR Text Extraction

Extract text and dimensions from CAD drawings.

```bash
curl -X POST "https://api.example.com/api/v2/ocr/extract" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@drawing.dxf"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "text_blocks": [
      {
        "text": "Part Name: Bracket Assembly",
        "confidence": 0.95,
        "bounding_box": {"x": 100, "y": 200, "width": 300, "height": 50}
      }
    ],
    "dimensions": [
      {
        "value": 25.4,
        "unit": "mm",
        "type": "linear",
        "confidence": 0.92
      }
    ],
    "processing_time_ms": 150
  }
}
```

### 3. Vision Analysis

Analyze CAD drawings using computer vision.

```bash
curl -X POST "https://api.example.com/api/v2/vision/analyze" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@drawing.dxf" \
  -F "options={\"detect_features\": true, \"classify_components\": true}"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "features": [
      {
        "type": "hole",
        "count": 4,
        "locations": [[100, 100], [200, 100], [100, 200], [200, 200]]
      },
      {
        "type": "slot",
        "count": 2,
        "dimensions": {"length": 50, "width": 10}
      }
    ],
    "classification": {
      "part_type": "bracket",
      "confidence": 0.89,
      "material_hint": "steel"
    }
  }
}
```

### 4. Material Classification

Identify materials from CAD data.

```bash
curl -X POST "https://api.example.com/api/v2/materials/classify" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "thickness": 2.5,
      "surface_area": 1250.5,
      "has_holes": true,
      "hole_count": 4
    },
    "text_hints": ["AISI 304", "stainless"]
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "material": {
      "name": "Stainless Steel 304",
      "code": "AISI-304",
      "category": "metal",
      "properties": {
        "density": 8.0,
        "yield_strength": 215
      }
    },
    "confidence": 0.94,
    "alternatives": [
      {"name": "Stainless Steel 316", "confidence": 0.78}
    ]
  }
}
```

## Batch Processing

### Submit Batch Job

Process multiple files asynchronously.

```bash
curl -X POST "https://api.example.com/api/v2/batch/submit" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "ocr_extract",
    "files": [
      {"url": "https://storage.example.com/drawings/part1.dxf"},
      {"url": "https://storage.example.com/drawings/part2.dxf"}
    ],
    "options": {
      "extract_dimensions": true,
      "language": "en"
    },
    "webhook_url": "https://your-app.com/webhook/batch-complete"
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "batch_abc123",
    "status": "queued",
    "total_items": 2,
    "estimated_completion": "2025-01-30T10:30:00Z"
  }
}
```

### Check Batch Status

```bash
curl -X GET "https://api.example.com/api/v2/batch/status/batch_abc123" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "batch_abc123",
    "status": "processing",
    "progress": {
      "completed": 1,
      "total": 2,
      "failed": 0
    },
    "results": [
      {
        "file": "part1.dxf",
        "status": "completed",
        "result_url": "https://api.example.com/api/v2/batch/result/batch_abc123/0"
      }
    ]
  }
}
```

## WebSocket Real-Time Updates

Connect to receive real-time notifications.

```javascript
const ws = new WebSocket('wss://api.example.com/api/v2/ws');

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-api-key'
  }));

  // Subscribe to channels
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['batch:batch_abc123', 'alerts']
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
  // { type: 'batch_progress', job_id: 'batch_abc123', progress: 50 }
};
```

## Rate Limits

| Tier | Requests/min | Burst | Batch Jobs/day |
|------|-------------|-------|----------------|
| Free | 60 | 10 | 10 |
| Pro | 300 | 50 | 100 |
| Enterprise | 1200 | 200 | Unlimited |

**Rate Limit Headers:**
```
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 298
X-RateLimit-Reset: 1706612400
```

## Error Handling

All errors follow a consistent format:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid file format. Expected DXF or DWG.",
    "details": {
      "field": "file",
      "allowed_formats": ["dxf", "dwg", "pdf"]
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `FILE_TOO_LARGE` | 413 | File exceeds size limit (100MB) |
| `PROCESSING_ERROR` | 500 | Internal processing failure |

## SDK Examples

### Python

```python
from cadml import Client

client = Client(api_key="your-api-key")

# OCR extraction
result = client.ocr.extract("drawing.dxf")
print(f"Found {len(result.text_blocks)} text blocks")

# Vision analysis
analysis = client.vision.analyze(
    "drawing.dxf",
    detect_features=True,
    classify_components=True
)
print(f"Part type: {analysis.classification.part_type}")

# Batch processing
job = client.batch.submit(
    operation="ocr_extract",
    files=["part1.dxf", "part2.dxf"],
    webhook_url="https://your-app.com/webhook"
)
print(f"Job ID: {job.id}")
```

### JavaScript/TypeScript

```typescript
import { CadMLClient } from '@cadml/sdk';

const client = new CadMLClient({ apiKey: 'your-api-key' });

// OCR extraction
const result = await client.ocr.extract(file);
console.log(`Found ${result.textBlocks.length} text blocks`);

// Vision analysis with streaming
const stream = await client.vision.analyzeStream(file, {
  detectFeatures: true,
  onProgress: (progress) => console.log(`${progress}% complete`)
});

for await (const feature of stream) {
  console.log(`Detected: ${feature.type}`);
}
```

## Webhooks

Configure webhooks for async notifications.

### Webhook Payload

```json
{
  "event": "batch.completed",
  "timestamp": "2025-01-30T10:30:00Z",
  "data": {
    "job_id": "batch_abc123",
    "status": "completed",
    "results_url": "https://api.example.com/api/v2/batch/results/batch_abc123"
  },
  "signature": "sha256=abc123..."
}
```

### Verifying Signatures

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## Best Practices

1. **Use batch processing** for multiple files (>5) to optimize throughput
2. **Implement webhooks** instead of polling for async operations
3. **Cache results** - analysis results are deterministic for unchanged files
4. **Handle rate limits** with exponential backoff
5. **Validate files locally** before uploading to catch format errors early

## Support

- **Documentation**: https://docs.example.com
- **API Status**: https://status.example.com
- **Support Email**: api-support@example.com
