# Phase 4 Day 1: Rolling Upgrade Compatibility

## Overview

This document describes the rolling upgrade strategy for transitioning from the legacy `file_bytes_b64` payload format to the new `file_ref` (external file storage) format.

## Payload Format Evolution

### Legacy Format (Pre-Phase 2.5)
```json
{
  "tenant_id": "abc123",
  "file_name": "drawing.dxf",
  "content_type": "application/dxf",
  "file_bytes_b64": "base64-encoded-file-content...",
  "query_geom": {...},
  "request_params": {...}
}
```

**Issues:**
- Large files cause Redis memory pressure
- Payload size limited by Redis string max (512MB)
- Network overhead for large base64 strings

### New Format (Phase 2.5+)
```json
{
  "tenant_id": "abc123",
  "file_name": "drawing.dxf",
  "content_type": "application/dxf",
  "file_ref": {
    "backend": "local",
    "path": "job-uuid/file-uuid_drawing.dxf"
  },
  "query_geom": {...},
  "request_params": {...}
}
```

**Benefits:**
- Files stored externally (local filesystem or S3)
- Minimal Redis payload size
- Supports files of any size
- Better separation of concerns

## Rolling Upgrade Strategy

### Step 1: Deploy New Workers First

Deploy workers that support **both** formats:
- If `file_ref` exists → load from file storage
- If `file_ref` is missing → fallback to `file_bytes_b64`

```python
# Worker logic (simplified)
if "file_ref" in payload:
    file_bytes = storage.load_bytes(file_ref)
elif "file_bytes_b64" in payload:
    file_bytes = base64.b64decode(payload["file_bytes_b64"])
else:
    raise ValueError("Missing file data")
```

### Step 2: (Optional) Enable Grayscale Mode

During the transition, you can enable dual-write mode where the API writes **both** `file_ref` and `file_bytes_b64`:

```bash
# Enable grayscale (both formats written)
DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64=1

# Only include b64 for files <= 10MB (default)
DEDUP2D_JOB_PAYLOAD_BYTES_B64_MAX_BYTES=10485760
```

This is useful when:
- You have a mixed fleet of old and new workers
- You need quick rollback capability

### Step 3: Disable Grayscale (file_ref-only)

Once all workers are upgraded:

```bash
# Disable grayscale (file_ref only, recommended for production)
DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64=0
```

### Step 4: Wait for TTL Expiration

Old jobs with `file_bytes_b64` will continue to work until their TTL expires (default: 1 hour). After that, all jobs will use `file_ref` exclusively.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64` | `0` (off) | Enable dual-write mode |
| `DEDUP2D_JOB_PAYLOAD_BYTES_B64_MAX_BYTES` | `10485760` (10MB) | Max file size for b64 inclusion |

## Rollback Plan

If issues arise after upgrading workers:

1. **Quick Rollback**: Enable `DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64=1` on API to support old workers
2. **Full Rollback**: Redeploy old workers (they can read `file_bytes_b64` payloads)

## Verification

### Unit Tests
```bash
./.venv/bin/python -m pytest -q tests/unit/test_dedup_2d_jobs_redis.py
```

### Integration Test
1. Submit job with new worker → verify `file_ref` is used
2. Submit job with grayscale enabled → verify both formats present
3. Process job with new worker → verify file loaded correctly

## Files Modified

- `src/core/dedupcad_2d_worker.py` - Added backward compatibility for `file_bytes_b64`
- `src/core/dedupcad_2d_jobs_redis.py` - Added `Dedup2DPayloadConfig` and dual-write support
- `.env.example` - Documented new environment variables

## Deployment Matrix

| Environment | File Storage | Grayscale | Notes |
|-------------|--------------|-----------|-------|
| Development | `local` | Off | Single machine, no shared storage needed |
| Staging | `local` (shared PVC) | Optional | Test rolling upgrade |
| Production | `s3` / `local` (shared PVC) | Off | Worker先升级完成后关闭 |

---

*Generated: Phase 4 Day 1*
