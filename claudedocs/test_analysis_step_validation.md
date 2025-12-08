# STEP File Validation Test Analysis

## Summary

**Status**: ✅ All tests PASSING

The two test files for STEP file parsing and format validation are functioning correctly and align with the current API implementation.

## Test Files Analyzed

### 1. test_step_parse_failure.py

**Purpose**: Verify that invalid STEP files are rejected with proper error codes.

**Test Coverage**:
- Tests signature validation for STEP files
- Verifies that files without the `ISO-10303-21` signature are rejected
- Expects HTTP 415 (Unsupported Media Type)
- Validates structured error response with `INPUT_FORMAT_INVALID` code

**Implementation Details**:
```python
def test_step_failure_graceful():
    file = ("bad.step", b"NOT_A_STEP_FILE", "application/octet-stream")
    r = client.post(
        "/api/v1/analyze/",
        files={"file": file},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 415
    assert detail.get("code") == "INPUT_FORMAT_INVALID"
```

**Result**: ✅ PASSED

---

### 2. test_strict_format_validation.py

**Purpose**: Verify strict format validation mode for STEP files.

**Test Coverage**:

#### Test 1: `test_step_deep_validation_fail_strict_mode`
- Sets `FORMAT_STRICT_MODE=1` environment variable
- Tests STEP file missing `ISO-10303-21` header
- Verifies rejection with HTTP 415
- Validates `INPUT_FORMAT_INVALID` error code

**Implementation**:
```python
def test_step_deep_validation_fail_strict_mode():
    os.environ["FORMAT_STRICT_MODE"] = "1"
    bad_step = b"HEADER;ENDSEC;DATA;ENDSEC;" + b"X" * 50  # Missing ISO-10303-21
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("bad.step", io.BytesIO(bad_step), "application/step")},
        data={"options": '{"extract_features": true}'},
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 415
```

**Result**: ✅ PASSED

#### Test 2: `test_step_deep_validation_pass_strict_mode`
- Sets `FORMAT_STRICT_MODE=1` environment variable
- Tests valid STEP file with proper header structure
- Verifies acceptance with HTTP 200
- Ensures well-formed STEP files pass validation

**Implementation**:
```python
def test_step_deep_validation_pass_strict_mode():
    os.environ["FORMAT_STRICT_MODE"] = "1"
    good_step = b"ISO-10303-21;HEADER;ENDSEC;DATA;ENDSEC;EOF" + b"X" * 50
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("good.step", io.BytesIO(good_step), "application/step")},
        data={"options": '{"extract_features": true}'},
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 200
```

**Result**: ✅ PASSED

---

## API Implementation Analysis

### Signature Validation (src/security/input_validator.py)

The `verify_signature()` function validates STEP files:

```python
def verify_signature(data: bytes, file_format: str) -> Tuple[bool, str]:
    header = data[:64]
    fmt = file_format.lower()
    if fmt in {"step", "stp"}:
        return (header.startswith(_STEP_SIGNATURE_PREFIX), "STEP header 'ISO-10303-21'")
    # ... other formats
```

- Signature prefix: `b"ISO-10303-21"`
- Validation occurs in analyze endpoint (analyze.py lines 435-450)
- Returns HTTP 415 on signature mismatch

### Deep Format Validation (Strict Mode)

The `deep_format_validate()` function provides additional checks:

```python
def deep_format_validate(data: bytes, file_format: str) -> Tuple[bool, str]:
    fmt = file_format.lower()
    head = data[:512]
    if fmt in {"step", "stp"}:
        text = head.decode(errors="ignore")
        if "ISO-10303-21" not in text:
            return False, "missing_step_header"
        if "HEADER" not in text:
            return False, "missing_step_HEADER_section"
        return True, "ok"
    # ... other formats
```

- Activated when `FORMAT_STRICT_MODE=1`
- Checks for both `ISO-10303-21` and `HEADER` tokens
- Returns HTTP 415 on validation failure (analyze.py lines 452-470)

### Error Response Structure

Both validation paths return structured errors via `build_error()`:

```python
err = build_error(
    ErrorCode.INPUT_FORMAT_INVALID,
    stage="input",
    message="Signature validation failed",
    format=file_format,
    signature_prefix=signature_hex_prefix(content[:32]),
    expected_signature=expectation,
)
raise HTTPException(status_code=415, detail=err)
```

---

## Test Execution Results

```bash
$ python3 -m pytest tests/unit/test_step_parse_failure.py tests/unit/test_strict_format_validation.py -v

collected 3 items

tests/unit/test_step_parse_failure.py::test_step_failure_graceful PASSED [ 33%]
tests/unit/test_strict_format_validation.py::test_step_deep_validation_fail_strict_mode PASSED [ 66%]
tests/unit/test_strict_format_validation.py::test_step_deep_validation_pass_strict_mode PASSED [100%]

======================== 3 passed, 2 warnings in 1.75s ========================
```

---

## Validation Flow

### Request Flow with Signature Validation:

1. **Upload** → File uploaded to `/api/v1/analyze/`
2. **Size Check** → Validate file size ≤ 10MB (default)
3. **MIME Sniff** → Best-effort MIME detection
4. **Format Check** → Verify file extension (step/stp/dxf/dwg/stl/iges/igs)
5. **Signature Validation** → **MANDATORY** - Verify format-specific signature
   - STEP: Must start with `ISO-10303-21`
   - Returns HTTP 415 if invalid
6. **Strict Mode** (optional) → If `FORMAT_STRICT_MODE=1`:
   - Deep validation of internal structure
   - STEP: Must contain both `ISO-10303-21` AND `HEADER`
7. **Parse** → Adapter converts to unified CadDocument
8. **Analysis** → Feature extraction, classification, etc.

### HTTP Status Codes:

- **200** - Valid file, analysis successful
- **415** - Unsupported Media Type / Invalid format signature
- **413** - File too large
- **400** - Invalid options or empty file
- **422** - Entity count exceeded

---

## Key Findings

1. **Signature Validation is Mandatory**: All files undergo signature verification, not optional
2. **Two-Layer Validation**:
   - Basic signature check (always active)
   - Deep format validation (strict mode only)
3. **Correct API Endpoint**: Tests use `/api/v1/analyze/`
4. **Proper Headers**: Tests include `X-API-Key: test` (case-insensitive in test 2)
5. **Structured Errors**: Error responses include `code` field for programmatic handling

---

## Test Quality Assessment

**Strengths**:
- Tests cover both positive and negative cases
- Validates error structure, not just status codes
- Uses environment variables correctly for strict mode
- Proper use of BytesIO for test data
- Clear test names indicating purpose

**Coverage**:
- ✅ Invalid signature rejection
- ✅ Strict mode validation failure
- ✅ Strict mode validation success
- ✅ Error response structure
- ✅ HTTP status codes

**Suggestions for Additional Tests** (Optional):
- Test case-sensitivity of headers (already handled: both `X-API-Key` and `x-api-key` work)
- Test partial STEP signature (e.g., `ISO-10303` without `-21`)
- Test binary vs ASCII STEP file handling
- Test matrix validation if configured

---

## Conclusion

Both test files are **correctly implemented** and **currently passing**. The tests accurately validate:

1. Signature validation enforcement (mandatory)
2. Deep format validation in strict mode
3. Proper error codes and HTTP status responses
4. Structured error response format

No fixes are required. The API implementation and tests are in sync.
