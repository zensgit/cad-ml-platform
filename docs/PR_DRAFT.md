# PR Draft: Observability, Metrics, Lint Phase 1, and Tests

## Summary
- Standardized error model (HTTP 200 + `{success, code, error}`) across Vision/OCR
- Expanded metrics: EMA error rates, input size histograms, detailed rejection reasons
- Health `/health` includes runtime + EMA + config
- Added dev tooling, CI non-blocking `lint-all` report, and Grafana/PromQL examples
- Phase 1 lint clean for `src/`, tests expanded; all tests pass

## Key Changes
- Metrics: `vision_image_size_bytes`, `ocr_image_size_bytes`, `*_error_rate_ema`
- Rejection reasons: Vision base64/url; OCR mime/size/pdf
- Config: `VISION_MAX_BASE64_BYTES`, `ERROR_EMA_ALPHA`, `OCR_MAX_PDF_PAGES`, `OCR_MAX_FILE_MB`
- Tests: Base64 cases, PDF rejections, provider down
- Docs: README, ALERT_RULES, runbooks, Grafana dashboard JSON

## How to Test
1. `uvicorn src.main:app --reload`
2. `GET /health` to see EMA and config
3. Hit `/api/v1/vision/analyze` with invalid base64 and check `/metrics`
4. Upload a big PDF to `/api/v1/ocr/extract` and check `ocr_input_rejected_total`

## Screenshots / Artifacts
- Grafana: `docs/grafana/observability_dashboard.json`
- Alert rules: `docs/ALERT_RULES.md`
- Runbooks: `docs/runbooks/*.md`

## CI
- Lint scope: `src` only for blocking lint/type; full-repo flake8 uploaded by `lint-all-report`
- `lint-type` job lints `src` and runs mypy
- `lint-all-report` uploads full flake8 output (non-blocking)
- Tests run on 3.10/3.11; 145 tests pass locally

## Highlights

- Metrics
  - Vision: `vision_image_size_bytes`、`vision_input_rejected_total{reason=*}`、`vision_error_rate_ema`
  - OCR: `ocr_image_size_bytes`、`ocr_input_rejected_total{reason=*}`、`ocr_error_rate_ema`
  - `/health` 暴露 `runtime.error_rate_ema.{ocr,vision}` 与 `runtime.config.error_ema_alpha`
- Error Model
  - 统一错误响应（HTTP 200 + `{ success, code, error }`）
  - Vision base64 细分：`base64_padding_error`、`base64_invalid_char`、`base64_decode_error`
  - Vision URL 细分：`url_invalid_scheme|format|not_found|forbidden|http_error|too_large_*|empty|timeout|network_error|download_error`
  - OCR 细分：`invalid_mime`、`file_too_large`、`pdf_pages_exceed`、`pdf_forbidden_token`
- CI/DevEx
  - 新增非阻断 `lint-all-report` 工件
  - `src` 范围 lint/type 保持增量低噪
  - Tests 扩展（145 全绿）

## Configuration

Update `.env` as needed (see `.env.example`):

- `VISION_MAX_BASE64_BYTES`（默认 1048576）：Vision Base64 输入大小上限（字节）
- `ERROR_EMA_ALPHA`（默认 0.2）：错误率 EMA 平滑系数（越大越敏感）
- `OCR_MAX_PDF_PAGES`（默认 20）：OCR 允许的 PDF 最大页数
- `OCR_MAX_FILE_MB`（默认 50）：OCR 文件大小上限（MB）

Trade-offs:

- 上限太小会导致输入拒绝激增；太大将带来延迟与内存压力
- EMA alpha 越大越敏感，建议生产在 0.1~0.3 之间

## Verification Steps

1) Run: `uvicorn src.main:app --reload`
2) GET `/health` 查看 `error_rate_ema` 与 `config.error_ema_alpha`
3) 触发 Vision base64 错误：

```bash
curl -s http://localhost:8000/api/v1/vision/analyze \
  -H 'Content-Type: application/json' \
  -d '{"image_base64":"@@@not_base64@@@","include_description":false}' | jq
```

4) 触发 OCR PDF 超限：

```bash
python - << 'PY'
import requests
pdf = b"%PDF-1.4\n" + b"%Page\n"*25 + b"%%EOF"
files = {"file": ("large.pdf", pdf, "application/pdf")}
print(requests.post("http://localhost:8000/api/v1/ocr/extract", files=files).json())
PY
```

5) 如启用 Prometheus，GET `/metrics` 检查新指标（`*_error_rate_ema`、`*_image_size_bytes`、`*_input_rejected_total`）

## PromQL Examples

- Vision 输入拒绝占比（5m）：

```
sum(rate(vision_input_rejected_total[5m])) / sum(rate(vision_requests_total[5m]))
```

- Vision 图像大小 P99（5m）：

```
histogram_quantile(0.99, rate(vision_image_size_bytes_bucket[5m]))
```

- OCR Provider Down（按 provider）：

```
sum by (provider) (rate(ocr_errors_total{code="provider_down"}[5m]))
```

- 错误率 EMA：

```
vision_error_rate_ema
ocr_error_rate_ema
```

## Rollback Plan

- 按需调低 `ERROR_EMA_ALPHA` 或恢复默认
- 回滚至上一个稳定 tag；Grafana 面板可独立移除不影响服务

## Risks & Mitigations

- Prometheus 未启用时 `/metrics` 返回 404（已在测试中兼容）
- PDF 解析使用 pypdf 优先，降级为原始字节扫描，以确保无依赖环境的健壮性

## Links

- Alert rules: `docs/ALERT_RULES.md`
- Runbooks: `docs/runbooks/*.md`
- Grafana: `docs/grafana/observability_dashboard.json`
