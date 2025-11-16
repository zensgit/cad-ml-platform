# OCR Quickstart Guide

## 1. 环境验证
```
python scripts/verify_environment.py --json
```
确认：PaddleOCR, Redis, Prometheus 可用；无 GPU 亦可运行。

## 1.1 真实 PaddleOCR 安装（可选）

当前系统支持无PaddleOCR运行（自动回退到示例数据）。如需真实OCR能力：

```bash
# 安装 PaddleOCR（需要 PaddlePaddle 依赖）
pip install paddlepaddle paddleocr

# 验证安装
python -c "from paddleocr import PaddleOCR; print('PaddleOCR installed')"

# 运行冒烟测试
python scripts/verify_environment.py
# 应显示: ✅ PaddleOCR: OK

# 测试真实图片
python -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')
result = ocr.ocr('examples/sample.png', cls=True)
print('OCR result:', result[:2] if result else 'No text found')
"
```

**注意事项**：
- PaddlePaddle 需要 Python 3.7-3.10（3.11+ 可能有兼容问题）
- 首次运行会下载模型文件（约 200MB）
- macOS M1/M2 用户建议使用 CPU 版本
- 无 PaddleOCR 时，系统自动使用 stub 数据（适合开发测试）

## 2. 运行服务
```
make run
```
或直接启动 FastAPI 应用（参考项目 README）。

## 3. 提交请求
```
curl -X POST -F "file=@examples/sample.png" \
  "http://localhost:8000/api/v1/ocr/extract?provider=auto"
```
响应包含：`provider/confidence/fallback_level/dimensions/symbols`。

## 4. 关键环境变量
- `OCR_PROVIDER=auto|paddle|deepseek_hf`
- `CONFIDENCE_FALLBACK=0.85`（已在运行时按 EMA 动态调整）
- `OCR_TIMEOUT_MS=30000`
- `REDIS_URL=redis://localhost:6379/0`
- 安全限制：`MAX_FILE_SIZE_MB=50`、`MAX_PDF_PAGES=20`、最大分辨率 2048px。

## 5. 监控与日志
- Prometheus 指标端点 `/metrics`（若已集成）
  - `ocr_requests_total`、`ocr_processing_duration_seconds`、`ocr_fallback_triggered`
  - `ocr_confidence_ema`、`ocr_confidence_fallback_threshold`
  - `ocr_rate_limited_total`、`ocr_circuit_state`
- 结构化日志：只包含 `image_hash`，不记录敏感内容。

## 6. 评测（可选）
```
python -m tests.ocr.golden.run_golden_evaluation
```
输出到 `reports/ocr_evaluation.md`，包含 Dimension Recall / Edge-F1 / Brier。

## 7. 常见问题
- DeepSeek GPU 不可用：自动退化到 CPU 与 Paddle。
- 回退频率高：检查 `ocr_confidence_ema` / Prompt 模板与版本号。
- 限流/熔断触发：查看 `ocr_rate_limited_total` 与 `ocr_circuit_state{key}`。

