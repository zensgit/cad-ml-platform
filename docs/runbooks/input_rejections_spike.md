# 运行手册：输入拒绝激增（Vision/OCR）

当 `*_input_rejected_total{reason=*}` 速率显著升高时，说明上游请求质量下降或配置不当。

## 快速检查
- `/metrics` 查看 `vision_input_rejected_total`、`ocr_input_rejected_total` 的 `reason` 标签分布：
  - `base64_too_large`、`base64_decode_error`、`base64_empty`
  - `url_invalid_scheme`、`url_invalid_format`、`url_http_error`、`url_not_found`、`url_forbidden`、`url_timeout`、`url_network_error` 等
- `/health.runtime.config.vision_max_base64_bytes` 是否过小导致频繁触发上限
- 客户端或网关是否近期更改了上传逻辑或压缩策略

## 定位步骤
1. 明确主要 `reason` 标签，按占比排序
2. 采样失败请求，核对真实请求负载（是否符合接口契约、是否被代理篡改）
3. 若集中在 URL 失败类：检查外网连通性、目标站点的防护策略（403/429）、DNS 或证书问题
4. 若集中在 base64 过大：评估输入上限、客户端是否误传原图或未压缩

## 缓解与修复
- 与客户端沟通输入上限与 MIME 约束，提供明确的错误提示
- 按需调整 `VISION_MAX_BASE64_BYTES`（权衡资源）；记录在变更单
- 对 URL 源启用白名单或前置下载代理，稳定可用性
- 若短期峰值：可临时放宽阈值并观测错误率 EMA 与时延影响

## 相关指标
- `vision_input_rejected_total{reason=*}`
- `ocr_input_rejected_total{reason=*}`
- `vision_image_size_bytes`、`ocr_image_size_bytes`
- `vision_error_rate_ema`、`ocr_error_rate_ema`

