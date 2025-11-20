# 运行手册：错误率 EMA 升高排障

当 `/health.runtime.error_rate_ema.ocr|vision` 持续升高时，说明近期错误比例增加。

## 快速检查
- 查看 `/metrics`：
  - `*_errors_total{code=*,...}` 是否激增
  - `*_input_rejected_total{reason=*}` 是否出现异常原因（如 base64 解码失败）
  - `vision_image_size_bytes` 分布是否异常（输入过大激增）
- 检查最近变更：部署、依赖升级、配置项（尤其 `VISION_MAX_BASE64_BYTES`、`ERROR_EMA_ALPHA`）
- 外部依赖：网络、Provider 可用性（PROVIDER_DOWN）

## 定位步骤
1. 错误类别
   - 通过 `*_errors_total{code}` 分解：`INPUT_ERROR`、`INTERNAL_ERROR`、`PROVIDER_DOWN` 等。
2. 输入问题
   - `*_input_rejected_total{reason}` 标签定位：base64 过大/解码失败/空输入。
3. Provider 问题
   - 检查 Provider 健康与凭据；查看 `ocr_circuit_state` 是否打开。
4. 回滚/变更验证
   - 如加载新模型或开启新路径，进行 A/B 或回滚验证。

## 缓解与修复
- 输入异常：增加前端校验或放宽/优化阈值（谨慎评估风险）。
- Provider 宕机：切换主备、提高重试与熔断策略、增加限流保护。
- 内部错误：检查近期代码变更与依赖升级，回滚或修复。

## 相关配置
- `ERROR_EMA_ALPHA`：EMA 平滑系数，默认 0.2。
- `VISION_MAX_BASE64_BYTES`：Vision Base64 输入大小上限。

