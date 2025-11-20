# 运行手册：Provider 宕机/不可用

当 `ocr_errors_total{code="provider_down"}` 或 `vision_errors_total{code="provider_down"}` 出现且持续时，可能是上游不可用或凭据/配额问题。

## 快速检查
- `/metrics`：
  - `*_errors_total{code="provider_down"}` 速率
  - `ocr_circuit_state{key=*}` 是否打开（2）
- 配置与凭据：API Key / Token 是否有效，是否达到速率或配额限制
- 供应商状态页与网络连通性（DNS、TLS、代理）

## 定位步骤
1. 小流量重试与替代 Provider 测试
2. 检查近期开关与版本变更（限流、重试、超时、熔断配置）
3. 如仅特定区域失败，排查网络路由与代理

## 缓解与修复
- 切换备份 Provider 或关闭依赖模块的强依赖路径
- 提高限流与重试上限（谨慎，避免雪崩）
- 与供应商联系并订阅状态通知

## 相关指标
- `ocr_errors_total{code="provider_down"}`、`vision_errors_total{code="provider_down"}`
- `ocr_circuit_state`
- `*_error_rate_ema`

