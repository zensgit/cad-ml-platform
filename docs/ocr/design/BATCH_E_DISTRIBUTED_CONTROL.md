# Batch E 分布式限流与熔断器设计总结

## 目标
在多实例部署与突发流量下保护后端 Provider（Paddle/DeepSeek），并在连续错误时快速熔断，避免级联失败。

## 实现
- 分布式限流（Token Bucket）
  - 文件：`src/utils/rate_limiter.py`
  - Redis + Lua 实现，键：`ocr:rl:{key}`，参数：`qps`、`burst`。
  - Redis 不可用时退化为进程内令牌桶。
  - 指标：`ocr_rate_limited_total` 计数。

- 熔断器（Circuit Breaker）
  - 文件：`src/utils/circuit_breaker.py`
  - 状态：0=closed，1=half_open，2=open；半开探测 `half_open_requests`。
  - 错误即开（简化版），超时后半开恢复；指标：`ocr_circuit_state{key}`。

- OcrManager 集成
  - 限流：每个 provider 一个 `RateLimiter` 实例，默认 `qps=10, burst=10`（可后续配置化）。
  - 熔断：每个 provider 一个 `CircuitBreaker` 实例，错误时 open；成功调用 reset。
  - 抛出 `OcrError` 到上层；metrics 同步更新。

## 测试
文件：`tests/ocr/test_distributed_control.py`
- `test_rate_limit_blocks`：极低 qps + burst 下，第二次请求被拒。
- `test_circuit_opens_and_blocks`：第一次失败触发打开，第二次立即被阻断。

## 局限与后续
- 目前熔断开关基于单次错误即开，后续可引入滑动窗口错误率与分类（timeout/parse/provider_down）。
- 限流未按用户/租户/端点细分；后续可扩展 key 维度。
- 未与配置中心/环境变量对齐（待接入 `Settings`）。

## 影响
- 在错误风暴下快速阻断，保护后端；在突发流量下平滑削峰。
- 指标可用于告警：`ocr_circuit_state`、`ocr_rate_limited_total`。

---
Batch E 完成，建议下一步进入 Batch F（多证据置信度校准 / Pydantic v2 清理）。

