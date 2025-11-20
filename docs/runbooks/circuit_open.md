# 运行手册：熔断器打开 (Circuit Open)

当 `ocr_circuit_state{key=*} == 2` 持续超过阈值时，表示该 Provider 的失败率较高，熔断器进入打开态。

## 快速检查
- `/metrics`：
  - `ocr_circuit_state` 是否为 2（打开）
  - 对应时间窗 `ocr_errors_total{provider=*}` 的错误码分布
- 最近变更：依赖版本、超时配置、重试配置、后端模型/服务变更

## 定位步骤
1. 查看失败类型：超时、网络错误、返回结构变化
2. 通过小流量灰度尝试重置熔断器（半开），观察 `on_success/on_error` 行为
3. 验证上游健康或切换备用通道

## 缓解与修复
- 增加保护：限流、退避重试、请求超时合理化
- 优化解析路径：对返回结构变化增强兼容
- 必要时临时降级：关闭依赖该 Provider 的功能或采用缓存结果

## 相关指标
- `ocr_circuit_state{key=*}`
- `ocr_errors_total{provider,code,stage}`
- `ocr_error_rate_ema`

