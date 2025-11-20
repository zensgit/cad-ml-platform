## Distributed Rate Limiting & Circuit Breaker Design (v1)

### 背景
当前限流为本地内存实现，不支持多实例。需要分布式令牌桶 + 熔断机制，保障高并发与异常场景稳定性。

### 目标
1. 分布式限流 (per IP + endpoint) 保证多实例一致 QPS 控制。
2. 熔断器防止故障 Provider 持续占用资源。
3. 统一错误指标：`ocr_errors_total{code="RATE_LIMIT|CIRCUIT_OPEN|PROVIDER_DOWN"}`。

### Redis 令牌桶实现
Key: `rate:{ip}:{endpoint}`
结构: `tokens`, `last_ts`
Lua 脚本 (原子更新):
```lua
local key=KEYS[1]
local now=tonumber(ARGV[1])
local rate=tonumber(ARGV[2])
local burst=tonumber(ARGV[3])
local ttl=tonumber(ARGV[4])
local data=redis.call('HMGET', key, 'tokens', 'ts')
local tokens=data[1] and tonumber(data[1]) or burst
local ts=data[2] and tonumber(data[2]) or now
local elapsed=now - ts
tokens=math.min(burst, tokens + elapsed*rate)
if tokens < 1 then
  redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
  redis.call('EXPIRE', key, ttl)
  return 0
end
tokens=tokens-1
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)
return 1
```
成功返回1，失败返回0。

### 熔断器
状态: CLOSED / OPEN / HALF_OPEN
Redis Key: `circuit:{provider}`
计数窗口 (最近 60s): requests_total, errors_total
错误率 = errors_total / requests_total
阈值: >0.30 且 requests_total≥20 → OPEN (记录打开时间)。
OPEN 持续时间 > cooldown (300s) → 转 HALF_OPEN，允许 N=2 探测请求：
 - 全成功 → CLOSED (重置计数)
 - 任一失败 → OPEN 重置时间。

### 指标
`ocr_circuit_state{provider}`: 0=closed,1=open,2=half_open
`ocr_rate_limit_hits_total{endpoint}` 计数 429 触发次数。
`ocr_threshold_changes_total{old,new}` 动态阈值变动计数。

### 错误映射
RATE_LIMIT → HTTP 429
CIRCUIT_OPEN → HTTP 503 (Provider unavailable)
PROVIDER_DOWN → HTTP 500 (内部错误)

### 风险与缓解
- Redis 不可用：降级到本地桶并发出告警日志。
- 过度限流：收集实际 QPS 与拒绝率，动态调节 `RATE_LIMIT_QPS`。
- 熔断误触发：增加最小请求数阈值 & 使用指数平滑。

### v2 拓展
- 加入成本权重 (GPU占用时间) 到熔断条件。
- 添加IP白名单/黑名单支持。

