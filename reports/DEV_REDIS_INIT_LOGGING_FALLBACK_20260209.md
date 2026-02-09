# DEV_REDIS_INIT_LOGGING_FALLBACK_20260209

## Goal
Avoid misleading startup logs when Redis is enabled but unavailable.

## Problem
`src/utils/cache.init_redis()` logs a warning and sets the Redis client to `None`
when it cannot connect. `src/main.py` previously logged `Redis initialized`
unconditionally after calling `init_redis()`, which could mislead operators.

## Change
File: `src/main.py`
- After `await init_redis()`, check whether a Redis client exists:
  - If present: log `Redis initialized`
  - If absent: log `Redis unavailable; using in-memory cache fallback`

## Validation
Command run:
```bash
python3 -c "import src.main; print('import_ok')"
```

