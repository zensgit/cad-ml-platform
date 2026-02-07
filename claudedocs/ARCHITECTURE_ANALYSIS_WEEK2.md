# Week 2 架构分析报告

**日期**: 2026-02-05
**目标**: Provider架构重构分析与计划

---

## 1. 现有Provider模式分析

### 1.1 Provider类型统计

| 类型 | 数量 | 基类/协议 |
|------|------|-----------|
| VisionProvider | 60+ | ABC (VisionProvider) |
| LLMProvider | 6 | ABC (BaseLLMProvider) |
| OcrClient | 3 | Protocol |
| VectorStore | 4 | ABC (BaseVectorStore) |
| ConfigSource | 4 | ABC |
| EmbeddingProvider | 2 | ABC |
| Store类 (通用) | 30+ | 各自ABC |

### 1.2 现有基类对比

#### VisionProvider (src/core/vision/base.py)
```python
class VisionProvider(ABC):
    @abstractmethod
    async def analyze_image(self, image_data: bytes, include_description: bool = True) -> VisionDescription:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
```
- **优点**: 简洁，职责单一
- **缺点**: 缺少 health_check, is_available, 错误处理

#### BaseLLMProvider (src/core/assistant/llm_providers.py)
```python
class BaseLLMProvider(ABC):
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
```
- **优点**: 配置注入，可用性检查
- **缺点**: 缺少 health_check, metrics

#### OcrClient (src/core/ocr/base.py)
```python
class OcrClient(Protocol):
    name: str
    async def warmup(self) -> None: ...
    async def extract(self, image_bytes: bytes, trace_id: Optional[str] = None) -> OcrResult: ...
    async def health_check(self) -> bool: ...
```
- **优点**: Protocol灵活性，warmup，health_check
- **缺点**: 无配置管理

#### BaseVectorStore (src/core/vectors/stores/base.py)
```python
class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, id: str, vector: List[float], meta: Optional[Dict[str, Any]] = None) -> bool: ...
    @abstractmethod
    def search(self, vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]: ...
    @abstractmethod
    def get_meta(self, id: str) -> Optional[Dict[str, Any]]: ...
    @abstractmethod
    def size(self) -> int: ...
    @abstractmethod
    def save(self, path: str): ...
    @abstractmethod
    def load(self, path: str): ...
```
- **优点**: 完整CRUD接口
- **缺点**: 缺少 health_check, is_available, metrics

---

## 2. 共同模式识别

### 2.1 重复出现的方法
| 方法 | VisionProvider | LLMProvider | OcrClient | VectorStore |
|------|----------------|-------------|-----------|-------------|
| health_check | ❌ | ❌ | ✅ | ❌ |
| is_available | ❌ | ✅ | ❌ | ❌ |
| warmup | ❌ | ❌ | ✅ | ❌ |
| provider_name | ✅ | ❌ | ✅ (name) | ❌ |
| config注入 | ❌ | ✅ | ❌ | ❌ |

### 2.2 缺失的共同功能
1. **健康检查** - 统一的 `health_check()` 接口
2. **可用性检查** - 统一的 `is_available()` 接口
3. **配置管理** - 标准化配置注入
4. **Metrics记录** - Prometheus指标集成
5. **错误处理** - 标准化错误码映射
6. **日志集成** - 统一的日志格式

---

## 3. 重构方案

### 3.1 统一BaseProvider抽象基类

```python
# src/core/providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar

T_Config = TypeVar("T_Config")
T_Result = TypeVar("T_Result")

@dataclass
class ProviderConfig:
    """Base configuration for all providers."""
    name: str = ""
    timeout: int = 30
    retry_count: int = 3
    metrics_enabled: bool = True
    log_level: str = "INFO"

class BaseProvider(ABC, Generic[T_Config, T_Result]):
    """Unified base class for all providers."""

    def __init__(self, config: Optional[T_Config] = None):
        self._config = config or self._default_config()
        self._initialized = False
        self._logger = logging.getLogger(f"{__name__}.{self.provider_name}")

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier."""
        pass

    @abstractmethod
    def _default_config(self) -> T_Config:
        """Return default configuration."""
        pass

    # === Health & Availability ===

    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        try:
            return await self._do_health_check()
        except Exception as e:
            self._logger.warning(f"Health check failed: {e}")
            return False

    async def _do_health_check(self) -> bool:
        """Override for custom health check logic."""
        return self.is_available()

    def is_available(self) -> bool:
        """Check if provider is available (configured properly)."""
        return self._initialized

    # === Lifecycle ===

    async def warmup(self) -> None:
        """Warm up provider (load models, establish connections)."""
        if not self._initialized:
            await self._do_warmup()
            self._initialized = True

    async def _do_warmup(self) -> None:
        """Override for custom warmup logic."""
        pass

    async def shutdown(self) -> None:
        """Gracefully shutdown provider."""
        await self._do_shutdown()
        self._initialized = False

    async def _do_shutdown(self) -> None:
        """Override for custom shutdown logic."""
        pass

    # === Metrics ===

    def _record_latency(self, operation: str, duration_ms: float) -> None:
        """Record operation latency metric."""
        # Prometheus integration point
        pass

    def _record_error(self, operation: str, error_code: str) -> None:
        """Record error metric."""
        pass
```

### 3.2 Provider Registry/Factory

```python
# src/core/providers/registry.py
from typing import Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T", bound="BaseProvider")

class ProviderRegistry:
    """Registry for provider factories."""

    _registries: Dict[str, Dict[str, Type]] = {}

    @classmethod
    def register(cls, category: str, name: str):
        """Decorator to register a provider class."""
        def decorator(provider_class: Type[T]) -> Type[T]:
            if category not in cls._registries:
                cls._registries[category] = {}
            cls._registries[category][name] = provider_class
            return provider_class
        return decorator

    @classmethod
    def get(cls, category: str, name: str, config: Optional[Any] = None) -> Optional[BaseProvider]:
        """Get provider instance by category and name."""
        if category not in cls._registries:
            return None
        provider_class = cls._registries[category].get(name)
        if provider_class is None:
            return None
        return provider_class(config)

    @classmethod
    def list_providers(cls, category: str) -> List[str]:
        """List registered providers for a category."""
        return list(cls._registries.get(category, {}).keys())

    @classmethod
    def get_best_available(cls, category: str, config: Optional[Any] = None) -> Optional[BaseProvider]:
        """Get best available provider for a category."""
        for name in cls._registries.get(category, {}):
            provider = cls.get(category, name, config)
            if provider and provider.is_available():
                return provider
        return None

# Usage example:
@ProviderRegistry.register("vision", "deepseek")
class DeepSeekVisionProvider(BaseProvider[VisionConfig, VisionDescription]):
    ...

@ProviderRegistry.register("llm", "claude")
class ClaudeProvider(BaseProvider[LLMConfig, str]):
    ...
```

### 3.3 配置管理集成

```python
# src/core/providers/config.py
from src.core.config.manager import get_config_manager

class ConfigurableProvider(BaseProvider):
    """Provider with ConfigManager integration."""

    async def load_config_from_manager(self, prefix: str) -> None:
        """Load configuration from ConfigManager."""
        manager = get_config_manager()
        # Load provider-specific config
        self._config.timeout = await manager.get_int(f"{prefix}.timeout", 30)
        self._config.retry_count = await manager.get_int(f"{prefix}.retry_count", 3)
        # Watch for changes
        manager.watch(f"{prefix}.*", self._on_config_change)

    def _on_config_change(self, event) -> None:
        """Handle configuration changes."""
        self._logger.info(f"Config changed: {event.key} = {event.new_value}")
```

---

## 4. 迁移计划

### Phase 1: 创建基础设施 (Day 1-2)
- [ ] 创建 `src/core/providers/base.py` - BaseProvider
- [ ] 创建 `src/core/providers/registry.py` - ProviderRegistry
- [ ] 创建 `src/core/providers/config.py` - ConfigurableProvider
- [ ] 添加单元测试

### Phase 2: 迁移LLM Providers (Day 3)
- [ ] 重构 BaseLLMProvider 继承 BaseProvider
- [ ] 更新 ClaudeProvider, OpenAIProvider, QwenProvider, OllamaProvider
- [ ] 注册到 ProviderRegistry
- [ ] 验证现有功能

### Phase 3: 迁移OCR Providers (Day 4)
- [ ] 创建 OcrProvider 基类继承 BaseProvider
- [ ] 迁移 DeepSeekHfProvider, PaddleOcrProvider
- [ ] 保持 Protocol 兼容性

### Phase 4: 迁移Vector Stores (Day 5)
- [ ] 创建 VectorStoreProvider 继承 BaseProvider
- [ ] 迁移 FaissStore, MemoryVectorStore
- [ ] 添加 health_check 和 metrics

---

## 5. 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 破坏现有功能 | 中 | 保持向后兼容，添加适配器 |
| 性能影响 | 低 | 基类方法轻量，避免过度抽象 |
| 测试覆盖不足 | 中 | 先写测试，后重构 |

---

## 6. 成功指标

- [ ] 代码重复减少 >30%
- [ ] 所有Provider有统一的health_check
- [ ] ProviderRegistry覆盖所有Provider类型
- [ ] 现有测试100%通过
- [ ] 新增单元测试覆盖BaseProvider

---

*Generated by Claude Code - 2026-02-05*
