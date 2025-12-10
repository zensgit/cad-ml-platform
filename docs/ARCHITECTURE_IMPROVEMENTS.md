# CAD-ML Platform 架构改进计划

> 本文档基于对现有代码库的深度分析，提出用工业级开源组件替代自研基础设施的改进方案。

## 实施状态

| Phase | 模块 | 状态 | 实现位置 |
|-------|------|------|----------|
| Phase 1 | Qdrant 向量存储 | ✅ 已完成 | `src/core/vector_stores/` |
| Phase 1 | Tenacity 重试逻辑 | ✅ 已完成 | `src/core/resilience/retry.py` |
| Phase 2 | Arq 任务队列 | ✅ 已完成 | `src/core/tasks/` |
| Phase 2 | OpenTelemetry 可观测性 | ✅ 已完成 | `src/core/observability/` |
| Phase 2 | Open3D 几何特征 | ✅ 已完成 | `src/core/geometry/` |
| Phase 3 | Temporal 工作流 | ⏸️ 按需 | 保留现有 `workflow_engine.py` |

*最后更新: 2024-12*

---

## 目录

- [现状分析](#现状分析)
- [改进方案总览](#改进方案总览)
- [Phase 1: 快速见效](#phase-1-快速见效1-2周)
- [Phase 2: 稳定优化](#phase-2-稳定优化2-4周)
- [Phase 3: 规模化](#phase-3-规模化按需)
- [实施风险与缓解](#实施风险与缓解)
- [参考资源](#参考资源)

---

## 现状分析

### 项目优势

- **Clean Architecture**: 代码结构清晰，接口抽象良好（如 `VectorStoreProtocol`）
- **功能完整**: 实现了完整的 CAD 分析流水线
- **稳健设计**: 包含降级处理、重试机制、熔断器等

### 核心问题

| 模块 | 当前实现 | 代码行数 | 问题 |
|------|----------|----------|------|
| 向量检索 | `similarity.py` | ~1000行 | 手动维护锁、TTL、Redis同步 |
| 工作流引擎 | `workflow_engine.py` | ~800行 | 内存状态，进程崩溃丢失 |
| 消息队列 | `message_queue.py` | ~600行 | 单机内存，无法分布式扩展 |
| 重试逻辑 | 多处手动实现 | 分散 | try-except-sleep 样板代码 |
| 可观测性 | `tracing.py` 等 | ~500行 | 手动埋点，易遗漏 |
| 几何特征 | `feature_extractor.py` | ~400行 | 宏观特征，区分度不足 |

---

## 改进方案总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        改进优先级矩阵                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   高收益 │  ★ Qdrant          │  ○ Temporal              │
│         │  (Phase 1)          │  (Phase 3)               │
│         │                     │                          │
│   ──────┼─────────────────────┼──────────────────────────│
│         │                     │                          │
│   低收益 │  ★ Tenacity        │  ○ Open3D                │
│         │  (Phase 1)          │  (Phase 2)               │
│         │                     │                          │
│         └─────────────────────┴──────────────────────────│
│              低复杂度                  高复杂度              │
└─────────────────────────────────────────────────────────────────┘

★ = 优先实施    ○ = 按需实施
```

---

## Phase 1: 快速见效（1-2周）

### 1.1 Qdrant 替代向量检索

**当前痛点**

```python
# similarity.py 中的复杂逻辑
_VECTOR_LOCK = threading.Lock()  # 手动锁管理
_REDIS_SYNC_ENABLED = True       # Redis 同步
_TTL_SECONDS = 3600              # 手动 TTL 管理
_DEGRADATION_HISTORY = []        # 降级历史追踪

async def register_vector(...):
    # 100+ 行代码处理：
    # - 锁获取/释放
    # - FAISS 索引更新
    # - Redis 备份
    # - 元数据存储
    # - TTL 管理
    # - 错误处理与降级
```

**替代方案**

```python
# 新增: src/core/vector_stores/qdrant_store.py
from qdrant_client import QdrantClient, models
from src.core.protocols import VectorStoreProtocol

class QdrantVectorStore(VectorStoreProtocol):
    """Qdrant 向量存储实现"""

    def __init__(self, url: str = "localhost", port: int = 6333):
        self.client = QdrantClient(url=url, port=port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """确保集合存在"""
        collections = self.client.get_collections().collections
        if "cad_vectors" not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name="cad_vectors",
                vectors_config=models.VectorParams(
                    size=128,  # 特征向量维度
                    distance=models.Distance.COSINE
                )
            )

    async def register_vector(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict | None = None
    ) -> bool:
        """注册向量 - 替代 ~100 行代码"""
        self.client.upsert(
            collection_name="cad_vectors",
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=metadata or {}
                )
            ]
        )
        return True

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_conditions: dict | None = None
    ) -> list[dict]:
        """相似度搜索 - 替代 ~150 行代码"""
        query_filter = None
        if filter_conditions:
            # 元数据过滤（如 material="steel"）
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
            )

        results = self.client.search(
            collection_name="cad_vectors",
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload
            }
            for hit in results
        ]

    async def delete_vector(self, vector_id: str) -> bool:
        """删除向量"""
        self.client.delete(
            collection_name="cad_vectors",
            points_selector=models.PointIdsList(points=[vector_id])
        )
        return True
```

**迁移步骤**

```bash
# 1. 安装依赖
pip install qdrant-client

# 2. 启动 Qdrant（开发环境）
docker run -p 6333:6333 qdrant/qdrant

# 3. 创建新实现
touch src/core/vector_stores/__init__.py
touch src/core/vector_stores/qdrant_store.py

# 4. 更新工厂函数
# src/adapters/factory.py
```

**工厂函数更新**

```python
# src/adapters/factory.py
from src.core.vector_stores.qdrant_store import QdrantVectorStore

def create_vector_store(backend: str = "qdrant") -> VectorStoreProtocol:
    """创建向量存储实例"""
    if backend == "qdrant":
        return QdrantVectorStore(
            url=os.getenv("QDRANT_URL", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
    elif backend == "faiss":
        # 保留原有实现作为降级选项
        from src.core.similarity import FAISSVectorStore
        return FAISSVectorStore()
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
```

**预期收益**

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 代码行数 | ~1000行 | ~100行 |
| 维护复杂度 | 高（手动锁/TTL/同步） | 低（SDK托管） |
| 持久化 | 手动 Redis 同步 | 原生支持 |
| 分布式 | 不支持 | 原生支持 |
| 元数据过滤 | 手动实现 | 原生支持 |

---

### 1.2 Tenacity 替代重试逻辑

**当前痛点**

```python
# 分散在多个文件中的重试逻辑
# src/core/vision/manager.py, src/core/similarity.py 等

async def call_provider_with_retry(self, ...):
    max_retries = 3
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            return await self._call_provider(...)
        except ProviderError as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2  # 指数退避
```

**替代方案**

```python
# src/core/resilience/retry.py
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

# 通用重试装饰器
def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 60,
    retry_exceptions: tuple = (Exception,)
):
    """可配置的重试装饰器"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(retry_exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )

# 特定场景的预设装饰器
provider_retry = with_retry(
    max_attempts=3,
    min_wait=2,
    max_wait=30,
    retry_exceptions=(ProviderError, TimeoutError)
)

database_retry = with_retry(
    max_attempts=5,
    min_wait=1,
    max_wait=10,
    retry_exceptions=(ConnectionError, TimeoutError)
)
```

**使用示例**

```python
# src/core/vision/manager.py
from src.core.resilience.retry import provider_retry

class VisionManager:
    @provider_retry
    async def analyze_image(self, image_data: bytes) -> VisionDescription:
        """分析图像 - 自动重试"""
        return await self._provider.analyze_image(image_data)

# src/core/similarity.py
from src.core.resilience.retry import database_retry

class VectorStore:
    @database_retry
    async def connect(self) -> None:
        """连接数据库 - 自动重试"""
        await self._client.connect()
```

**迁移步骤**

```bash
# 1. 安装依赖
pip install tenacity

# 2. 创建统一重试模块
touch src/core/resilience/retry.py

# 3. 逐步替换现有重试逻辑
# 搜索所有手动重试代码
grep -r "for attempt in range" src/
grep -r "time.sleep" src/
grep -r "asyncio.sleep.*backoff" src/
```

**预期收益**

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 重试代码 | 分散在 10+ 文件 | 集中在 1 个模块 |
| 可配置性 | 硬编码 | 声明式配置 |
| 可观测性 | 手动日志 | 自动日志钩子 |
| 测试难度 | 需要 mock sleep | 可配置测试模式 |

---

## Phase 2: 稳定优化（2-4周）

### 2.1 Arq 替代消息队列

**当前痛点**

```python
# src/core/vision/message_queue.py
class InMemoryMessageStore:
    """内存消息存储 - 单机限制"""
    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._dlq: dict[str, list] = {}  # Dead Letter Queue
        self._ack_pending: dict[str, Message] = {}
```

**替代方案**

```python
# src/core/tasks/worker.py
from arq import create_pool
from arq.connections import RedisSettings

# 任务定义
async def analyze_cad_file(ctx, file_path: str, options: dict) -> dict:
    """CAD 文件分析任务"""
    analyzer = ctx["analyzer"]
    result = await analyzer.analyze(file_path, **options)
    return result.to_dict()

async def extract_features(ctx, document_id: str) -> dict:
    """特征提取任务"""
    extractor = ctx["extractor"]
    features = await extractor.extract(document_id)
    return features

# Worker 配置
class WorkerSettings:
    functions = [analyze_cad_file, extract_features]
    redis_settings = RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379))
    )
    max_jobs = 10
    job_timeout = 300  # 5 分钟超时

    @staticmethod
    async def on_startup(ctx):
        """Worker 启动时初始化"""
        ctx["analyzer"] = CADAnalyzer()
        ctx["extractor"] = FeatureExtractor()

    @staticmethod
    async def on_shutdown(ctx):
        """Worker 关闭时清理"""
        await ctx["analyzer"].close()
```

```python
# src/core/tasks/client.py
from arq import create_pool
from arq.connections import RedisSettings

class TaskClient:
    """任务客户端"""

    def __init__(self):
        self._pool = None

    async def connect(self):
        self._pool = await create_pool(RedisSettings())

    async def submit_analysis(
        self,
        file_path: str,
        options: dict | None = None
    ) -> str:
        """提交分析任务"""
        job = await self._pool.enqueue_job(
            "analyze_cad_file",
            file_path,
            options or {}
        )
        return job.job_id

    async def get_result(self, job_id: str, timeout: float = 60) -> dict:
        """获取任务结果"""
        job = Job(job_id, self._pool)
        return await job.result(timeout=timeout)
```

**迁移步骤**

```bash
# 1. 安装依赖
pip install arq

# 2. 创建任务模块
mkdir -p src/core/tasks
touch src/core/tasks/__init__.py
touch src/core/tasks/worker.py
touch src/core/tasks/client.py

# 3. 启动 Worker
arq src.core.tasks.worker.WorkerSettings
```

---

### 2.2 OpenTelemetry 替代手动追踪

**当前痛点**

```python
# src/core/vision/tracing.py - 手动 Span 管理
class Tracer:
    def start_span(self, name: str) -> Span:
        span = Span(name=name, trace_id=self._trace_id)
        self._active_spans.append(span)
        return span
```

**替代方案**

```python
# src/core/observability/telemetry.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

def setup_telemetry(app, service_name: str = "cad-ml-platform"):
    """配置 OpenTelemetry"""

    # 设置 TracerProvider
    provider = TracerProvider()
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=os.getenv("OTEL_ENDPOINT", "localhost:4317"))
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # 自动埋点
    FastAPIInstrumentor.instrument_app(app)  # HTTP 请求
    RedisInstrumentor().instrument()          # Redis 调用
    HTTPXClientInstrumentor().instrument()    # HTTP 客户端

    return trace.get_tracer(service_name)

# 业务代码中的使用
tracer = trace.get_tracer(__name__)

async def analyze_document(document_id: str):
    with tracer.start_as_current_span("analyze_document") as span:
        span.set_attribute("document_id", document_id)

        # 自动记录子调用
        features = await extract_features(document_id)
        similarity = await find_similar(features)

        span.set_attribute("similar_count", len(similarity))
        return similarity
```

**迁移步骤**

```bash
# 1. 安装依赖
pip install opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-redis \
    opentelemetry-instrumentation-httpx

# 2. 启动 Jaeger（开发环境）
docker run -d --name jaeger \
    -p 16686:16686 \
    -p 4317:4317 \
    jaegertracing/all-in-one:latest

# 3. 在 main.py 中初始化
```

```python
# src/main.py
from src.core.observability.telemetry import setup_telemetry

app = FastAPI()
tracer = setup_telemetry(app)
```

---

### 2.3 Open3D 增强几何特征

**当前痛点**

```python
# src/core/feature_extractor.py - 宏观特征
def compute_features(mesh) -> dict:
    return {
        "surface_count": len(mesh.faces),
        "bbox_volume": mesh.bounding_box.volume,
        "aspect_ratio": max(dims) / min(dims),  # 旋转敏感
    }
```

**替代方案**

```python
# src/core/geometry/advanced_features.py
import open3d as o3d
import numpy as np

class GeometricFeatureExtractor:
    """高级几何特征提取器"""

    def __init__(self, num_points: int = 2048):
        self.num_points = num_points

    def mesh_to_pointcloud(self, mesh) -> o3d.geometry.PointCloud:
        """网格转点云"""
        # 从 trimesh 转换
        vertices = np.asarray(mesh.vertices)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        # 均匀采样
        pcd = pcd.farthest_point_down_sample(self.num_points)

        # 估计法线
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        return pcd

    def compute_fpfh(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """计算 FPFH 特征（旋转不变）"""
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
        )
        # 返回全局特征（均值池化）
        return np.mean(fpfh.data, axis=1)

    def align_to_canonical(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """对齐到标准姿态（解决旋转敏感问题）"""
        # PCA 对齐
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid

        # 协方差矩阵特征分解
        cov = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 按特征值排序
        idx = np.argsort(eigenvalues)[::-1]
        rotation = eigenvectors[:, idx]

        # 应用旋转
        aligned_points = points_centered @ rotation

        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)
        aligned_pcd.normals = pcd.normals

        return aligned_pcd

    def extract_features(self, mesh) -> dict:
        """提取完整特征集"""
        # 转换并对齐
        pcd = self.mesh_to_pointcloud(mesh)
        aligned_pcd = self.align_to_canonical(pcd)

        # 计算特征
        fpfh = self.compute_fpfh(aligned_pcd)

        # 几何统计（对齐后的）
        points = np.asarray(aligned_pcd.points)
        bbox = aligned_pcd.get_axis_aligned_bounding_box()

        return {
            "fpfh_descriptor": fpfh.tolist(),  # 33维 FPFH
            "point_count": len(points),
            "bbox_extent": bbox.get_extent().tolist(),
            "centroid": np.mean(points, axis=0).tolist(),
            "surface_area": mesh.area if hasattr(mesh, 'area') else 0,
            "volume": mesh.volume if hasattr(mesh, 'volume') else 0,
        }
```

**迁移步骤**

```bash
# 1. 安装依赖
pip install open3d

# 2. 创建高级特征模块
mkdir -p src/core/geometry
touch src/core/geometry/__init__.py
touch src/core/geometry/advanced_features.py

# 3. 更新特征提取器工厂
```

---

## Phase 3: 规模化（按需）

### 3.1 Temporal 替代工作流引擎

**适用场景**
- 工作流需要跨服务/跨进程执行
- 需要长时间运行的工作流（小时/天级别）
- 需要复杂的错误处理和补偿逻辑
- 需要工作流可视化和调试

**当前 vs Temporal 对比**

```python
# 当前实现：workflow_engine.py
class WorkflowExecution:
    """内存工作流 - 进程崩溃丢失状态"""
    status: WorkflowStatus
    current_task: int
    context: dict  # 内存中，无持久化

# Temporal 实现
from temporalio import workflow, activity

@activity.defn
async def extract_features(document_id: str) -> dict:
    """特征提取活动"""
    extractor = FeatureExtractor()
    return await extractor.extract(document_id)

@activity.defn
async def find_similar(features: dict) -> list:
    """相似度搜索活动"""
    store = VectorStore()
    return await store.search(features["vector"])

@workflow.defn
class CADAnalysisWorkflow:
    """CAD 分析工作流 - 自动持久化"""

    @workflow.run
    async def run(self, document_id: str) -> dict:
        # 步骤1：提取特征
        features = await workflow.execute_activity(
            extract_features,
            document_id,
            start_to_close_timeout=timedelta(minutes=5)
        )

        # 步骤2：相似度搜索
        similar = await workflow.execute_activity(
            find_similar,
            features,
            start_to_close_timeout=timedelta(minutes=2)
        )

        return {
            "document_id": document_id,
            "features": features,
            "similar_documents": similar
        }
```

**部署要求**

```yaml
# docker-compose.temporal.yml
version: '3.8'
services:
  temporal:
    image: temporalio/auto-setup:latest
    ports:
      - "7233:7233"
    environment:
      - DB=postgresql
      - DB_PORT=5432
      - POSTGRES_USER=temporal
      - POSTGRES_PWD=temporal
      - POSTGRES_SEEDS=postgresql
    depends_on:
      - postgresql

  temporal-ui:
    image: temporalio/ui:latest
    ports:
      - "8080:8080"
    environment:
      - TEMPORAL_ADDRESS=temporal:7233

  postgresql:
    image: postgres:13
    environment:
      - POSTGRES_USER=temporal
      - POSTGRES_PASSWORD=temporal
```

**何时引入 Temporal**

| 信号 | 建议 |
|------|------|
| 单机部署，工作流简单 | 保持现有实现 |
| 多节点部署，需要协调 | 考虑 Temporal |
| 工作流执行 > 1小时 | 强烈建议 Temporal |
| 需要 SAGA 补偿逻辑 | 强烈建议 Temporal |
| 需要工作流可视化 | 强烈建议 Temporal |

---

## 实施风险与缓解

### 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Qdrant 服务不可用 | 中 | 高 | 保留 FAISS 降级路径 |
| 数据迁移丢失 | 低 | 高 | 双写期 + 数据校验 |
| 性能回退 | 低 | 中 | 基准测试对比 |
| 学习曲线 | 中 | 低 | 渐进式迁移 |

### 回滚策略

```python
# src/core/vector_stores/__init__.py
from src.core.protocols import VectorStoreProtocol

def get_vector_store() -> VectorStoreProtocol:
    """获取向量存储（带降级）"""
    backend = os.getenv("VECTOR_STORE_BACKEND", "qdrant")

    if backend == "qdrant":
        try:
            from .qdrant_store import QdrantVectorStore
            store = QdrantVectorStore()
            store.health_check()  # 健康检查
            return store
        except Exception as e:
            logger.warning(f"Qdrant unavailable, falling back to FAISS: {e}")
            backend = "faiss"

    if backend == "faiss":
        from src.core.similarity import FAISSVectorStore
        return FAISSVectorStore()

    raise ValueError(f"Unknown backend: {backend}")
```

### 迁移检查清单

```markdown
## Qdrant 迁移检查清单

- [ ] 开发环境 Qdrant 部署完成
- [ ] QdrantVectorStore 单元测试通过
- [ ] 与现有 VectorStoreProtocol 接口兼容
- [ ] 数据迁移脚本准备就绪
- [ ] 双写期配置完成
- [ ] 性能基准测试完成
- [ ] 降级路径测试通过
- [ ] 监控告警配置完成
- [ ] 文档更新完成
- [ ] 生产环境 Qdrant 部署完成
- [ ] 生产数据迁移完成
- [ ] 双写期观察（1周）
- [ ] 切换到 Qdrant 单写
- [ ] 旧代码清理
```

---

## 参考资源

### 官方文档

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Tenacity Documentation](https://tenacity.readthedocs.io/)
- [Arq Documentation](https://arq-docs.helpmanual.io/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Open3D Documentation](http://www.open3d.org/docs/)
- [Temporal Documentation](https://docs.temporal.io/)

### 示例代码

```bash
# 克隆示例仓库
git clone https://github.com/qdrant/examples.git qdrant-examples
git clone https://github.com/temporalio/samples-python.git temporal-examples
```

### 社区支持

- Qdrant Discord: https://discord.gg/qdrant
- Temporal Slack: https://temporal.io/slack

---

## 附录：依赖更新

```txt
# requirements.txt 新增
qdrant-client>=1.7.0
tenacity>=8.2.0
arq>=0.25.0
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
opentelemetry-instrumentation-fastapi>=0.41b0
open3d>=0.17.0  # Phase 2
temporalio>=1.4.0  # Phase 3
```

```txt
# requirements-dev.txt 新增
pytest-asyncio>=0.21.0
testcontainers>=3.7.0  # 用于集成测试
```

---

*文档版本: 1.0*
*创建日期: 2024-12*
*维护者: CAD-ML Platform Team*
