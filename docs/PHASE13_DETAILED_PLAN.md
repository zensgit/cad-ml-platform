# Phase 13+ 详细改进计划

**版本**: v3.0 Roadmap
**基于**: Phase 11 (协作引擎) + Phase 12 (前端集成) 完成状态
**日期**: 2025-12-06

---

## 📊 当前系统状态总结

### 已完成功能
| Phase | 功能模块 | 完成度 |
|-------|---------|--------|
| 1-9 | 核心特征提取 (v1-v9) | ✅ 100% |
| 10 | LLM推理 + 混合搜索 + 视觉增强 | ✅ 100% |
| 11 | 实时协作引擎 | ✅ 100% |
| 12 | Web前端客户端 | ✅ 95% |

### 待完善项
1. **Line Hit Test** - 前端线条点击检测 (代码有 TODO)
2. **Optimistic UI** - 前端本地即时更新
3. **身份认证** - 当前使用 mock username
4. **Active Learning Loop** - 主动学习闭环 (代码骨架存在但未激活)

---

## 🎯 Phase 13: 生产化加固 (Week 1-2)

### 执行顺序调整说明

根据生产就绪原则，调整后的执行顺序：

```
13.1 身份认证系统 (Day 1-2)     ← 安全基石，必须最先
    ↓
13.1.5 前端容器化与部署 (Day 3)  ← 新增：基础设施层就绪
    ↓
13.2 前端功能完善 (Day 4-5)     ← 在稳固环境下开发
    ↓
13.3 协作功能增强 (Day 6-7)     ← 最后完成高级功能
```

---

### 13.1 身份认证系统

**目标**: 集成真实用户认证，替换 mock username

**任务清单**:
```yaml
任务 13.1.1: JWT 认证中间件
  文件: src/core/auth/jwt_middleware.py
  内容:
    - JWT 令牌验证
    - 用户上下文注入
    - 刷新令牌支持
  依赖: PyJWT, python-jose

任务 13.1.2: 认证 API 端点
  文件: src/api/v1/auth.py
  内容:
    - POST /auth/login
    - POST /auth/register
    - POST /auth/refresh
    - GET /auth/me

任务 13.1.3: 协作端点认证集成
  文件: src/api/v1/collaboration.py
  修改:
    - WebSocket 连接验证 JWT
    - 用户 ID 从令牌提取
    - 权限检查 (文档访问控制)

任务 13.1.4: 前端认证流程
  文件: clients/web-collaboration/js/auth.js
  内容:
    - 登录/注册 UI
    - 令牌存储 (localStorage/Cookie)
    - 自动刷新机制
```

**代码示例**:
```python
# src/core/auth/jwt_middleware.py
from fastapi import Request, HTTPException
from jose import jwt, JWTError
import os

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret")
ALGORITHM = "HS256"

async def verify_token(request: Request) -> dict:
    """验证 JWT 令牌"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing authorization header")

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(401, "Invalid token")
```

**配置参数**:
```bash
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

---

### 13.1.5 前端容器化与部署架构 (新增)

**目标**: 基础设施层生产就绪，前端可独立部署

**任务清单**:
```yaml
任务 13.1.5.1: 前端 Dockerfile
  文件: clients/web-collaboration/Dockerfile
  内容:
    - 多阶段构建 (build + nginx)
    - 静态资源优化 (gzip)
    - 健康检查端点

任务 13.1.5.2: Nginx 反向代理配置
  文件: deployments/nginx/nginx.conf
  内容:
    - API 路由 (/api/* → backend:8000)
    - WebSocket 代理 (/api/v1/collaboration/ws/* → backend:8000)
    - 静态资源服务 (/ → frontend)
    - SSL 终止 (生产环境)
    - 请求限流

任务 13.1.5.3: Docker Compose 整合
  文件: deployments/docker/docker-compose.full.yml
  内容:
    - frontend 服务
    - backend 服务
    - nginx 服务
    - redis 服务
    - 网络配置

任务 13.1.5.4: 环境配置管理
  文件: clients/web-collaboration/config/
  内容:
    - config.development.js
    - config.production.js
    - 构建时环境变量注入
```

**代码示例** (前端 Dockerfile):
```dockerfile
# clients/web-collaboration/Dockerfile

# === 构建阶段 (如果有构建步骤) ===
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production 2>/dev/null || true
COPY . .
# 如果使用构建工具: RUN npm run build

# === 生产阶段 ===
FROM nginx:alpine

# 复制静态文件
COPY --from=builder /app /usr/share/nginx/html
# 或直接: COPY . /usr/share/nginx/html

# 复制 Nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

**代码示例** (Nginx 配置):
```nginx
# deployments/nginx/nginx.conf

upstream backend {
    server backend:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name _;

    # 静态文件
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;

        # 缓存控制
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # API 代理
    location /api/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # WebSocket 代理 (协作功能)
    location /api/v1/collaboration/ws/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # WebSocket 超时
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # 健康检查
    location /health {
        access_log off;
        return 200 "OK";
        add_header Content-Type text/plain;
    }

    # 压缩
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml;
    gzip_min_length 1000;
}
```

**代码示例** (Docker Compose):
```yaml
# deployments/docker/docker-compose.full.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro  # 生产环境 SSL
    depends_on:
      - backend
      - frontend
    networks:
      - cad-network
    restart: unless-stopped

  frontend:
    build:
      context: ../../clients/web-collaboration
      dockerfile: Dockerfile
    networks:
      - cad-network
    restart: unless-stopped

  backend:
    build:
      context: ../..
      dockerfile: deployments/docker/Dockerfile
    environment:
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - redis
    networks:
      - cad-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    networks:
      - cad-network
    restart: unless-stopped

networks:
  cad-network:
    driver: bridge

volumes:
  redis-data:
```

**配置参数**:
```bash
# 前端配置
VITE_API_BASE_URL=/api
VITE_WS_BASE_URL=/api/v1/collaboration/ws

# Nginx
NGINX_WORKER_CONNECTIONS=1024
NGINX_CLIENT_MAX_BODY_SIZE=50m
```

---

### 13.2 前端功能完善

**目标**: 补全前端待完善功能

**任务清单**:
```yaml
任务 13.2.1: Line Hit Test 实现
  文件: clients/web-collaboration/js/renderer.js
  修改: hitTest() 方法
  算法: 点到线段距离 < threshold

任务 13.2.2: Optimistic UI
  文件: clients/web-collaboration/js/app.js
  内容:
    - 本地立即应用操作
    - 服务端确认后标记
    - 冲突时回滚

任务 13.2.3: 更多实体类型
  文件: clients/web-collaboration/js/renderer.js
  新增:
    - CIRCLE 渲染
    - ARC 渲染
    - TEXT 渲染

任务 13.2.4: 游标同步
  文件: clients/web-collaboration/js/cursors.js
  内容:
    - 其他用户游标位置显示
    - 实时更新
    - 颜色区分
```

**代码示例** (Line Hit Test):
```javascript
// clients/web-collaboration/js/renderer.js
hitTest(x, y) {
    const THRESHOLD = 5; // 像素容差

    for (const id in this.entities) {
        const ent = this.entities[id];
        const data = ent.data;

        if (data.type === 'LINE') {
            // 点到线段距离
            const dist = this.pointToLineDistance(
                x, y,
                data.start[0], data.start[1],
                data.end[0], data.end[1]
            );
            if (dist < THRESHOLD) return id;
        }
        // ... 其他类型
    }
    return null;
}

pointToLineDistance(px, py, x1, y1, x2, y2) {
    const A = px - x1;
    const B = py - y1;
    const C = x2 - x1;
    const D = y2 - y1;

    const dot = A * C + B * D;
    const len_sq = C * C + D * D;
    let param = -1;

    if (len_sq !== 0) param = dot / len_sq;

    let xx, yy;
    if (param < 0) {
        xx = x1; yy = y1;
    } else if (param > 1) {
        xx = x2; yy = y2;
    } else {
        xx = x1 + param * C;
        yy = y1 + param * D;
    }

    const dx = px - xx;
    const dy = py - yy;
    return Math.sqrt(dx * dx + dy * dy);
}
```

---

### 13.3 协作功能增强

**目标**: 完善实时协作体验

**任务清单**:
```yaml
任务 13.3.1: 撤销/重做支持
  文件: src/core/collaboration/operations.py
  新增:
    - undo(doc_id, user_id) 方法
    - redo(doc_id, user_id) 方法
    - 操作栈管理

任务 13.3.2: 评论/标注系统
  文件: src/core/collaboration/comments.py
  内容:
    - 实体关联评论
    - 区域标注
    - @用户提及

任务 13.3.3: 版本历史
  文件: src/api/v1/collaboration.py
  新增:
    - GET /collaboration/{doc_id}/history
    - GET /collaboration/{doc_id}/snapshot/{version}
    - POST /collaboration/{doc_id}/restore/{version}

任务 13.3.4: 冲突可视化
  文件: clients/web-collaboration/js/conflicts.js
  内容:
    - 冲突高亮显示
    - 解决方案选择 UI
```

---

## 🧠 Phase 14: Active Learning 闭环激活 (Week 3-4)

### 14.1 核心组件激活

**目标**: 激活现有 `src/core/active_learning.py` 骨架代码

**任务清单**:
```yaml
任务 14.1.1: 不确定性采样激活
  文件: src/core/knowledge/enhanced_classifier.py
  修改:
    - 分类结果检测不确定性
    - 自动标记待审核样本

任务 14.1.2: 反馈 API 端点
  文件: src/api/v1/active_learning.py
  新增:
    - GET /active-learning/pending - 获取待审核样本
    - POST /active-learning/feedback - 提交用户反馈
    - GET /active-learning/stats - 获取统计信息
    - POST /active-learning/export - 导出训练数据

任务 14.1.3: 标注界面
  文件: examples/labeling_ui.html
  内容:
    - 样本展示
    - 分类选择
    - 批量标注
```

**代码示例** (API 端点):
```python
# src/api/v1/active_learning.py
from fastapi import APIRouter, HTTPException
from src.core.active_learning import get_active_learner
from pydantic import BaseModel

router = APIRouter()

class FeedbackRequest(BaseModel):
    doc_id: str
    predicted_type: str
    true_type: str
    confidence: float
    user_id: str | None = None

@router.get("/pending")
async def get_pending_samples(limit: int = 20):
    """获取待审核的不确定样本"""
    learner = get_active_learner()
    samples = learner.get_uncertain_samples(limit)
    return {"samples": samples, "total": len(samples)}

@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """提交用户反馈"""
    learner = get_active_learner()
    result = learner.submit_feedback(
        doc_id=req.doc_id,
        predicted_type=req.predicted_type,
        true_type=req.true_type,
        confidence=req.confidence,
        user_id=req.user_id,
    )
    return result

@router.get("/stats")
async def get_stats():
    """获取 Active Learning 统计"""
    learner = get_active_learner()
    return learner.get_stats()

@router.post("/export")
async def export_training_data():
    """导出训练数据"""
    learner = get_active_learner()
    return learner.export_training_data()
```

---

### 14.2 自动微调流程

**目标**: 实现自动化模型更新

**任务清单**:
```yaml
任务 14.2.1: 微调触发器
  文件: src/core/active_learning.py
  新增:
    - check_retrain_threshold() 方法
    - trigger_retrain() 方法

任务 14.2.2: 微调脚本增强
  文件: scripts/finetune_from_feedback.py
  内容:
    - 从 Redis 导出反馈数据
    - 构建 Triplet 数据集
    - 微调 Metric Embedder
    - 验证并部署

任务 14.2.3: 模型版本管理
  文件: src/core/model_registry.py
  内容:
    - 模型版本号管理
    - A/B 测试支持
    - 回滚机制
```

**配置参数**:
```bash
ACTIVE_LEARNING_ENABLED=true
ACTIVE_LEARNING_STORE=redis
ACTIVE_LEARNING_RETRAIN_THRESHOLD=100
UNCERTAINTY_LOW=0.4
UNCERTAINTY_HIGH=0.7
RETRAIN_AUTO_TRIGGER=false  # 需要手动确认
```

---

## 🔍 Phase 15: 高级分析功能 (Week 5-6)

### 15.1 相似零件聚类分析

**目标**: 自动发现相似零件族

**任务清单**:
```yaml
任务 15.1.1: 聚类引擎
  文件: src/core/clustering.py
  内容:
    - HDBSCAN 聚类
    - 聚类质量指标
    - 自动确定 K 值

任务 15.1.2: 聚类 API
  文件: src/api/v1/clustering.py
  新增:
    - POST /clustering/run - 执行聚类
    - GET /clustering/results/{job_id} - 获取结果
    - GET /clustering/clusters - 获取所有聚类

任务 15.1.3: 可视化
  文件: examples/cluster_visualization.html
  内容:
    - t-SNE 降维展示
    - 聚类交互选择
    - 代表性样本展示
```

**代码示例**:
```python
# src/core/clustering.py
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ClusteringEngine:
    """零件聚类引擎"""

    def __init__(self, min_cluster_size: int = 5):
        self.min_cluster_size = min_cluster_size
        self._hdbscan = None

    def _init_hdbscan(self):
        try:
            import hdbscan
            self._hdbscan = hdbscan
        except ImportError:
            logger.warning("hdbscan not available")

    def cluster(
        self,
        vectors: List[List[float]],
        doc_ids: List[str],
    ) -> Dict[str, Any]:
        """执行聚类分析"""
        if self._hdbscan is None:
            self._init_hdbscan()

        if self._hdbscan is None:
            return {"error": "hdbscan not available"}

        X = np.array(vectors)

        clusterer = self._hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
        )
        labels = clusterer.fit_predict(X)

        # 组织结果
        clusters: Dict[int, List[str]] = {}
        for doc_id, label in zip(doc_ids, labels):
            if label == -1:
                continue  # 噪声点
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doc_id)

        return {
            "n_clusters": len(clusters),
            "clusters": clusters,
            "noise_count": int((labels == -1).sum()),
            "probabilities": clusterer.probabilities_.tolist(),
        }
```

---

### 15.2 设计意图推理增强

**目标**: 深度理解 CAD 设计意图

**任务清单**:
```yaml
任务 15.2.1: 设计模式识别
  文件: src/core/design_patterns.py
  内容:
    - 常见设计模式库
    - 模式匹配算法
    - 置信度评估

任务 15.2.2: 装配关系推理
  文件: src/core/assembly_inference.py
  内容:
    - 配合关系检测
    - 装配序列推断
    - 干涉检查

任务 15.2.3: 功能推断 API
  文件: src/api/v1/inference.py
  新增:
    - POST /inference/design-intent
    - POST /inference/assembly-relations
    - POST /inference/function-analysis
```

---

## 🚀 Phase 16: 性能与扩展性 (Week 7-8)

### 16.1 缓存策略优化

**目标**: 多级缓存减少延迟

**任务清单**:
```yaml
任务 16.1.1: L1 内存缓存
  文件: src/core/cache.py
  新增:
    - LRU 内存缓存
    - TTL 支持
    - 缓存预热

任务 16.1.2: L2 Redis 缓存
  修改: src/core/cache.py
  内容:
    - 缓存穿透保护
    - 热点数据检测
    - 异步刷新

任务 16.1.3: 特征缓存
  修改: src/core/feature_extractor.py
  内容:
    - 特征向量缓存
    - 版本化缓存 key
    - 批量预计算
```

---

### 16.2 水平扩展支持

**目标**: 支持多实例部署

**任务清单**:
```yaml
任务 16.2.1: 会话亲和性
  文件: deployments/nginx/nginx.conf
  内容:
    - WebSocket 路由
    - 负载均衡策略

任务 16.2.2: 分布式锁优化
  修改: src/core/collaboration/locking.py
  内容:
    - Redlock 算法
    - 锁续期机制
    - 故障转移

任务 16.2.3: Kubernetes 配置更新
  文件: charts/cad-ml-platform/values.yaml
  修改:
    - HPA 配置
    - Pod 反亲和性
    - 资源限制调优
```

---

## 📈 实施时间线

```
Week 1-2: Phase 13 (生产化加固) - 调整后顺序
├── Day 1-2: 13.1 身份认证系统 (安全基石)
├── Day 3:   13.1.5 前端容器化与部署 (基础设施就绪)
├── Day 4-5: 13.2 前端功能完善 (在稳固环境下开发)
├── Day 6-7: 13.3 协作功能增强 (高级功能)
└── Day 8:   集成测试 + 端到端验证

Week 3-4: Phase 14 (Active Learning)
├── Day 1-2: 核心组件激活
├── Day 3-4: 自动微调流程
├── Day 5-6: 标注界面开发
└── Day 7: 端到端验证

Week 5-6: Phase 15 (高级分析)
├── Day 1-3: 聚类分析
├── Day 4-6: 设计意图推理
└── Day 7: API 文档

Week 7-8: Phase 16 (性能扩展)
├── Day 1-3: 缓存策略
├── Day 4-6: 水平扩展
└── Day 7: 负载测试
```

---

## 📊 成功指标

| 指标 | 当前基线 | Phase 13后 | Phase 14后 | Phase 15后 | Phase 16后 |
|------|---------|-----------|-----------|-----------|-----------|
| 分类准确率 | 85% | 85% | 90% | 92% | 92% |
| 协作延迟 | ~100ms | ~80ms | ~80ms | ~80ms | ~50ms |
| 并发用户 | 50 | 100 | 100 | 100 | 500+ |
| 月度改进率 | 0% | 5% | 15% | 20% | 20% |
| 前端功能完整度 | 95% | 100% | 100% | 100% | 100% |

---

## 🔧 配置汇总

```bash
# Phase 13: 身份认证
JWT_SECRET_KEY=your-secure-secret
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Phase 14: Active Learning
ACTIVE_LEARNING_ENABLED=true
ACTIVE_LEARNING_STORE=redis
ACTIVE_LEARNING_RETRAIN_THRESHOLD=100
UNCERTAINTY_LOW=0.4
UNCERTAINTY_HIGH=0.7

# Phase 15: 聚类分析
CLUSTERING_ENABLED=true
CLUSTERING_MIN_SIZE=5
CLUSTERING_ALGORITHM=hdbscan

# Phase 16: 缓存优化
CACHE_L1_ENABLED=true
CACHE_L1_MAX_SIZE=1000
CACHE_L2_ENABLED=true
CACHE_TTL_SECONDS=3600
```

---

## 📦 新增依赖

```txt
# requirements-phase13+.txt

# 身份认证
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4

# 聚类分析
hdbscan>=0.8.33
umap-learn>=0.5.3

# 性能优化
cachetools>=5.3.0
aiocache>=0.12.0
```

---

**文档版本**: v3.0
**更新日期**: 2025-12-06
**作者**: Claude Code Analysis
