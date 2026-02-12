# CAD ML Platform - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Load Balancer                                   │
│                           (Nginx Ingress / ALB)                             │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
            ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
            │   API Pod 1   │     │   API Pod 2   │     │   API Pod N   │
            │  (FastAPI)    │     │  (FastAPI)    │     │  (FastAPI)    │
            └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          │
        ┌─────────────────┬───────────────┼───────────────┬─────────────────┐
        │                 │               │               │                 │
        ▼                 ▼               ▼               ▼                 ▼
  ┌───────────┐   ┌───────────────┐ ┌───────────┐ ┌───────────────┐ ┌───────────┐
  │   Redis   │   │  ML Models    │ │  Vector   │ │   External    │ │  Object   │
  │  Cluster  │   │  (PyTorch)    │ │   Store   │ │   Vision API  │ │  Storage  │
  │           │   │               │ │  (FAISS)  │ │               │ │   (S3)    │
  └───────────┘   └───────────────┘ └───────────┘ └───────────────┘ └───────────┘
        │
        │  Job Queue
        ▼
  ┌───────────────┐     ┌───────────────┐
  │  Worker Pod 1 │     │  Worker Pod N │
  │    (ARQ)      │     │    (ARQ)      │
  └───────────────┘     └───────────────┘
```

## Core Components

### 1. API Layer (FastAPI)

**Location**: `src/api/`

- **Router Registration**: Modular endpoint organization
- **Middleware Stack**: Rate limiting, audit logging, error handling
- **API Versioning**: v1 (legacy), v2 (current)
- **WebSocket Support**: Real-time notifications

```python
# Request flow
Request → Rate Limiter → Auth → Router → Handler → Response
            ↓                              ↓
         Audit Log                    Telemetry
```

### 2. ML Processing Pipeline

**Location**: `src/ml/`

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  DXF/DWG    │───▶│  Graph       │───▶│  GNN Model  │───▶│  Post-       │
│  Parser     │    │  Builder     │    │  Inference  │    │  Processing  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          │                   │
                          ▼                   ▼
                   ┌──────────────┐    ┌─────────────┐
                   │  Feature     │    │  Hot Reload │
                   │  Extraction  │    │  Manager    │
                   └──────────────┘    └─────────────┘
```

**Key Models**:
- **DXF Graph GNN**: Entity classification for CAD drawings
- **OCR Pipeline**: Text and dimension extraction
- **Material Classifier**: Material identification from features

### 3. Caching Architecture

**Location**: `src/core/cache/`

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cache Hierarchy                          │
├─────────────────────────────────────────────────────────────────┤
│  L1 Cache (In-Memory LRU)                                       │
│  ├── TTL: 5 minutes                                             │
│  ├── Max Size: 2000 entries                                     │
│  └── Hit Rate: ~85%                                             │
├─────────────────────────────────────────────────────────────────┤
│  L2 Cache (Redis)                                               │
│  ├── TTL: 1 hour                                                │
│  ├── Serialization: MessagePack                                 │
│  └── Hit Rate: ~95%                                             │
├─────────────────────────────────────────────────────────────────┤
│  Model Cache (Hot Reload Manager)                               │
│  ├── Version-based invalidation                                 │
│  └── Graceful model swapping                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Background Job Processing

**Location**: `src/core/batch/`

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  API Submit  │────▶│  Priority Queue  │────▶│  ARQ Worker  │
│              │     │  (Redis)         │     │              │
└──────────────┘     └──────────────────┘     └──────────────┘
                              │                       │
                              │                       ▼
                              │                ┌──────────────┐
                              │                │  Processing  │
                              │                │  Pipeline    │
                              │                └──────────────┘
                              │                       │
                              ▼                       ▼
                     ┌──────────────────┐     ┌──────────────┐
                     │  Job Status      │◀────│  Results     │
                     │  Tracking        │     │  Storage     │
                     └──────────────────┘     └──────────────┘
```

**Job Types**:
- `ocr_extract`: Batch OCR processing
- `vision_analyze`: Batch vision analysis
- `dedup_2d`: 2D drawing deduplication

### 5. Rate Limiting

**Location**: `src/api/middleware/rate_limiting.py`

**Algorithm**: Sliding Window with Redis backend

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rate Limit Tiers                              │
├──────────────┬────────────┬─────────┬───────────────────────────┤
│     Tier     │  Rate/min  │  Burst  │  Identification           │
├──────────────┼────────────┼─────────┼───────────────────────────┤
│  Anonymous   │     60     │    5    │  IP Address               │
│  Free        │    300     │   10    │  API Key                  │
│  Pro         │   1200     │   50    │  API Key                  │
│  Enterprise  │   6000     │  200    │  API Key + Tenant         │
└──────────────┴────────────┴─────────┴───────────────────────────┘
```

### 6. Observability Stack

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Application│────▶│ Prometheus  │────▶│   Grafana   │
│  Metrics    │     │             │     │  Dashboard  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       │                   ▼
       │            ┌─────────────┐
       │            │ AlertManager│
       │            └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Structured │────▶│   Loki /    │
│    Logs     │     │   ELK       │
└─────────────┘     └─────────────┘
```

**Metrics Exported**:
- `http_requests_total`: Request count by endpoint, status
- `http_request_duration_seconds`: Request latency histogram
- `model_inference_duration_seconds`: ML inference latency
- `cache_hits_total`, `cache_misses_total`: Cache statistics
- `rate_limit_hits_total`: Rate limiting events

## Data Flow

### OCR Processing Flow

```
1. Client uploads DXF file
          │
          ▼
2. File validation & virus scan
          │
          ▼
3. Parse DXF structure (ezdxf)
          │
          ▼
4. Extract text entities
   ├── TEXT blocks
   ├── MTEXT blocks
   └── DIMENSION entities
          │
          ▼
5. Apply OCR model (if needed)
          │
          ▼
6. Post-process & structure results
          │
          ▼
7. Cache results (content-hash key)
          │
          ▼
8. Return JSON response
```

### Vision Analysis Flow

```
1. Receive CAD file
          │
          ▼
2. Build entity graph
   ├── Nodes: Entities (LINE, CIRCLE, ARC, etc.)
   └── Edges: Spatial relationships
          │
          ▼
3. Extract node features (20-dim vector)
   ├── Entity type one-hot
   ├── Geometric properties
   └── Layer/color information
          │
          ▼
4. GNN inference (message passing)
          │
          ▼
5. Aggregate predictions
          │
          ▼
6. Return structured analysis
```

## Design Decisions

### 1. Why FastAPI?

- **Async Support**: Critical for I/O-bound operations
- **Type Safety**: Pydantic models for validation
- **OpenAPI**: Auto-generated documentation
- **Performance**: Comparable to Go/Node.js

### 2. Why Graph Neural Networks?

- **Structural Awareness**: CAD drawings are inherently graph-structured
- **Permutation Invariance**: Entity order shouldn't matter
- **Relation Learning**: Captures spatial relationships between entities

### 3. Why Two-Level Cache?

- **L1 (Memory)**: Ultra-low latency for hot data
- **L2 (Redis)**: Shared across pods, survives restarts
- **Trade-off**: Memory cost vs. latency reduction

### 4. Why ARQ for Background Jobs?

- **Redis-native**: Minimal infrastructure
- **Async-first**: Built on asyncio
- **Retry Support**: Automatic failure handling
- **Simple**: Easy to understand and debug

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Security Layers                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Network Layer                                               │
│     ├── TLS 1.3 termination at ingress                         │
│     ├── Network policies (namespace isolation)                  │
│     └── WAF rules (OWASP top 10)                               │
├─────────────────────────────────────────────────────────────────┤
│  2. Authentication Layer                                        │
│     ├── API Key validation                                      │
│     ├── JWT tokens (for WebSocket)                             │
│     └── Admin token (privileged operations)                     │
├─────────────────────────────────────────────────────────────────┤
│  3. Authorization Layer                                         │
│     ├── Tier-based rate limiting                               │
│     ├── Endpoint permissions                                    │
│     └── Tenant isolation                                        │
├─────────────────────────────────────────────────────────────────┤
│  4. Application Layer                                           │
│     ├── Input validation (Pydantic)                            │
│     ├── File type verification                                  │
│     ├── Size limits                                             │
│     └── Sanitization                                            │
├─────────────────────────────────────────────────────────────────┤
│  5. Audit Layer                                                 │
│     ├── Request logging                                         │
│     ├── Action tracking                                         │
│     └── Security event alerting                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Scalability Considerations

### Horizontal Scaling

| Component | Strategy | Notes |
|-----------|----------|-------|
| API Pods | HPA on CPU/Memory | Stateless, scale freely |
| Workers | HPA on queue depth | Match to job volume |
| Redis | Cluster mode | For high availability |
| Models | Shared volume | NFS/EFS for consistency |

### Vertical Scaling

| Workload | Resource Profile |
|----------|------------------|
| API (standard) | 500m-2000m CPU, 1-4Gi RAM |
| API (ML-heavy) | 1000m-4000m CPU, 2-8Gi RAM |
| Worker | 200m-1000m CPU, 512Mi-2Gi RAM |
| GPU Worker | 1-4 NVIDIA GPUs |

### Bottleneck Analysis

1. **Model Loading**: ~10s cold start → mitigated by hot reload
2. **Large File Processing**: >100MB files → chunked processing
3. **Complex Drawings**: >10k entities → importance sampling
4. **Redis Memory**: High cardinality → TTL management

## Future Considerations

1. **Multi-Region Deployment**: Active-active with Redis Global
2. **Model Serving**: Dedicated Triton/TensorRT inference servers
3. **Event Streaming**: Kafka for high-volume async processing
4. **Vector Database**: Milvus for production vector search
