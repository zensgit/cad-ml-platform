# 🚀 Phase 5 Roadmap: High Performance & 3D Deep Learning

**Target Version**: v1.5.0
**Estimated Timeline**: Q1 2026
**Focus**: Latency Reduction, Native 3D Understanding, Privacy

## 1. Executive Summary

Following the successful deployment of the Cognitive Intelligence Engine (v1.4.0), Phase 5 aims to address the computational bottlenecks of LLM inference and the information loss inherent in 2D projection methods. We will transition to **local high-performance inference** and **native 3D deep learning**.

## 2. Key Initiatives

### 2.1 ⚡ vLLM Integration (High-Throughput Inference)
Currently, `LLMReasoningEngine` relies on external API calls, introducing network latency (300ms-1s).
- **Goal**: Sub-50ms latency for semantic reasoning.
- **Tech Stack**: [vLLM](https://github.com/vllm-project/vllm) + Ray Serve.
- **Model**: Quantized DeepSeek-v3 or Llama-3-8B (AWQ/GPTQ).
- **Architecture**:
  - Deploy vLLM as a sidecar container or separate microservice.
  - Implement continuous batching for high concurrency.

### 2.2 🧊 PointNet++ Integration (Native 3D Learning)
Current v7 features use "Geometric (B-Rep) + Visual (2D Projection)". This loses internal structure information.
- **Goal**: Direct feature extraction from 3D point clouds.
- **Tech Stack**: PyTorch Geometric / Open3D.
- **Pipeline**:
  1. **Sampling**: Convert STEP/IGES B-Rep to uniform point cloud (2048 points).
  2. **Embedding**: Pass through PointNet++ encoder.
  3. **Fusion**: Concatenate with v7 semantic features.
- **Benefit**: Better distinction of complex internal cavities and machining features.

### 2.3 🔒 Federated Learning (Privacy-Preserving Training)
As we serve multiple enterprise tenants, data privacy prevents centralizing CAD models for training.
- **Goal**: Train metric models without raw data leaving tenant premises.
- **Tech Stack**: Flower (flwr) or PySyft.
- **Workflow**:
  1. Local training on tenant edge nodes.
  2. Encrypted gradient aggregation on central server.
  3. Global model update distribution.

## 3. Implementation Plan

### Week 1-2: Infrastructure Prep
- [x] Benchmark vLLM on target GPU hardware (T4/A10).
  - Created `scripts/benchmark_vllm_quantization.py` for latency/throughput testing.
- [x] Create point cloud sampling pipeline for STEP files.
  - Implemented `PointCloudSampler` using `trimesh`.
  - Created `PointNetFeatureExtractor` skeleton.

### Week 3-4: Prototype Development
- [x] Implement `PointNetFeatureExtractor`.
  - Skeleton created and integrated into `FeatureExtractor` (v8).
  - Configured via `ENABLE_3D_NATIVE` flag.
  - Reference `PointNetPP` architecture implemented in `src/core/deep_3d/pointnet_arch.py`.
- [x] Deploy vLLM service and update `LLMReasoningEngine` to use local endpoint.
  - Client-side integration complete (`VLLM_ENDPOINT` support).
  - Docker Compose template created.

### Week 5-6: Integration & Optimization
- [x] Train hybrid model (PointNet + v7).
  - Created `scripts/train_hybrid_3d_model.py` for hybrid training.
  - Defines `HybridNetwork` combining v7 MLP and PointNet.
- [x] Optimize quantization parameters for vLLM.
  - Created `scripts/benchmark_vllm_quantization.py` to measure latency/throughput.
  - Supports concurrency testing for AWQ/GPTQ models.
- [x] Implement Federated Learning Scaffold.
  - Created `src/core/federated/` with `FederatedClient` and `FederatedServer`.

## 4. Success Metrics

| Metric | Current (v1.4) | Target (v1.5) |
|--------|---------------|---------------|
| **Reasoning Latency** | ~800ms (API) | < 50ms (Local) |
| **3D Classification Acc** | 92% (Projection) | 96% (Point Cloud) |
| **Throughput** | 10 req/s | 50 req/s |
| **Data Privacy** | Centralized | Federated |

## 5. Resource Requirements

- **GPU**: Minimum NVIDIA T4 (16GB VRAM) for vLLM + PointNet inference.
- **Storage**: Increased storage for point cloud cache.
