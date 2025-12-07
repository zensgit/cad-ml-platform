# Phase 15-16 Hardening Roadmap: Bridging the Gap to Autonomous Manufacturing

**Status**: Planned
**Target**: Q1-Q2 2026
**Duration**: 16 Weeks
**Goal**: Transform "Scaffold" prototypes into production-grade infrastructure to support Phase 17.

## 1. Strategic Context
Phase 17 (Robotics) requires a robust Digital Twin (for telemetry) and a Physics-Aware Assembly Engine (for path planning). Currently, these are prototypes. This hardening phase bridges that gap.

## 2. Workstream 1: Digital Twin Connectivity (Weeks 1-6)
**Objective**: Replace stubs with real-time, high-throughput IoT connectivity.

### Week 1-2: MQTT Infrastructure
- [ ] **Broker Deployment**: Add `mosquitto` or `emqx` to `docker-compose.yml`.
- [ ] **Client Implementation**:
  - Create `src/core/twin/connectivity.py` using `aiomqtt`.
  - Implement robust reconnection logic and QoS 1/2 support.
  - Add TLS/SSL support for secure edge communication.

### Week 3-4: Telemetry Data Pipeline
- [ ] **Data Modeling**:
  - Define `TelemetryFrame` Pydantic model (Timestamp, DeviceID, Sensors, Metrics).
  - Implement binary serialization (Protobuf/MsgPack) for efficiency.
- [ ] **Ingestion Service**:
  - Create `TelemetryIngestor` actor to consume MQTT topics.
  - Implement backpressure handling for high-load scenarios.

### Week 5-6: Time-Series Storage
- [ ] **Storage Backend**:
  - Integrate `InfluxDB` or `TimescaleDB` into the stack.
  - Implement `TimeSeriesStore` adapter in `src/core/storage/timeseries.py`.
- [ ] **Query API**:
  - Expose `/api/v1/twin/history` for retrieving historical sensor data.

## 3. Workstream 2: Physics-Aware Assembly Engine (Weeks 7-12)
**Objective**: Enable physical validation of CAD assemblies (Collision, Mass, Constraints).

### Week 7-8: Collision Detection
- [ ] **Engine Integration**:
  - Integrate `trimesh.collision` or `python-fcl` (Flexible Collision Library).
  - Implement `CollisionManager` in `src/core/physics/collision.py`.
- [ ] **Interference Check**:
  - Implement `detect_interference(assembly_id)` API.
  - Visualize colliding mesh faces in the frontend.

### Week 9-10: Constraint Solving
- [ ] **Geometric Constraints**:
  - Implement solver for: `Coincident`, `Concentric`, `Parallel`, `Distance`.
  - Validate if a proposed assembly configuration is geometrically valid.
- [ ] **DOF Analysis**:
  - Calculate Degrees of Freedom for assembly components.

### Week 11-12: Physical Properties
- [ ] **Material Database**:
  - Expand `src/core/materials.py` with density, elasticity, thermal properties.
- [ ] **Mass Properties**:
  - Calculate Center of Mass (CoM) and Inertia Tensor for assemblies.
  - Essential for robot arm load estimation.

## 4. Workstream 3: Edge-Ready Architecture (Weeks 13-16)
**Objective**: Prepare ML models for deployment on constrained edge devices.

### Week 13-14: Model Optimization
- [ ] **ONNX Export**:
  - Create pipeline to export `MetricMLP` and `PointNet++` to ONNX format.
- [ ] **Quantization**:
  - Implement Post-Training Quantization (PTQ) to INT8.
  - Benchmark accuracy loss vs. size reduction.

### Week 15-16: Edge Client SDK
- [ ] **Lightweight Client**:
  - Create `cad-ml-edge` Python package (minimal dependencies).
  - Support local inference using `onnxruntime`.
- [ ] **Sync Protocol**:
  - Implement "Store-and-Forward" logic for offline edge devices.

## 5. Deliverables
- **Infrastructure**: Production-ready MQTT Broker & Time-Series DB.
- **Code**: `src/core/physics/`, `src/core/twin/connectivity.py`.
- **SDK**: `clients/edge_sdk/`.
- **Validation**: Physics engine passing 95% of collision tests on standard assemblies.
