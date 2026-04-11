# Phase 17 Roadmap: Autonomous Manufacturing (Revised Strategy)

**Status**: Planning
**Target Start**: Q3 2026 (Post-Hardening)
**Duration**: 9-12 Months
**Focus**: Robotics, Edge AI, Predictive Maintenance

## 1. Strategic Assessment
The initial plan for Phase 17 was deemed too aggressive given the current maturity of the Digital Twin and Assembly Inference modules. This revised roadmap adopts a phased approach, extending the timeline to ensure robust infrastructure before advanced robotics implementation.

## 2. Prerequisites (Phase 15-16 Hardening)
Before starting Phase 17, the following "scaffolds" must be hardened:
- **Digital Twin**: Implement real MQTT broker connection (replacing stubs) and sensor data modeling.
- **Assembly Inference**: Add physical collision detection and constraint solving to the inference engine.

## 3. Implementation Plan (Sub-Phases)

### Phase 17A: Kinematic Foundation (8 Weeks)
**Goal**: Establish the mathematical and data foundation for robotics.
- [ ] **URDF/SDF Parser**: Import standard robot descriptions.
- [ ] **Kinematic Modeling**: Implement forward/inverse kinematics solvers.
- [ ] **Visualization**: Basic 3D robot rendering in the web dashboard.

### Phase 17B: Path Planning (8 Weeks)
**Goal**: Automated trajectory generation.
- [ ] **RRT* Algorithm**: Implement collision-free path finding.
- [ ] **Constraint Integration**: Link with Assembly Inference for valid assembly sequences.
- [ ] **Simulation**: Full physics-based simulation of assembly tasks.

### Phase 17C: Edge Deployment (8 Weeks)
**Goal**: Move intelligence to the manufacturing edge.
- [ ] **Model Quantization**: Convert models to INT8 (TFLite/TensorRT).
- [ ] **Edge Service**: Lightweight inference service with MQTT support.
- [ ] **Hardware Validation**: Benchmark on Jetson Nano / Edge TPU.

### Phase 17D: Predictive Maintenance (8 Weeks)
**Goal**: Data-driven equipment health monitoring.
- [ ] **Telemetry Pipeline**: High-throughput time-series ingestion.
- [ ] **Anomaly Detection**: LSTM/Transformer models for failure prediction.
- [ ] **Alerting**: Integration with maintenance ticketing systems.

## 4. Revised Success Metrics
- **Path Planning**: < 10s for standard assembly tasks (relaxed from 5s).
- **Inspection Accuracy**: > 95% defect detection rate (relaxed from 99%).
- **Edge Latency**: < 100ms inference time (relaxed from 50ms).
