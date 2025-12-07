# Phase 15-16 Functional Design Document

**Version**: 1.0
**Status**: Draft
**Author**: CAD-ML Platform Team
**Date**: 2025-12-07
**Target**: Q1-Q2 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Workstream 1: Digital Twin Connectivity](#3-workstream-1-digital-twin-connectivity)
4. [Workstream 2: Physics-Aware Assembly Engine](#4-workstream-2-physics-aware-assembly-engine)
5. [Workstream 3: Edge-Ready Architecture](#5-workstream-3-edge-ready-architecture)
6. [Data Models & Interfaces](#6-data-models--interfaces)
7. [API Specifications](#7-api-specifications)
8. [Integration Points](#8-integration-points)
9. [Success Metrics & Validation](#9-success-metrics--validation)
10. [Risk Analysis & Mitigation](#10-risk-analysis--mitigation)

---

## 1. Executive Summary

### 1.1 Purpose

This document defines the functional specifications for Phase 15-16 "Hardening" of the CAD-ML Platform. The goal is to transform prototype-level scaffolds into production-grade infrastructure, enabling Phase 17's Autonomous Manufacturing & Robotics capabilities.

### 1.2 Scope

| Workstream | Objective | Duration |
|------------|-----------|----------|
| Digital Twin Connectivity | Real-time IoT telemetry pipeline | Weeks 1-6 |
| Physics-Aware Assembly Engine | 3D collision & constraint solving | Weeks 7-12 |
| Edge-Ready Architecture | Model optimization & Edge SDK | Weeks 13-16 |

### 1.3 Current State Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION STATUS                            │
├─────────────────────────────────────────────────────────────────────┤
│  Component                      │ Status      │ Completeness       │
├─────────────────────────────────┼─────────────┼────────────────────┤
│  TelemetryFrame (Pydantic)      │ DONE        │ ████████████ 100%  │
│  MqttTelemetryClient            │ DONE        │ ████████████ 100%  │
│  TelemetryIngestor              │ DONE        │ ████████████ 100%  │
│  TimeSeriesStore (Memory)       │ DONE        │ ████████████ 100%  │
│  TimeSeriesStore (InfluxDB)     │ PLANNED     │ ░░░░░░░░░░░░   0%  │
│  TimeSeriesStore (TimescaleDB)  │ PLANNED     │ ░░░░░░░░░░░░   0%  │
│  CollisionManager               │ PLANNED     │ ░░░░░░░░░░░░   0%  │
│  ConstraintSolver               │ PLANNED     │ ░░░░░░░░░░░░   0%  │
│  PhysicalPropertiesCalculator   │ PLANNED     │ ░░░░░░░░░░░░   0%  │
│  ONNX Export Pipeline           │ PLANNED     │ ░░░░░░░░░░░░   0%  │
│  Edge Client SDK                │ PLANNED     │ ░░░░░░░░░░░░   0%  │
└─────────────────────────────────┴─────────────┴────────────────────┘
```

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CAD-ML PLATFORM                                   │
│                       Phase 15-16 Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │   EDGE DEVICES   │    │   MQTT BROKER    │    │   API GATEWAY    │      │
│  │  (cad-ml-edge)   │───▶│   (Mosquitto)    │───▶│   (FastAPI)      │      │
│  │                  │    │   Port: 1883     │    │   Port: 8000     │      │
│  └──────────────────┘    └──────────────────┘    └────────┬─────────┘      │
│           │                       │                        │                │
│           │              ┌────────┴────────┐               │                │
│           │              │                 │               │                │
│           ▼              ▼                 ▼               ▼                │
│  ┌──────────────┐  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │  ONNX        │  │  Telemetry   │ │  Time-Series │ │  Physics     │      │
│  │  Runtime     │  │  Ingestor    │ │  Store       │ │  Engine      │      │
│  │              │  │  (Backpres.) │ │  (InfluxDB)  │ │  (trimesh)   │      │
│  └──────────────┘  └──────────────┘ └──────────────┘ └──────────────┘      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                     CORE SERVICES                               │       │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │       │
│  │  │ Feature    │ │ Assembly   │ │ Similarity │ │ Active     │   │       │
│  │  │ Extractor  │ │ Inference  │ │ Search     │ │ Learning   │   │       │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TELEMETRY DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EDGE DEVICE          MQTT          INGESTOR         STORAGE        API    │
│      │                 │                │               │            │     │
│      │  TelemetryFrame │                │               │            │     │
│      │  (MsgPack)      │                │               │            │     │
│      │─────────────────▶│                │               │            │     │
│      │                 │  Subscribe     │               │            │     │
│      │                 │────────────────▶│               │            │     │
│      │                 │                │  Decode       │            │     │
│      │                 │                │──────────┐    │            │     │
│      │                 │                │          │    │            │     │
│      │                 │                │◀─────────┘    │            │     │
│      │                 │                │               │            │     │
│      │                 │                │  Queue        │            │     │
│      │                 │                │  (Backpress.) │            │     │
│      │                 │                │───────────────▶│            │     │
│      │                 │                │               │  Write     │     │
│      │                 │                │               │──────┐     │     │
│      │                 │                │               │      │     │     │
│      │                 │                │               │◀─────┘     │     │
│      │                 │                │               │            │     │
│      │                 │                │               │   Query    │     │
│      │                 │                │               │◀───────────│     │
│      │                 │                │               │            │     │
│      │                 │                │               │   Results  │     │
│      │                 │                │               │───────────▶│     │
│      │                 │                │               │            │     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Workstream 1: Digital Twin Connectivity

### 3.1 Objectives

Transform the existing stub Digital Twin sync into a production-grade, high-throughput IoT connectivity layer capable of:

- **10,000+ msg/sec** MQTT message throughput
- **Sub-100ms** end-to-end latency
- **TLS 1.3** secure communication
- **Graceful degradation** under load

### 3.2 Existing Implementation

#### 3.2.1 TelemetryFrame (DONE)

**Location**: `src/core/twin/connectivity.py:38-76`

```python
class TelemetryFrame(BaseModel):
    """Canonical telemetry envelope."""
    timestamp: float       # Unix timestamp seconds
    device_id: str         # Source device / asset identifier
    sensors: Dict[str, float]   # Raw sensor readings
    metrics: Dict[str, float]   # Derived metrics/health signals
    status: Dict[str, Any]      # Additional status/labels

    def to_bytes(self) -> bytes:
        """Serialize to MsgPack (preferred) or JSON fallback."""
        ...

    @classmethod
    def from_bytes(cls, data: bytes) -> "TelemetryFrame":
        """Deserialize from MsgPack or JSON bytes."""
        ...
```

**Features**:
- MsgPack serialization (7x smaller than JSON)
- JSON fallback for compatibility
- Pydantic validation with `timestamp >= 0`

#### 3.2.2 MqttTelemetryClient (DONE)

**Location**: `src/core/twin/connectivity.py:90-163`

```python
class MqttTelemetryClient:
    """MQTT client wrapper with graceful degradation."""

    async def start(self, topics: list[str], handler: Callable) -> None:
        """Start subscription loop; no-op if aiomqtt unavailable."""

    async def wait_subscribed(self, timeout: float = 2.0) -> bool:
        """Wait until subscription is established."""

    async def stop(self) -> None:
        """Stop subscription loop."""
```

**Features**:
- TLS/SSL support via `ssl.create_default_context()`
- QoS 1/2 support
- Graceful degradation when `aiomqtt` unavailable
- Reconnection logic

#### 3.2.3 TelemetryIngestor (DONE)

**Location**: `src/core/twin/ingest.py:22-84`

```python
class TelemetryIngestor:
    """Telemetry ingestion with backpressure handling."""

    def __init__(self, store: TimeSeriesStore, max_queue: int = 1000):
        self.queue: asyncio.Queue[TelemetryFrame]
        self.drop_count = 0  # Backpressure metric

    async def handle_payload(self, payload: Any) -> Dict[str, Any]:
        """Decode and enqueue for persistence."""
        # Returns: {"status": "queued|dropped|rejected"}
```

**Features**:
- Bounded queue (default 1000) for backpressure
- Drop count tracking for metrics
- Payload coercion (bytes/dict/TelemetryFrame)

### 3.3 Planned Implementation

#### 3.3.1 InfluxDB Adapter

**Location**: `src/core/storage/influxdb_adapter.py` (NEW)

```python
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

class InfluxDBTimeSeriesStore:
    """InfluxDB 2.x adapter for telemetry storage."""

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        batch_size: int = 500,
        flush_interval_ms: int = 1000,
    ):
        self._client = InfluxDBClient(url=url, token=token, org=org)
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
        self._query_api = self._client.query_api()
        self._bucket = bucket
        self._org = org

    async def append(self, frame: TelemetryFrame) -> None:
        """Write telemetry frame to InfluxDB."""
        point = Point("telemetry") \
            .tag("device_id", frame.device_id) \
            .time(int(frame.timestamp * 1e9), WritePrecision.NS)

        for key, value in frame.sensors.items():
            point = point.field(f"sensor_{key}", value)
        for key, value in frame.metrics.items():
            point = point.field(f"metric_{key}", value)

        self._write_api.write(bucket=self._bucket, org=self._org, record=point)

    async def history(
        self,
        device_id: str,
        limit: int = 100,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> List[TelemetryFrame]:
        """Query historical telemetry data."""
        query = f'''
        from(bucket: "{self._bucket}")
            |> range(start: {start or "-1h"}, stop: {stop or "now()"})
            |> filter(fn: (r) => r["device_id"] == "{device_id}")
            |> limit(n: {limit})
        '''
        tables = self._query_api.query(query, org=self._org)
        return self._tables_to_frames(tables)

    async def aggregate(
        self,
        device_id: str,
        field: str,
        window: str = "5m",
        fn: str = "mean",
    ) -> List[Dict[str, Any]]:
        """Aggregate telemetry data over time windows."""
        query = f'''
        from(bucket: "{self._bucket}")
            |> range(start: -1h)
            |> filter(fn: (r) => r["device_id"] == "{device_id}")
            |> filter(fn: (r) => r["_field"] == "{field}")
            |> aggregateWindow(every: {window}, fn: {fn})
        '''
        tables = self._query_api.query(query, org=self._org)
        return self._tables_to_aggregates(tables)
```

**Interface Contract**:

```python
class TimeSeriesStore(Protocol):
    """Time-series storage interface."""

    async def append(self, frame: TelemetryFrame) -> None:
        """Append a single telemetry frame."""
        ...

    async def history(
        self,
        device_id: str,
        limit: int = 100,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> List[TelemetryFrame]:
        """Retrieve historical frames for a device."""
        ...

    async def aggregate(
        self,
        device_id: str,
        field: str,
        window: str = "5m",
        fn: str = "mean",
    ) -> List[Dict[str, Any]]:
        """Aggregate telemetry data."""
        ...

    async def health_check(self) -> bool:
        """Check storage backend connectivity."""
        ...
```

#### 3.3.2 TimescaleDB Adapter

**Location**: `src/core/storage/timescale_adapter.py` (NEW)

```python
import asyncpg

class TimescaleDBTimeSeriesStore:
    """TimescaleDB adapter for SQL-native telemetry storage."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """Create connection pool and ensure hypertable exists."""
        self._pool = await asyncpg.create_pool(self._dsn)
        await self._ensure_schema()

    async def _ensure_schema(self) -> None:
        """Create hypertable if not exists."""
        async with self._pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS telemetry (
                    time        TIMESTAMPTZ NOT NULL,
                    device_id   TEXT NOT NULL,
                    sensors     JSONB,
                    metrics     JSONB,
                    status      JSONB
                );
                SELECT create_hypertable('telemetry', 'time',
                    if_not_exists => TRUE);
                CREATE INDEX IF NOT EXISTS idx_telemetry_device
                    ON telemetry (device_id, time DESC);
            ''')

    async def append(self, frame: TelemetryFrame) -> None:
        """Insert telemetry frame."""
        async with self._pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO telemetry (time, device_id, sensors, metrics, status)
                VALUES ($1, $2, $3, $4, $5)
            ''',
                datetime.fromtimestamp(frame.timestamp, tz=timezone.utc),
                frame.device_id,
                json.dumps(frame.sensors),
                json.dumps(frame.metrics),
                json.dumps(frame.status),
            )

    async def history(
        self,
        device_id: str,
        limit: int = 100,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> List[TelemetryFrame]:
        """Query with time range."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT time, device_id, sensors, metrics, status
                FROM telemetry
                WHERE device_id = $1
                  AND ($2::timestamptz IS NULL OR time >= $2)
                  AND ($3::timestamptz IS NULL OR time <= $3)
                ORDER BY time DESC
                LIMIT $4
            ''', device_id, start, stop, limit)
            return [self._row_to_frame(row) for row in rows]
```

#### 3.3.3 Factory Pattern

**Location**: `src/core/storage/__init__.py` (UPDATE)

```python
def create_timeseries_store(settings: Settings) -> TimeSeriesStore:
    """Factory for time-series store based on configuration."""
    backend = settings.TELEMETRY_STORE_BACKEND.lower()

    if backend == "memory":
        return InMemoryTimeSeriesStore()

    elif backend == "influx":
        if not settings.INFLUX_URL:
            raise ValueError("INFLUX_URL required for influx backend")
        return InfluxDBTimeSeriesStore(
            url=settings.INFLUX_URL,
            token=settings.INFLUX_TOKEN,
            org=settings.INFLUX_ORG,
            bucket=settings.INFLUX_BUCKET,
        )

    elif backend == "timescale":
        if not settings.TIMESCALE_DSN:
            raise ValueError("TIMESCALE_DSN required for timescale backend")
        return TimescaleDBTimeSeriesStore(dsn=settings.TIMESCALE_DSN)

    elif backend == "none":
        return NullTimeSeriesStore()

    else:
        logger.warning(f"Unknown backend '{backend}', using memory")
        return InMemoryTimeSeriesStore()
```

### 3.4 Configuration

**Location**: `src/core/config.py` (EXISTING)

```python
# Telemetry / MQTT (Phase 15 hardening)
TELEMETRY_MQTT_ENABLED: bool = False
MQTT_HOST: str = "localhost"
MQTT_PORT: int = 1883
MQTT_USERNAME: Optional[str] = None
MQTT_PASSWORD: Optional[str] = None
MQTT_TLS_CA: Optional[str] = None
MQTT_TOPIC: str = "twin/telemetry/#"
MQTT_QOS: int = 1
MQTT_CLIENT_ID: str = "cad-ml-platform"

TELEMETRY_STORE_BACKEND: str = "memory"  # memory|influx|timescale|none
INFLUX_URL: Optional[str] = None
INFLUX_TOKEN: Optional[str] = None
INFLUX_ORG: Optional[str] = None
INFLUX_BUCKET: Optional[str] = None
TIMESCALE_DSN: Optional[str] = None
```

### 3.5 Infrastructure

**Location**: `deployments/docker/docker-compose.yml` (EXISTING)

```yaml
# MQTT Broker for telemetry (Digital Twin ingest)
mosquitto:
  image: eclipse-mosquitto:2
  ports:
    - "1883:1883"
  volumes:
    - ./mosquitto.conf:/mosquitto/config/mosquitto.conf:ro

# InfluxDB Time Series Database
influxdb:
  image: influxdb:2
  ports:
    - "8086:8086"
  environment:
    - DOCKER_INFLUXDB_INIT_MODE=setup
    - DOCKER_INFLUXDB_INIT_USERNAME=admin
    - DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword
    - DOCKER_INFLUXDB_INIT_ORG=cad-ml
    - DOCKER_INFLUXDB_INIT_BUCKET=telemetry
    - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-auth-token
```

---

## 4. Workstream 2: Physics-Aware Assembly Engine

### 4.1 Objectives

Enable physical validation of CAD assemblies to support Phase 17's robotic path planning:

- **Precise collision detection** using mesh-based algorithms
- **Geometric constraint solving** (Coincident, Concentric, Parallel, Distance)
- **Mass property calculation** (CoM, Inertia Tensor)
- **DOF analysis** for assembly components

### 4.2 Existing Implementation

#### 4.2.1 AssemblyInferenceEngine (EXISTING)

**Location**: `src/core/assembly_inference.py`

The current implementation provides:
- Mating relationship detection (COINCIDENT, PARALLEL, CONCENTRIC, etc.)
- Fit type analysis (CLEARANCE, TRANSITION, INTERFERENCE)
- Assembly sequence inference
- Basic bounding-box interference check (NOT mesh-based)

**Limitations**:
- Uses bounding-box overlap, not true mesh collision
- No constraint solver
- No mass property calculation

### 4.3 Planned Implementation

#### 4.3.1 CollisionManager

**Location**: `src/core/physics/collision.py` (NEW)

```python
"""
Collision detection engine using trimesh or python-fcl.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import trimesh
    from trimesh.collision import CollisionManager as TrimeshCollisionManager
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False
    logger.warning("trimesh not installed; collision detection disabled")

try:
    import fcl
    _FCL_AVAILABLE = True
except ImportError:
    _FCL_AVAILABLE = False


class CollisionBackend(str, Enum):
    TRIMESH = "trimesh"
    FCL = "fcl"
    NONE = "none"


@dataclass
class CollisionResult:
    """Result of collision check between two meshes."""
    has_collision: bool
    collision_pairs: List[Tuple[str, str]]  # (part_a_id, part_b_id)
    contact_points: List[np.ndarray]        # Contact point coordinates
    penetration_depths: List[float]         # Penetration depth per contact
    collision_volume_estimate: float        # Approximate intersection volume


@dataclass
class ClearanceResult:
    """Result of minimum clearance calculation."""
    min_clearance: float                    # Minimum distance between parts
    closest_point_a: np.ndarray
    closest_point_b: np.ndarray
    clearance_vector: np.ndarray           # Direction of minimum clearance


class CollisionManager:
    """
    Manages collision detection for assembly analysis.

    Supports:
    - Pairwise collision detection
    - Batch collision detection
    - Minimum clearance calculation
    - Interference volume estimation
    """

    def __init__(self, backend: CollisionBackend = CollisionBackend.TRIMESH):
        self._backend = backend
        self._meshes: Dict[str, Any] = {}
        self._manager: Optional[TrimeshCollisionManager] = None

        if backend == CollisionBackend.TRIMESH and _TRIMESH_AVAILABLE:
            self._manager = TrimeshCollisionManager()
        elif backend == CollisionBackend.FCL and _FCL_AVAILABLE:
            self._fcl_manager = fcl.DynamicAABBTreeCollisionManager()

    def add_mesh(
        self,
        part_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        """Add a mesh to the collision manager."""
        if self._backend == CollisionBackend.TRIMESH and _TRIMESH_AVAILABLE:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if transform is not None:
                mesh.apply_transform(transform)
            self._meshes[part_id] = mesh
            self._manager.add_object(part_id, mesh)

    def remove_mesh(self, part_id: str) -> None:
        """Remove a mesh from the collision manager."""
        if part_id in self._meshes:
            del self._meshes[part_id]
            if self._manager:
                self._manager.remove_object(part_id)

    def check_collision(
        self,
        part_a_id: str,
        part_b_id: str,
    ) -> CollisionResult:
        """Check collision between two specific parts."""
        if not _TRIMESH_AVAILABLE:
            return CollisionResult(
                has_collision=False,
                collision_pairs=[],
                contact_points=[],
                penetration_depths=[],
                collision_volume_estimate=0.0,
            )

        mesh_a = self._meshes.get(part_a_id)
        mesh_b = self._meshes.get(part_b_id)

        if mesh_a is None or mesh_b is None:
            raise ValueError(f"Mesh not found: {part_a_id if mesh_a is None else part_b_id}")

        # Compute collision
        collision = mesh_a.intersection(mesh_b, engine="blender")
        has_collision = collision.volume > 1e-10  # Small threshold

        contact_points = []
        penetration_depths = []

        if has_collision:
            # Get contact points from intersection
            contact_points = [collision.centroid]
            penetration_depths = [self._estimate_penetration(mesh_a, mesh_b)]

        return CollisionResult(
            has_collision=has_collision,
            collision_pairs=[(part_a_id, part_b_id)] if has_collision else [],
            contact_points=contact_points,
            penetration_depths=penetration_depths,
            collision_volume_estimate=collision.volume if has_collision else 0.0,
        )

    def check_all_collisions(self) -> List[CollisionResult]:
        """Check collisions between all registered meshes."""
        if not self._manager:
            return []

        # Get collision pairs
        in_collision = self._manager.in_collision_internal(return_names=True)

        results = []
        for pair in in_collision:
            result = self.check_collision(pair[0], pair[1])
            if result.has_collision:
                results.append(result)

        return results

    def calculate_clearance(
        self,
        part_a_id: str,
        part_b_id: str,
    ) -> ClearanceResult:
        """Calculate minimum clearance between two parts."""
        if not _TRIMESH_AVAILABLE:
            return ClearanceResult(
                min_clearance=float('inf'),
                closest_point_a=np.zeros(3),
                closest_point_b=np.zeros(3),
                clearance_vector=np.zeros(3),
            )

        mesh_a = self._meshes.get(part_a_id)
        mesh_b = self._meshes.get(part_b_id)

        if mesh_a is None or mesh_b is None:
            raise ValueError("Mesh not found")

        # Use proximity query
        closest, distance, _ = trimesh.proximity.closest_point(
            mesh_a, mesh_b.vertices
        )

        min_idx = np.argmin(distance)
        min_clearance = distance[min_idx]
        closest_a = closest[min_idx]
        closest_b = mesh_b.vertices[min_idx]

        clearance_vector = closest_b - closest_a
        if np.linalg.norm(clearance_vector) > 0:
            clearance_vector = clearance_vector / np.linalg.norm(clearance_vector)

        return ClearanceResult(
            min_clearance=min_clearance,
            closest_point_a=closest_a,
            closest_point_b=closest_b,
            clearance_vector=clearance_vector,
        )

    def _estimate_penetration(
        self,
        mesh_a: "trimesh.Trimesh",
        mesh_b: "trimesh.Trimesh",
    ) -> float:
        """Estimate maximum penetration depth."""
        # Simplified: use intersection volume as proxy
        intersection = mesh_a.intersection(mesh_b, engine="blender")
        if intersection.is_empty:
            return 0.0

        # Approximate penetration as cube root of volume
        return intersection.volume ** (1/3)
```

#### 4.3.2 ConstraintSolver

**Location**: `src/core/physics/constraints.py` (NEW)

```python
"""
Geometric constraint solver for assembly validation.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of geometric constraints."""
    COINCIDENT = "coincident"       # Points or planes touch
    CONCENTRIC = "concentric"       # Cylindrical axes aligned
    PARALLEL = "parallel"           # Planes or axes parallel
    PERPENDICULAR = "perpendicular" # Planes or axes at 90 degrees
    DISTANCE = "distance"           # Fixed distance between entities
    ANGLE = "angle"                 # Fixed angle between entities
    TANGENT = "tangent"            # Surfaces touch at single point/line
    FIXED = "fixed"                # Entity locked in place


@dataclass
class GeometricEntity:
    """Represents a geometric entity for constraint solving."""
    entity_id: str
    entity_type: str  # "point", "line", "plane", "cylinder", "sphere"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    direction: Optional[np.ndarray] = None  # For lines/axes
    normal: Optional[np.ndarray] = None      # For planes
    radius: Optional[float] = None           # For cylinders/spheres
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Constraint:
    """A single constraint between entities."""
    constraint_id: str
    constraint_type: ConstraintType
    entity_a: str  # Entity ID
    entity_b: Optional[str] = None  # Entity ID (None for FIXED)
    value: Optional[float] = None  # For DISTANCE/ANGLE constraints
    tolerance: float = 0.001  # Acceptable error


@dataclass
class ConstraintSolverResult:
    """Result of constraint solving."""
    is_valid: bool                          # All constraints satisfied
    error: float                            # Total constraint error
    transforms: Dict[str, np.ndarray]       # Entity transforms (4x4 matrices)
    violated_constraints: List[str]         # IDs of violated constraints
    dof_remaining: int                     # Degrees of freedom remaining
    iterations: int                        # Solver iterations used


class ConstraintSolver:
    """
    Geometric constraint solver for assembly validation.

    Uses numerical optimization to find entity positions
    satisfying all constraints.
    """

    def __init__(self, tolerance: float = 0.001, max_iterations: int = 1000):
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._entities: Dict[str, GeometricEntity] = {}
        self._constraints: List[Constraint] = []

    def add_entity(self, entity: GeometricEntity) -> None:
        """Add a geometric entity to the solver."""
        self._entities[entity.entity_id] = entity

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the solver."""
        self._constraints.append(constraint)

    def solve(self) -> ConstraintSolverResult:
        """Solve the constraint system."""
        if not self._constraints:
            return ConstraintSolverResult(
                is_valid=True,
                error=0.0,
                transforms={},
                violated_constraints=[],
                dof_remaining=self._calculate_total_dof(),
                iterations=0,
            )

        # Build initial state vector (position + rotation for each entity)
        x0 = self._build_state_vector()

        # Define objective function (sum of squared constraint errors)
        def objective(x):
            self._apply_state_vector(x)
            return self._compute_total_error()

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            options={'maxiter': self._max_iterations, 'ftol': self._tolerance}
        )

        # Apply final state
        self._apply_state_vector(result.x)

        # Check which constraints are violated
        violated = []
        for c in self._constraints:
            error = self._compute_constraint_error(c)
            if error > c.tolerance:
                violated.append(c.constraint_id)

        # Build transforms
        transforms = {}
        for eid, entity in self._entities.items():
            transforms[eid] = self._entity_to_transform(entity)

        return ConstraintSolverResult(
            is_valid=len(violated) == 0,
            error=result.fun,
            transforms=transforms,
            violated_constraints=violated,
            dof_remaining=self._calculate_remaining_dof(),
            iterations=result.nit,
        )

    def _compute_constraint_error(self, constraint: Constraint) -> float:
        """Compute error for a single constraint."""
        entity_a = self._entities.get(constraint.entity_a)
        entity_b = self._entities.get(constraint.entity_b) if constraint.entity_b else None

        if entity_a is None:
            return float('inf')

        if constraint.constraint_type == ConstraintType.COINCIDENT:
            if entity_b is None:
                return float('inf')
            return np.linalg.norm(entity_a.position - entity_b.position)

        elif constraint.constraint_type == ConstraintType.CONCENTRIC:
            if entity_b is None or entity_a.direction is None or entity_b.direction is None:
                return float('inf')
            # Distance between axes + angle between directions
            axis_dist = self._axis_distance(entity_a, entity_b)
            angle = self._angle_between(entity_a.direction, entity_b.direction)
            return axis_dist + min(angle, np.pi - angle)

        elif constraint.constraint_type == ConstraintType.PARALLEL:
            if entity_b is None:
                return float('inf')
            dir_a = entity_a.direction if entity_a.direction is not None else entity_a.normal
            dir_b = entity_b.direction if entity_b.direction is not None else entity_b.normal
            if dir_a is None or dir_b is None:
                return float('inf')
            angle = self._angle_between(dir_a, dir_b)
            return min(angle, np.pi - angle)

        elif constraint.constraint_type == ConstraintType.DISTANCE:
            if entity_b is None or constraint.value is None:
                return float('inf')
            actual_dist = np.linalg.norm(entity_a.position - entity_b.position)
            return abs(actual_dist - constraint.value)

        elif constraint.constraint_type == ConstraintType.ANGLE:
            if entity_b is None or constraint.value is None:
                return float('inf')
            dir_a = entity_a.direction if entity_a.direction is not None else entity_a.normal
            dir_b = entity_b.direction if entity_b.direction is not None else entity_b.normal
            if dir_a is None or dir_b is None:
                return float('inf')
            actual_angle = self._angle_between(dir_a, dir_b)
            return abs(actual_angle - constraint.value)

        elif constraint.constraint_type == ConstraintType.FIXED:
            # Fixed entities should not move from initial position
            return 0.0  # Handled separately

        return 0.0

    def _compute_total_error(self) -> float:
        """Compute total error across all constraints."""
        return sum(self._compute_constraint_error(c) ** 2 for c in self._constraints)

    def _axis_distance(self, entity_a: GeometricEntity, entity_b: GeometricEntity) -> float:
        """Compute shortest distance between two axes."""
        p1, d1 = entity_a.position, entity_a.direction
        p2, d2 = entity_b.position, entity_b.direction

        n = np.cross(d1, d2)
        if np.linalg.norm(n) < 1e-10:
            # Parallel axes
            return np.linalg.norm(np.cross(p2 - p1, d1)) / np.linalg.norm(d1)

        # Skew axes
        return abs(np.dot(p2 - p1, n)) / np.linalg.norm(n)

    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def _build_state_vector(self) -> np.ndarray:
        """Build state vector from entity positions."""
        state = []
        for entity in self._entities.values():
            state.extend(entity.position.tolist())
            # Add rotation (Euler angles) if entity has direction
            if entity.direction is not None:
                state.extend([0.0, 0.0, 0.0])  # Initial rotation
        return np.array(state)

    def _apply_state_vector(self, x: np.ndarray) -> None:
        """Apply state vector to entities."""
        idx = 0
        for entity in self._entities.values():
            entity.position = np.array(x[idx:idx+3])
            idx += 3
            if entity.direction is not None:
                # Apply rotation (simplified)
                idx += 3

    def _entity_to_transform(self, entity: GeometricEntity) -> np.ndarray:
        """Convert entity state to 4x4 transform matrix."""
        T = np.eye(4)
        T[:3, 3] = entity.position
        return T

    def _calculate_total_dof(self) -> int:
        """Calculate total degrees of freedom."""
        # Each entity: 6 DOF (3 translation + 3 rotation)
        return len(self._entities) * 6

    def _calculate_remaining_dof(self) -> int:
        """Calculate remaining DOF after constraints."""
        total_dof = self._calculate_total_dof()
        constrained_dof = 0

        for c in self._constraints:
            if c.constraint_type == ConstraintType.COINCIDENT:
                constrained_dof += 3
            elif c.constraint_type == ConstraintType.CONCENTRIC:
                constrained_dof += 4  # 2 translation + 2 rotation
            elif c.constraint_type == ConstraintType.PARALLEL:
                constrained_dof += 2
            elif c.constraint_type == ConstraintType.DISTANCE:
                constrained_dof += 1
            elif c.constraint_type == ConstraintType.ANGLE:
                constrained_dof += 1
            elif c.constraint_type == ConstraintType.FIXED:
                constrained_dof += 6

        return max(0, total_dof - constrained_dof)
```

#### 4.3.3 PhysicalPropertiesCalculator

**Location**: `src/core/physics/mass_properties.py` (NEW)

```python
"""
Physical properties calculator for assemblies.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False


@dataclass
class MaterialProperties:
    """Material physical properties."""
    name: str
    density: float  # kg/m^3
    youngs_modulus: float = 0.0  # Pa (optional)
    poissons_ratio: float = 0.0  # dimensionless (optional)
    yield_strength: float = 0.0  # Pa (optional)


# Common materials database
MATERIALS_DB: Dict[str, MaterialProperties] = {
    "steel": MaterialProperties("Steel", 7850, 200e9, 0.3, 250e6),
    "aluminum": MaterialProperties("Aluminum", 2700, 70e9, 0.33, 270e6),
    "titanium": MaterialProperties("Titanium", 4500, 110e9, 0.34, 900e6),
    "copper": MaterialProperties("Copper", 8960, 120e9, 0.34, 210e6),
    "brass": MaterialProperties("Brass", 8500, 100e9, 0.34, 200e6),
    "abs_plastic": MaterialProperties("ABS Plastic", 1050, 2.3e9, 0.35, 40e6),
    "pla_plastic": MaterialProperties("PLA Plastic", 1250, 3.5e9, 0.36, 60e6),
    "nylon": MaterialProperties("Nylon", 1150, 2.7e9, 0.4, 75e6),
}


@dataclass
class MassProperties:
    """Mass properties result for a part or assembly."""
    mass: float                           # kg
    volume: float                         # m^3
    center_of_mass: np.ndarray            # [x, y, z] in meters
    inertia_tensor: np.ndarray            # 3x3 matrix in kg*m^2
    principal_moments: np.ndarray         # [Ixx, Iyy, Izz] principal moments
    principal_axes: np.ndarray            # 3x3 rotation matrix to principal axes
    bounding_box_volume: float            # m^3
    surface_area: float                   # m^2


class PhysicalPropertiesCalculator:
    """
    Calculate mass properties for meshes and assemblies.

    Essential for:
    - Robot arm load estimation
    - Gripper force calculation
    - Motion planning inertia compensation
    """

    def __init__(self):
        self._parts: Dict[str, Dict[str, Any]] = {}

    def add_part(
        self,
        part_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        material: str = "steel",
        transform: Optional[np.ndarray] = None,
    ) -> None:
        """Add a part with its mesh and material."""
        if not _TRIMESH_AVAILABLE:
            logger.warning("trimesh not available; mass properties disabled")
            return

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if transform is not None:
            mesh.apply_transform(transform)

        mat = MATERIALS_DB.get(material.lower(), MATERIALS_DB["steel"])

        self._parts[part_id] = {
            "mesh": mesh,
            "material": mat,
            "transform": transform or np.eye(4),
        }

    def calculate_part_properties(self, part_id: str) -> MassProperties:
        """Calculate mass properties for a single part."""
        if not _TRIMESH_AVAILABLE or part_id not in self._parts:
            return self._empty_properties()

        part = self._parts[part_id]
        mesh = part["mesh"]
        material = part["material"]

        # Ensure mesh is watertight for accurate volume
        if not mesh.is_watertight:
            logger.warning(f"Mesh {part_id} is not watertight; results may be inaccurate")
            mesh = mesh.convex_hull

        # Volume in m^3 (assuming mesh is in mm, convert)
        volume_mm3 = abs(mesh.volume)
        volume_m3 = volume_mm3 * 1e-9  # mm^3 to m^3

        # Mass
        mass = volume_m3 * material.density

        # Center of mass (convert mm to m)
        com = mesh.center_mass * 1e-3

        # Inertia tensor (needs scaling)
        # trimesh returns inertia in mesh units^5 * density
        # We need to convert and apply material density
        inertia = mesh.moment_inertia * material.density * (1e-3)**5

        # Principal moments and axes
        eigenvalues, eigenvectors = np.linalg.eig(inertia)
        sorted_indices = np.argsort(eigenvalues)
        principal_moments = eigenvalues[sorted_indices]
        principal_axes = eigenvectors[:, sorted_indices]

        return MassProperties(
            mass=mass,
            volume=volume_m3,
            center_of_mass=com,
            inertia_tensor=inertia,
            principal_moments=principal_moments,
            principal_axes=principal_axes,
            bounding_box_volume=mesh.bounding_box.volume * 1e-9,
            surface_area=mesh.area * 1e-6,  # mm^2 to m^2
        )

    def calculate_assembly_properties(
        self,
        part_ids: Optional[List[str]] = None,
    ) -> MassProperties:
        """Calculate combined mass properties for an assembly."""
        if part_ids is None:
            part_ids = list(self._parts.keys())

        if not part_ids:
            return self._empty_properties()

        # Calculate individual properties
        part_props = [self.calculate_part_properties(pid) for pid in part_ids]

        # Total mass
        total_mass = sum(p.mass for p in part_props)

        if total_mass < 1e-10:
            return self._empty_properties()

        # Combined center of mass (weighted average)
        combined_com = np.zeros(3)
        for p in part_props:
            combined_com += p.mass * p.center_of_mass
        combined_com /= total_mass

        # Combined inertia tensor (parallel axis theorem)
        combined_inertia = np.zeros((3, 3))
        for p in part_props:
            # Inertia about part's own CoM
            I_part = p.inertia_tensor

            # Offset from assembly CoM
            r = p.center_of_mass - combined_com

            # Parallel axis theorem
            I_offset = p.mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

            combined_inertia += I_part + I_offset

        # Principal moments and axes
        eigenvalues, eigenvectors = np.linalg.eig(combined_inertia)
        sorted_indices = np.argsort(eigenvalues)
        principal_moments = eigenvalues[sorted_indices]
        principal_axes = eigenvectors[:, sorted_indices]

        # Totals
        total_volume = sum(p.volume for p in part_props)
        total_surface = sum(p.surface_area for p in part_props)
        total_bbox = sum(p.bounding_box_volume for p in part_props)

        return MassProperties(
            mass=total_mass,
            volume=total_volume,
            center_of_mass=combined_com,
            inertia_tensor=combined_inertia,
            principal_moments=principal_moments,
            principal_axes=principal_axes,
            bounding_box_volume=total_bbox,
            surface_area=total_surface,
        )

    def _empty_properties(self) -> MassProperties:
        """Return empty/zero mass properties."""
        return MassProperties(
            mass=0.0,
            volume=0.0,
            center_of_mass=np.zeros(3),
            inertia_tensor=np.zeros((3, 3)),
            principal_moments=np.zeros(3),
            principal_axes=np.eye(3),
            bounding_box_volume=0.0,
            surface_area=0.0,
        )

    def estimate_gripper_force(
        self,
        assembly_props: MassProperties,
        friction_coefficient: float = 0.3,
        safety_factor: float = 2.0,
        acceleration: float = 9.81,  # m/s^2 (gravity default)
    ) -> float:
        """
        Estimate minimum gripper force for lifting an assembly.

        F_grip = (m * a * safety_factor) / (2 * friction)

        Returns force in Newtons.
        """
        weight_force = assembly_props.mass * acceleration
        required_grip = (weight_force * safety_factor) / (2 * friction_coefficient)
        return required_grip

    def check_robot_payload(
        self,
        assembly_props: MassProperties,
        robot_payload_kg: float,
        robot_reach_m: float,
    ) -> Dict[str, Any]:
        """
        Check if assembly is within robot arm payload limits.

        Returns feasibility assessment.
        """
        mass_ok = assembly_props.mass <= robot_payload_kg

        # Check if CoM is within reach
        com_distance = np.linalg.norm(assembly_props.center_of_mass)
        reach_ok = com_distance <= robot_reach_m

        # Torque check (simplified)
        max_torque = assembly_props.mass * 9.81 * com_distance

        return {
            "feasible": mass_ok and reach_ok,
            "mass_kg": assembly_props.mass,
            "mass_limit_kg": robot_payload_kg,
            "mass_ok": mass_ok,
            "com_distance_m": com_distance,
            "reach_limit_m": robot_reach_m,
            "reach_ok": reach_ok,
            "estimated_torque_nm": max_torque,
        }
```

---

## 5. Workstream 3: Edge-Ready Architecture

### 5.1 Objectives

Prepare ML models for deployment on resource-constrained edge devices:

- **ONNX export** for cross-platform inference
- **INT8 quantization** for reduced model size
- **Edge SDK** with offline capability

### 5.2 Planned Implementation

#### 5.2.1 ONNX Export Pipeline

**Location**: `src/ml/export/onnx_exporter.py` (NEW)

```python
"""
ONNX model export pipeline for edge deployment.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import onnx
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    logger.warning("ONNX dependencies not available")


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""
    opset_version: int = 17
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    input_names: List[str] = None
    output_names: List[str] = None
    optimize: bool = True
    quantize: bool = False
    quantization_mode: str = "dynamic"  # dynamic | static


@dataclass
class ExportResult:
    """Result of ONNX export."""
    success: bool
    output_path: Path
    original_size_mb: float
    exported_size_mb: float
    quantized_size_mb: Optional[float]
    inference_time_ms: float
    accuracy_delta: Optional[float]  # Difference from original
    warnings: List[str]


class OnnxExporter:
    """
    Exports PyTorch models to ONNX format with optional quantization.

    Supports:
    - MetricMLP (feature embedding)
    - PointNet++ (3D classification)
    - Classifier heads
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        self._validation_inputs: Dict[str, np.ndarray] = {}

    def export_model(
        self,
        model: "torch.nn.Module",
        output_path: Path,
        sample_input: "torch.Tensor",
        model_name: str = "model",
    ) -> ExportResult:
        """Export a PyTorch model to ONNX."""
        if not _ONNX_AVAILABLE:
            return ExportResult(
                success=False,
                output_path=output_path,
                original_size_mb=0,
                exported_size_mb=0,
                quantized_size_mb=None,
                inference_time_ms=0,
                accuracy_delta=None,
                warnings=["ONNX dependencies not installed"],
            )

        warnings = []

        # Get original model size
        original_size = self._get_model_size(model)

        # Set model to eval mode
        model.eval()

        # Export to ONNX
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_names = self.config.input_names or ["input"]
        output_names = self.config.output_names or ["output"]

        dynamic_axes = self.config.dynamic_axes or {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

        try:
            torch.onnx.export(
                model,
                sample_input,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=self.config.opset_version,
                do_constant_folding=True,
            )
        except Exception as e:
            return ExportResult(
                success=False,
                output_path=output_path,
                original_size_mb=original_size,
                exported_size_mb=0,
                quantized_size_mb=None,
                inference_time_ms=0,
                accuracy_delta=None,
                warnings=[f"Export failed: {str(e)}"],
            )

        # Verify exported model
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            warnings.append(f"Model verification warning: {str(e)}")

        exported_size = output_path.stat().st_size / (1024 * 1024)

        # Optimize if requested
        if self.config.optimize:
            optimized_path = output_path.with_suffix(".optimized.onnx")
            self._optimize_model(output_path, optimized_path)
            if optimized_path.exists():
                output_path = optimized_path
                exported_size = output_path.stat().st_size / (1024 * 1024)

        # Quantize if requested
        quantized_size = None
        if self.config.quantize:
            quantized_path = output_path.with_suffix(".int8.onnx")
            quant_success = self._quantize_model(output_path, quantized_path)
            if quant_success:
                quantized_size = quantized_path.stat().st_size / (1024 * 1024)

        # Benchmark inference
        inference_time = self._benchmark_inference(output_path, sample_input.numpy())

        # Calculate accuracy delta if we have validation inputs
        accuracy_delta = None
        if self._validation_inputs:
            accuracy_delta = self._calculate_accuracy_delta(
                model, output_path, sample_input
            )

        return ExportResult(
            success=True,
            output_path=output_path,
            original_size_mb=original_size,
            exported_size_mb=exported_size,
            quantized_size_mb=quantized_size,
            inference_time_ms=inference_time,
            accuracy_delta=accuracy_delta,
            warnings=warnings,
        )

    def _get_model_size(self, model: "torch.nn.Module") -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def _optimize_model(self, input_path: Path, output_path: Path) -> None:
        """Optimize ONNX model using onnxruntime."""
        try:
            from onnxruntime.transformers import optimizer
            optimized = optimizer.optimize_model(
                str(input_path),
                model_type="bert",  # Generic optimization
                num_heads=0,
                hidden_size=0,
            )
            optimized.save_model_to_file(str(output_path))
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")

    def _quantize_model(self, input_path: Path, output_path: Path) -> bool:
        """Quantize ONNX model to INT8."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(
                model_input=str(input_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
            )
            return output_path.exists()
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return False

    def _benchmark_inference(
        self,
        model_path: Path,
        sample_input: np.ndarray,
        num_runs: int = 100,
    ) -> float:
        """Benchmark ONNX inference time in milliseconds."""
        try:
            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name

            # Warmup
            for _ in range(10):
                session.run(None, {input_name: sample_input})

            # Benchmark
            import time
            start = time.perf_counter()
            for _ in range(num_runs):
                session.run(None, {input_name: sample_input})
            elapsed = time.perf_counter() - start

            return (elapsed / num_runs) * 1000  # Convert to ms
        except Exception as e:
            logger.warning(f"Benchmark failed: {e}")
            return 0.0

    def _calculate_accuracy_delta(
        self,
        original_model: "torch.nn.Module",
        onnx_path: Path,
        sample_input: "torch.Tensor",
    ) -> float:
        """Calculate accuracy difference between original and ONNX model."""
        try:
            # Original inference
            with torch.no_grad():
                original_output = original_model(sample_input).numpy()

            # ONNX inference
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            onnx_output = session.run(None, {input_name: sample_input.numpy()})[0]

            # Calculate mean absolute difference
            return float(np.mean(np.abs(original_output - onnx_output)))
        except Exception as e:
            logger.warning(f"Accuracy comparison failed: {e}")
            return None
```

#### 5.2.2 Edge Client SDK

**Location**: `clients/edge_sdk/` (NEW PACKAGE)

```
clients/edge_sdk/
├── pyproject.toml
├── README.md
├── src/
│   └── cad_ml_edge/
│       ├── __init__.py
│       ├── client.py
│       ├── inference.py
│       ├── sync.py
│       └── models/
│           └── __init__.py
└── tests/
    └── test_client.py
```

**`clients/edge_sdk/src/cad_ml_edge/client.py`**:

```python
"""
CAD-ML Edge Client - Lightweight inference and sync SDK.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EdgeConfig:
    """Edge client configuration."""
    server_url: str = "http://localhost:8000"
    model_dir: Path = Path("./models")
    cache_dir: Path = Path("./cache")
    offline_db: Path = Path("./offline.db")
    sync_interval_sec: int = 60
    max_offline_queue: int = 1000


@dataclass
class InferenceResult:
    """Result of local inference."""
    part_type: str
    confidence: float
    features: Dict[str, float]
    latency_ms: float
    model_version: str
    offline: bool = False


class CadMlEdgeClient:
    """
    Lightweight edge client for CAD-ML Platform.

    Features:
    - Local ONNX inference
    - Store-and-forward for offline operation
    - Automatic sync when online
    - Model version management
    """

    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()
        self._inference_engine: Optional["OnnxInferenceEngine"] = None
        self._offline_queue: List[Dict[str, Any]] = []
        self._is_online = False
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize local storage."""
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite for offline queue
        conn = sqlite3.connect(str(self.config.offline_db))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS offline_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                request_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                synced INTEGER DEFAULT 0
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                model_name TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                downloaded_at REAL NOT NULL,
                file_path TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    async def initialize(self) -> None:
        """Initialize client and check server connectivity."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.config.server_url}/api/v1/health",
                    timeout=5.0,
                )
                self._is_online = resp.status_code == 200
        except Exception:
            self._is_online = False
            logger.info("Server unreachable; operating in offline mode")

        # Load local models
        self._load_models()

    def _load_models(self) -> None:
        """Load ONNX models for local inference."""
        from .inference import OnnxInferenceEngine

        model_path = self.config.model_dir / "classifier.onnx"
        if model_path.exists():
            self._inference_engine = OnnxInferenceEngine(model_path)
            logger.info(f"Loaded inference model: {model_path}")
        else:
            logger.warning("No local model found; inference disabled")

    async def classify(
        self,
        features: Dict[str, float],
        prefer_server: bool = True,
    ) -> InferenceResult:
        """
        Classify a CAD part.

        Args:
            features: Extracted feature dict
            prefer_server: If True and online, use server inference

        Returns:
            Classification result
        """
        start_time = time.perf_counter()

        # Try server inference if online and preferred
        if prefer_server and self._is_online:
            try:
                result = await self._server_classify(features)
                result.latency_ms = (time.perf_counter() - start_time) * 1000
                return result
            except Exception as e:
                logger.warning(f"Server inference failed: {e}")
                self._is_online = False

        # Fall back to local inference
        if self._inference_engine:
            result = self._local_classify(features)
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            result.offline = True
            return result

        # No inference available
        return InferenceResult(
            part_type="unknown",
            confidence=0.0,
            features=features,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            model_version="none",
            offline=True,
        )

    async def _server_classify(self, features: Dict[str, float]) -> InferenceResult:
        """Send classification request to server."""
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.config.server_url}/api/v1/inference/classify",
                json={"features": features},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

            return InferenceResult(
                part_type=data["part_type"],
                confidence=data["confidence"],
                features=features,
                latency_ms=0,  # Will be set by caller
                model_version=data.get("model_version", "server"),
            )

    def _local_classify(self, features: Dict[str, float]) -> InferenceResult:
        """Run local ONNX inference."""
        if not self._inference_engine:
            raise RuntimeError("No local model loaded")

        result = self._inference_engine.classify(features)
        return InferenceResult(
            part_type=result["part_type"],
            confidence=result["confidence"],
            features=features,
            latency_ms=0,
            model_version=self._inference_engine.model_version,
            offline=True,
        )

    async def submit_telemetry(
        self,
        device_id: str,
        sensors: Dict[str, float],
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """
        Submit telemetry data (store-and-forward).

        If offline, queues for later sync.
        """
        payload = {
            "timestamp": time.time(),
            "device_id": device_id,
            "sensors": sensors,
            "metrics": metrics or {},
        }

        if self._is_online:
            try:
                return await self._send_telemetry(payload)
            except Exception as e:
                logger.warning(f"Telemetry send failed: {e}")
                self._is_online = False

        # Queue for offline sync
        self._queue_offline(request_type="telemetry", payload=payload)
        return {"status": "queued", "offline": True}

    async def _send_telemetry(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Send telemetry to server."""
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.config.server_url}/api/v1/twin/{payload['device_id']}/telemetry",
                json=payload,
                timeout=5.0,
            )
            resp.raise_for_status()
            return resp.json()

    def _queue_offline(self, request_type: str, payload: Dict[str, Any]) -> None:
        """Queue request for offline sync."""
        conn = sqlite3.connect(str(self.config.offline_db))
        conn.execute(
            "INSERT INTO offline_queue (timestamp, request_type, payload) VALUES (?, ?, ?)",
            (time.time(), request_type, json.dumps(payload)),
        )
        conn.commit()

        # Check queue size limit
        count = conn.execute("SELECT COUNT(*) FROM offline_queue WHERE synced = 0").fetchone()[0]
        if count > self.config.max_offline_queue:
            # Remove oldest entries
            conn.execute('''
                DELETE FROM offline_queue
                WHERE id IN (
                    SELECT id FROM offline_queue
                    WHERE synced = 0
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
            ''', (count - self.config.max_offline_queue,))
            conn.commit()

        conn.close()

    async def sync(self) -> Dict[str, Any]:
        """
        Sync offline queue with server.

        Returns sync statistics.
        """
        if not self._is_online:
            # Check connectivity first
            await self.initialize()

        if not self._is_online:
            return {"status": "offline", "synced": 0, "failed": 0}

        conn = sqlite3.connect(str(self.config.offline_db))
        rows = conn.execute(
            "SELECT id, request_type, payload FROM offline_queue WHERE synced = 0 ORDER BY timestamp ASC"
        ).fetchall()

        synced = 0
        failed = 0

        for row_id, req_type, payload_json in rows:
            payload = json.loads(payload_json)
            try:
                if req_type == "telemetry":
                    await self._send_telemetry(payload)

                # Mark as synced
                conn.execute("UPDATE offline_queue SET synced = 1 WHERE id = ?", (row_id,))
                conn.commit()
                synced += 1
            except Exception as e:
                logger.warning(f"Sync failed for {req_type}: {e}")
                failed += 1

        conn.close()
        return {"status": "ok", "synced": synced, "failed": failed}

    async def download_model(self, model_name: str = "classifier") -> bool:
        """Download latest model from server."""
        if not self._is_online:
            return False

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                # Get model info
                info_resp = await client.get(
                    f"{self.config.server_url}/api/v1/model/{model_name}/info",
                    timeout=10.0,
                )
                info_resp.raise_for_status()
                info = info_resp.json()

                # Download model
                model_resp = await client.get(
                    f"{self.config.server_url}/api/v1/model/{model_name}/download",
                    timeout=60.0,
                )
                model_resp.raise_for_status()

                # Save model
                model_path = self.config.model_dir / f"{model_name}.onnx"
                model_path.write_bytes(model_resp.content)

                # Update version tracking
                conn = sqlite3.connect(str(self.config.offline_db))
                conn.execute('''
                    INSERT OR REPLACE INTO model_versions
                    (model_name, version, downloaded_at, file_path)
                    VALUES (?, ?, ?, ?)
                ''', (model_name, info.get("version", "unknown"), time.time(), str(model_path)))
                conn.commit()
                conn.close()

                # Reload models
                self._load_models()
                return True

        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False
```

---

## 6. Data Models & Interfaces

### 6.1 Core Data Models

```python
# src/models/telemetry.py

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class TelemetryFrame(BaseModel):
    """Canonical telemetry envelope."""
    timestamp: float = Field(..., description="Unix timestamp seconds")
    device_id: str = Field(..., description="Source device identifier")
    sensors: Dict[str, float] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    status: Dict[str, Any] = Field(default_factory=dict)


class TelemetryQuery(BaseModel):
    """Query parameters for telemetry history."""
    device_id: str
    start: Optional[datetime] = None
    stop: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=10000)
    fields: Optional[List[str]] = None


class AggregateQuery(BaseModel):
    """Query parameters for aggregated telemetry."""
    device_id: str
    field: str
    window: str = "5m"  # InfluxDB duration format
    fn: str = "mean"    # mean, sum, min, max, count
    start: Optional[datetime] = None
    stop: Optional[datetime] = None
```

### 6.2 Physics Models

```python
# src/models/physics.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np


class Vector3(BaseModel):
    """3D vector."""
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class Matrix3x3(BaseModel):
    """3x3 matrix (e.g., inertia tensor)."""
    values: List[List[float]] = Field(..., min_length=3, max_length=3)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.values)


class MassPropertiesResponse(BaseModel):
    """API response for mass properties."""
    mass_kg: float
    volume_m3: float
    center_of_mass: Vector3
    inertia_tensor: Matrix3x3
    principal_moments: List[float]
    surface_area_m2: float


class CollisionCheckRequest(BaseModel):
    """Request for collision check."""
    assembly_id: str
    part_ids: Optional[List[str]] = None  # None = check all


class CollisionCheckResponse(BaseModel):
    """Response for collision check."""
    has_collisions: bool
    collision_count: int
    collisions: List[Dict[str, Any]]
    check_time_ms: float


class ConstraintValidationRequest(BaseModel):
    """Request for constraint validation."""
    assembly_id: str
    constraints: List[Dict[str, Any]]


class ConstraintValidationResponse(BaseModel):
    """Response for constraint validation."""
    is_valid: bool
    total_error: float
    violated_constraints: List[str]
    dof_remaining: int
```

---

## 7. API Specifications

### 7.1 Digital Twin API

**Base Path**: `/api/v1/twin`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/history` | Query telemetry history |
| GET | `/history/aggregate` | Query aggregated telemetry |
| POST | `/{asset_id}/telemetry` | Ingest telemetry (HTTP fallback) |
| GET | `/{asset_id}/state` | Get current state snapshot |
| WS | `/ws/{asset_id}` | Real-time WebSocket updates |

**Example: Query History**

```http
GET /api/v1/twin/history?device_id=robot-arm-01&limit=100&start=2025-01-01T00:00:00Z
Authorization: Bearer <api_key>

Response:
{
  "device_id": "robot-arm-01",
  "count": 100,
  "frames": [
    {
      "timestamp": 1735689600.123,
      "device_id": "robot-arm-01",
      "sensors": {"temperature": 45.2, "current": 2.5},
      "metrics": {"efficiency": 0.92},
      "status": {"mode": "running"}
    },
    ...
  ]
}
```

### 7.2 Physics API

**Base Path**: `/api/v1/physics`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/collision/check` | Check assembly collisions |
| POST | `/collision/clearance` | Calculate minimum clearance |
| POST | `/constraints/validate` | Validate geometric constraints |
| POST | `/mass-properties` | Calculate mass properties |
| POST | `/robot-check` | Check robot payload feasibility |

**Example: Collision Check**

```http
POST /api/v1/physics/collision/check
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "assembly_id": "asm-001",
  "part_ids": ["part-a", "part-b", "part-c"]
}

Response:
{
  "has_collisions": true,
  "collision_count": 1,
  "collisions": [
    {
      "part_a": "part-a",
      "part_b": "part-b",
      "volume_estimate_mm3": 12.5,
      "contact_points": [[10.0, 20.0, 5.0]],
      "severity": "minor"
    }
  ],
  "check_time_ms": 45.2
}
```

### 7.3 Model API

**Base Path**: `/api/v1/model`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/{name}/info` | Get model metadata |
| GET | `/{name}/download` | Download ONNX model |
| POST | `/{name}/export` | Trigger ONNX export |
| GET | `/versions` | List all model versions |

---

## 8. Integration Points

### 8.1 Phase 17 Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 17 INTEGRATION MATRIX                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 15-16 Component        Phase 17 Consumer        Integration Point   │
│  ─────────────────────        ─────────────────        ─────────────────   │
│                                                                             │
│  TelemetryIngestor    ────▶   Predictive Maint.   ────▶  Sensor streams    │
│  TimeSeriesStore      ────▶   Anomaly Detection   ────▶  Historical data   │
│  MqttTelemetryClient  ────▶   Robot Controller    ────▶  Real-time cmds    │
│                                                                             │
│  CollisionManager     ────▶   Path Planner        ────▶  Obstacle avoidance│
│  ConstraintSolver     ────▶   Assembly Sequence   ────▶  Valid configs     │
│  MassProperties       ────▶   Gripper Control     ────▶  Force calculation │
│                                                                             │
│  OnnxExporter         ────▶   Visual Inspection   ────▶  Edge TPU models   │
│  EdgeClient           ────▶   Edge Gateway        ────▶  Local inference   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 External System Integration

| System | Protocol | Purpose |
|--------|----------|---------|
| PLC/SCADA | MQTT | Real-time sensor data |
| Robot Controller | ROS2 / gRPC | Motion commands |
| MES | REST API | Production orders |
| ERP | Webhook | Inventory updates |
| Edge Gateway | ONNX Runtime | Local inference |

---

## 9. Success Metrics & Validation

### 9.1 Workstream 1: Digital Twin

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| MQTT Throughput | 10,000 msg/sec | Load test with `mosquitto_pub` |
| End-to-end Latency | < 100ms (P95) | Prometheus histogram |
| Storage Write Rate | 5,000 points/sec | InfluxDB metrics |
| Query Response Time | < 500ms (1M points) | API benchmark |
| Backpressure Drop Rate | < 1% | Ingestor metrics |

### 9.2 Workstream 2: Physics Engine

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Collision Detection Accuracy | > 99% | CAD test suite |
| Collision Check Latency | < 50ms (100 parts) | Benchmark |
| Constraint Solver Convergence | > 95% | Solver metrics |
| Mass Property Accuracy | < 1% error | CAD software comparison |
| API Response Time | < 200ms | Load test |

### 9.3 Workstream 3: Edge

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| ONNX Export Success Rate | 100% | CI pipeline |
| Model Size Reduction (INT8) | > 50% | File size comparison |
| Inference Accuracy Loss | < 2% | Validation set |
| Edge Inference Latency | < 20ms | Raspberry Pi 4 |
| Offline Queue Durability | 100% | Crash recovery test |

### 9.4 Test Plan

```yaml
test_suites:
  unit_tests:
    - test_telemetry_frame_serialization
    - test_mqtt_client_reconnection
    - test_ingestor_backpressure
    - test_timeseries_store_history
    - test_collision_manager_basic
    - test_constraint_solver_convergence
    - test_mass_properties_simple_shapes
    - test_onnx_export_basic
    - test_edge_client_offline_queue

  integration_tests:
    - test_mqtt_to_influxdb_pipeline
    - test_physics_api_endpoints
    - test_model_download_and_inference
    - test_edge_sync_roundtrip

  performance_tests:
    - benchmark_mqtt_throughput
    - benchmark_collision_detection_scaling
    - benchmark_onnx_inference_latency

  e2e_tests:
    - test_full_telemetry_flow
    - test_assembly_validation_workflow
    - test_edge_offline_online_transition
```

---

## 10. Risk Analysis & Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| InfluxDB performance bottleneck | Medium | High | Pre-size retention policies; use continuous queries |
| trimesh collision accuracy | Low | Medium | Validate against CAD software; fallback to FCL |
| ONNX compatibility issues | Medium | Medium | Pin ONNX opset version; extensive testing |
| Edge device resource limits | High | Medium | Profile memory/CPU; implement model pruning |
| MQTT message loss | Low | High | QoS 2 for critical; implement acknowledgments |

### 10.2 Dependency Risks

| Dependency | Risk | Mitigation |
|------------|------|------------|
| aiomqtt | API changes | Pin version; abstract client interface |
| trimesh | Performance | Benchmark; consider FCL fallback |
| onnxruntime | Platform support | Test on all target platforms |
| influxdb-client | Breaking changes | Pin major version; use Protocol abstraction |

### 10.3 Schedule Risks

| Phase | Risk | Mitigation |
|-------|------|------------|
| Weeks 1-6 | MQTT infrastructure delay | Start with Mosquitto; defer EMQX HA |
| Weeks 7-12 | Physics engine complexity | Prioritize collision > constraints > mass |
| Weeks 13-16 | Edge SDK scope creep | MVP first; defer advanced sync features |

---

## Appendix A: File Structure

```
src/
├── core/
│   ├── twin/
│   │   ├── connectivity.py    [DONE]
│   │   ├── ingest.py          [DONE]
│   │   └── sync.py            [EXISTING]
│   ├── storage/
│   │   ├── timeseries.py      [DONE]
│   │   ├── influxdb_adapter.py  [NEW]
│   │   └── timescale_adapter.py [NEW]
│   ├── physics/               [NEW DIR]
│   │   ├── __init__.py
│   │   ├── collision.py
│   │   ├── constraints.py
│   │   └── mass_properties.py
│   └── config.py              [UPDATED]
├── ml/
│   └── export/                [NEW DIR]
│       ├── __init__.py
│       └── onnx_exporter.py
├── api/v1/
│   ├── twin.py                [UPDATED]
│   └── physics.py             [NEW]
├── models/
│   ├── telemetry.py           [NEW]
│   └── physics.py             [NEW]
│
clients/
└── edge_sdk/                  [NEW PACKAGE]
    ├── pyproject.toml
    ├── src/cad_ml_edge/
    │   ├── __init__.py
    │   ├── client.py
    │   ├── inference.py
    │   └── sync.py
    └── tests/
```

---

## Appendix B: Configuration Reference

```yaml
# .env.production

# MQTT / Digital Twin
TELEMETRY_MQTT_ENABLED=true
MQTT_HOST=mosquitto
MQTT_PORT=1883
MQTT_TLS_CA=/etc/ssl/certs/ca.crt
MQTT_TOPIC=twin/telemetry/#
MQTT_QOS=1

# Time-Series Storage
TELEMETRY_STORE_BACKEND=influx
INFLUX_URL=http://influxdb:8086
INFLUX_TOKEN=<secret>
INFLUX_ORG=cad-ml
INFLUX_BUCKET=telemetry

# Alternative: TimescaleDB
# TELEMETRY_STORE_BACKEND=timescale
# TIMESCALE_DSN=postgresql://user:pass@timescale:5432/telemetry

# Physics Engine
PHYSICS_COLLISION_BACKEND=trimesh
PHYSICS_CONSTRAINT_TOLERANCE=0.001

# Edge/Model Export
ONNX_OPSET_VERSION=17
ONNX_QUANTIZE_DEFAULT=true
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-07 | CAD-ML Team | Initial draft |

---

**End of Document**
