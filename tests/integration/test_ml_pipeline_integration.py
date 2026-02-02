"""
Integration tests for ML Pipeline and CAD Processing.

Tests cover:
- End-to-end ML pipeline workflow
- CAD geometry analysis integration
- Model monitoring and alerting
- Multi-source fusion with confidence calibration
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# ML Pipeline imports
from src.ml.pipeline import (
    PipelineStage,
    StageStatus,
    StageResult,
    Pipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStatus,
    PipelineBuilder,
)

# ML Hybrid imports
from src.ml.hybrid import (
    FusionStrategy,
    SourcePrediction,
    FusionResult,
    MultiSourceFusion,
)
from src.ml.hybrid.calibration import (
    CalibrationMethod,
    ConfidenceCalibrator,
)
from src.ml.hybrid.explainer import (
    HybridExplainer,
    Explanation,
)

# ML Monitoring imports
from src.ml.monitoring import MetricsCollector
from src.ml.monitoring.drift import (
    DriftResult,
    DriftType,
    KSTestDetector,
    PSIDetector,
    DriftMonitor,
)
from src.ml.monitoring.alerts import (
    AlertSeverity,
    AlertManager,
)

# CAD Geometry imports
from src.core.cad.geometry import (
    GeometryExtractor,
    BoundingBox,
)
from src.core.cad.geometry.topology import (
    TopologyGraph,
    TopologyAnalyzer,
    TopologicalNode,
    ConnectionType,
)
from src.core.cad.geometry.spatial import (
    GridIndex,
    SpatialQuery,
)


class TestMLPipelineEndToEnd:
    """End-to-end tests for ML pipeline workflow."""

    @pytest.fixture
    def sample_cad_data(self) -> Dict[str, Any]:
        """Create sample CAD data for testing."""
        return {
            "filename": "test_part.dxf",
            "entities": [
                {"type": "LINE", "start": (0, 0), "end": (100, 0)},
                {"type": "LINE", "start": (100, 0), "end": (100, 50)},
                {"type": "CIRCLE", "center": (50, 25), "radius": 10},
            ],
            "layers": ["0", "DIMENSIONS", "GEOMETRY"],
        }

    @pytest.fixture
    def pipeline_config(self) -> PipelineConfig:
        """Create pipeline configuration."""
        return PipelineConfig(
            name="test_pipeline",
            description="Test pipeline",
            stop_on_error=True,
            save_intermediate=False,
        )

    def test_pipeline_config_creation(self, pipeline_config: PipelineConfig):
        """Test creating pipeline configuration."""
        assert pipeline_config.name == "test_pipeline"
        assert pipeline_config.stop_on_error is True

    def test_pipeline_creation(self, pipeline_config: PipelineConfig):
        """Test creating a pipeline."""
        pipeline = Pipeline(config=pipeline_config)
        assert pipeline.config.name == "test_pipeline"
        assert pipeline.status == PipelineStatus.PENDING
        assert len(pipeline.stages) == 0

    def test_pipeline_builder_exists(self):
        """Test pipeline builder creation."""
        builder = PipelineBuilder()
        assert builder is not None

    def test_pipeline_status_enum(self):
        """Test pipeline status enumeration."""
        assert PipelineStatus.PENDING.value == "pending"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"

    def test_stage_status_enum(self):
        """Test stage status enumeration."""
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.RUNNING.value == "running"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.FAILED.value == "failed"

    def test_stage_result_creation(self):
        """Test creating stage result."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.COMPLETED,
            output={"data": "test"},
            execution_time=1.5,
        )
        assert result.stage_name == "test_stage"
        assert result.success is True
        assert result.execution_time == 1.5

    def test_stage_result_to_dict(self):
        """Test stage result serialization."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.COMPLETED,
            execution_time=1.234,
        )
        data = result.to_dict()
        assert data["stage_name"] == "test_stage"
        assert data["status"] == "completed"
        assert data["execution_time"] == 1.234


class TestMultiSourceFusionIntegration:
    """Integration tests for multi-source fusion with calibration."""

    @pytest.fixture
    def fusion_predictions(self) -> List[SourcePrediction]:
        """Create sample predictions from multiple sources."""
        return [
            SourcePrediction(source="filename", label="零件图", confidence=0.92),
            SourcePrediction(source="graph2d", label="零件图", confidence=0.78),
            SourcePrediction(source="titleblock", label="装配图", confidence=0.65),
            SourcePrediction(source="ocr", label="零件图", confidence=0.85),
        ]

    def test_fusion_strategy_enum(self):
        """Test fusion strategy enumeration."""
        assert FusionStrategy.WEIGHTED_AVERAGE is not None
        assert FusionStrategy.VOTING is not None

    def test_source_prediction_creation(self):
        """Test source prediction creation."""
        pred = SourcePrediction(source_name="test", label="零件图", confidence=0.9)
        assert pred.source_name == "test"
        assert pred.label == "零件图"
        assert pred.confidence == 0.9

    def test_multi_source_fusion_creation(self):
        """Test multi-source fusion creation."""
        fusion = MultiSourceFusion(default_strategy=FusionStrategy.WEIGHTED_AVERAGE)
        assert fusion is not None

    def test_calibrator_creation(self):
        """Test confidence calibrator creation."""
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE_SCALING)
        assert calibrator is not None

    def test_calibration_methods_enum(self):
        """Test calibration method enumeration."""
        assert CalibrationMethod.TEMPERATURE_SCALING is not None
        assert CalibrationMethod.PLATT_SCALING is not None

    def test_explainer_creation(self):
        """Test hybrid explainer creation."""
        explainer = HybridExplainer()
        assert explainer is not None


class TestCADGeometryIntegration:
    """Integration tests for CAD geometry analysis."""

    def test_geometry_extractor_creation(self):
        """Test geometry extractor creation."""
        extractor = GeometryExtractor()
        assert extractor is not None

    def test_bounding_box_operations(self):
        """Test bounding box calculations."""
        bbox = BoundingBox(min_x=0, min_y=0, max_x=100, max_y=50)

        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.center == (50, 25)
        assert bbox.area == 5000

    def test_bounding_box_contains(self):
        """Test bounding box containment check."""
        bbox = BoundingBox(min_x=0, min_y=0, max_x=100, max_y=50)

        assert bbox.contains(50, 25) is True
        assert bbox.contains(150, 25) is False

    def test_topology_graph_creation(self):
        """Test creating topology graph."""
        graph = TopologyGraph()
        assert graph is not None

    def test_topology_node_creation(self):
        """Test creating topology nodes with correct API."""
        node = TopologicalNode(entity_id="n1", entity_type="LINE", position=(0, 0))
        assert node.entity_id == "n1"
        assert node.entity_type == "LINE"

    def test_topology_graph_operations(self):
        """Test topology graph operations."""
        graph = TopologyGraph()

        # Add nodes with correct API - add_node takes entity_id and entity_type
        node1 = graph.add_node(entity_id="n1", entity_type="LINE", position=(0, 0))
        node2 = graph.add_node(entity_id="n2", entity_type="LINE", position=(100, 0))
        graph.add_edge(node1.entity_id, node2.entity_id, ConnectionType.ENDPOINT)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_spatial_index_creation(self):
        """Test creating spatial index."""
        index = GridIndex(cell_size=10.0)
        assert index is not None

    def test_topology_analyzer_creation(self):
        """Test topology analyzer creation."""
        analyzer = TopologyAnalyzer()
        assert analyzer is not None


class TestMonitoringIntegration:
    """Integration tests for model monitoring and alerting."""

    def test_metrics_collector_creation(self):
        """Test metrics collector creation."""
        collector = MetricsCollector()
        assert collector is not None

    def test_drift_types_enum(self):
        """Test drift type enumeration."""
        assert DriftType.DATA_DRIFT is not None
        assert DriftType.PREDICTION_DRIFT is not None

    def test_ks_detector_creation(self):
        """Test KS test detector creation."""
        detector = KSTestDetector(threshold=0.05)
        assert detector is not None

    def test_psi_detector_creation(self):
        """Test PSI detector creation."""
        detector = PSIDetector(threshold=0.2)
        assert detector is not None

    def test_drift_monitor_creation(self):
        """Test drift monitor creation."""
        monitor = DriftMonitor(window_size=100)
        assert monitor is not None

    def test_alert_severity_enum(self):
        """Test alert severity enumeration."""
        assert AlertSeverity.INFO is not None
        assert AlertSeverity.WARNING is not None
        assert AlertSeverity.CRITICAL is not None

    def test_alert_manager_creation(self):
        """Test alert manager creation."""
        manager = AlertManager()
        assert manager is not None

    def test_alert_manager_fire_alert(self):
        """Test firing an alert."""
        manager = AlertManager()
        alert = manager.fire_alert(
            name="test_alert",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            source="test",
        )
        assert alert is not None
        assert alert.name == "test_alert"


class TestEndToEndWorkflow:
    """End-to-end integration tests combining all components."""

    def test_pipeline_and_fusion_workflow(self):
        """Test pipeline creation with fusion."""
        # Create pipeline
        config = PipelineConfig(name="classification_pipeline")
        pipeline = Pipeline(config=config)
        assert pipeline is not None

        # Create fusion
        fusion = MultiSourceFusion(default_strategy=FusionStrategy.WEIGHTED_AVERAGE)
        predictions = [
            SourcePrediction("source1", "零件图", 0.88),
            SourcePrediction("source2", "零件图", 0.82),
        ]
        result = fusion.fuse(predictions)
        assert result.label == "零件图"

    def test_geometry_and_topology_workflow(self):
        """Test geometry and topology analysis workflow."""
        # Create geometry tools
        extractor = GeometryExtractor()
        bbox = BoundingBox(min_x=0, min_y=0, max_x=100, max_y=100)

        # Create topology
        graph = TopologyGraph()
        node = graph.add_node(entity_id="test", entity_type="LINE")

        assert bbox.area == 10000
        assert len(graph.nodes) == 1

    def test_monitoring_workflow(self):
        """Test monitoring workflow."""
        # Create monitoring components
        collector = MetricsCollector()
        monitor = DriftMonitor(window_size=50)
        alert_manager = AlertManager()

        # Fire a test alert
        alert = alert_manager.fire_alert(
            name="test",
            severity=AlertSeverity.INFO,
            message="Test",
            source="test",
        )

        assert collector is not None
        assert monitor is not None
        assert alert is not None

    def test_calibration_workflow(self):
        """Test calibration workflow."""
        # Create calibrator
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE_SCALING)

        # Train with sample data
        train_confidences = np.random.uniform(0.5, 1.0, 100)
        train_labels = (np.random.random(100) > 0.3).astype(int)
        calibrator.fit(train_confidences, train_labels)

        # Calibrate a confidence value
        cal_confidence = calibrator.calibrate(0.8, source="test")
        assert 0.0 <= cal_confidence <= 1.0


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
