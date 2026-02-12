"""
Tests for new modules: M6 Hybrid, C3 Geometry, I4 Monitoring.

Covers:
- M6: Multi-source fusion, calibration, explainability
- C3: Geometry features, topology, spatial indexing
- I4: Metrics, drift detection, alerts
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# M6: HybridClassifier Enhancement Tests
# ============================================================================

class TestM6Fusion:
    """Tests for multi-source fusion."""

    def test_fusion_imports(self):
        """Test fusion module imports."""
        from src.ml.hybrid import (
            FusionStrategy,
            SourcePrediction,
            FusionResult,
            MultiSourceFusion,
            WeightedAverageFusion,
            VotingFusion,
        )
        assert FusionStrategy is not None
        assert MultiSourceFusion is not None

    def test_source_prediction_creation(self):
        """Test SourcePrediction creation."""
        from src.ml.hybrid.fusion import SourcePrediction

        pred = SourcePrediction(
            source_name="filename",
            label="零件图",
            confidence=0.85,
        )
        assert pred.source_name == "filename"
        assert pred.is_valid is True

    def test_weighted_average_fusion(self):
        """Test weighted average fusion."""
        from src.ml.hybrid.fusion import WeightedAverageFusion, SourcePrediction

        fusion = WeightedAverageFusion()
        predictions = [
            SourcePrediction("source1", "A", 0.8),
            SourcePrediction("source2", "A", 0.7),
            SourcePrediction("source3", "B", 0.6),
        ]
        result = fusion.fuse(predictions)
        assert result.label == "A"
        assert result.agreement_score > 0.5

    def test_voting_fusion(self):
        """Test voting fusion."""
        from src.ml.hybrid.fusion import VotingFusion, SourcePrediction

        fusion = VotingFusion(voting_type="soft")
        predictions = [
            SourcePrediction("s1", "A", 0.9),
            SourcePrediction("s2", "B", 0.8),
            SourcePrediction("s3", "A", 0.7),
        ]
        result = fusion.fuse(predictions)
        assert result.label == "A"

    def test_multi_source_fusion(self):
        """Test MultiSourceFusion manager."""
        from src.ml.hybrid.fusion import MultiSourceFusion, FusionStrategy, SourcePrediction

        fusion = MultiSourceFusion(default_strategy=FusionStrategy.WEIGHTED_AVERAGE)
        predictions = [
            SourcePrediction("filename", "零件图", 0.9),
            SourcePrediction("graph2d", "零件图", 0.75),
        ]
        result = fusion.fuse(predictions)
        assert result.label == "零件图"
        assert result.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE


class TestM6Calibration:
    """Tests for confidence calibration."""

    def test_calibration_imports(self):
        """Test calibration module imports."""
        from src.ml.hybrid import (
            CalibrationMethod,
            ConfidenceCalibrator,
            PlattScaling,
            TemperatureScaling,
        )
        assert CalibrationMethod is not None
        assert ConfidenceCalibrator is not None

    def test_platt_scaling(self):
        """Test Platt scaling calibrator."""
        from src.ml.hybrid.calibration import PlattScaling

        calibrator = PlattScaling()
        # Fit on synthetic data
        confidences = np.array([0.3, 0.5, 0.7, 0.9, 0.95])
        labels = np.array([0, 0, 1, 1, 1])
        calibrator.fit(confidences, labels)

        # Calibrate
        calibrated = calibrator.calibrate(0.8)
        assert 0 <= calibrated <= 1

    def test_temperature_scaling(self):
        """Test temperature scaling calibrator."""
        from src.ml.hybrid.calibration import TemperatureScaling

        calibrator = TemperatureScaling()
        confidences = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([0, 1, 1, 1])
        calibrator.fit(confidences, labels)

        calibrated = calibrator.calibrate(0.85)
        assert 0 <= calibrated <= 1

    def test_confidence_calibrator(self):
        """Test main ConfidenceCalibrator."""
        from src.ml.hybrid.calibration import ConfidenceCalibrator, CalibrationMethod

        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE_SCALING)
        confidences = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        calibrator.fit(confidences, labels)

        metrics = calibrator.evaluate(confidences, labels)
        assert metrics.n_samples == 5
        assert metrics.brier_score >= 0


class TestM6Explainer:
    """Tests for explainability."""

    def test_explainer_imports(self):
        """Test explainer module imports."""
        from src.ml.hybrid import (
            ExplanationType,
            Explanation,
            HybridExplainer,
            FeatureContribution,
        )
        assert ExplanationType is not None
        assert HybridExplainer is not None

    def test_feature_contribution(self):
        """Test FeatureContribution creation."""
        from src.ml.hybrid.explainer import FeatureContribution

        contrib = FeatureContribution(
            feature_name="filename_label",
            feature_value="零件图",
            contribution=0.5,
            source="filename",
            description="文件名分类为 零件图",
        )
        assert contrib.contribution == 0.5

    def test_explanation_creation(self):
        """Test Explanation creation."""
        from src.ml.hybrid.explainer import Explanation, ExplanationType

        explanation = Explanation(
            prediction_label="零件图",
            prediction_confidence=0.85,
            explanation_type=ExplanationType.DECISION_PATH,
            summary="基于文件名特征分类",
        )
        assert explanation.prediction_label == "零件图"

    def test_explanation_natural_language(self):
        """Test natural language generation."""
        from src.ml.hybrid.explainer import Explanation, ExplanationType

        explanation = Explanation(
            prediction_label="零件图",
            prediction_confidence=0.85,
            explanation_type=ExplanationType.DECISION_PATH,
            summary="文件名特征明确指向零件图",
            top_positive_features=["文件名匹配零件图模式"],
            source_contributions={"文件名": 0.7},
        )
        text = explanation.to_natural_language()
        assert "零件图" in text
        assert "85" in text


# ============================================================================
# C3: Geometry Analysis Tests
# ============================================================================

class TestC3Features:
    """Tests for geometric feature extraction."""

    def test_geometry_imports(self):
        """Test geometry module imports."""
        from src.core.cad.geometry import (
            GeometryType,
            BoundingBox,
            GeometricFeatures,
            GeometryExtractor,
            DrawingStatistics,
            DrawingAnalyzer,
        )
        assert GeometryType is not None
        assert GeometryExtractor is not None

    def test_bounding_box(self):
        """Test BoundingBox creation."""
        from src.core.cad.geometry.features import BoundingBox

        bbox = BoundingBox(0, 0, 100, 50)
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000
        assert bbox.aspect_ratio == 2.0
        assert bbox.center == (50, 25)

    def test_bounding_box_operations(self):
        """Test BoundingBox operations."""
        from src.core.cad.geometry.features import BoundingBox

        bbox1 = BoundingBox(0, 0, 50, 50)
        bbox2 = BoundingBox(25, 25, 75, 75)

        assert bbox1.intersects(bbox2)
        assert bbox1.contains(25, 25)

        union = bbox1.union(bbox2)
        assert union.min_x == 0
        assert union.max_x == 75

    def test_geometric_features(self):
        """Test GeometricFeatures creation."""
        from src.core.cad.geometry.features import GeometricFeatures, GeometryType

        features = GeometricFeatures(
            entity_type=GeometryType.LINE,
            length=100.0,
            layer="GEOMETRY",
        )
        assert features.entity_type == GeometryType.LINE
        assert features.length == 100.0

        vec = features.to_vector()
        assert len(vec) == 12

    def test_drawing_statistics(self):
        """Test DrawingStatistics creation."""
        from src.core.cad.geometry.features import DrawingStatistics

        stats = DrawingStatistics(
            total_entities=100,
            line_count=50,
            circle_count=20,
            text_count=30,
        )
        assert stats.total_entities == 100


class TestC3Topology:
    """Tests for topological analysis."""

    def test_topology_imports(self):
        """Test topology module imports."""
        from src.core.cad.geometry import (
            ConnectionType,
            TopologicalNode,
            TopologyGraph,
            TopologyAnalyzer,
        )
        assert ConnectionType is not None
        assert TopologyGraph is not None

    def test_topology_graph(self):
        """Test TopologyGraph creation."""
        from src.core.cad.geometry.topology import TopologyGraph, ConnectionType

        graph = TopologyGraph()
        graph.add_node("e1", "LINE", (0, 0))
        graph.add_node("e2", "LINE", (10, 0))
        graph.add_edge("e1", "e2", ConnectionType.ENDPOINT, (5, 0))

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_connected_components(self):
        """Test finding connected components."""
        from src.core.cad.geometry.topology import TopologyGraph, ConnectionType

        graph = TopologyGraph()
        # Component 1
        graph.add_node("a1", "LINE")
        graph.add_node("a2", "LINE")
        graph.add_edge("a1", "a2", ConnectionType.ENDPOINT)

        # Component 2 (isolated)
        graph.add_node("b1", "CIRCLE")

        components = graph.find_connected_components()
        assert len(components) == 2

    def test_topology_analysis(self):
        """Test TopologyAnalysis result."""
        from src.core.cad.geometry.topology import TopologyAnalysis

        analysis = TopologyAnalysis(
            node_count=10,
            edge_count=8,
            connected_components=[],
            isolated_nodes=2,
            max_degree=4,
            mean_degree=1.6,
            clustering_coefficient=0.3,
            density=0.2,
            junction_points=[],
            endpoint_pairs=[],
            chains=[],
        )
        assert analysis.node_count == 10


class TestC3Spatial:
    """Tests for spatial indexing."""

    def test_spatial_imports(self):
        """Test spatial module imports."""
        from src.core.cad.geometry import (
            SpatialBounds,
            SpatialIndex,
            GridIndex,
            RTreeIndex,
            SpatialQuery,
        )
        assert SpatialBounds is not None
        assert GridIndex is not None

    def test_spatial_bounds(self):
        """Test SpatialBounds operations."""
        from src.core.cad.geometry.spatial import SpatialBounds

        bounds = SpatialBounds(0, 0, 100, 100)
        assert bounds.contains_point(50, 50)
        assert not bounds.contains_point(150, 50)

    def test_grid_index(self):
        """Test GridIndex operations."""
        from src.core.cad.geometry.spatial import GridIndex, SpatialBounds

        index = GridIndex(cell_size=50.0)
        index.insert("e1", SpatialBounds(0, 0, 40, 40))
        index.insert("e2", SpatialBounds(60, 60, 100, 100))

        assert index.count == 2

        # Query point
        results = index.query_point(20, 20)
        assert len(results) == 1
        assert results[0].entity_id == "e1"

    def test_spatial_query_bounds(self):
        """Test spatial query by bounds."""
        from src.core.cad.geometry.spatial import GridIndex, SpatialBounds

        index = GridIndex(cell_size=50.0)
        index.insert("e1", SpatialBounds(10, 10, 30, 30))
        index.insert("e2", SpatialBounds(20, 20, 50, 50))
        index.insert("e3", SpatialBounds(100, 100, 120, 120))

        results = index.query_bounds(SpatialBounds(0, 0, 40, 40))
        assert len(results) == 2  # e1 and e2

    def test_nearest_query(self):
        """Test nearest neighbor query."""
        from src.core.cad.geometry.spatial import GridIndex, SpatialBounds

        index = GridIndex()
        index.insert("e1", SpatialBounds(0, 0, 10, 10))
        index.insert("e2", SpatialBounds(100, 100, 110, 110))

        nearest = index.nearest(5, 5, k=1)
        assert len(nearest) == 1
        assert nearest[0][0].entity_id == "e1"


# ============================================================================
# I4: Model Monitoring Tests
# ============================================================================

class TestI4Metrics:
    """Tests for metrics collection."""

    def test_metrics_imports(self):
        """Test metrics module imports."""
        from src.ml.monitoring import (
            MetricType,
            Counter,
            Gauge,
            Histogram,
            MetricsCollector,
            get_metrics_collector,
        )
        assert MetricType is not None
        assert MetricsCollector is not None

    def test_counter(self):
        """Test Counter metric."""
        from src.ml.monitoring.metrics import Counter

        counter = Counter("requests_total")
        counter.inc()
        counter.inc(5)
        assert counter.get() == 6

    def test_gauge(self):
        """Test Gauge metric."""
        from src.ml.monitoring.metrics import Gauge

        gauge = Gauge("memory_usage")
        gauge.set(100)
        gauge.inc(50)
        gauge.dec(25)
        assert gauge.get() == 125

    def test_histogram(self):
        """Test Histogram metric."""
        from src.ml.monitoring.metrics import Histogram

        hist = Histogram("latency_seconds")
        hist.observe(0.1)
        hist.observe(0.5)
        hist.observe(1.5)

        summary = hist.get_summary()
        assert summary.count == 3

    def test_sliding_window(self):
        """Test SlidingWindowMetric."""
        from src.ml.monitoring.metrics import SlidingWindowMetric

        metric = SlidingWindowMetric("latency", window_seconds=60)
        metric.add(0.1)
        metric.add(0.2)
        metric.add(0.3)

        summary = metric.get_summary()
        assert summary.count == 3
        assert abs(summary.mean - 0.2) < 0.01

    def test_metrics_collector(self):
        """Test MetricsCollector."""
        from src.ml.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_prediction(
            latency_seconds=0.1,
            success=True,
            confidence=0.9,
            label="A",
        )

        metrics = collector.get_all_metrics()
        assert "counters" in metrics
        assert "histograms" in metrics


class TestI4Drift:
    """Tests for drift detection."""

    def test_drift_imports(self):
        """Test drift module imports."""
        from src.ml.monitoring import (
            DriftType,
            DriftSeverity,
            DriftResult,
            KSTestDetector,
            PSIDetector,
            DriftMonitor,
        )
        assert DriftType is not None
        assert DriftMonitor is not None

    def test_drift_result(self):
        """Test DriftResult creation."""
        from src.ml.monitoring.drift import DriftResult, DriftType, DriftSeverity

        result = DriftResult(
            drift_type=DriftType.DATA_DRIFT,
            detected=True,
            severity=DriftSeverity.MEDIUM,
            score=0.35,
            p_value=0.02,
        )
        assert result.detected is True
        assert result.drift_type == DriftType.DATA_DRIFT

    def test_ks_detector(self):
        """Test KS test detector."""
        from src.ml.monitoring.drift import KSTestDetector

        detector = KSTestDetector()
        reference = np.random.normal(0, 1, 1000)
        detector.fit(reference)

        # No drift
        current_same = np.random.normal(0, 1, 500)
        result = detector.detect(current_same)
        assert result.score < 0.2

        # With drift
        current_drift = np.random.normal(2, 1, 500)
        result = detector.detect(current_drift)
        assert result.score > 0.3

    def test_psi_detector(self):
        """Test PSI detector."""
        from src.ml.monitoring.drift import PSIDetector

        detector = PSIDetector(n_bins=10)
        reference = np.random.uniform(0, 1, 1000)
        detector.fit(reference)

        current = np.random.uniform(0.3, 1.3, 500)
        result = detector.detect(current)
        assert result.score >= 0

    def test_drift_monitor(self):
        """Test DriftMonitor."""
        from src.ml.monitoring.drift import DriftMonitor

        monitor = DriftMonitor(window_size=100, check_interval=50)
        reference = np.random.normal(0, 1, (500, 5))
        monitor.set_reference(reference)

        status = monitor.get_status()
        assert status["reference_size"] == 500


class TestI4Alerts:
    """Tests for alerting system."""

    def test_alerts_imports(self):
        """Test alerts module imports."""
        from src.ml.monitoring import (
            AlertSeverity,
            AlertStatus,
            Alert,
            AlertRule,
            AlertManager,
            get_alert_manager,
        )
        assert AlertSeverity is not None
        assert AlertManager is not None

    def test_alert_creation(self):
        """Test Alert creation."""
        from src.ml.monitoring.alerts import Alert, AlertSeverity, AlertStatus
        import time

        alert = Alert(
            alert_id="test-1",
            name="test_alert",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            message="Test alert message",
            source="test",
            created_at=time.time(),
            updated_at=time.time(),
        )
        assert alert.name == "test_alert"
        assert alert.severity == AlertSeverity.WARNING

    def test_alert_rule(self):
        """Test AlertRule evaluation."""
        from src.ml.monitoring.alerts import AlertRule, AlertSeverity

        rule = AlertRule(
            name="high_latency",
            condition=lambda ctx: ctx.get("latency", 0) > 1.0,
            severity=AlertSeverity.WARNING,
            message_template="High latency: {latency}s",
            cooldown_seconds=0,
        )

        # Should trigger
        result = rule.evaluate({"latency": 2.0})
        assert result is not None
        assert "2.0" in result

        # Should not trigger
        result = rule.evaluate({"latency": 0.5})
        assert result is None

    def test_alert_manager(self):
        """Test AlertManager."""
        from src.ml.monitoring.alerts import AlertManager, AlertSeverity

        manager = AlertManager()

        alert = manager.fire_alert(
            name="test_alert",
            severity=AlertSeverity.ERROR,
            message="Test error",
            source="test",
        )

        assert alert.alert_id is not None
        active = manager.get_active_alerts()
        assert len(active) >= 1

    def test_alert_lifecycle(self):
        """Test alert acknowledge and resolve."""
        from src.ml.monitoring.alerts import AlertManager, AlertSeverity, AlertStatus

        manager = AlertManager()
        alert = manager.fire_alert(
            name="lifecycle_test",
            severity=AlertSeverity.WARNING,
            message="Test",
            source="test",
        )

        # Acknowledge
        manager.acknowledge(alert.alert_id, by="tester")
        updated = manager.get_alert(alert.alert_id)
        assert updated.status == AlertStatus.ACKNOWLEDGED

        # Resolve
        manager.resolve(alert.alert_id)
        resolved = manager.get_alert(alert.alert_id)
        assert resolved.status == AlertStatus.RESOLVED


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
