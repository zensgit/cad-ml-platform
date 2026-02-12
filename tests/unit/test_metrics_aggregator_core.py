"""Tests for metrics_aggregator core module to improve coverage."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.core.metrics_aggregator.core import (
    MetricType,
    MetricLabels,
    MetricValue,
    MetricMetadata,
    Metric,
    LabeledMetric,
    Counter,
    Gauge,
    Histogram,
    Summary,
    Timer,
    MetricSample,
)


class TestMetricLabels:
    """Tests for MetricLabels class."""

    def test_hash(self):
        """Test MetricLabels.__hash__."""
        labels1 = MetricLabels({"a": "1", "b": "2"})
        labels2 = MetricLabels({"b": "2", "a": "1"})

        # Same labels should have same hash
        assert hash(labels1) == hash(labels2)

        # Can be used as dict key
        d = {labels1: "value"}
        assert d[labels2] == "value"

    def test_eq_with_non_metric_labels(self):
        """Test MetricLabels.__eq__ with non-MetricLabels object (line 39)."""
        labels = MetricLabels({"a": "1"})

        # Comparing with non-MetricLabels should return False
        assert labels != "string"
        assert labels != {"a": "1"}
        assert labels != 123
        assert labels != None

    def test_eq_with_different_labels(self):
        """Test MetricLabels.__eq__ with different labels."""
        labels1 = MetricLabels({"a": "1"})
        labels2 = MetricLabels({"a": "2"})

        assert labels1 != labels2

    def test_to_string_with_labels(self):
        """Test MetricLabels.to_string with labels (lines 45-46)."""
        labels = MetricLabels({"method": "GET", "path": "/api"})

        result = labels.to_string()

        # Labels should be sorted alphabetically
        assert result == '{method="GET",path="/api"}'

    def test_to_string_empty(self):
        """Test MetricLabels.to_string with empty labels."""
        labels = MetricLabels()

        result = labels.to_string()

        assert result == ""


class TestMetricValue:
    """Tests for MetricValue class."""

    def test_default_timestamp(self):
        """Test MetricValue gets default timestamp."""
        value = MetricValue(value=42.0)

        assert value.value == 42.0
        assert value.timestamp is not None
        assert isinstance(value.timestamp, datetime)

    def test_default_labels(self):
        """Test MetricValue gets default empty labels."""
        value = MetricValue(value=42.0)

        assert isinstance(value.labels, MetricLabels)
        assert value.labels.labels == {}


class TestMetricMetadata:
    """Tests for MetricMetadata class."""

    def test_metadata_creation(self):
        """Test MetricMetadata creation."""
        metadata = MetricMetadata(
            name="test_metric",
            metric_type=MetricType.COUNTER,
            description="Test description",
            unit="bytes",
            label_names=["method", "status"],
        )

        assert metadata.name == "test_metric"
        assert metadata.metric_type == MetricType.COUNTER
        assert metadata.description == "Test description"
        assert metadata.unit == "bytes"
        assert metadata.label_names == ["method", "status"]


class TestMetricAbstract:
    """Tests for abstract Metric class."""

    def test_metric_type_is_abstract(self):
        """Test metric_type is abstract (line 92)."""
        # Cannot instantiate Metric directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Metric("test")

    def test_collect_is_abstract(self):
        """Test collect is abstract (line 97)."""
        # Verified through Counter/Gauge/etc implementations
        counter = Counter("test")
        result = counter.collect()
        assert isinstance(result, list)


class TestMetricLabelsMethod:
    """Tests for Metric.labels method."""

    def test_labels_returns_labeled_metric(self):
        """Test Metric.labels returns LabeledMetric."""
        counter = Counter("requests", label_names=["method"])

        labeled = counter.labels(method="GET")

        assert isinstance(labeled, LabeledMetric)

    def test_validate_labels_mismatch(self):
        """Test _validate_labels raises on mismatch (lines 105-108)."""
        counter = Counter("requests", label_names=["method", "status"])

        # Wrong labels
        wrong_labels = MetricLabels({"wrong": "label"})

        with pytest.raises(ValueError, match="Label mismatch"):
            counter._validate_labels(wrong_labels)

    def test_validate_labels_correct(self):
        """Test _validate_labels passes with correct labels."""
        counter = Counter("requests", label_names=["method"])

        correct_labels = MetricLabels({"method": "GET"})

        # Should not raise
        counter._validate_labels(correct_labels)


class TestLabeledMetric:
    """Tests for LabeledMetric class."""

    def test_inc(self):
        """Test LabeledMetric.inc."""
        counter = Counter("test", label_names=["method"])
        labeled = counter.labels(method="GET")

        labeled.inc()
        labeled.inc(5)

        values = counter.collect()
        assert len(values) == 1
        assert values[0].value == 6

    def test_dec_with_gauge(self):
        """Test LabeledMetric.dec with Gauge (lines 123-124)."""
        gauge = Gauge("test", label_names=["method"])
        labeled = gauge.labels(method="GET")

        labeled.set(10)
        labeled.dec(3)

        values = gauge.collect()
        assert len(values) == 1
        assert values[0].value == 7

    def test_dec_without_dec_method(self):
        """Test LabeledMetric.dec when metric has no _dec method."""
        # Counter doesn't have _dec
        counter = Counter("test", label_names=["method"])
        labeled = counter.labels(method="GET")

        # Should not raise, just do nothing
        labeled.dec(1)

    def test_set(self):
        """Test LabeledMetric.set."""
        gauge = Gauge("test", label_names=["method"])
        labeled = gauge.labels(method="GET")

        labeled.set(42)

        values = gauge.collect()
        assert values[0].value == 42

    def test_observe_with_histogram(self):
        """Test LabeledMetric.observe with Histogram (lines 131-132)."""
        histogram = Histogram("test", label_names=["method"])
        labeled = histogram.labels(method="GET")

        labeled.observe(0.5)

        values = histogram.collect()
        # Should have bucket values, sum, and count
        assert len(values) > 0

    def test_observe_without_observe_method(self):
        """Test LabeledMetric.observe when metric has no _observe method."""
        counter = Counter("test", label_names=["method"])
        labeled = counter.labels(method="GET")

        # Should not raise, just do nothing
        labeled.observe(1.0)


class TestCounter:
    """Tests for Counter class."""

    def test_metric_type(self):
        """Test Counter.metric_type."""
        counter = Counter("test")
        assert counter.metric_type == MetricType.COUNTER

    def test_inc_negative_raises(self):
        """Test Counter.inc raises on negative value."""
        counter = Counter("test")

        with pytest.raises(ValueError, match="Counter can only be incremented"):
            counter.inc(-1)

    def test_collect_empty(self):
        """Test Counter.collect with no values."""
        counter = Counter("test")

        values = counter.collect()

        assert values == []


class TestGauge:
    """Tests for Gauge class."""

    def test_metric_type(self):
        """Test Gauge.metric_type."""
        gauge = Gauge("test")
        assert gauge.metric_type == MetricType.GAUGE

    def test_set_and_collect(self):
        """Test Gauge.set and collect."""
        gauge = Gauge("test")

        gauge.set(42)

        values = gauge.collect()
        assert len(values) == 1
        assert values[0].value == 42

    def test_inc_and_dec(self):
        """Test Gauge.inc and dec."""
        gauge = Gauge("test")

        gauge.set(10)
        gauge.inc(5)
        gauge.dec(3)

        values = gauge.collect()
        assert values[0].value == 12


class TestHistogram:
    """Tests for Histogram class."""

    def test_metric_type(self):
        """Test Histogram.metric_type (line 245)."""
        histogram = Histogram("test")
        assert histogram.metric_type == MetricType.HISTOGRAM

    def test_observe_and_collect(self):
        """Test Histogram.observe and collect."""
        histogram = Histogram("test")

        histogram.observe(0.5)
        histogram.observe(1.5)

        values = histogram.collect()

        # Should have bucket values (with le labels), sum, and count
        bucket_values = [v for v in values if "le" in v.labels.labels]
        sum_values = [v for v in values if v.labels.labels.get("_type") == "sum"]
        count_values = [v for v in values if v.labels.labels.get("_type") == "count"]

        assert len(bucket_values) > 0
        assert len(sum_values) == 1
        assert sum_values[0].value == 2.0  # 0.5 + 1.5
        assert len(count_values) == 1
        assert count_values[0].value == 2

    def test_custom_buckets(self):
        """Test Histogram with custom buckets."""
        buckets = (0.1, 0.5, 1.0, float('inf'))
        histogram = Histogram("test", buckets=buckets)

        histogram.observe(0.3)

        values = histogram.collect()
        bucket_values = [v for v in values if "le" in v.labels.labels]

        # Should have 4 buckets
        assert len(bucket_values) == 4

    def test_timer_context_manager(self):
        """Test Histogram.time returns Timer."""
        histogram = Histogram("test")

        timer = histogram.time()

        assert isinstance(timer, Timer)


class TestSummary:
    """Tests for Summary class."""

    def test_metric_type(self):
        """Test Summary.metric_type (line 324)."""
        summary = Summary("test")
        assert summary.metric_type == MetricType.SUMMARY

    def test_observe_and_collect(self):
        """Test Summary.observe and collect."""
        summary = Summary("test")

        for i in range(10):
            summary.observe(i * 0.1)

        values = summary.collect()

        # Should have quantile values, sum, and count
        quantile_values = [v for v in values if "quantile" in v.labels.labels]
        sum_values = [v for v in values if v.labels.labels.get("_type") == "sum"]
        count_values = [v for v in values if v.labels.labels.get("_type") == "count"]

        assert len(quantile_values) == 4  # Default quantiles: 0.5, 0.9, 0.95, 0.99
        assert len(sum_values) == 1
        assert len(count_values) == 1
        assert count_values[0].value == 10

    def test_max_samples_exceeded(self):
        """Test Summary respects max_samples (lines 341-342)."""
        summary = Summary("test", max_samples=5)

        for i in range(10):
            summary.observe(i)

        # Internal samples should be capped at 5
        assert len(summary._samples[MetricLabels()]) == 5

    def test_collect_empty_samples(self):
        """Test Summary.collect skips empty samples (lines 353-354)."""
        summary = Summary("test")

        # Don't observe anything
        values = summary.collect()

        assert values == []

    def test_custom_quantiles(self):
        """Test Summary with custom quantiles."""
        summary = Summary("test", quantiles=(0.25, 0.5, 0.75))

        for i in range(100):
            summary.observe(i)

        values = summary.collect()
        quantile_values = [v for v in values if "quantile" in v.labels.labels]

        assert len(quantile_values) == 3


class TestTimer:
    """Tests for Timer class."""

    def test_timer_as_context_manager(self):
        """Test Timer as context manager."""
        histogram = Histogram("test")

        with histogram.time():
            pass  # Do something

        values = histogram.collect()

        # Should have recorded a duration
        sum_values = [v for v in values if v.labels.labels.get("_type") == "sum"]
        assert len(sum_values) == 1
        assert sum_values[0].value >= 0

    def test_timer_with_labels(self):
        """Test Timer with custom labels."""
        histogram = Histogram("test")
        labels = MetricLabels({"method": "GET"})

        timer = Timer(histogram, labels)

        with timer:
            pass

        values = histogram.collect()
        # Should have recorded with labels
        assert len(values) > 0


class TestMetricSample:
    """Tests for MetricSample class."""

    def test_sample_creation(self):
        """Test MetricSample creation."""
        values = [MetricValue(value=42.0)]
        sample = MetricSample(
            name="test_metric",
            metric_type=MetricType.COUNTER,
            description="Test description",
            values=values,
        )

        assert sample.name == "test_metric"
        assert sample.metric_type == MetricType.COUNTER
        assert sample.description == "Test description"
        assert sample.values == values
        assert sample.timestamp is not None


class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_types(self):
        """Test all metric types exist."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


class TestMetricProperties:
    """Tests for Metric base class properties."""

    def test_name_property(self):
        """Test Metric.name property."""
        counter = Counter("my_counter")
        assert counter.name == "my_counter"

    def test_description_property(self):
        """Test Metric.description property."""
        counter = Counter("test", description="Test counter")
        assert counter.description == "Test counter"


class TestThreadSafety:
    """Tests for thread safety of metrics."""

    def test_counter_thread_safe(self):
        """Test Counter is thread-safe."""
        import threading

        counter = Counter("test")

        def increment():
            for _ in range(100):
                counter.inc()

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        values = counter.collect()
        assert values[0].value == 1000

    def test_gauge_thread_safe(self):
        """Test Gauge is thread-safe."""
        import threading

        gauge = Gauge("test")
        gauge.set(0)

        def modify():
            for _ in range(100):
                gauge.inc()

        threads = [threading.Thread(target=modify) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        values = gauge.collect()
        assert values[0].value == 1000
