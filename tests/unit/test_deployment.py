"""Tests for Deployment Management (Blue-Green, A/B, Canary)."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.core.deployment.blue_green import (
    BlueGreenManager,
    DeploymentSlot,
    SlotStatus,
)
from src.core.deployment.ab_testing import (
    ABTestManager,
    Experiment,
    ExperimentStatus,
    Variant,
)
from src.core.deployment.canary import (
    CanaryManager,
    CanaryConfig,
    CanaryMetrics,
    CanaryRelease,
    CanaryStatus,
    RolloutPhase,
    RolloutStep,
)


class TestDeploymentSlot:
    """Tests for DeploymentSlot."""

    def test_slot_creation(self):
        """Test slot creation with defaults."""
        slot = DeploymentSlot(name="blue")
        assert slot.name == "blue"
        assert slot.status == SlotStatus.STANDBY
        assert slot.version == ""

    def test_slot_is_healthy(self):
        """Test health check logic."""
        slot = DeploymentSlot(name="blue", replicas=3, ready_replicas=3)
        assert slot.is_healthy() is True

        slot.ready_replicas = 2
        assert slot.is_healthy() is False

        slot.ready_replicas = 3
        slot.status = SlotStatus.UNHEALTHY
        assert slot.is_healthy() is False

    def test_slot_to_dict(self):
        """Test serialization."""
        slot = DeploymentSlot(
            name="green",
            status=SlotStatus.ACTIVE,
            version="v1.2.3",
            replicas=3,
            ready_replicas=3,
        )
        data = slot.to_dict()
        assert data["name"] == "green"
        assert data["status"] == "active"
        assert data["version"] == "v1.2.3"
        assert data["is_healthy"] is True


class TestBlueGreenManager:
    """Tests for BlueGreenManager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = BlueGreenManager()
        assert manager.blue is not None
        assert manager.green is not None
        assert manager._active_slot == "blue"

    def test_active_standby_slots(self):
        """Test active/standby slot properties."""
        manager = BlueGreenManager()
        assert manager.active_slot.name == "blue"
        assert manager.standby_slot.name == "green"

    @pytest.mark.asyncio
    async def test_deploy_to_standby(self):
        """Test deployment to standby slot."""
        deploy_called = {"called": False, "version": None}

        def mock_deploy(slot, version):
            deploy_called["called"] = True
            deploy_called["version"] = version
            return True

        def mock_health(slot):
            return True

        manager = BlueGreenManager(
            deploy_fn=mock_deploy,
            health_check_fn=mock_health,
        )

        # Make standby healthy
        manager.green.ready_replicas = manager.green.replicas

        result = await manager.deploy("v2.0.0", health_check_retries=1, health_check_interval=0.1)
        assert result is True
        assert deploy_called["called"] is True
        assert deploy_called["version"] == "v2.0.0"
        assert manager.green.version == "v2.0.0"

    @pytest.mark.asyncio
    async def test_switch_traffic(self):
        """Test traffic switching."""
        switch_called = {"slot": None}

        def mock_switch(slot):
            switch_called["slot"] = slot
            return True

        manager = BlueGreenManager(switch_traffic_fn=mock_switch)

        # Make standby healthy
        manager.green.ready_replicas = manager.green.replicas
        manager.green.status = SlotStatus.STANDBY

        result = await manager.switch_traffic()
        assert result is True
        assert manager._active_slot == "green"
        assert manager.green.status == SlotStatus.ACTIVE
        assert manager.blue.status == SlotStatus.STANDBY

    def test_get_status(self):
        """Test status retrieval."""
        manager = BlueGreenManager()
        status = manager.get_status()

        assert "active_slot" in status
        assert "blue" in status
        assert "green" in status
        assert "can_switch" in status


class TestVariant:
    """Tests for A/B test Variant."""

    def test_variant_creation(self):
        """Test variant creation."""
        variant = Variant(name="control", is_control=True)
        assert variant.name == "control"
        assert variant.is_control is True
        assert variant.weight == 1.0

    def test_conversion_rate(self):
        """Test conversion rate calculation."""
        variant = Variant(name="treatment")
        variant.impressions = 100
        variant.conversions = 10
        assert variant.conversion_rate == 0.1

    def test_conversion_rate_zero_impressions(self):
        """Test conversion rate with zero impressions."""
        variant = Variant(name="treatment")
        assert variant.conversion_rate == 0.0


class TestExperiment:
    """Tests for A/B test Experiment."""

    def test_experiment_creation(self):
        """Test experiment creation with defaults."""
        exp = Experiment(id="test-1", name="Test Experiment")
        assert exp.id == "test-1"
        assert len(exp.variants) == 2  # Default control + treatment
        assert exp.status == ExperimentStatus.DRAFT

    def test_experiment_get_control(self):
        """Test getting control variant."""
        exp = Experiment(id="test-1", name="Test")
        control = exp.get_control()
        assert control is not None
        assert control.is_control is True

    def test_experiment_total_impressions(self):
        """Test total impressions calculation."""
        exp = Experiment(id="test-1", name="Test")
        exp.variants[0].impressions = 100
        exp.variants[1].impressions = 100
        assert exp.total_impressions == 200


class TestABTestManager:
    """Tests for ABTestManager."""

    def test_create_experiment(self):
        """Test experiment creation."""
        manager = ABTestManager()
        exp = manager.create_experiment(
            id="test-exp",
            name="Test Experiment",
        )
        assert exp.id == "test-exp"
        assert manager.get_experiment("test-exp") is not None

    def test_start_experiment(self):
        """Test starting an experiment."""
        manager = ABTestManager()
        manager.create_experiment(id="test-exp", name="Test")

        result = manager.start_experiment("test-exp")
        assert result is True

        exp = manager.get_experiment("test-exp")
        assert exp.status == ExperimentStatus.RUNNING

    def test_variant_assignment(self):
        """Test consistent variant assignment."""
        manager = ABTestManager()
        manager.create_experiment(id="test-exp", name="Test")
        manager.start_experiment("test-exp")

        # Same user should get same variant
        variant1 = manager.get_variant_for_user("test-exp", "user-123")
        variant2 = manager.get_variant_for_user("test-exp", "user-123")
        assert variant1 == variant2

    def test_record_metrics(self):
        """Test recording impressions and conversions."""
        manager = ABTestManager()
        manager.create_experiment(id="test-exp", name="Test")
        manager.start_experiment("test-exp")

        # Get variant and record
        variant_name = manager.get_variant_for_user("test-exp", "user-123")
        manager.record_impression("test-exp", "user-123")
        manager.record_conversion("test-exp", "user-123", value=10.0)

        exp = manager.get_experiment("test-exp")
        variant = exp.get_variant(variant_name)
        assert variant.impressions == 1
        assert variant.conversions == 1
        assert variant.total_value == 10.0

    def test_get_results(self):
        """Test getting experiment results."""
        manager = ABTestManager()
        manager.create_experiment(id="test-exp", name="Test")
        manager.start_experiment("test-exp")

        # Record some data
        for i in range(50):
            manager.record_impression("test-exp", f"user-{i}")
            if i % 10 == 0:
                manager.record_conversion("test-exp", f"user-{i}")

        results = manager.get_results("test-exp")
        assert results is not None
        assert "variants" in results
        assert "total_impressions" in results


class TestCanaryConfig:
    """Tests for CanaryConfig."""

    def test_config_defaults(self):
        """Test config with default steps."""
        config = CanaryConfig(version="v2.0.0")
        assert len(config.steps) == 7
        assert config.steps[0].percentage == 1
        assert config.steps[-1].percentage == 100

    def test_config_custom_steps(self):
        """Test config with custom steps."""
        steps = [
            RolloutStep(5, RolloutPhase.CANARY),
            RolloutStep(50, RolloutPhase.PROGRESSIVE),
            RolloutStep(100, RolloutPhase.FULL),
        ]
        config = CanaryConfig(version="v2.0.0", steps=steps)
        assert len(config.steps) == 3


class TestCanaryMetrics:
    """Tests for CanaryMetrics."""

    def test_error_rate(self):
        """Test error rate calculation."""
        metrics = CanaryMetrics(requests=100, errors=5)
        assert metrics.error_rate == 0.05

    def test_avg_latency(self):
        """Test average latency calculation."""
        metrics = CanaryMetrics(requests=100, latency_sum_ms=5000.0)
        assert metrics.avg_latency_ms == 50.0


class TestCanaryManager:
    """Tests for CanaryManager."""

    @pytest.mark.asyncio
    async def test_create_release(self):
        """Test creating a canary release."""
        manager = CanaryManager()
        release = await manager.create_release(
            release_id="release-1",
            version="v2.0.0",
            baseline_version="v1.0.0",
        )
        assert release.id == "release-1"
        assert release.config.version == "v2.0.0"
        assert release.status == CanaryStatus.PENDING

    @pytest.mark.asyncio
    async def test_start_release(self):
        """Test starting a canary release."""
        set_traffic_called = {"version": None, "percentage": None}

        def mock_set_traffic(version, percentage):
            set_traffic_called["version"] = version
            set_traffic_called["percentage"] = percentage
            return True

        manager = CanaryManager(set_traffic_fn=mock_set_traffic)
        await manager.create_release("release-1", "v2.0.0")

        result = await manager.start_release("release-1")
        assert result is True

        release = manager.get_release("release-1")
        assert release.status == CanaryStatus.ROLLING_OUT
        assert set_traffic_called["percentage"] == 1  # First step

    @pytest.mark.asyncio
    async def test_pause_resume_release(self):
        """Test pausing and resuming a release."""
        manager = CanaryManager()
        await manager.create_release("release-1", "v2.0.0")
        await manager.start_release("release-1")

        # Pause
        result = await manager.pause_release("release-1")
        assert result is True
        assert manager.get_release("release-1").status == CanaryStatus.PAUSED

        # Resume
        result = await manager.resume_release("release-1")
        assert result is True
        assert manager.get_release("release-1").status == CanaryStatus.ROLLING_OUT

    @pytest.mark.asyncio
    async def test_rollback_release(self):
        """Test rolling back a release."""
        rollback_called = {"version": None}

        def mock_rollback(version):
            rollback_called["version"] = version
            return True

        manager = CanaryManager(rollback_fn=mock_rollback)
        await manager.create_release("release-1", "v2.0.0")
        await manager.start_release("release-1")

        result = await manager.rollback_release("release-1", "High error rate")
        assert result is True
        assert manager.get_release("release-1").status == CanaryStatus.ROLLED_BACK
        assert rollback_called["version"] == "v2.0.0"

    def test_get_status(self):
        """Test getting canary status."""
        manager = CanaryManager()
        status = manager.get_status()
        assert "active_release" in status
        assert "total_releases" in status
