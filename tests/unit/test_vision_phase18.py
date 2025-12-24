"""Tests for Phase 18: Advanced ML Pipeline & AutoML.

This module tests the Phase 18 Vision components:
- AutoML Engine
- Feature Store
- Model Registry
- Pipeline Orchestrator
- Experiment Tracker
"""

from datetime import datetime

import pytest


class TestAutoMLEngine:
    """Tests for AutoML Engine module."""

    def test_search_strategy_enum(self):
        """Test SearchStrategy enum values."""
        from src.core.vision import SearchStrategy

        assert SearchStrategy.GRID_SEARCH.value == "grid_search"
        assert SearchStrategy.RANDOM_SEARCH.value == "random_search"
        assert SearchStrategy.BAYESIAN.value == "bayesian"
        assert SearchStrategy.EVOLUTIONARY.value == "evolutionary"
        assert SearchStrategy.HYPERBAND.value == "hyperband"

    def test_optimization_objective_enum(self):
        """Test OptimizationObjective enum values."""
        from src.core.vision import OptimizationObjective

        assert OptimizationObjective.ACCURACY.value == "accuracy"
        assert OptimizationObjective.F1_SCORE.value == "f1_score"
        assert OptimizationObjective.MSE.value == "mse"
        assert OptimizationObjective.AUC_ROC.value == "auc_roc"

    def test_automl_model_type_enum(self):
        """Test AutoMLModelType enum values."""
        from src.core.vision import AutoMLModelType

        assert AutoMLModelType.LINEAR.value == "linear"
        assert AutoMLModelType.TREE.value == "tree"
        assert AutoMLModelType.ENSEMBLE.value == "ensemble"
        assert AutoMLModelType.NEURAL_NETWORK.value == "neural_network"

    def test_automl_task_type_enum(self):
        """Test AutoMLTaskType enum values."""
        from src.core.vision import AutoMLTaskType

        assert AutoMLTaskType.CLASSIFICATION.value == "classification"
        assert AutoMLTaskType.REGRESSION.value == "regression"
        assert AutoMLTaskType.CLUSTERING.value == "clustering"

    def test_trial_status_enum(self):
        """Test TrialStatus enum values."""
        from src.core.vision import TrialStatus

        assert TrialStatus.PENDING.value == "pending"
        assert TrialStatus.RUNNING.value == "running"
        assert TrialStatus.COMPLETED.value == "completed"
        assert TrialStatus.FAILED.value == "failed"

    def test_hyperparameter_space_creation(self):
        """Test HyperparameterSpace dataclass."""
        from src.core.vision import HyperparameterSpace

        space = HyperparameterSpace(
            name="learning_rate",
            param_type="float",
            low=0.001,
            high=0.1,
            log_scale=True,
        )
        assert space.name == "learning_rate"
        assert space.param_type == "float"
        assert space.low == 0.001
        assert space.high == 0.1
        assert space.log_scale is True

    def test_hyperparameter_space_sample(self):
        """Test HyperparameterSpace sampling."""
        from src.core.vision import HyperparameterSpace

        # Test float sampling
        float_space = HyperparameterSpace(name="lr", param_type="float", low=0.0, high=1.0)
        value = float_space.sample()
        assert 0.0 <= value <= 1.0

        # Test int sampling
        int_space = HyperparameterSpace(name="epochs", param_type="int", low=1, high=100)
        value = int_space.sample()
        assert 1 <= value <= 100
        assert isinstance(value, int)

        # Test categorical sampling
        cat_space = HyperparameterSpace(
            name="optimizer", param_type="categorical", choices=["adam", "sgd"]
        )
        value = cat_space.sample()
        assert value in ["adam", "sgd"]

    def test_create_automl_engine(self):
        """Test create_automl_engine factory."""
        from src.core.vision import AutoMLTaskType, OptimizationObjective, create_automl_engine

        engine = create_automl_engine(
            task_type=AutoMLTaskType.CLASSIFICATION,
            objective=OptimizationObjective.ACCURACY,
        )
        assert engine is not None

    def test_create_search_config(self):
        """Test create_search_config factory."""
        from src.core.vision import (
            AutoMLTaskType,
            OptimizationObjective,
            SearchStrategy,
            create_search_config,
        )

        config = create_search_config(
            search_id="test-search",
            strategy=SearchStrategy.BAYESIAN,
            objective=OptimizationObjective.ACCURACY,
            task_type=AutoMLTaskType.CLASSIFICATION,
            max_trials=50,
        )
        assert config.search_id == "test-search"
        assert config.strategy == SearchStrategy.BAYESIAN
        assert config.max_trials == 50

    def test_create_model_selector(self):
        """Test create_model_selector factory."""
        from src.core.vision import AutoMLTaskType, create_model_selector

        selector = create_model_selector(AutoMLTaskType.CLASSIFICATION)
        candidates = selector.get_candidates()
        assert len(candidates) > 0

    def test_create_nas(self):
        """Test create_nas factory."""
        from src.core.vision import create_nas

        nas = create_nas(search_space="default", max_epochs=100)
        assert nas is not None

        architecture = nas.sample_architecture()
        assert "layers" in architecture
        assert "architecture_id" in architecture


class TestFeatureStore:
    """Tests for Feature Store module."""

    def test_feature_type_enum(self):
        """Test FeatureType enum values."""
        from src.core.vision import FeatureType

        assert FeatureType.NUMERICAL.value == "numerical"
        assert FeatureType.CATEGORICAL.value == "categorical"
        assert FeatureType.BOOLEAN.value == "boolean"
        assert FeatureType.TEXT.value == "text"
        assert FeatureType.EMBEDDING.value == "embedding"

    def test_feature_status_enum(self):
        """Test FeatureStatus enum values."""
        from src.core.vision import FeatureStatus

        assert FeatureStatus.DRAFT.value == "draft"
        assert FeatureStatus.ACTIVE.value == "active"
        assert FeatureStatus.DEPRECATED.value == "deprecated"

    def test_computation_mode_enum(self):
        """Test ComputationMode enum values."""
        from src.core.vision import ComputationMode

        assert ComputationMode.BATCH.value == "batch"
        assert ComputationMode.STREAMING.value == "streaming"
        assert ComputationMode.ON_DEMAND.value == "on_demand"

    def test_transformation_type_enum(self):
        """Test TransformationType enum values."""
        from src.core.vision import TransformationType

        assert TransformationType.NORMALIZE.value == "normalize"
        assert TransformationType.STANDARDIZE.value == "standardize"
        assert TransformationType.ONE_HOT.value == "one_hot"
        assert TransformationType.BUCKETIZE.value == "bucketize"

    def test_feature_definition_creation(self):
        """Test FeatureDefinition dataclass."""
        from src.core.vision import FeatureType, create_feature_definition

        feature = create_feature_definition(
            feature_id="user_age",
            name="User Age",
            feature_type=FeatureType.NUMERICAL,
            description="Age of the user",
        )
        assert feature.feature_id == "user_age"
        assert feature.name == "User Age"
        assert feature.feature_type == FeatureType.NUMERICAL

    def test_feature_group_creation(self):
        """Test FeatureGroup dataclass."""
        from src.core.vision import create_feature_group

        group = create_feature_group(
            group_id="user_features",
            name="User Features",
            features=["age", "gender", "location"],
        )
        assert group.group_id == "user_features"
        assert len(group.features) == 3

    def test_create_feature_registry(self):
        """Test create_feature_registry factory."""
        from src.core.vision import create_feature_registry

        registry = create_feature_registry()
        assert registry is not None

    def test_create_feature_store(self):
        """Test create_feature_store factory."""
        from src.core.vision import create_feature_store

        store = create_feature_store()
        assert store is not None

    def test_feature_registry_operations(self):
        """Test FeatureRegistry CRUD operations."""
        from src.core.vision import FeatureType, create_feature_definition, create_feature_registry

        registry = create_feature_registry()
        feature = create_feature_definition(
            feature_id="test_feature",
            name="Test Feature",
            feature_type=FeatureType.NUMERICAL,
        )

        registry.register_feature(feature)
        retrieved = registry.get_feature("test_feature")
        assert retrieved is not None
        assert retrieved.feature_id == "test_feature"


class TestModelRegistry:
    """Tests for Model Registry module."""

    def test_model_stage_enum(self):
        """Test ModelStage enum values."""
        from src.core.vision import ModelStage

        assert ModelStage.DEVELOPMENT.value == "development"
        assert ModelStage.STAGING.value == "staging"
        assert ModelStage.PRODUCTION.value == "production"
        assert ModelStage.ARCHIVED.value == "archived"

    def test_model_registry_status_enum(self):
        """Test ModelRegistryStatus enum values."""
        from src.core.vision import ModelRegistryStatus

        assert ModelRegistryStatus.PENDING.value == "pending"
        assert ModelRegistryStatus.TRAINING.value == "training"
        assert ModelRegistryStatus.READY.value == "ready"
        assert ModelRegistryStatus.DEPLOYED.value == "deployed"

    def test_deployment_strategy_enum(self):
        """Test DeploymentStrategy enum values."""
        from src.core.vision import DeploymentStrategy

        assert DeploymentStrategy.DIRECT.value == "direct"
        assert DeploymentStrategy.CANARY.value == "canary"
        assert DeploymentStrategy.BLUE_GREEN.value == "blue_green"
        assert DeploymentStrategy.A_B_TEST.value == "ab_test"

    def test_serving_format_enum(self):
        """Test ServingFormat enum values."""
        from src.core.vision import ServingFormat

        assert ServingFormat.ONNX.value == "onnx"
        assert ServingFormat.TENSORFLOW.value == "tensorflow"
        assert ServingFormat.PYTORCH.value == "pytorch"

    def test_model_version_creation(self):
        """Test ModelVersion dataclass."""
        from src.core.vision import ModelStage, create_model_version

        version = create_model_version(
            model_id="model1",
            version="1.0.0",
            name="Test Model",
            stage=ModelStage.DEVELOPMENT,
        )
        assert version.model_id == "model1"
        assert version.version == "1.0.0"
        assert version.stage == ModelStage.DEVELOPMENT

    def test_model_metadata_creation(self):
        """Test ModelMetadata dataclass."""
        from src.core.vision import create_model_metadata

        metadata = create_model_metadata(
            model_id="model1",
            name="Test Model",
            description="A test model",
            owner="test_user",
        )
        assert metadata.model_id == "model1"
        assert metadata.owner == "test_user"

    def test_create_model_registry(self):
        """Test create_model_registry factory."""
        from src.core.vision import create_model_registry

        registry = create_model_registry()
        assert registry is not None

    def test_create_ab_test_config(self):
        """Test create_ab_test_config factory."""
        from src.core.vision import create_ab_test_config

        config = create_ab_test_config(
            test_id="test1",
            name="AB Test 1",
            control_model="model_a",
            control_version="1.0",
            treatment_model="model_b",
            treatment_version="1.0",
            traffic_split=0.5,
        )
        assert config.test_id == "test1"
        assert config.traffic_split == 0.5

    def test_model_registry_operations(self):
        """Test ModelRegistry CRUD operations."""
        from src.core.vision import (
            ModelStage,
            create_model_metadata,
            create_model_registry,
            create_model_version,
        )

        registry = create_model_registry()
        metadata = create_model_metadata(
            model_id="test_model",
            name="Test Model",
        )
        registry.register_model(metadata)

        version = create_model_version(
            model_id="test_model",
            version="1.0.0",
            name="Version 1",
        )
        registry.create_version(version)

        retrieved = registry.get_version("test_model", "1.0.0")
        assert retrieved is not None
        assert retrieved.version == "1.0.0"


class TestPipelineOrchestrator:
    """Tests for Pipeline Orchestrator module."""

    def test_pipeline_task_status_enum(self):
        """Test PipelineTaskStatus enum values."""
        from src.core.vision import PipelineTaskStatus

        assert PipelineTaskStatus.PENDING.value == "pending"
        assert PipelineTaskStatus.RUNNING.value == "running"
        assert PipelineTaskStatus.COMPLETED.value == "completed"
        assert PipelineTaskStatus.FAILED.value == "failed"

    def test_pipeline_status_enum(self):
        """Test PipelineStatus enum values."""
        from src.core.vision import PipelineStatus

        assert PipelineStatus.CREATED.value == "created"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.CANCELLED.value == "cancelled"

    def test_trigger_type_enum(self):
        """Test TriggerType enum values."""
        from src.core.vision import TriggerType

        assert TriggerType.MANUAL.value == "manual"
        assert TriggerType.SCHEDULED.value == "scheduled"
        assert TriggerType.EVENT.value == "event"
        assert TriggerType.API.value == "api"

    def test_retry_policy_enum(self):
        """Test RetryPolicy enum values."""
        from src.core.vision import RetryPolicy

        assert RetryPolicy.NONE.value == "none"
        assert RetryPolicy.FIXED.value == "fixed"
        assert RetryPolicy.EXPONENTIAL.value == "exponential"

    def test_execution_mode_enum(self):
        """Test ExecutionMode enum values."""
        from src.core.vision import ExecutionMode

        assert ExecutionMode.SEQUENTIAL.value == "sequential"
        assert ExecutionMode.PARALLEL.value == "parallel"
        assert ExecutionMode.HYBRID.value == "hybrid"

    def test_task_definition_creation(self):
        """Test TaskDefinition dataclass."""
        from src.core.vision import create_task_definition

        task = create_task_definition(
            task_id="task1",
            name="Task 1",
            dependencies=["task0"],
        )
        assert task.task_id == "task1"
        assert "task0" in task.dependencies

    def test_pipeline_definition_creation(self):
        """Test PipelineDefinition dataclass."""
        from src.core.vision import (
            ExecutionMode,
            create_pipeline_definition,
            create_task_definition,
        )

        tasks = [
            create_task_definition(task_id="t1", name="Task 1"),
            create_task_definition(task_id="t2", name="Task 2", dependencies=["t1"]),
        ]
        pipeline = create_pipeline_definition(
            pipeline_id="pipeline1",
            name="Test Pipeline",
            tasks=tasks,
            execution_mode=ExecutionMode.HYBRID,
        )
        assert pipeline.pipeline_id == "pipeline1"
        assert len(pipeline.tasks) == 2

    def test_dag_builder(self):
        """Test DAGBuilder operations."""
        from src.core.vision import create_dag_builder

        dag = create_dag_builder()
        dag.add_task("task1")
        dag.add_task("task2", dependencies=["task1"])
        dag.add_task("task3", dependencies=["task1"])
        dag.add_task("task4", dependencies=["task2", "task3"])

        valid, errors = dag.validate()
        assert valid is True
        assert len(errors) == 0

        execution_order = dag.get_execution_order()
        assert len(execution_order) > 0
        assert "task1" in execution_order[0]

    def test_create_pipeline_orchestrator(self):
        """Test create_pipeline_orchestrator factory."""
        from src.core.vision import create_pipeline_orchestrator

        orchestrator = create_pipeline_orchestrator()
        assert orchestrator is not None

    def test_pipeline_execution(self):
        """Test pipeline execution."""
        from src.core.vision import (
            PipelineStatus,
            create_pipeline_definition,
            create_pipeline_orchestrator,
            create_task_definition,
        )

        orchestrator = create_pipeline_orchestrator()
        orchestrator._executor.start()

        tasks = [
            create_task_definition(task_id="t1", name="Task 1"),
            create_task_definition(task_id="t2", name="Task 2", dependencies=["t1"]),
        ]
        pipeline = create_pipeline_definition(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
            tasks=tasks,
        )
        orchestrator.register_pipeline(pipeline)

        run = orchestrator.create_run("test_pipeline")
        result = orchestrator.execute_run(run.run_id)

        assert result.status == PipelineStatus.COMPLETED
        orchestrator._executor.stop()


class TestExperimentTracker:
    """Tests for Experiment Tracker module."""

    def test_run_status_enum(self):
        """Test RunStatus enum values."""
        from src.core.vision import RunStatus

        assert RunStatus.CREATED.value == "created"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"

    def test_metric_goal_enum(self):
        """Test MetricGoal enum values."""
        from src.core.vision import MetricGoal

        assert MetricGoal.MAXIMIZE.value == "maximize"
        assert MetricGoal.MINIMIZE.value == "minimize"

    def test_artifact_type_enum(self):
        """Test ArtifactType enum values."""
        from src.core.vision import ArtifactType

        assert ArtifactType.MODEL.value == "model"
        assert ArtifactType.DATASET.value == "dataset"
        assert ArtifactType.PLOT.value == "plot"
        assert ArtifactType.CHECKPOINT.value == "checkpoint"

    def test_experiment_creation(self):
        """Test Experiment dataclass."""
        from src.core.vision import create_experiment

        experiment = create_experiment(
            name="Test Experiment",
            description="A test experiment",
            tags=["test", "ml"],
        )
        assert experiment.name == "Test Experiment"
        assert "test" in experiment.tags

    def test_experiment_run_creation(self):
        """Test Run dataclass."""
        from src.core.vision import RunStatus, create_experiment_run

        run = create_experiment_run(
            run_id="run1",
            experiment_id="exp1",
            name="Run 1",
            status=RunStatus.CREATED,
        )
        assert run.run_id == "run1"
        assert run.status == RunStatus.CREATED

    def test_metric_creation(self):
        """Test Metric dataclass."""
        from src.core.vision import create_experiment_metric

        metric = create_experiment_metric(
            key="accuracy",
            value=0.95,
            run_id="run1",
            step=10,
        )
        assert metric.key == "accuracy"
        assert metric.value == 0.95
        assert metric.step == 10

    def test_create_experiment_tracker(self):
        """Test create_experiment_tracker factory."""
        from src.core.vision import create_experiment_tracker

        tracker = create_experiment_tracker()
        assert tracker is not None

    def test_experiment_tracker_operations(self):
        """Test ExperimentTracker CRUD operations."""
        from src.core.vision import create_experiment_tracker

        tracker = create_experiment_tracker()

        # Create experiment
        experiment = tracker.create_experiment(
            name="Test Experiment",
            description="Testing",
        )
        assert experiment is not None

        # Start run and log
        with tracker.start_run(experiment.experiment_id, "Test Run") as run_context:
            run_context.log_param("learning_rate", 0.01)
            run_context.log_metric("loss", 0.5, step=0)
            run_context.log_metric("loss", 0.3, step=1)

        # Verify run was created
        runs = tracker.list_runs(experiment.experiment_id)
        assert len(runs) == 1

    def test_run_comparison(self):
        """Test comparing runs."""
        from src.core.vision import MetricGoal, create_experiment_tracker

        tracker = create_experiment_tracker()
        experiment = tracker.create_experiment(name="Comparison Test")

        # Create multiple runs
        run_ids = []
        for i in range(3):
            with tracker.start_run(experiment.experiment_id, f"Run {i}") as ctx:
                ctx.log_param("run_index", i)
                ctx.log_metric("accuracy", 0.8 + i * 0.05)
                run_ids.append(ctx.run_id)

        # Compare runs
        result = tracker.compare_runs(
            run_ids,
            metric_key="accuracy",
            goal=MetricGoal.MAXIMIZE,
        )
        assert result is not None
        assert len(result.run_ids) == 3


class TestPhase18Integration:
    """Integration tests for Phase 18 components."""

    def test_all_phase18_exports_available(self):
        """Test that all Phase 18 exports are available."""
        from src.core.vision import (  # AutoML; Feature Store; Model Registry; Pipeline; Experiment
            AutoMLEngine,
            DAGBuilder,
            DeploymentStrategy,
            ExecutionMode,
            ExperimentTracker,
            FeatureDefinition,
            FeatureRegistry,
            FeatureStore,
            FeatureType,
            HyperparameterSpace,
            MetricGoal,
            ModelDeployer,
            ModelRegistry,
            ModelStage,
            OptimizationObjective,
            PipelineOrchestrator,
            PipelineStatus,
            RunStatus,
            SearchStrategy,
        )

        # Verify all imports worked
        assert AutoMLEngine is not None
        assert FeatureRegistry is not None
        assert ModelRegistry is not None
        assert PipelineOrchestrator is not None
        assert ExperimentTracker is not None

    def test_factory_functions_available(self):
        """Test that all factory functions are available."""
        from src.core.vision import (  # AutoML factories; Feature Store factories; Model Registry factories; Pipeline factories; Experiment factories
            create_automl_engine,
            create_dag_builder,
            create_deployment_config,
            create_experiment,
            create_experiment_tracker,
            create_feature_definition,
            create_feature_registry,
            create_feature_store,
            create_hyperparameter_space,
            create_model_registry,
            create_model_selector,
            create_model_version,
            create_nas,
            create_pipeline_definition,
            create_pipeline_orchestrator,
            create_search_config,
            create_task_definition,
        )

        assert callable(create_automl_engine)
        assert callable(create_feature_registry)
        assert callable(create_model_registry)
        assert callable(create_pipeline_orchestrator)
        assert callable(create_experiment_tracker)

    def test_vision_providers_available(self):
        """Test that all Vision Provider factory functions are available."""
        from src.core.vision import (
            create_automl_provider,
            create_experiment_tracker_provider,
            create_feature_store_provider,
            create_model_registry_provider,
            create_pipeline_orchestrator_provider,
        )

        # Test factory functions exist and are callable
        assert callable(create_automl_provider)
        assert callable(create_feature_store_provider)
        assert callable(create_model_registry_provider)
        assert callable(create_pipeline_orchestrator_provider)
        assert callable(create_experiment_tracker_provider)
