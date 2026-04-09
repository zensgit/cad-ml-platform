"""Integration tests for newly enabled hybrid classifier features.

Validates that graph2d, history_sequence, rejection, and distillation
branches work correctly after being enabled in hybrid_classifier.yaml.
"""

import pytest
import yaml
from pathlib import Path

CONFIG_PATH = Path("config/hybrid_classifier.yaml")


@pytest.fixture
def hybrid_config():
    """Load the hybrid classifier configuration."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class TestEnabledFeatures:
    """Verify all previously disabled features are now enabled in config."""

    def test_graph2d_enabled(self, hybrid_config):
        assert hybrid_config["graph2d"]["enabled"] is True

    def test_history_sequence_enabled(self, hybrid_config):
        assert hybrid_config["history_sequence"]["enabled"] is True

    def test_rejection_enabled(self, hybrid_config):
        assert hybrid_config["rejection"]["enabled"] is True

    def test_distillation_enabled(self, hybrid_config):
        assert hybrid_config["distillation"]["enabled"] is True

    def test_filename_still_enabled(self, hybrid_config):
        assert hybrid_config["filename"]["enabled"] is True

    def test_titleblock_still_enabled(self, hybrid_config):
        assert hybrid_config["titleblock"]["enabled"] is True

    def test_process_still_enabled(self, hybrid_config):
        assert hybrid_config["process"]["enabled"] is True


class TestGraph2DConfig:
    """Validate graph2d branch configuration."""

    def test_min_confidence_reasonable(self, hybrid_config):
        conf = hybrid_config["graph2d"]["min_confidence"]
        assert 0.0 < conf < 1.0, f"min_confidence {conf} out of range"

    def test_fusion_weight_positive(self, hybrid_config):
        weight = hybrid_config["graph2d"]["fusion_weight"]
        assert weight > 0, "fusion_weight must be positive"

    def test_drawing_type_labels_present(self, hybrid_config):
        labels = hybrid_config["graph2d"].get("drawing_type_labels", [])
        assert len(labels) > 0, "drawing_type_labels should not be empty"


class TestHistorySequenceConfig:
    """Validate history sequence branch configuration."""

    def test_min_confidence_reasonable(self, hybrid_config):
        conf = hybrid_config["history_sequence"]["min_confidence"]
        assert 0.0 < conf < 1.0

    def test_fusion_weight_positive(self, hybrid_config):
        weight = hybrid_config["history_sequence"]["fusion_weight"]
        assert weight > 0

    def test_prototypes_path_configured(self, hybrid_config):
        path = hybrid_config["history_sequence"].get("prototypes_path", "")
        assert path, "prototypes_path should be configured"


class TestRejectionConfig:
    """Validate rejection mechanism configuration."""

    def test_min_confidence_threshold(self, hybrid_config):
        conf = hybrid_config["rejection"]["min_confidence"]
        assert 0.0 < conf < 1.0, f"rejection min_confidence {conf} out of range"

    def test_rejection_threshold_below_classifier_thresholds(self, hybrid_config):
        """Rejection threshold should be lower than classifier thresholds."""
        rejection_conf = hybrid_config["rejection"]["min_confidence"]
        filename_conf = hybrid_config["filename"]["min_confidence"]
        assert rejection_conf < filename_conf, (
            f"Rejection threshold ({rejection_conf}) should be below "
            f"filename threshold ({filename_conf})"
        )


class TestDistillationConfig:
    """Validate distillation configuration."""

    def test_alpha_in_range(self, hybrid_config):
        alpha = hybrid_config["distillation"]["alpha"]
        assert 0.0 <= alpha <= 1.0, f"alpha {alpha} out of [0, 1] range"

    def test_temperature_positive(self, hybrid_config):
        temp = hybrid_config["distillation"]["temperature"]
        assert temp > 0, "temperature must be positive"

    def test_teacher_type_valid(self, hybrid_config):
        valid_types = {"hybrid", "graph2d", "ensemble", "single"}
        teacher = hybrid_config["distillation"]["teacher_type"]
        assert teacher in valid_types, f"Unknown teacher_type: {teacher}"


class TestFusionWeightsConsistency:
    """Validate that fusion weights are consistent across all enabled branches."""

    def test_all_weights_positive(self, hybrid_config):
        branches = ["filename", "graph2d", "titleblock", "process", "history_sequence"]
        for branch in branches:
            if hybrid_config[branch]["enabled"]:
                weight = hybrid_config[branch]["fusion_weight"]
                assert weight > 0, f"{branch} fusion_weight should be positive"

    def test_no_single_branch_dominates(self, hybrid_config):
        """No single branch should have weight > 0.8."""
        branches = ["filename", "graph2d", "titleblock", "process", "history_sequence"]
        for branch in branches:
            if hybrid_config[branch]["enabled"]:
                weight = hybrid_config[branch]["fusion_weight"]
                assert weight <= 0.8, f"{branch} weight {weight} too dominant"


class TestMultimodalConfig:
    """Validate multimodal fusion settings."""

    def test_multimodal_enabled(self, hybrid_config):
        assert hybrid_config["multimodal"]["enabled"] is True

    def test_weights_sum_to_one(self, hybrid_config):
        mm = hybrid_config["multimodal"]
        total = mm["geometry_weight"] + mm["text_weight"] + mm["rule_weight"]
        assert abs(total - 1.0) < 0.01, f"Multimodal weights sum to {total}, expected 1.0"

    def test_gate_type_valid(self, hybrid_config):
        valid_gates = {"weighted", "attention", "learned", "weighted_average"}
        gate = hybrid_config["multimodal"]["gate_type"]
        assert gate in valid_gates, f"Unknown gate_type: {gate}"


class TestClassBalanceConfig:
    """Validate class balance settings for training."""

    def test_focal_loss_params(self, hybrid_config):
        cb = hybrid_config["class_balance"]
        assert cb["strategy"] == "focal"
        assert 0.0 < cb["focal_alpha"] < 1.0
        assert cb["focal_gamma"] > 0


class TestFeatureExtractorV4:
    """Validate V4 feature extraction functions exist and work."""

    def test_shape_entropy_import(self):
        from src.core.feature_extractor import compute_shape_entropy

        assert callable(compute_shape_entropy)

    def test_shape_entropy_empty(self):
        from src.core.feature_extractor import compute_shape_entropy

        assert compute_shape_entropy({}) == 0.0

    def test_shape_entropy_single_type(self):
        from src.core.feature_extractor import compute_shape_entropy

        assert compute_shape_entropy({"LINE": 100}) == 0.0

    def test_shape_entropy_uniform(self):
        from src.core.feature_extractor import compute_shape_entropy

        result = compute_shape_entropy({"LINE": 50, "CIRCLE": 50, "ARC": 50})
        assert 0.9 < result <= 1.0, f"Uniform distribution entropy {result} should be near 1.0"

    def test_shape_entropy_skewed(self):
        from src.core.feature_extractor import compute_shape_entropy

        result = compute_shape_entropy({"LINE": 1000, "CIRCLE": 1, "ARC": 1})
        assert result < 0.3, f"Skewed distribution entropy {result} should be low"

    def test_surface_count_import(self):
        from src.core.feature_extractor import compute_surface_count

        assert callable(compute_surface_count)


class TestSecurityFeatures:
    """Validate security features are properly configured."""

    def test_opcode_mode_configured(self):
        from config.feature_flags import OPCODE_MODE

        assert OPCODE_MODE in {"audit", "blocklist", "whitelist"}

    def test_opcode_scan_enabled(self):
        from config.feature_flags import OPCODE_SCAN_ENABLED

        assert OPCODE_SCAN_ENABLED is True

    def test_opcode_audit_snapshot_callable(self):
        from src.ml.classifier import get_opcode_audit_snapshot

        snapshot = get_opcode_audit_snapshot()
        assert "opcodes" in snapshot
        assert "counts" in snapshot
        assert "total_samples" in snapshot


class TestLevel3Rollback:
    """Validate Level 3 rollback infrastructure exists."""

    def test_prev3_variables_exist(self):
        from src.ml import classifier

        assert hasattr(classifier, "_MODEL_PREV3")
        assert hasattr(classifier, "_MODEL_PREV3_HASH")
        assert hasattr(classifier, "_MODEL_PREV3_VERSION")
        assert hasattr(classifier, "_MODEL_PREV3_PATH")

    def test_reload_model_callable(self):
        from src.ml.classifier import reload_model

        assert callable(reload_model)
