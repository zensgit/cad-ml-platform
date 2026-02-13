from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


def test_ensemble_soft_voting_averages_prob_vectors(monkeypatch) -> None:
    import src.ml.vision_2d as vision_2d

    # Force-enable the ensemble path without requiring torch in this unit test.
    monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

    class _StubClf:
        def __init__(self, label_map, probs):  # noqa: ANN001
            self.label_map = dict(label_map)
            self._probs = list(probs)

        def _predict_probs(self, data, file_name):  # noqa: ANN001, ANN201
            return {"status": "ok", "probs": list(self._probs)}

    label_map = {"A": 0, "B": 1, "C": 2}
    clf1 = _StubClf(label_map, [0.6, 0.3, 0.1])
    clf2 = _StubClf(label_map, [0.4, 0.5, 0.1])

    ens = vision_2d.EnsembleGraph2DClassifier(model_paths=[], voting="soft")
    ens.classifiers = [clf1, clf2]
    ens._loaded = True

    out = ens.predict_from_bytes(b"0", "x.dxf")
    assert out.get("status") == "ok"
    assert out.get("voting") == "soft"
    assert out.get("label_map_mismatch") is False
    assert out.get("label_map_size") == 3
    assert out.get("label") == "A"
    assert abs(float(out.get("confidence") or 0.0) - 0.5) < 1e-9
    assert abs(float(out.get("top2_confidence") or 0.0) - 0.4) < 1e-9
    assert abs(float(out.get("margin") or 0.0) - 0.1) < 1e-9
    assert out.get("ensemble_size") == 2


def test_ensemble_soft_voting_falls_back_on_label_map_mismatch(monkeypatch) -> None:
    import src.ml.vision_2d as vision_2d

    monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

    class _StubClf:
        def __init__(self, label_map, probs):  # noqa: ANN001
            self.label_map = dict(label_map)
            self._probs = list(probs)

        def _predict_probs(self, data, file_name):  # noqa: ANN001, ANN201
            return {"status": "ok", "probs": list(self._probs)}

    clf1 = _StubClf({"A": 0, "B": 1}, [0.9, 0.1])
    clf2 = _StubClf({"X": 0, "Y": 1}, [0.9, 0.1])

    ens = vision_2d.EnsembleGraph2DClassifier(model_paths=[], voting="soft")
    ens.classifiers = [clf1, clf2]
    ens._loaded = True

    out = ens.predict_from_bytes(b"0", "x.dxf")
    assert out.get("status") == "ok"
    assert str(out.get("voting") or "").startswith("hard_fallback")
    assert out.get("label_map_mismatch") is True
    assert out.get("ensemble_size") == 2


class TestGraph2DClassifier:
    """Tests for Graph2DClassifier class."""

    def test_init_no_torch(self, monkeypatch) -> None:
        """Test Graph2DClassifier init when torch is not available."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)

        clf = vision_2d.Graph2DClassifier(model_path="/nonexistent/path.pth")

        assert clf._loaded is False
        assert clf.model is None

    def test_init_no_model_file(self, monkeypatch) -> None:
        """Test Graph2DClassifier init when model file doesn't exist."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

        clf = vision_2d.Graph2DClassifier(model_path="/nonexistent/path.pth")

        assert clf._loaded is False

    def test_init_model_load_failure_is_best_effort(self, monkeypatch) -> None:
        """Model loading errors should not crash Graph2DClassifier init."""
        import src.ml.vision_2d as vision_2d

        class _StubTorch:
            @staticmethod
            def load(*_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
                raise RuntimeError("boom")

        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)
        monkeypatch.setattr(vision_2d, "torch", _StubTorch)
        monkeypatch.setattr(vision_2d.os.path, "exists", lambda _p: True)

        clf = vision_2d.Graph2DClassifier(model_path="/tmp/fake_model.pth")
        assert clf._loaded is False

    def test_predict_probs_model_unavailable(self, monkeypatch) -> None:
        """Test _predict_probs when model is not loaded."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)

        clf = vision_2d.Graph2DClassifier()
        result = clf._predict_probs(b"data", "test.dxf")

        assert result == {"status": "model_unavailable"}

    def test_predict_probs_empty_input(self, monkeypatch) -> None:
        """Test _predict_probs with empty input."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

        clf = vision_2d.Graph2DClassifier()
        clf._loaded = True
        clf.model = MagicMock()

        result = clf._predict_probs(b"", "test.dxf")

        assert result == {"status": "empty_input"}

    def test_load_temperature_from_env(self, monkeypatch) -> None:
        """Test loading temperature from environment variable."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE", "1.5")

        clf = vision_2d.Graph2DClassifier()

        assert clf.temperature == 1.5
        assert clf.temperature_source == "env"

    def test_load_temperature_invalid_env(self, monkeypatch) -> None:
        """Test loading invalid temperature from environment variable."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE", "invalid")

        clf = vision_2d.Graph2DClassifier()

        assert clf.temperature == 1.0  # Default
        assert clf.temperature_source is None

    def test_load_temperature_negative_env(self, monkeypatch) -> None:
        """Test loading negative temperature from environment variable."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE", "-1.0")

        clf = vision_2d.Graph2DClassifier()

        assert clf.temperature == 1.0  # Default
        assert clf.temperature_source is None

    def test_load_temperature_from_calibration_file(self, monkeypatch, tmp_path) -> None:
        """Test loading temperature from calibration file."""
        import json

        import src.ml.vision_2d as vision_2d

        # Create calibration file
        cal_file = tmp_path / "calibration.json"
        cal_file.write_text(json.dumps({"temperature": 2.5}))

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)
        monkeypatch.delenv("GRAPH2D_TEMPERATURE", raising=False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH", str(cal_file))

        clf = vision_2d.Graph2DClassifier()

        assert clf.temperature == 2.5
        assert str(cal_file) in clf.temperature_source

    def test_load_temperature_calibration_file_not_found(self, monkeypatch) -> None:
        """Test loading temperature from nonexistent calibration file."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)
        monkeypatch.delenv("GRAPH2D_TEMPERATURE", raising=False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH", "/nonexistent/cal.json")

        clf = vision_2d.Graph2DClassifier()

        assert clf.temperature == 1.0
        assert clf.temperature_source is None

    def test_load_temperature_calibration_invalid_json(self, monkeypatch, tmp_path) -> None:
        """Test loading temperature from invalid calibration file."""
        import src.ml.vision_2d as vision_2d

        # Create invalid calibration file
        cal_file = tmp_path / "calibration.json"
        cal_file.write_text("not valid json")

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)
        monkeypatch.delenv("GRAPH2D_TEMPERATURE", raising=False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH", str(cal_file))

        clf = vision_2d.Graph2DClassifier()

        assert clf.temperature == 1.0
        assert clf.temperature_source is None

    def test_load_temperature_calibration_invalid_value(self, monkeypatch, tmp_path) -> None:
        """Test loading temperature with invalid value in calibration file."""
        import json

        import src.ml.vision_2d as vision_2d

        # Create calibration file with invalid temperature
        cal_file = tmp_path / "calibration.json"
        cal_file.write_text(json.dumps({"temperature": "not_a_number"}))

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)
        monkeypatch.delenv("GRAPH2D_TEMPERATURE", raising=False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH", str(cal_file))

        clf = vision_2d.Graph2DClassifier()

        assert clf.temperature == 1.0
        assert clf.temperature_source is None

    def test_load_temperature_calibration_negative_value(self, monkeypatch, tmp_path) -> None:
        """Test loading temperature with negative value in calibration file."""
        import json

        import src.ml.vision_2d as vision_2d

        # Create calibration file with negative temperature
        cal_file = tmp_path / "calibration.json"
        cal_file.write_text(json.dumps({"temperature": -0.5}))

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)
        monkeypatch.delenv("GRAPH2D_TEMPERATURE", raising=False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH", str(cal_file))

        clf = vision_2d.Graph2DClassifier()

        assert clf.temperature == 1.0
        assert clf.temperature_source is None


class TestEnsembleGraph2DClassifier:
    """Tests for EnsembleGraph2DClassifier class."""

    def test_init_no_torch(self, monkeypatch) -> None:
        """Test EnsembleGraph2DClassifier init when torch is not available."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)

        ens = vision_2d.EnsembleGraph2DClassifier()

        assert ens._loaded is False
        assert ens.classifiers == []

    def test_init_from_env_paths(self, monkeypatch) -> None:
        """Test EnsembleGraph2DClassifier init with paths from env."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)
        monkeypatch.setenv("GRAPH2D_ENSEMBLE_MODELS", "model1.pth, model2.pth")

        ens = vision_2d.EnsembleGraph2DClassifier()

        assert ens.model_paths == ["model1.pth", "model2.pth"]

    def test_predict_no_torch(self, monkeypatch) -> None:
        """Test predict when torch is not available."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", False)

        ens = vision_2d.EnsembleGraph2DClassifier()
        result = ens.predict_from_bytes(b"data", "test.dxf")

        assert result == {"status": "model_unavailable"}

    def test_predict_empty_input(self, monkeypatch) -> None:
        """Test predict with empty input."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

        ens = vision_2d.EnsembleGraph2DClassifier(model_paths=[])
        ens._loaded = True

        result = ens.predict_from_bytes(b"", "test.dxf")

        assert result == {"status": "empty_input"}

    def test_predict_all_models_failed(self, monkeypatch) -> None:
        """Test predict when all models fail."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

        class _FailingClf:
            label_map = {}

        ens = vision_2d.EnsembleGraph2DClassifier(model_paths=[])
        ens.classifiers = [_FailingClf()]
        ens._loaded = True

        result = ens.predict_from_bytes(b"data", "test.dxf")

        assert result == {"status": "all_models_failed"}

    def test_predict_single_model(self, monkeypatch) -> None:
        """Test predict with single model returns that model's prediction."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

        class _StubClf:
            label_map = {"A": 0, "B": 1}

            def _predict_probs(self, data, file_name):
                return {"status": "ok", "probs": [0.8, 0.2]}

        ens = vision_2d.EnsembleGraph2DClassifier(model_paths=[])
        ens.classifiers = [_StubClf()]
        ens._loaded = True

        result = ens.predict_from_bytes(b"data", "test.dxf")

        assert result.get("status") == "ok"
        assert result.get("label") == "A"

    def test_predict_hard_voting(self, monkeypatch) -> None:
        """Test predict with hard voting strategy."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

        class _StubClf:
            def __init__(self, label_map, probs):
                self.label_map = dict(label_map)
                self._probs = list(probs)

            def _predict_probs(self, data, file_name):
                return {"status": "ok", "probs": self._probs}

        # Two vote for A, one for B
        clf1 = _StubClf({"A": 0, "B": 1}, [0.9, 0.1])
        clf2 = _StubClf({"A": 0, "B": 1}, [0.8, 0.2])
        clf3 = _StubClf({"A": 0, "B": 1}, [0.4, 0.6])

        ens = vision_2d.EnsembleGraph2DClassifier(model_paths=[], voting="hard")
        ens.classifiers = [clf1, clf2, clf3]
        ens._loaded = True

        result = ens.predict_from_bytes(b"data", "test.dxf")

        assert result.get("status") == "ok"
        assert result.get("voting") == "hard"
        assert result.get("label") == "A"

    def test_predict_fallback_to_public_interface(self, monkeypatch) -> None:
        """Test predict falls back to predict_from_bytes when _predict_probs fails."""
        import src.ml.vision_2d as vision_2d

        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

        class _StubClf:
            label_map = {"A": 0, "B": 1}

            def _predict_probs(self, data, file_name):
                return {"status": "error"}  # Fails

            def predict_from_bytes(self, data, file_name):
                return {"status": "ok", "label": "A", "confidence": 0.9}

        ens = vision_2d.EnsembleGraph2DClassifier(model_paths=[])
        ens.classifiers = [_StubClf()]
        ens._loaded = True

        result = ens.predict_from_bytes(b"data", "test.dxf")

        assert result.get("status") == "ok"
        assert result.get("label") == "A"

    def test_top2_empty_probs(self, monkeypatch) -> None:
        """Test _top2 with empty probs list."""
        import src.ml.vision_2d as vision_2d

        # Access the inner _top2 function through ensemble's predict
        monkeypatch.setattr(vision_2d, "HAS_TORCH", True)

        class _StubClf:
            label_map = {"A": 0}

            def _predict_probs(self, data, file_name):
                return {"status": "ok", "probs": []}  # Empty probs

        ens = vision_2d.EnsembleGraph2DClassifier(model_paths=[])
        ens.classifiers = [_StubClf()]
        ens._loaded = True

        # This should handle empty probs gracefully
        result = ens.predict_from_bytes(b"data", "test.dxf")
        # With empty probs, the _top2 function handles it
        # The result may be 'ok' with None values or 'all_models_failed'
        # depending on how alignment works
        assert result.get("status") in ["ok", "all_models_failed"]


class TestGetClassifiers:
    """Tests for get_2d_classifier and get_ensemble_2d_classifier functions."""

    def test_get_2d_classifier(self, monkeypatch) -> None:
        """Test get_2d_classifier returns singleton."""
        import src.ml.vision_2d as vision_2d

        clf = vision_2d.get_2d_classifier()

        assert clf is vision_2d._graph2d

    def test_get_ensemble_2d_classifier(self, monkeypatch) -> None:
        """Test get_ensemble_2d_classifier creates singleton."""
        import src.ml.vision_2d as vision_2d

        # Reset singleton
        vision_2d._ensemble_graph2d = None

        ens = vision_2d.get_ensemble_2d_classifier()

        assert ens is vision_2d._ensemble_graph2d
        assert vision_2d._ensemble_graph2d is not None

    def test_get_ensemble_2d_classifier_returns_same_instance(self, monkeypatch) -> None:
        """Test get_ensemble_2d_classifier returns same instance."""
        import src.ml.vision_2d as vision_2d

        # Reset singleton
        vision_2d._ensemble_graph2d = None

        ens1 = vision_2d.get_ensemble_2d_classifier()
        ens2 = vision_2d.get_ensemble_2d_classifier()

        assert ens1 is ens2
