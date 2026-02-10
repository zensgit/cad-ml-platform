from __future__ import annotations


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

