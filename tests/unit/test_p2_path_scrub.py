"""P2 path-scrub regression tests (L3 model-activation membrane design lock).

Design lock: NO filesystem paths in health payloads, telemetry, or logs. These
tests pin two round-2 NO-GO leaks closed:

1. The health payload must not surface any filesystem-path-looking string for the
   model/config artifacts under ``config.ml`` (previously ``graph2d_model_path`` /
   ``hybrid_config_path`` / ``graph2d_temperature_calibration_path`` and the
   ``graph2d_temperature_source`` calibration-file branch leaked store locations).
2. The Graph2D load-failure log must not contain ``self.model_path``.
"""

from __future__ import annotations

import logging

import pytest


def _iter_strings(obj):
    """Yield every string value nested under a JSON-like structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)
    elif isinstance(obj, str):
        yield obj


class TestHealthPayloadPathScrub:
    """The health payload must emit model NAMES + present/degraded state only."""

    def test_no_model_path_keys_in_classification(self, monkeypatch, tmp_path):
        """The removed ``*_path`` keys must be gone; name-only keys present.

        A REAL calibration file is written and selected (with GRAPH2D_TEMPERATURE
        cleared so it does not short-circuit to the ``"env"`` source). This
        genuinely drives ``load_graph2d_temperature_settings`` down the branch
        that returns ``str(<path>)`` (graph2d_temperature.py:95) — the exact
        leak vector the source-scrub closes. A non-existent path would return
        ``(1.0, None)`` and never exercise the scrub.
        """
        cal = tmp_path / "graph2d_calibration.json"
        cal.write_text('{"temperature": 1.5}')
        monkeypatch.delenv("GRAPH2D_TEMPERATURE", raising=False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH", str(cal))

        from src.api.health_utils import build_health_payload

        classification = build_health_payload(metrics_enabled_override=False)[
            "config"
        ]["ml"]["classification"]

        # Path-bearing keys are dropped.
        assert "graph2d_model_path" not in classification
        assert "hybrid_config_path" not in classification
        assert "graph2d_temperature_calibration_path" not in classification

        # Path-free NAME replacements are present and carry no separator.
        assert "graph2d_model_name" in classification
        assert "/" not in classification["graph2d_model_name"]
        assert classification["graph2d_model_name"].endswith(".pth")
        assert "/" not in classification["hybrid_config_name"]
        # calibration name is a basename (no directory), never the full path.
        cal_name = classification["graph2d_temperature_calibration_name"]
        assert cal_name == "graph2d_calibration.json"
        assert "/" not in cal_name
        # The calibration-file source path must collapse to a fixed token, never
        # the resolved path (str(cal) contains the tmp_path directory separators).
        assert classification["graph2d_temperature_source"] == "calibration"

    def test_no_filesystem_path_string_under_config_ml(self, monkeypatch, tmp_path):
        """No string value anywhere under ``config.ml`` may look like a path.

        A directory separator ("/") is the discriminating signal — model NAMES,
        hashes, statuses and degraded-reason tokens carry none; a leaked artifact
        path ("models/...pth", "config/...yaml", the resolved calibration path)
        does. The real calibration file forces the source-path branch so a
        reverted scrub would surface ``str(<tmp_path>/...)`` here and fail.
        """
        cal = tmp_path / "graph2d_calibration.json"
        cal.write_text('{"temperature": 1.5}')
        monkeypatch.delenv("GRAPH2D_TEMPERATURE", raising=False)
        monkeypatch.setenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH", str(cal))
        from src.api.health_utils import build_health_payload

        ml = build_health_payload(metrics_enabled_override=False)["config"]["ml"]

        leaking = [s for s in _iter_strings(ml) if "/" in s]
        assert leaking == [], f"filesystem-path-looking values leaked into config.ml: {leaking}"


class TestGraph2DLoadFailureLogPathScrub:
    """The Graph2D load-failure log must not contain the model path."""

    def test_load_failure_log_excludes_model_path(self, caplog):
        pytest.importorskip("torch")
        from src.ml.vision_2d import Graph2DClassifier

        sentinel = "/SENTINEL/leak/graph2d_model.pth"

        # activate_file returning bytes that torch cannot load drives the
        # __init__ except branch (line 76). A None return is a graceful degrade
        # and never hits the failure log, so we feed invalid bytes.
        with caplog.at_level(logging.WARNING, logger="src.ml.vision_2d"):
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr(
                    "src.ml.vision_2d.activate_file",
                    lambda *a, **k: b"not-a-torch-checkpoint",
                )
                clf = Graph2DClassifier(model_path=sentinel)

        # Load failed (invalid bytes) so the warning fired...
        assert clf.model is None
        assert clf._load_error is not None
        assert any(
            "Failed to load Graph2D model" in r.getMessage() for r in caplog.records
        ), "expected the Graph2D load-failure warning to fire"
        # ...but the sentinel model path must NOT appear anywhere in the log text.
        assert sentinel not in caplog.text
        assert "/SENTINEL/leak" not in caplog.text
