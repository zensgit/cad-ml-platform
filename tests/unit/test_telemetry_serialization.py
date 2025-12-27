import pytest

from src.core.twin.connectivity import TelemetryFrame


def test_telemetry_roundtrip_serialization():
    frame = TelemetryFrame(
        timestamp=1730000000.0,
        device_id="dev-123",
        sensors={"temp": 36.5},
        metrics={"load": 0.9},
        status={"ok": True},
    )
    data = frame.to_bytes()
    restored = TelemetryFrame.from_bytes(data)
    assert restored == frame


def test_telemetry_rejects_empty_payload():
    with pytest.raises(ValueError):
        TelemetryFrame.from_bytes(b"")
