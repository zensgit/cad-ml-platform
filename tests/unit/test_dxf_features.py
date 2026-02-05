import numpy as np
import pytest


def test_extract_features_v6_dimensions(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    from src.utils.dxf_features import extract_features_v6

    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_circle((5, 5), 2)

    dxf_path = tmp_path / "sample.dxf"
    doc.saveas(dxf_path)

    features = extract_features_v6(str(dxf_path))
    assert features is not None
    assert features.shape == (48,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert np.any(features != 0)
