import asyncio
import hashlib

from src.core.feature_pipeline import run_feature_pipeline


class _DummyFeatureCache:
    def __init__(self, initial=None):
        self._store = dict(initial or {})

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def size(self):
        return len(self._store)


class _DummyGeometryCache:
    def __init__(self, cached=None):
        self.cached = cached
        self.last_key = None

    def generate_key(self, content, version):
        self.last_key = f"{version}:{len(content)}"
        return self.last_key

    def get(self, key):
        return self.cached

    def set(self, key, value):
        self.cached = value


class _DummyFeatureExtractor:
    def __init__(self):
        self.extract_calls = 0
        self.rehydrate_calls = 0

    async def extract(self, doc, brep_features=None):
        self.extract_calls += 1
        return {"geometric": [1.0, 2.0], "semantic": [3.0]}

    def flatten(self, features):
        return list(features["geometric"]) + list(features["semantic"])

    def rehydrate(self, vector, version="v1"):
        self.rehydrate_calls += 1
        return {"geometric": list(vector[:2]), "semantic": list(vector[2:])}

    def slots(self, version):
        return {"geometric": 2, "semantic": 1}


def test_run_feature_pipeline_uses_2d_cache_hit_and_rehydrates(monkeypatch):
    monkeypatch.setenv("FEATURE_VERSION", "v_test")
    cached_vector = [9.0, 8.0, 7.0]
    cache_key = f"{hashlib.sha256(b'abc').hexdigest()}:v_test:layout_v2"
    feature_cache = _DummyFeatureCache(
        {cache_key: cached_vector}
    )
    extractor = _DummyFeatureExtractor()

    result = asyncio.run(
        run_feature_pipeline(
            extract_features=True,
            file_format="dxf",
            file_name="sample.dxf",
            content=b"abc",
            doc=object(),
            started_at=0.0,
            stage_times={"parse": 0.2},
            feature_extractor_factory=lambda: extractor,
            feature_cache_factory=lambda: feature_cache,
        )
    )

    assert result["features"] == {"geometric": [9.0, 8.0], "semantic": [7.0]}
    assert result["features_3d"] == {}
    assert extractor.extract_calls == 0
    assert extractor.rehydrate_calls == 1
    assert result["results_patch"]["features"]["combined"] == cached_vector
    assert result["results_patch"]["features"]["cache_hit"] is True
    assert result["results_patch"]["features"]["feature_version"] == "v_test"
    assert result["features_stage_duration"] is not None


def test_run_feature_pipeline_extracts_3d_features_and_embedding_result():
    feature_cache = _DummyFeatureCache()
    geometry_cache = _DummyGeometryCache()
    extractor = _DummyFeatureExtractor()

    class _DummyGeometryEngine:
        def load_step(self, content, file_name=""):
            assert file_name == "sample.step"
            return object()

        def extract_brep_features(self, shape):
            return {"surface_area": 12.0}

        def extract_dfm_features(self, shape):
            return {"thin_walls_detected": False}

    class _DummyEncoder:
        # Verified model output: declares a non-degraded encode so the pipeline
        # labels the embedding as real (mirrors a loaded UVNetEncoder on the
        # graph-data path).
        last_encode_degraded = False

        def encode(self, features_3d):
            return [0.1, 0.2, 0.3]

    result = asyncio.run(
        run_feature_pipeline(
            extract_features=True,
            file_format="step",
            file_name="sample.step",
            content=b"step-content",
            doc=object(),
            started_at=0.0,
            stage_times={"parse": 0.1},
            feature_extractor_factory=lambda: extractor,
            feature_cache_factory=lambda: feature_cache,
            geometry_cache_factory=lambda: geometry_cache,
            geometry_engine_factory=lambda: _DummyGeometryEngine(),
            encoder_3d_factory=lambda: _DummyEncoder(),
        )
    )

    assert result["features_3d"]["surface_area"] == 12.0
    assert result["features_3d"]["thin_walls_detected"] is False
    assert result["features_3d"]["embedding_vector"] == [0.1, 0.2, 0.3]
    assert result["results_patch"]["features_3d"]["surface_area"] == 12.0
    assert result["results_patch"]["features_3d"]["embedding_dim"] == 3
    assert result["features_3d_stage_duration"] is not None
    assert result["results_patch"]["features"]["cache_hit"] is False
    # Positive control: a verified encode is labeled real AND cached tagged verified.
    assert result["features_3d"]["embedding_degraded"] is False
    assert result["features_3d"]["embedding_provenance"] == "uvnet_model"
    assert result["results_patch"]["features_3d"]["embedding_degraded"] is False
    assert result["results_patch"]["features_3d"]["embedding_provenance"] == "uvnet_model"
    # Cached entry carries the verified marker.
    assert geometry_cache.cached is not None
    assert geometry_cache.cached["embedding_degraded"] is False


def _run_3d_pipeline_with_encoder(encoder, geometry_cache):
    """Drive run_feature_pipeline through the 3D branch with a given encoder."""
    feature_cache = _DummyFeatureCache()
    extractor = _DummyFeatureExtractor()

    class _DummyGeometryEngine:
        def load_step(self, content, file_name=""):
            return object()

        def extract_brep_features(self, shape):
            return {"surface_area": 12.0, "faces": 6}

        def extract_dfm_features(self, shape):
            return {"thin_walls_detected": False}

    return asyncio.run(
        run_feature_pipeline(
            extract_features=True,
            file_format="step",
            file_name="sample.step",
            content=b"step-content",
            doc=object(),
            started_at=0.0,
            stage_times={"parse": 0.1},
            feature_extractor_factory=lambda: extractor,
            feature_cache_factory=lambda: feature_cache,
            geometry_cache_factory=lambda: geometry_cache,
            geometry_engine_factory=lambda: _DummyGeometryEngine(),
            encoder_3d_factory=lambda: encoder,
        )
    )


def test_degraded_embedding_is_marked_and_not_cached_as_verified():
    """F4 discriminator: an unpinned/mock encode must not surface as verified.

    Mirrors ``UVNetEncoder`` under no pin (``_loaded`` False / legacy-dict path):
    ``encode`` returns a heuristic vector and flips ``last_encode_degraded`` True.
    The pipeline must (a) attach an explicit degrade/provenance marker to the
    result AND to the downstream ``results_patch``, and (b) never cache the mock
    as a verified embedding — it is cached TAGGED degraded so a later cache HIT
    (design lock: "a later read cannot mistake it for a verified embedding")
    still carries the marker.
    """

    class _DegradedEncoder:
        # Same contract as the real UVNetEncoder's mock/legacy-dict path.
        last_encode_degraded = True

        def encode(self, features_3d):
            return [0.0, 0.0, 0.0]

    geometry_cache = _DummyGeometryCache()
    result = _run_3d_pipeline_with_encoder(_DegradedEncoder(), geometry_cache)

    # (a) marker propagates with the result and to downstream/health.
    assert result["features_3d"]["embedding_degraded"] is True
    assert result["features_3d"]["embedding_provenance"] == "mock_heuristic"
    assert result["results_patch"]["features_3d"]["embedding_degraded"] is True
    assert result["results_patch"]["features_3d"]["embedding_provenance"] == "mock_heuristic"

    # (b) it IS cached, but tagged degraded — a later HIT cannot mistake it for
    # a verified embedding (the whole reason the marker lives inside the dict).
    assert geometry_cache.cached is not None
    assert geometry_cache.cached["embedding_degraded"] is True
    assert geometry_cache.cached["embedding_provenance"] == "mock_heuristic"


def test_absent_marker_defaults_to_degraded_fail_closed():
    """An encoder that does not expose ``last_encode_degraded`` is treated as
    degraded, never fail-open to "verified"."""

    class _MarkerlessEncoder:
        def encode(self, features_3d):
            return [0.5, 0.5, 0.5]

    geometry_cache = _DummyGeometryCache()
    result = _run_3d_pipeline_with_encoder(_MarkerlessEncoder(), geometry_cache)

    assert result["features_3d"]["embedding_degraded"] is True
    assert result["features_3d"]["embedding_provenance"] == "mock_heuristic"
    assert geometry_cache.cached["embedding_degraded"] is True


def test_cache_hit_preserves_degraded_marker():
    """A 3D cache HIT on a previously-degraded entry re-surfaces the marker,
    so a cached mock can never be re-read as a verified embedding."""
    degraded_cached = {
        "surface_area": 12.0,
        "faces": 6,
        "embedding_vector": [0.0, 0.0, 0.0],
        "embedding_degraded": True,
        "embedding_provenance": "mock_heuristic",
    }
    geometry_cache = _DummyGeometryCache(cached=degraded_cached)

    class _ShouldNotEncode:
        last_encode_degraded = False

        def encode(self, features_3d):  # pragma: no cover - must not run on a hit
            raise AssertionError("encode() must not run on a cache HIT")

    result = _run_3d_pipeline_with_encoder(_ShouldNotEncode(), geometry_cache)

    assert result["features_3d"]["embedding_degraded"] is True
    assert result["results_patch"]["features_3d"]["embedding_degraded"] is True
    assert result["results_patch"]["features_3d"]["embedding_provenance"] == "mock_heuristic"
