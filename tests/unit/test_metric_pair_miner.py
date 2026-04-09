"""Tests for scripts/mine_metric_pairs.py.

All tests use synthetic numpy feature vectors and temporary directories -- no
real DXF files or ezdxf are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.mine_metric_pairs import MetricPairMiner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_class_dirs(root: Path, classes: dict[str, int]) -> dict[str, list[str]]:
    """Create temp subdirectories with dummy .dxf files.

    Args:
        root: Parent directory.
        classes: ``{label: num_files}``

    Returns:
        ``{label: [file_path, ...]}``
    """
    result: dict[str, list[str]] = {}
    for label, count in classes.items():
        d = root / label
        d.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i in range(count):
            fp = d / f"sample_{i:04d}.dxf"
            fp.write_text("")  # empty placeholder
            paths.append(str(fp))
        result[label] = paths
    return result


def _synth_features(
    label_to_paths: dict[str, list[str]],
    dim: int = 48,
    *,
    rng: np.random.RandomState | None = None,
    class_centers: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Build a synthetic feature list (no DXF parsing).

    Each class gets a random cluster centre; per-sample noise is added so that
    intra-class similarity is higher than inter-class similarity.
    """
    rng = rng or np.random.RandomState(0)
    if class_centers is None:
        class_centers = {label: rng.randn(dim).astype(np.float32) * 5 for label in label_to_paths}

    items: list[dict] = []
    for label, paths in label_to_paths.items():
        centre = class_centers[label]
        for p in paths:
            noise = rng.randn(dim).astype(np.float32) * 0.3
            items.append({"path": p, "label": label, "features": centre + noise})
    return items


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScanDirectoryStructure:
    def test_basic_scan(self, tmp_path: Path):
        _make_class_dirs(tmp_path, {"classA": 3, "classB": 2})
        miner = MetricPairMiner(data_dir=str(tmp_path))
        scanned = miner.scan_labeled_files()
        assert set(scanned.keys()) == {"classA", "classB"}
        assert len(scanned["classA"]) == 3
        assert len(scanned["classB"]) == 2

    def test_manifest_takes_precedence(self, tmp_path: Path):
        _make_class_dirs(tmp_path, {"alpha": 2})
        manifest = [
            {"file": "alpha/sample_0000.dxf", "category": "ALPHA"},
            {"file": "alpha/sample_0001.dxf", "category": "ALPHA"},
        ]
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        miner = MetricPairMiner(data_dir=str(tmp_path))
        scanned = miner.scan_labeled_files()
        # Label comes from manifest category, not directory name
        assert "ALPHA" in scanned
        assert len(scanned["ALPHA"]) == 2


class TestMinePositivePairs:
    def test_same_class(self, tmp_path: Path):
        l2p = _make_class_dirs(tmp_path, {"X": 10, "Y": 10})
        feats = _synth_features(l2p)
        miner = MetricPairMiner(data_dir=str(tmp_path))
        pairs = miner.mine_positive_pairs(feats, pairs_per_class=5)
        assert len(pairs) > 0
        for p in pairs:
            assert p["anchor"] != p["positive"]
            assert p["label"] in {"X", "Y"}
            # Both paths must belong to the declared label
            assert Path(p["anchor"]).parent.name == p["label"]
            assert Path(p["positive"]).parent.name == p["label"]

    def test_respects_limit(self, tmp_path: Path):
        l2p = _make_class_dirs(tmp_path, {"A": 50})
        feats = _synth_features(l2p)
        miner = MetricPairMiner(data_dir=str(tmp_path))
        pairs = miner.mine_positive_pairs(feats, pairs_per_class=10)
        assert len(pairs) == 10


class TestMineHardNegatives:
    def test_different_class(self, tmp_path: Path):
        l2p = _make_class_dirs(tmp_path, {"cat": 5, "dog": 5})
        feats = _synth_features(l2p)
        miner = MetricPairMiner(data_dir=str(tmp_path))
        neg_pairs = miner.mine_hard_negatives(feats, negatives_per_anchor=2)
        assert len(neg_pairs) > 0
        for p in neg_pairs:
            assert p["anchor_label"] != p["negative_label"]
            assert "similarity" in p

    def test_returns_up_to_k_negatives(self, tmp_path: Path):
        l2p = _make_class_dirs(tmp_path, {"A": 3, "B": 3})
        feats = _synth_features(l2p)
        miner = MetricPairMiner(data_dir=str(tmp_path))
        neg_pairs = miner.mine_hard_negatives(feats, negatives_per_anchor=2)
        # Each of the 6 anchors should get up to 2 negatives
        from collections import Counter
        counts = Counter(p["anchor"] for p in neg_pairs)
        for cnt in counts.values():
            assert cnt <= 2


class TestMineTriplets:
    def test_margin_positive(self, tmp_path: Path):
        """Positive similarity should generally exceed negative similarity
        when classes are well-separated."""
        rng = np.random.RandomState(123)
        # Create well-separated clusters
        centers = {
            "round": rng.randn(48).astype(np.float32) * 10,
            "flat": rng.randn(48).astype(np.float32) * 10 + 50,
        }
        l2p = _make_class_dirs(tmp_path, {"round": 15, "flat": 15})
        feats = _synth_features(l2p, rng=rng, class_centers=centers)
        miner = MetricPairMiner(data_dir=str(tmp_path), seed=123)
        triplets = miner.mine_triplets(feats, triplets_per_class=20)
        assert len(triplets) > 0
        positive_margins = [t["margin"] for t in triplets]
        # With well-separated clusters most margins should be positive
        frac_positive = sum(1 for m in positive_margins if m > 0) / len(positive_margins)
        assert frac_positive > 0.5, f"Only {frac_positive:.0%} of triplets have positive margin"

    def test_triplet_labels_correct(self, tmp_path: Path):
        l2p = _make_class_dirs(tmp_path, {"A": 8, "B": 8})
        feats = _synth_features(l2p)
        miner = MetricPairMiner(data_dir=str(tmp_path))
        triplets = miner.mine_triplets(feats, triplets_per_class=5)
        for t in triplets:
            anchor_label = t["anchor_label"]
            # Positive must be from same class
            assert Path(t["positive"]).parent.name == anchor_label
            # Negative must be from a different class
            assert Path(t["negative"]).parent.name != anchor_label


class TestSavePairsJsonl:
    def test_round_trip(self, tmp_path: Path):
        pairs = [
            {"anchor": "a.dxf", "positive": "b.dxf", "label": "X", "similarity": 0.95},
            {"anchor": "c.dxf", "positive": "d.dxf", "label": "Y", "similarity": 0.88},
        ]
        miner = MetricPairMiner(data_dir=str(tmp_path))
        out_path = str(tmp_path / "pairs.jsonl")
        count = miner.save_pairs(out_path, pairs)
        assert count == 2

        # Reload and verify
        loaded = []
        with open(out_path) as fh:
            for line in fh:
                loaded.append(json.loads(line))
        assert len(loaded) == 2
        assert loaded[0]["similarity"] == 0.95
        assert loaded[1]["label"] == "Y"

    def test_creates_parent_dirs(self, tmp_path: Path):
        miner = MetricPairMiner(data_dir=str(tmp_path))
        deep_path = str(tmp_path / "a" / "b" / "c" / "out.jsonl")
        count = miner.save_pairs(deep_path, [{"x": 1}])
        assert count == 1
        assert Path(deep_path).exists()


class TestEmptyDirectoryGraceful:
    def test_empty_dir(self, tmp_path: Path):
        miner = MetricPairMiner(data_dir=str(tmp_path))
        scanned = miner.scan_labeled_files()
        assert scanned == {}

    def test_nonexistent_dir(self, tmp_path: Path):
        miner = MetricPairMiner(data_dir=str(tmp_path / "does_not_exist"))
        scanned = miner.scan_labeled_files()
        assert scanned == {}

    def test_generate_training_set_empty(self, tmp_path: Path):
        miner = MetricPairMiner(data_dir=str(tmp_path / "empty"))
        stats = miner.generate_training_set(str(tmp_path / "output"))
        assert stats["total_pairs"] == 0
        assert stats["total_files_scanned"] == 0


class TestManifestHasStatistics:
    def test_manifest_written(self, tmp_path: Path):
        """generate_training_set should write a manifest.json with expected keys."""
        l2p = _make_class_dirs(tmp_path / "data", {"P": 8, "Q": 8})
        feats = _synth_features(l2p)

        miner = MetricPairMiner(data_dir=str(tmp_path / "data"))

        # We bypass extract_features_batch by injecting features directly.
        # Monkey-patch the method so generate_training_set works end-to-end.
        feature_map = {item["path"]: item["features"] for item in feats}
        original_extract = miner.extract_features_batch
        miner.extract_features_batch = lambda paths: MetricPairMiner.inject_features(
            miner.scan_labeled_files(), feature_map,
        )

        out_dir = tmp_path / "output"
        stats = miner.generate_training_set(str(out_dir))

        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as fh:
            manifest = json.load(fh)

        assert "total_pairs" in manifest
        assert "positive_pairs" in manifest
        assert "hard_negatives" in manifest
        assert "triplets" in manifest
        assert "class_distribution" in manifest
        assert "generated_at" in manifest
        assert set(manifest["class_distribution"].keys()) == {"P", "Q"}
        assert manifest["total_pairs"] == (
            manifest["positive_pairs"] + manifest["hard_negatives"] + manifest["triplets"]
        )

        # Verify JSONL files were created
        assert (out_dir / "positive_pairs.jsonl").exists()
        assert (out_dir / "hard_negatives.jsonl").exists()
        assert (out_dir / "triplets.jsonl").exists()
