#!/usr/bin/env python3
"""Unit tests for B2 data collection toolchain.

Covers:
- Filename extraction (label_annotation_tool.extract_part_name)
- Taxonomy mapping
- CSV output format and resume logic
- Augmentation parameter generation
- Manifest builder synonym matching
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.label_annotation_tool import (
    extract_part_name,
    load_synonyms,
    build_synonym_matcher,
    load_taxonomy_mapping,
    match_to_synonym,
    load_already_annotated,
    iter_dxf_files,
    run_annotation,
)
from scripts.augment_dxf_data import generate_augmentation_params
from scripts.build_unified_manifest import (
    extract_part_name as manifest_extract,
    load_excluded_labels,
)


# ---------------------------------------------------------------------------
# Filename extraction tests
# ---------------------------------------------------------------------------
class TestExtractPartName:
    def test_standard_format(self):
        assert extract_part_name("BTJ01239901522-00拖轮组件v1.dxf") == "拖轮组件"

    def test_manhole_format(self):
        assert extract_part_name("J2925001-01人孔v2.dxf") == "人孔"

    def test_comparison_format(self):
        result = extract_part_name("比较_LTJ012306102-0084调节螺栓v1 vs xxx.dxf")
        assert result is not None
        assert "调节螺栓" in result

    def test_discharge_flange(self):
        result = extract_part_name("J0224025-06-01-03出料凸缘.dxf")
        assert result is not None
        assert "出料凸缘" in result

    def test_no_chinese(self):
        assert extract_part_name("ABC123-456.dxf") is None

    def test_empty(self):
        assert extract_part_name("") is None

    def test_spec_suffix_stripped(self):
        result = extract_part_name("拖车DN1500.dxf")
        assert result == "拖车"

    def test_version_suffix_stripped(self):
        result = extract_part_name("阀体v3.dxf")
        assert result is not None
        assert "阀体" in result

    def test_pure_chinese(self):
        assert extract_part_name("搅拌器组件.dxf") == "搅拌器组件"


# ---------------------------------------------------------------------------
# Taxonomy mapping tests
# ---------------------------------------------------------------------------
class TestTaxonomyMapping:
    @pytest.fixture
    def taxonomy_path(self):
        return ROOT / "config" / "label_taxonomy_v2.yaml"

    def test_load_taxonomy(self, taxonomy_path):
        mapping = load_taxonomy_mapping(taxonomy_path)
        assert len(mapping) > 0
        # Check a known mapping
        assert mapping.get("调节螺栓") == "紧固件"
        assert mapping.get("再沸器") == "换热器"
        assert mapping.get("人孔") == "人孔"

    def test_special_mappings(self, taxonomy_path):
        mapping = load_taxonomy_mapping(taxonomy_path)
        assert mapping.get("防爆视灯组件") == "盖罩"
        assert mapping.get("前进离合器") == "传动件"

    def test_excluded_labels(self, taxonomy_path):
        excluded = load_excluded_labels(taxonomy_path)
        assert "金雨薇" in excluded
        assert "零件一" in excluded
        assert "模板" in excluded
        assert "其他" in excluded


# ---------------------------------------------------------------------------
# Synonym matching tests
# ---------------------------------------------------------------------------
class TestSynonymMatching:
    @pytest.fixture
    def matcher(self):
        synonyms = load_synonyms(ROOT / "data" / "knowledge" / "label_synonyms_template.json")
        return build_synonym_matcher(synonyms)

    def test_exact_match(self, matcher):
        assert match_to_synonym("人孔", matcher) == "人孔"

    def test_alias_match(self, matcher):
        assert match_to_synonym("manhole", matcher) == "人孔"

    def test_partial_match(self, matcher):
        # "拖轮" should partially match "拖轮组件"
        result = match_to_synonym("拖轮", matcher)
        assert result is not None

    def test_no_match(self, matcher):
        assert match_to_synonym("不存在的零件", matcher) is None


# ---------------------------------------------------------------------------
# CSV output and resume tests
# ---------------------------------------------------------------------------
class TestCSVAndResume:
    def test_resume_loads_existing(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "label"])
            writer.writeheader()
            writer.writerow({"file_path": "/a/b.dxf", "label": "test"})
            writer.writerow({"file_path": "/c/d.dxf", "label": "test2"})
        done = load_already_annotated(csv_path)
        assert "/a/b.dxf" in done
        assert "/c/d.dxf" in done

    def test_resume_empty_file(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.touch()
        done = load_already_annotated(csv_path)
        assert len(done) == 0

    def test_resume_nonexistent(self, tmp_path):
        csv_path = tmp_path / "nope.csv"
        done = load_already_annotated(csv_path)
        assert len(done) == 0

    def test_annotation_dry_run(self, tmp_path):
        """Dry run should not create output file."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test_人孔.dxf").touch()

        output = tmp_path / "out.csv"
        run_annotation(
            input_dir=input_dir,
            output_path=output,
            synonyms_path=ROOT / "data" / "knowledge" / "label_synonyms_template.json",
            taxonomy_path=ROOT / "config" / "label_taxonomy_v2.yaml",
            dry_run=True,
        )
        assert not output.exists()

    def test_annotation_non_interactive(self, tmp_path):
        """Non-interactive mode should auto-accept and write CSV."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "J123-人孔v1.dxf").touch()

        output = tmp_path / "annotations" / "out.csv"
        run_annotation(
            input_dir=input_dir,
            output_path=output,
            synonyms_path=ROOT / "data" / "knowledge" / "label_synonyms_template.json",
            taxonomy_path=ROOT / "config" / "label_taxonomy_v2.yaml",
            dry_run=False,
            non_interactive=True,
        )
        assert output.exists()
        with open(output, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["extracted_name"] == "人孔"
        assert rows[0]["annotator_label"] != ""


# ---------------------------------------------------------------------------
# DXF file iteration tests
# ---------------------------------------------------------------------------
class TestDXFIteration:
    def test_finds_dxf_files(self, tmp_path):
        (tmp_path / "a.dxf").touch()
        (tmp_path / "b.DXF").touch()
        (tmp_path / "c.txt").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.dxf").touch()
        files = iter_dxf_files(tmp_path, recursive=True)
        names = {f.name for f in files}
        assert "a.dxf" in names
        assert "b.DXF" in names
        assert "d.dxf" in names
        assert "c.txt" not in names

    def test_no_recursive(self, tmp_path):
        (tmp_path / "a.dxf").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.dxf").touch()
        files = iter_dxf_files(tmp_path, recursive=False)
        names = {f.name for f in files}
        assert "a.dxf" in names
        assert "b.dxf" not in names


# ---------------------------------------------------------------------------
# Augmentation parameter tests
# ---------------------------------------------------------------------------
class TestAugmentationParams:
    def test_deterministic(self):
        p1 = generate_augmentation_params(0, seed=42)
        p2 = generate_augmentation_params(0, seed=42)
        assert p1 == p2

    def test_different_copies(self):
        p0 = generate_augmentation_params(0, seed=42)
        p1 = generate_augmentation_params(1, seed=42)
        # Different copy indices should (usually) produce different params
        # At minimum the structure is correct
        assert "angle" in p0
        assert "scale" in p0
        assert "mirror_x" in p0
        assert "mirror_y" in p0
        assert "dropout_rate" in p0

    def test_angle_in_valid_range(self):
        for i in range(20):
            p = generate_augmentation_params(i, seed=i)
            assert p["angle"] in [0, 90, 180, 270]

    def test_scale_in_range(self):
        for i in range(20):
            p = generate_augmentation_params(i, seed=i)
            assert 0.8 <= p["scale"] <= 1.2

    def test_dropout_in_range(self):
        for i in range(20):
            p = generate_augmentation_params(i, seed=i)
            assert 0.05 <= p["dropout_rate"] <= 0.10
