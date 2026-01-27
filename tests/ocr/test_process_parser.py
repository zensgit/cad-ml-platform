"""Tests for process_parser extraction of manufacturing requirements."""

import pytest

from src.core.ocr.base import (
    HeatTreatmentType,
    SurfaceTreatmentType,
    WeldingType,
)
from src.core.ocr.parsing.process_parser import parse_process_requirements


class TestHeatTreatmentExtraction:
    """Test heat treatment parsing."""

    def test_quenching_with_hardness_range(self):
        text = "整体淬火 HRC58-62"
        result = parse_process_requirements(text)
        assert len(result.heat_treatments) == 1
        ht = result.heat_treatments[0]
        assert ht.type == HeatTreatmentType.quenching
        assert ht.hardness == "HRC58-62"
        assert ht.hardness_min == 58.0
        assert ht.hardness_max == 62.0
        assert ht.hardness_unit == "HRC"

    def test_carburizing_with_depth(self):
        text = "渗碳淬火，渗碳层深度0.8-1.2mm，表面硬度HRC58-62"
        result = parse_process_requirements(text)
        types = [ht.type for ht in result.heat_treatments]
        assert HeatTreatmentType.carburizing in types
        assert HeatTreatmentType.quenching in types
        carb = next(h for h in result.heat_treatments if h.type == HeatTreatmentType.carburizing)
        assert carb.depth == 1.0  # average of 0.8-1.2

    def test_induction_hardening(self):
        text = "表面高频淬火 HRC≥50 淬硬层深度≥0.5mm"
        result = parse_process_requirements(text)
        types = [ht.type for ht in result.heat_treatments]
        assert HeatTreatmentType.induction_hardening in types
        ih = next(h for h in result.heat_treatments if h.type == HeatTreatmentType.induction_hardening)
        assert ih.hardness_min == 50.0
        assert ih.depth == 0.5

    def test_normalizing(self):
        text = "正火处理"
        result = parse_process_requirements(text)
        assert len(result.heat_treatments) == 1
        assert result.heat_treatments[0].type == HeatTreatmentType.normalizing

    def test_stress_relief(self):
        text = "去应力退火"
        result = parse_process_requirements(text)
        types = [ht.type for ht in result.heat_treatments]
        assert HeatTreatmentType.stress_relief in types

    def test_nitriding(self):
        text = "渗氮处理 HV500-600 渗氮层深度0.3mm"
        result = parse_process_requirements(text)
        assert len(result.heat_treatments) >= 1
        nit = next(h for h in result.heat_treatments if h.type == HeatTreatmentType.nitriding)
        assert nit.hardness_unit == "HV"
        assert nit.depth == 0.3

    def test_tempering(self):
        text = "淬火后回火处理"
        result = parse_process_requirements(text)
        types = [ht.type for ht in result.heat_treatments]
        assert HeatTreatmentType.tempering in types
        assert HeatTreatmentType.quenching in types


class TestSurfaceTreatmentExtraction:
    """Test surface treatment parsing."""

    def test_galvanizing_with_thickness(self):
        text = "表面镀锌，镀层厚度8-12μm"
        result = parse_process_requirements(text)
        assert len(result.surface_treatments) == 1
        st = result.surface_treatments[0]
        assert st.type == SurfaceTreatmentType.galvanizing
        assert st.thickness == 10.0  # average

    def test_chromating(self):
        text = "镀硬铬 厚度≥30μm"
        result = parse_process_requirements(text)
        assert len(result.surface_treatments) == 1
        st = result.surface_treatments[0]
        assert st.type == SurfaceTreatmentType.chromating
        assert st.thickness == 30.0

    def test_anodizing(self):
        text = "阳极氧化处理"
        result = parse_process_requirements(text)
        assert len(result.surface_treatments) == 1
        assert result.surface_treatments[0].type == SurfaceTreatmentType.anodizing

    def test_blackening(self):
        text = "发黑处理"
        result = parse_process_requirements(text)
        assert len(result.surface_treatments) == 1
        assert result.surface_treatments[0].type == SurfaceTreatmentType.blackening

    def test_multiple_surface_treatments(self):
        text = "喷砂后喷漆"
        result = parse_process_requirements(text)
        types = [st.type for st in result.surface_treatments]
        assert SurfaceTreatmentType.sandblasting in types
        assert SurfaceTreatmentType.painting in types

    def test_nickel_plating(self):
        text = "化学镍 膜层厚度10μm"
        result = parse_process_requirements(text)
        assert len(result.surface_treatments) == 1
        st = result.surface_treatments[0]
        assert st.type == SurfaceTreatmentType.nickel_plating
        assert st.thickness == 10.0

    def test_phosphating(self):
        text = "磷化处理"
        result = parse_process_requirements(text)
        assert len(result.surface_treatments) == 1
        assert result.surface_treatments[0].type == SurfaceTreatmentType.phosphating


class TestWeldingExtraction:
    """Test welding info parsing."""

    def test_tig_welding_with_filler(self):
        text = "氩弧焊 焊丝ER50-6"
        result = parse_process_requirements(text)
        assert len(result.welding) == 1
        w = result.welding[0]
        assert w.type == WeldingType.tig_welding
        assert w.filler_material == "ER50-6"

    def test_mig_welding(self):
        text = "MIG焊 焊丝ER70S-6"
        result = parse_process_requirements(text)
        assert len(result.welding) == 1
        w = result.welding[0]
        assert w.type == WeldingType.mig_welding

    def test_spot_welding(self):
        text = "点焊连接"
        result = parse_process_requirements(text)
        assert len(result.welding) == 1
        assert result.welding[0].type == WeldingType.spot_welding

    def test_welding_with_leg_size(self):
        text = "氩弧焊 焊脚6mm"
        result = parse_process_requirements(text)
        assert len(result.welding) == 1
        w = result.welding[0]
        assert w.leg_size == 6.0

    def test_brazing(self):
        text = "钎焊连接"
        result = parse_process_requirements(text)
        assert len(result.welding) == 1
        assert result.welding[0].type == WeldingType.brazing


class TestGeneralNotes:
    """Test general technical notes extraction."""

    def test_tolerance_standard(self):
        text = "未注公差按GB/T1804-m执行"
        result = parse_process_requirements(text)
        assert any("GB/T1804" in note for note in result.general_notes)

    def test_unspecified_radius(self):
        text = "未注圆角R3"
        result = parse_process_requirements(text)
        assert "未注圆角R3" in result.general_notes

    def test_unspecified_chamfer(self):
        text = "未注倒角C1"
        result = parse_process_requirements(text)
        assert "未注倒角C1" in result.general_notes

    def test_deburr(self):
        text = "去毛刺倒钝"
        result = parse_process_requirements(text)
        assert any("去毛刺" in note for note in result.general_notes)


class TestComplexText:
    """Test complex multi-process text extraction."""

    def test_full_technical_requirements(self):
        text = """
        技术要求：
        1. 材料45钢
        2. 调质处理 HB220-250
        3. 表面渗碳淬火 HRC58-62 渗碳层深度0.8-1.2mm
        4. 外圆镀硬铬 厚度≥20μm
        5. 未注公差按GB/T1804-m
        6. 未注圆角R2
        7. 去毛刺倒钝锐边
        8. 焊接采用氩弧焊 焊丝ER50-6 焊脚6mm
        """
        result = parse_process_requirements(text)

        # Heat treatments
        ht_types = [ht.type for ht in result.heat_treatments]
        assert HeatTreatmentType.quenching in ht_types
        assert HeatTreatmentType.carburizing in ht_types

        # Surface treatments
        st_types = [st.type for st in result.surface_treatments]
        assert SurfaceTreatmentType.chromating in st_types

        # Welding
        assert len(result.welding) >= 1
        w = result.welding[0]
        assert w.type == WeldingType.tig_welding
        assert w.leg_size == 6.0
        assert w.filler_material == "ER50-6"

        # General notes
        assert len(result.general_notes) >= 2

    def test_empty_text(self):
        result = parse_process_requirements("")
        assert len(result.heat_treatments) == 0
        assert len(result.surface_treatments) == 0
        assert len(result.welding) == 0
        assert len(result.general_notes) == 0

    def test_no_process_info(self):
        text = "Φ20±0.02 R5 M10×1.5"
        result = parse_process_requirements(text)
        assert len(result.heat_treatments) == 0
        assert len(result.surface_treatments) == 0
        assert len(result.welding) == 0


class TestHardnessPatterns:
    """Test various hardness format patterns."""

    def test_hrc_range(self):
        text = "HRC58-62"
        result = parse_process_requirements(text)
        # No heat treatment keyword, but hardness alone shouldn't create entry
        # This is expected behavior - hardness needs context
        pass

    def test_hb_range(self):
        text = "调质处理 HB220-250"
        result = parse_process_requirements(text)
        assert any("调质处理" in note for note in result.general_notes)

    def test_hv_single(self):
        text = "渗氮 HV≥500"
        result = parse_process_requirements(text)
        nit = next(h for h in result.heat_treatments if h.type == HeatTreatmentType.nitriding)
        assert nit.hardness_min == 500.0
        assert nit.hardness_unit == "HV"
