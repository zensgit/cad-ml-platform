"""Tests for ProcessRouteGenerator."""

import pytest

from src.core.ocr.base import (
    HeatTreatmentInfo,
    HeatTreatmentType,
    ProcessRequirements,
    SurfaceTreatmentInfo,
    SurfaceTreatmentType,
    WeldingInfo,
    WeldingType,
)
from src.core.process.route_generator import (
    ProcessRoute,
    ProcessRouteGenerator,
    ProcessStage,
    ProcessStep,
    generate_process_route,
    get_route_generator,
)


class TestProcessStep:
    """Test ProcessStep dataclass."""

    def test_to_dict(self):
        """to_dict returns correct structure."""
        step = ProcessStep(
            stage=ProcessStage.rough_machining,
            name="粗加工",
            description="车削粗加工",
            parameters={"tool": "T01"},
            source="图纸要求",
            sequence=2,
        )
        d = step.to_dict()
        assert d["stage"] == "rough_machining"
        assert d["name"] == "粗加工"
        assert d["description"] == "车削粗加工"
        assert d["parameters"]["tool"] == "T01"
        assert d["source"] == "图纸要求"
        assert d["sequence"] == 2


class TestProcessRoute:
    """Test ProcessRoute dataclass."""

    def test_to_dict(self):
        """to_dict returns correct structure."""
        route = ProcessRoute(
            steps=[
                ProcessStep(stage=ProcessStage.blank_preparation, name="毛坯准备", sequence=1),
                ProcessStep(stage=ProcessStage.inspection, name="检验", sequence=2),
            ],
            confidence=0.8,
            warnings=["警告1"],
        )
        d = route.to_dict()
        assert d["total_steps"] == 2
        assert d["confidence"] == 0.8
        assert len(d["warnings"]) == 1
        assert d["steps"][0]["name"] == "毛坯准备"


class TestProcessRouteGenerator:
    """Test ProcessRouteGenerator core functionality."""

    def test_none_requirements_returns_basic_route(self):
        """None requirements returns basic 4-step route."""
        gen = ProcessRouteGenerator()
        route = gen.generate(None)

        assert len(route.steps) == 4
        assert route.steps[0].name == "毛坯准备"
        assert route.steps[-1].name == "检验"
        assert route.confidence == 0.3

    def test_empty_requirements_returns_basic_route(self):
        """Empty requirements returns basic route."""
        gen = ProcessRouteGenerator()
        route = gen.generate(ProcessRequirements())

        # Should still have basic steps
        assert len(route.steps) >= 4
        assert route.confidence < 0.5

    def test_heat_treatment_pre_inserted_after_rough(self):
        """Pre heat treatment (调质) inserted after rough machining."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            heat_treatments=[
                HeatTreatmentInfo(
                    type=HeatTreatmentType.quench_temper,
                    hardness="HB220-250",
                )
            ]
        )
        route = gen.generate(proc)

        step_names = [s.name for s in route.steps]
        assert "调质处理" in step_names
        # 调质应该在粗加工后
        rough_idx = step_names.index("粗加工")
        quench_temper_idx = step_names.index("调质处理")
        assert quench_temper_idx > rough_idx

    def test_heat_treatment_post_adds_grinding(self):
        """Post heat treatment (淬火) adds grinding step."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            heat_treatments=[
                HeatTreatmentInfo(
                    type=HeatTreatmentType.quenching,
                    hardness="HRC58-62",
                )
            ]
        )
        route = gen.generate(proc)

        step_names = [s.name for s in route.steps]
        assert "淬火" in step_names
        assert "磨削" in step_names
        # 磨削应该在淬火后
        quench_idx = step_names.index("淬火")
        grinding_idx = step_names.index("磨削")
        assert grinding_idx > quench_idx

    def test_surface_treatment_before_inspection(self):
        """Surface treatment inserted before inspection."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            surface_treatments=[
                SurfaceTreatmentInfo(
                    type=SurfaceTreatmentType.chromating,
                    thickness=20.0,
                )
            ]
        )
        route = gen.generate(proc)

        step_names = [s.name for s in route.steps]
        assert "镀铬" in step_names
        assert "检验" in step_names
        chrome_idx = step_names.index("镀铬")
        inspect_idx = step_names.index("检验")
        assert chrome_idx < inspect_idx

    def test_welding_generates_warning(self):
        """Welding without stress relief generates warning."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            welding=[
                WeldingInfo(
                    type=WeldingType.tig_welding,
                    filler_material="ER50-6",
                )
            ]
        )
        route = gen.generate(proc)

        step_names = [s.name for s in route.steps]
        assert "氩弧焊" in step_names
        assert any("去应力" in w for w in route.warnings)

    def test_welding_with_stress_relief_no_warning(self):
        """Welding with stress relief has no warning."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            welding=[
                WeldingInfo(type=WeldingType.tig_welding)
            ],
            heat_treatments=[
                HeatTreatmentInfo(type=HeatTreatmentType.stress_relief)
            ]
        )
        route = gen.generate(proc)

        assert not any("去应力" in w for w in route.warnings)

    def test_parameters_captured(self):
        """Process parameters are captured in steps."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            heat_treatments=[
                HeatTreatmentInfo(
                    type=HeatTreatmentType.quenching,
                    hardness="HRC58-62",
                    hardness_min=58.0,
                    hardness_max=62.0,
                    hardness_unit="HRC",
                    depth=1.0,
                )
            ],
            surface_treatments=[
                SurfaceTreatmentInfo(
                    type=SurfaceTreatmentType.chromating,
                    thickness=20.0,
                )
            ]
        )
        route = gen.generate(proc)

        # Find quenching step
        quench_step = next(s for s in route.steps if s.name == "淬火")
        assert quench_step.parameters.get("hardness_min") == 58.0
        assert quench_step.parameters.get("depth_mm") == 1.0

        # Find chrome step
        chrome_step = next(s for s in route.steps if s.name == "镀铬")
        assert chrome_step.parameters.get("thickness_um") == 20.0

    def test_inspection_items_reflect_processes(self):
        """Inspection description reflects processes used."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            heat_treatments=[HeatTreatmentInfo(type=HeatTreatmentType.quenching)],
            surface_treatments=[SurfaceTreatmentInfo(type=SurfaceTreatmentType.painting)],
            welding=[WeldingInfo(type=WeldingType.spot_welding)],
        )
        route = gen.generate(proc)

        inspect_step = next(s for s in route.steps if s.name == "检验")
        assert "硬度检验" in inspect_step.description
        assert "镀层" in inspect_step.description or "涂层" in inspect_step.description
        assert "焊缝检验" in inspect_step.description

    def test_confidence_increases_with_features(self):
        """Confidence increases with more process features."""
        gen = ProcessRouteGenerator()

        # Basic
        route_basic = gen.generate(ProcessRequirements())

        # With heat treatment
        route_ht = gen.generate(ProcessRequirements(
            heat_treatments=[HeatTreatmentInfo(type=HeatTreatmentType.quenching)]
        ))

        # With multiple features
        route_multi = gen.generate(ProcessRequirements(
            heat_treatments=[HeatTreatmentInfo(type=HeatTreatmentType.quenching)],
            surface_treatments=[SurfaceTreatmentInfo(type=SurfaceTreatmentType.chromating)],
            welding=[WeldingInfo(type=WeldingType.tig_welding)],
            general_notes=["去毛刺"]
        ))

        assert route_ht.confidence > route_basic.confidence
        assert route_multi.confidence > route_ht.confidence

    def test_step_sequence_numbers(self):
        """Step sequence numbers are assigned correctly."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            heat_treatments=[HeatTreatmentInfo(type=HeatTreatmentType.quenching)],
        )
        route = gen.generate(proc)

        for i, step in enumerate(route.steps):
            assert step.sequence == i + 1


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_route_generator_singleton(self):
        """get_route_generator returns singleton."""
        g1 = get_route_generator()
        g2 = get_route_generator()
        assert g1 is g2

    def test_generate_process_route(self):
        """generate_process_route convenience function works."""
        proc = ProcessRequirements(
            heat_treatments=[HeatTreatmentInfo(type=HeatTreatmentType.normalizing)]
        )
        route = generate_process_route(proc)
        assert isinstance(route, ProcessRoute)
        assert len(route.steps) > 0


class TestComplexScenarios:
    """Test complex multi-process scenarios."""

    def test_full_process_chain(self):
        """Test full process chain: 调质 + 渗碳淬火 + 镀铬."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            heat_treatments=[
                HeatTreatmentInfo(type=HeatTreatmentType.quench_temper, hardness="HB220-250"),
                HeatTreatmentInfo(type=HeatTreatmentType.carburizing, depth=1.0),
                HeatTreatmentInfo(type=HeatTreatmentType.quenching, hardness="HRC58-62"),
            ],
            surface_treatments=[
                SurfaceTreatmentInfo(type=SurfaceTreatmentType.chromating, thickness=20.0),
            ],
            general_notes=["未注公差按GB/T1804-m"]
        )
        route = gen.generate(proc)

        step_names = [s.name for s in route.steps]

        # Verify order
        assert "毛坯准备" in step_names
        assert "粗加工" in step_names
        assert "调质处理" in step_names
        assert "渗碳" in step_names
        assert "淬火" in step_names
        assert "磨削" in step_names
        assert "镀铬" in step_names
        assert "检验" in step_names

        # 调质 before 淬火
        qt_idx = step_names.index("调质处理")
        q_idx = step_names.index("淬火")
        assert qt_idx < q_idx

        # 镀铬 after 磨削
        chrome_idx = step_names.index("镀铬")
        grind_idx = step_names.index("磨削")
        assert chrome_idx > grind_idx

    def test_welded_assembly_with_painting(self):
        """Test welded assembly with painting."""
        gen = ProcessRouteGenerator()
        proc = ProcessRequirements(
            welding=[
                WeldingInfo(type=WeldingType.tig_welding, leg_size=6.0),
                WeldingInfo(type=WeldingType.spot_welding),
            ],
            surface_treatments=[
                SurfaceTreatmentInfo(type=SurfaceTreatmentType.sandblasting),
                SurfaceTreatmentInfo(type=SurfaceTreatmentType.painting),
            ],
            heat_treatments=[
                HeatTreatmentInfo(type=HeatTreatmentType.stress_relief),
            ]
        )
        route = gen.generate(proc)

        step_names = [s.name for s in route.steps]
        assert "氩弧焊" in step_names
        assert "点焊" in step_names
        assert "去应力退火" in step_names
        assert "喷砂" in step_names
        assert "喷漆" in step_names

        # No warning since stress relief is present
        assert not any("去应力" in w for w in route.warnings)
