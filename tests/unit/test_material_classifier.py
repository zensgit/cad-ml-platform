"""Tests for the detailed material classification system."""

import pytest

from src.core.materials import (
    MATERIAL_DATABASE,
    MaterialCategory,
    MaterialInfo,
    MaterialSubCategory,
    classify_material_detailed,
    get_material_info,
    get_process_recommendations,
)
from src.core.materials.classifier import (
    MaterialGroup,
    MaterialProperties,
    ProcessRecommendation,
    classify_material_simple,
)


class TestMaterialInfo:
    """Test MaterialInfo dataclass."""

    def test_to_dict(self):
        """to_dict returns correct structure."""
        info = MATERIAL_DATABASE.get("45")
        assert info is not None

        d = info.to_dict()
        assert d["grade"] == "45"
        assert d["name"] == "优质碳素结构钢"
        assert d["category"] == "metal"
        assert d["sub_category"] == "ferrous"
        assert d["group"] == "carbon_steel"
        assert "properties" in d
        assert "process" in d

    def test_material_has_required_fields(self):
        """All materials have required fields."""
        for grade, info in MATERIAL_DATABASE.items():
            assert info.grade == grade
            assert info.name
            assert info.category in MaterialCategory
            assert info.sub_category in MaterialSubCategory
            assert info.group in MaterialGroup


class TestMaterialDatabase:
    """Test material database contents."""

    def test_database_has_materials(self):
        """Database contains materials."""
        assert len(MATERIAL_DATABASE) > 30

    def test_common_materials_exist(self):
        """Common materials exist in database."""
        common_materials = [
            "S30408", "S31603", "Q235B", "45", "40Cr", "42CrMo",
            "HT200", "QT400", "6061", "7075", "H62", "TA2", "TC4",
            "C276", "C22", "PTFE",
        ]
        for mat in common_materials:
            assert mat in MATERIAL_DATABASE, f"{mat} not found in database"

    def test_materials_have_properties(self):
        """Materials have properties defined."""
        for grade, info in MATERIAL_DATABASE.items():
            if info.category == MaterialCategory.METAL:
                assert info.properties.density is not None, f"{grade} missing density"

    def test_materials_have_process_recommendations(self):
        """Materials have process recommendations defined."""
        for grade, info in MATERIAL_DATABASE.items():
            assert info.process is not None
            assert isinstance(info.process, ProcessRecommendation)


class TestClassifyMaterialDetailed:
    """Test classify_material_detailed function."""

    def test_exact_match(self):
        """Exact match returns correct info."""
        info = classify_material_detailed("S30408")
        assert info is not None
        assert info.grade == "S30408"
        assert info.group == MaterialGroup.STAINLESS_STEEL

    def test_case_insensitive_match(self):
        """Case insensitive match works."""
        info = classify_material_detailed("s30408")
        assert info is not None
        assert info.grade == "S30408"

    def test_alias_match(self):
        """Alias match works."""
        info = classify_material_detailed("304")
        assert info is not None
        assert info.grade == "S30408"

        info = classify_material_detailed("SUS304")
        assert info is not None
        assert info.grade == "S30408"

    def test_pattern_match(self):
        """Pattern match works."""
        info = classify_material_detailed("Q235A")
        assert info is not None
        assert info.grade == "Q235B"

        info = classify_material_detailed("45#")
        assert info is not None
        assert info.grade == "45"

    def test_hastelloy_c22_patterns(self):
        """C22 alloy various patterns work."""
        patterns = ["C22", "c22", "C-22", "C 22", "Hastelloy C-22", "N06022"]
        for pat in patterns:
            info = classify_material_detailed(pat)
            assert info is not None, f"Pattern '{pat}' not recognized"
            assert info.grade == "C22", f"Pattern '{pat}' returned {info.grade}"

    def test_hastelloy_c276_patterns(self):
        """C276 alloy various patterns work."""
        patterns = ["C276", "C-276", "C 276", "Hastelloy C-276", "N10276"]
        for pat in patterns:
            info = classify_material_detailed(pat)
            assert info is not None, f"Pattern '{pat}' not recognized"
            assert info.grade == "C276", f"Pattern '{pat}' returned {info.grade}"

    def test_stainless_steel_fallback(self):
        """Generic stainless steel falls back to 304."""
        info = classify_material_detailed("不锈钢")
        assert info is not None
        assert info.grade == "S30408"

    def test_none_material(self):
        """None material returns None."""
        assert classify_material_detailed(None) is None

    def test_empty_material(self):
        """Empty material returns None."""
        assert classify_material_detailed("") is None

    def test_unknown_material(self):
        """Unknown material returns None."""
        assert classify_material_detailed("XYZ123") is None

    def test_cast_iron_patterns(self):
        """Cast iron patterns work."""
        info = classify_material_detailed("HT250")
        assert info is not None
        assert info.group == MaterialGroup.CAST_IRON

        info = classify_material_detailed("球墨铸铁")
        assert info is not None
        assert info.group == MaterialGroup.CAST_IRON

    def test_aluminum_patterns(self):
        """Aluminum patterns work."""
        info = classify_material_detailed("6061-T6")
        assert info is not None
        assert info.group == MaterialGroup.ALUMINUM

        info = classify_material_detailed("铝合金")
        assert info is not None
        assert info.group == MaterialGroup.ALUMINUM

    def test_titanium_patterns(self):
        """Titanium patterns work."""
        info = classify_material_detailed("Ti-6Al-4V")
        assert info is not None
        assert info.grade == "TC4"


class TestGetMaterialInfo:
    """Test get_material_info function."""

    def test_found_material(self):
        """Found material returns full info."""
        info = get_material_info("S30408")
        assert info["grade"] == "S30408"
        assert info["name"] == "奥氏体不锈钢"
        assert info["category"] == "metal"
        assert "properties" in info
        assert "process" in info

    def test_not_found_material(self):
        """Not found material returns minimal info."""
        info = get_material_info("UNKNOWN123")
        assert info["grade"] == "UNKNOWN123"
        assert info["found"] is False


class TestGetProcessRecommendations:
    """Test get_process_recommendations function."""

    def test_found_material(self):
        """Found material returns process recommendations."""
        rec = get_process_recommendations("S30408")
        assert isinstance(rec, ProcessRecommendation)
        assert "固溶处理" in rec.heat_treatments
        assert "钝化" in rec.surface_treatments
        assert len(rec.warnings) > 0

    def test_not_found_material(self):
        """Not found material returns empty recommendations."""
        rec = get_process_recommendations("UNKNOWN123")
        assert isinstance(rec, ProcessRecommendation)
        assert rec.blank_hint == ""

    def test_stainless_steel_forbidden_treatments(self):
        """Stainless steel has forbidden treatments."""
        rec = get_process_recommendations("304")
        assert "淬火" in rec.forbidden_heat_treatments
        assert "渗碳" in rec.forbidden_heat_treatments

    def test_titanium_special_tooling(self):
        """Titanium requires special tooling."""
        rec = get_process_recommendations("TC4")
        assert rec.special_tooling is True
        assert rec.coolant_required is True


class TestClassifyMaterialSimple:
    """Test classify_material_simple function."""

    def test_returns_group_value(self):
        """Returns material group as string."""
        result = classify_material_simple("S30408")
        assert result == "stainless_steel"

    def test_returns_none_for_unknown(self):
        """Returns None for unknown material."""
        assert classify_material_simple("UNKNOWN123") is None

    def test_various_materials(self):
        """Various materials classified correctly."""
        test_cases = [
            ("HT200", "cast_iron"),
            ("Q235B", "carbon_steel"),
            ("40Cr", "alloy_steel"),
            ("6061", "aluminum"),
            ("TC4", "titanium"),
            ("H62", "copper"),
            ("C276", "corrosion_resistant"),
        ]
        for mat, expected in test_cases:
            result = classify_material_simple(mat)
            assert result == expected, f"{mat} expected {expected}, got {result}"


class TestMaterialCategories:
    """Test material category coverage."""

    def test_metal_subcategories(self):
        """Metal subcategories exist."""
        ferrous = [m for m in MATERIAL_DATABASE.values()
                   if m.sub_category == MaterialSubCategory.FERROUS]
        non_ferrous = [m for m in MATERIAL_DATABASE.values()
                       if m.sub_category == MaterialSubCategory.NON_FERROUS]

        assert len(ferrous) > 10
        assert len(non_ferrous) > 5

    def test_non_metal_exists(self):
        """Non-metal materials exist."""
        non_metals = [m for m in MATERIAL_DATABASE.values()
                      if m.category == MaterialCategory.NON_METAL]
        assert len(non_metals) > 5

    def test_composite_exists(self):
        """Composite/assembly materials exist."""
        composites = [m for m in MATERIAL_DATABASE.values()
                      if m.category == MaterialCategory.COMPOSITE]
        assert len(composites) >= 2


class TestRealWorldMaterials:
    """Test materials found in real DXF files."""

    @pytest.mark.parametrize("material,expected_group", [
        ("S30408", "stainless_steel"),
        ("C276", "corrosion_resistant"),
        ("HT200", "cast_iron"),
        ("PTFE", "fluoropolymer"),
        ("A2-70", "stainless_steel"),  # Fastener material
        ("QSn4-3", "copper"),
        ("VMQ", "rubber"),
        ("EPDM", "rubber"),
        ("聚氨酯", "polyurethane"),
        ("硼硅玻璃", "borosilicate"),
        ("组焊件", "welded_assembly"),
        ("C22", "corrosion_resistant"),
        ("C-22", "corrosion_resistant"),
    ])
    def test_real_material_classification(self, material, expected_group):
        """Real materials from DXF files are classified correctly."""
        info = classify_material_detailed(material)
        assert info is not None, f"Material '{material}' not found"
        assert info.group.value == expected_group, \
            f"Material '{material}' expected group '{expected_group}', got '{info.group.value}'"


class TestHighEndMaterials:
    """Test high-end materials: nickel superalloys and duplex stainless steels."""

    @pytest.mark.parametrize("pattern,expected_grade", [
        # Inconel 625
        ("Inconel625", "Inconel625"),
        ("Inconel 625", "Inconel625"),
        ("IN625", "Inconel625"),
        ("IN-625", "Inconel625"),
        ("N06625", "Inconel625"),
        ("NCF625", "Inconel625"),
        ("Alloy 625", "Inconel625"),
        # Inconel 718
        ("Inconel718", "Inconel718"),
        ("Inconel 718", "Inconel718"),
        ("IN718", "Inconel718"),
        ("IN-718", "Inconel718"),
        ("N07718", "Inconel718"),
        ("GH4169", "Inconel718"),
        ("GH-4169", "Inconel718"),
        ("Alloy 718", "Inconel718"),
    ])
    def test_inconel_patterns(self, pattern, expected_grade):
        """Inconel superalloy patterns are recognized."""
        info = classify_material_detailed(pattern)
        assert info is not None, f"Pattern '{pattern}' not recognized"
        assert info.grade == expected_grade, f"'{pattern}' returned {info.grade}"

    @pytest.mark.parametrize("pattern,expected_grade", [
        # 2205 duplex
        ("2205", "2205"),
        ("S31803", "2205"),
        ("S32205", "2205"),
        ("SAF2205", "2205"),
        ("SAF 2205", "2205"),
        ("1.4462", "2205"),
        # 2507 super duplex
        ("2507", "2507"),
        ("S32750", "2507"),
        ("SAF2507", "2507"),
        ("SAF 2507", "2507"),
        ("1.4410", "2507"),
    ])
    def test_duplex_stainless_patterns(self, pattern, expected_grade):
        """Duplex stainless steel patterns are recognized."""
        info = classify_material_detailed(pattern)
        assert info is not None, f"Pattern '{pattern}' not recognized"
        assert info.grade == expected_grade, f"'{pattern}' returned {info.grade}"

    def test_inconel625_properties(self):
        """Inconel 625 has correct properties."""
        info = classify_material_detailed("Inconel625")
        assert info is not None
        assert info.group == MaterialGroup.CORROSION_RESISTANT
        assert info.properties.density == 8.44
        assert info.process.special_tooling is True
        assert info.process.coolant_required is True
        assert len(info.process.warnings) > 0

    def test_inconel718_properties(self):
        """Inconel 718 has correct properties."""
        info = classify_material_detailed("Inconel718")
        assert info is not None
        assert info.group == MaterialGroup.CORROSION_RESISTANT  # Nickel superalloy in corrosion-resistant group
        assert info.properties.density == 8.19
        assert info.process.special_tooling is True

    def test_duplex_2205_properties(self):
        """2205 duplex has correct properties."""
        info = classify_material_detailed("2205")
        assert info is not None
        assert info.group == MaterialGroup.STAINLESS_STEEL
        assert info.properties.tensile_strength >= 620
        assert "固溶处理" in info.process.heat_treatments

    def test_super_duplex_2507_properties(self):
        """2507 super duplex has correct properties."""
        info = classify_material_detailed("2507")
        assert info is not None
        assert info.group == MaterialGroup.STAINLESS_STEEL
        assert info.properties.tensile_strength >= 795
        # Super duplex is harder to machine
        assert info.process.special_tooling is True

    def test_no_conflict_with_20_steel(self):
        """2205/2507 patterns don't conflict with 20 steel."""
        # 20 steel should still work
        info = classify_material_detailed("20")
        assert info is not None
        assert info.grade == "20"
        assert info.group == MaterialGroup.CARBON_STEEL

        info = classify_material_detailed("20#")
        assert info is not None
        assert info.grade == "20"

        info = classify_material_detailed("20钢")
        assert info is not None
        assert info.grade == "20"


class TestEngineeringPlastics:
    """Test engineering plastics: PEEK, POM, PA, PC, UHMWPE."""

    def test_engineering_plastics_exist(self):
        """Engineering plastics exist in database."""
        plastics = ["PEEK", "POM", "PA66", "PC", "UHMWPE"]
        for p in plastics:
            assert p in MATERIAL_DATABASE, f"{p} not found in database"

    @pytest.mark.parametrize("pattern,expected_grade", [
        # PEEK
        ("PEEK", "PEEK"),
        ("聚醚醚酮", "PEEK"),
        # POM
        ("POM", "POM"),
        ("POM-C", "POM"),
        ("POM-H", "POM"),
        ("Delrin", "POM"),
        ("赛钢", "POM"),
        ("聚甲醛", "POM"),
        # PA (Nylon)
        ("PA66", "PA66"),
        ("PA6", "PA66"),
        ("Nylon66", "PA66"),
        ("Nylon6", "PA66"),
        ("尼龙", "PA66"),
        # PC
        ("PC", "PC"),
        ("聚碳酸酯", "PC"),
        ("Polycarbonate", "PC"),
        # UHMWPE
        ("UHMWPE", "UHMWPE"),
        ("UHMW-PE", "UHMWPE"),
        ("UPE", "UHMWPE"),
        ("超高分子量聚乙烯", "UHMWPE"),
    ])
    def test_engineering_plastic_patterns(self, pattern, expected_grade):
        """Engineering plastic patterns are recognized."""
        info = classify_material_detailed(pattern)
        assert info is not None, f"Pattern '{pattern}' not recognized"
        assert info.grade == expected_grade, f"'{pattern}' returned {info.grade}"

    def test_peek_properties(self):
        """PEEK has correct properties."""
        info = classify_material_detailed("PEEK")
        assert info is not None
        assert info.group == MaterialGroup.ENGINEERING_PLASTIC
        assert info.properties.density == 1.30
        assert info.properties.melting_point == 343
        assert info.process.coolant_required is True

    def test_pom_properties(self):
        """POM has correct properties."""
        info = classify_material_detailed("POM")
        assert info is not None
        assert info.group == MaterialGroup.ENGINEERING_PLASTIC
        assert info.properties.machinability == "excellent"
        assert info.process.coolant_required is False

    def test_engineering_plastic_group(self):
        """All engineering plastics are in engineering_plastic group."""
        plastics = ["PEEK", "POM", "PA66", "PC", "UHMWPE"]
        for p in plastics:
            info = classify_material_detailed(p)
            assert info.group == MaterialGroup.ENGINEERING_PLASTIC, \
                f"{p} expected engineering_plastic group, got {info.group}"
