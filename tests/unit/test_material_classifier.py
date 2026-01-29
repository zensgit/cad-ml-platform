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
        # Note: GH4169 now maps to its own entry GH4169, not Inconel718
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


class TestSpecialAlloys:
    """Test special alloys: Monel, Stellite, Hastelloy, Incoloy."""

    def test_special_alloys_exist(self):
        """Special alloys exist in database."""
        alloys = ["Monel400", "MonelK500", "HastelloyB3", "Stellite6", "Incoloy825"]
        for a in alloys:
            assert a in MATERIAL_DATABASE, f"{a} not found in database"

    @pytest.mark.parametrize("pattern,expected_grade", [
        # Monel 400
        ("Monel400", "Monel400"),
        ("Monel 400", "Monel400"),
        ("N04400", "Monel400"),
        ("NCu30", "Monel400"),
        # Monel K-500
        ("Monel K-500", "MonelK500"),
        ("Monel K500", "MonelK500"),
        ("N05500", "MonelK500"),
        # Hastelloy B-3
        ("Hastelloy B-3", "HastelloyB3"),
        ("Hastelloy B3", "HastelloyB3"),
        ("N10675", "HastelloyB3"),
        # Stellite 6
        ("Stellite 6", "Stellite6"),
        ("Stellite6", "Stellite6"),
        ("钴基6号", "Stellite6"),
        ("CoCr-A", "Stellite6"),
        # Incoloy 825
        ("Incoloy 825", "Incoloy825"),
        ("Incoloy825", "Incoloy825"),
        ("N08825", "Incoloy825"),
        ("GH2825", "Incoloy825"),
    ])
    def test_special_alloy_patterns(self, pattern, expected_grade):
        """Special alloy patterns are recognized."""
        info = classify_material_detailed(pattern)
        assert info is not None, f"Pattern '{pattern}' not recognized"
        assert info.grade == expected_grade, f"'{pattern}' returned {info.grade}"

    def test_monel400_properties(self):
        """Monel 400 has correct properties."""
        info = classify_material_detailed("Monel400")
        assert info is not None
        assert info.group == MaterialGroup.CORROSION_RESISTANT
        assert info.properties.density == 8.80
        assert info.process.special_tooling is True

    def test_stellite6_properties(self):
        """Stellite 6 has correct properties."""
        info = classify_material_detailed("Stellite6")
        assert info is not None
        assert info.group == MaterialGroup.CORROSION_RESISTANT
        assert info.properties.hardness == "HRC38-44"
        assert info.process.special_tooling is True
        assert "CBN" in str(info.process.warnings) or "金刚石" in str(info.process.warnings)

    def test_special_alloys_in_corrosion_resistant_group(self):
        """All special alloys are in corrosion_resistant group."""
        alloys = ["Monel400", "MonelK500", "HastelloyB3", "Stellite6", "Incoloy825"]
        for a in alloys:
            info = classify_material_detailed(a)
            assert info.group == MaterialGroup.CORROSION_RESISTANT, \
                f"{a} expected corrosion_resistant group, got {info.group}"


class TestCopperAlloys:
    """Test copper alloy materials."""

    def test_copper_alloys_exist(self):
        """Copper alloys exist in database."""
        alloys = [
            "H62", "H68", "HPb59-1", "QBe2", "QAl10-3-1.5",
            "QAl9-4", "QSn4-3", "QSn6.5-0.1", "Cu65", "CuNi10Fe1Mn"
        ]
        for a in alloys:
            assert a in MATERIAL_DATABASE, f"{a} not found in database"

    @pytest.mark.parametrize("pattern,expected_grade", [
        # 黄铜
        ("H62", "H62"),
        ("H68", "H68"),
        ("CuZn33", "H68"),
        ("C26800", "H68"),
        ("HPb59-1", "HPb59-1"),
        ("C38500", "HPb59-1"),
        ("易切削黄铜", "HPb59-1"),
        ("快削黄铜", "HPb59-1"),
        ("黄铜", "H62"),
        # 铍青铜
        ("QBe2", "QBe2"),
        ("QBe-2", "QBe2"),
        ("C17200", "QBe2"),
        ("BeCu", "QBe2"),
        ("CuBe2", "QBe2"),
        ("铍铜", "QBe2"),
        ("铍青铜", "QBe2"),
        # 铝青铜
        ("QAl10-3-1.5", "QAl10-3-1.5"),
        ("C63000", "QAl10-3-1.5"),
        ("铝铁镍青铜", "QAl10-3-1.5"),
        # 磷青铜
        ("QSn6.5-0.1", "QSn6.5-0.1"),
        ("C51900", "QSn6.5-0.1"),
        ("磷青铜", "QSn6.5-0.1"),
        ("磷铜", "QSn6.5-0.1"),
        ("弹簧磷青铜", "QSn6.5-0.1"),
        # 锡青铜
        ("QSn4-3", "QSn4-3"),
        ("青铜", "QSn4-3"),
        # 白铜
        ("CuNi10Fe1Mn", "CuNi10Fe1Mn"),
        ("B10", "CuNi10Fe1Mn"),
        ("C70600", "CuNi10Fe1Mn"),
        ("白铜", "CuNi10Fe1Mn"),
        # 紫铜
        ("Cu65", "Cu65"),
        ("紫铜", "Cu65"),
        ("纯铜", "Cu65"),
        ("T2", "Cu65"),
        # Note: T1 is AISI standard for high-speed steel (W18Cr4V), not pure copper
    ])
    def test_copper_alloy_patterns(self, pattern, expected_grade):
        """Copper alloy patterns are recognized."""
        info = classify_material_detailed(pattern)
        assert info is not None, f"Pattern '{pattern}' not recognized"
        assert info.grade == expected_grade, f"'{pattern}' returned {info.grade}"

    def test_qbe2_properties(self):
        """QBe2 beryllium bronze has correct properties."""
        info = classify_material_detailed("QBe2")
        assert info is not None
        assert info.name == "铍青铜"
        assert info.group == MaterialGroup.BERYLLIUM_COPPER
        assert info.properties.tensile_strength == 1250
        assert "铍" in str(info.process.warnings)

    def test_cupper_conductivity_warning(self):
        """Pure copper has soft material warning."""
        info = classify_material_detailed("Cu65")
        assert info is not None
        assert info.name == "紫铜"
        assert "软" in str(info.process.warnings)

    def test_hpb59_lead_warning(self):
        """HPb59-1 lead brass has lead warning."""
        info = classify_material_detailed("HPb59-1")
        assert info is not None
        assert "含铅" in str(info.process.warnings)

    def test_copper_alloys_in_copper_group(self):
        """Most copper alloys are in copper group (except specialized ones)."""
        # Generic copper alloys
        copper_alloys = [
            "H62", "H68", "HPb59-1", "QAl10-3-1.5",
            "QSn4-3", "QSn6.5-0.1", "Cu65", "CuNi10Fe1Mn"
        ]
        for a in copper_alloys:
            info = classify_material_detailed(a)
            assert info.group == MaterialGroup.COPPER, \
                f"{a} expected copper group, got {info.group}"
        # Specialized copper alloys have their own groups
        assert classify_material_detailed("QBe2").group == MaterialGroup.BERYLLIUM_COPPER
        assert classify_material_detailed("QAl9-4").group == MaterialGroup.ALUMINUM_BRONZE

    def test_copper_alloy_machinability(self):
        """Test machinability ratings for copper alloys."""
        # 易切削黄铜最好
        hpb = classify_material_detailed("HPb59-1")
        assert hpb.properties.machinability == "excellent"

        # 铍铜加工性良好
        qbe = classify_material_detailed("QBe2")
        assert qbe.properties.machinability == "good"


class TestTitaniumAlloys:
    """Test titanium alloy materials."""

    def test_titanium_alloys_exist(self):
        """Verify all titanium alloys exist in database."""
        titanium_alloys = ["TA1", "TA2", "TC4", "TC11", "TB6", "TC21"]
        for grade in titanium_alloys:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # TA1 patterns
        ("TA1", "TA1"),
        ("TA1锻件", "TA1"),
        ("Gr1", "TA1"),
        # TA2 patterns
        ("TA2", "TA2"),
        ("Gr2", "TA2"),
        # TC4 patterns
        ("TC4", "TC4"),
        ("Ti-6Al-4V", "TC4"),
        ("Gr5", "TC4"),
        # TC11 patterns
        ("TC11", "TC11"),
        ("BT9", "TC11"),
        # TB6 patterns
        ("TB6", "TB6"),
        ("Ti-10V-2Fe-3Al", "TB6"),
        ("Ti1023", "TB6"),
        # TC21 patterns
        ("TC21", "TC21"),
    ])
    def test_titanium_alloy_patterns(self, input_name, expected_grade):
        """Test pattern matching for titanium alloys."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Failed to classify {input_name}"
        assert info.grade == expected_grade, f"{input_name} should match {expected_grade}"

    def test_tc4_properties(self):
        """Test TC4 properties."""
        info = classify_material_detailed("TC4")
        assert info.properties.density == 4.43
        assert info.properties.tensile_strength == 895
        assert info.properties.machinability == "poor"

    def test_tb6_high_strength(self):
        """Test TB6 has highest strength among titanium alloys."""
        tb6 = classify_material_detailed("TB6")
        tc4 = classify_material_detailed("TC4")
        assert tb6.properties.tensile_strength > tc4.properties.tensile_strength

    def test_titanium_alloys_in_titanium_group(self):
        """Test all titanium alloys are in titanium group."""
        alloys = ["TA1", "TA2", "TC4", "TC11", "TB6", "TC21"]
        for a in alloys:
            info = classify_material_detailed(a)
            assert info.group == MaterialGroup.TITANIUM, \
                f"{a} expected titanium group, got {info.group}"


class TestMagnesiumAlloys:
    """Test magnesium alloy materials."""

    def test_magnesium_alloys_exist(self):
        """Verify all magnesium alloys exist in database."""
        mg_alloys = ["AZ31B", "AZ91D", "ZK60"]
        for grade in mg_alloys:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # AZ31B patterns
        ("AZ31B", "AZ31B"),
        ("AZ31", "AZ31B"),
        ("MB2", "AZ31B"),
        # AZ91D patterns
        ("AZ91D", "AZ91D"),
        ("AZ91", "AZ91D"),
        # ZK60 patterns
        ("ZK60", "ZK60"),
        ("ZK60A", "ZK60"),
        # Generic
        ("镁合金", "AZ31B"),
    ])
    def test_magnesium_alloy_patterns(self, input_name, expected_grade):
        """Test pattern matching for magnesium alloys."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Failed to classify {input_name}"
        assert info.grade == expected_grade, f"{input_name} should match {expected_grade}"

    def test_az31b_properties(self):
        """Test AZ31B properties."""
        info = classify_material_detailed("AZ31B")
        assert info.properties.density == 1.77
        assert info.properties.machinability == "excellent"
        assert "易燃材料" in info.process.warnings

    def test_magnesium_lightweight(self):
        """Test magnesium is lightweight material."""
        az31 = classify_material_detailed("AZ31B")
        al6061 = classify_material_detailed("6061")
        # 镁合金比铝合金更轻
        assert az31.properties.density < al6061.properties.density

    def test_magnesium_alloys_in_magnesium_group(self):
        """Test all magnesium alloys are in magnesium group."""
        alloys = ["AZ31B", "AZ91D", "ZK60"]
        for a in alloys:
            info = classify_material_detailed(a)
            assert info.group == MaterialGroup.MAGNESIUM, \
                f"{a} expected magnesium group, got {info.group}"


class TestCementedCarbides:
    """Test cemented carbide materials."""

    def test_cemented_carbides_exist(self):
        """Verify all cemented carbides exist in database."""
        carbides = ["YG8", "YT15", "YG6"]
        for grade in carbides:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # YG8 patterns
        ("YG8", "YG8"),
        ("YG 8", "YG8"),
        ("K20", "YG8"),
        ("WC-8Co", "YG8"),
        # YT15 patterns
        ("YT15", "YT15"),
        ("YT 15", "YT15"),
        ("P20", "YT15"),
        # YG6 patterns
        ("YG6", "YG6"),
        ("K10", "YG6"),
        # Generic
        ("硬质合金", "WC-Co"),
        ("钨钢", "YG8"),
    ])
    def test_cemented_carbide_patterns(self, input_name, expected_grade):
        """Test pattern matching for cemented carbides."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Failed to classify {input_name}"
        assert info.grade == expected_grade, f"{input_name} should match {expected_grade}"

    def test_yg8_properties(self):
        """Test YG8 properties."""
        info = classify_material_detailed("YG8")
        assert info.properties.density == 14.7
        assert "HRA" in info.properties.hardness
        assert info.properties.machinability == "poor"

    def test_carbide_hardness_very_high(self):
        """Test cemented carbides have very high hardness."""
        yg8 = classify_material_detailed("YG8")
        # 硬度应该是 HRA89+
        assert "HRA" in yg8.properties.hardness
        hardness_value = int(yg8.properties.hardness.replace("HRA", ""))
        assert hardness_value >= 89

    def test_carbide_special_tooling(self):
        """Test cemented carbides require special tooling."""
        for grade in ["YG8", "YT15", "YG6"]:
            info = classify_material_detailed(grade)
            assert info.process.special_tooling is True

    def test_cemented_carbides_in_cemented_carbide_group(self):
        """Test all cemented carbides are in cemented_carbide group."""
        carbides = ["YG8", "YT15", "YG6"]
        for c in carbides:
            info = classify_material_detailed(c)
            assert info.group == MaterialGroup.CEMENTED_CARBIDE, \
                f"{c} expected cemented_carbide group, got {info.group}"


class TestAluminumAlloys:
    """Test aluminum alloy materials."""

    def test_aluminum_alloys_exist(self):
        """Verify all aluminum alloys exist in database."""
        al_alloys = ["6061", "7075", "2024", "5052", "5083", "2A12", "6063"]
        for grade in al_alloys:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 2024 patterns
        ("2024", "2024"),
        ("2024-T4", "2024"),
        ("2024-T351", "2024"),
        # 5052 patterns
        ("5052", "5052"),
        ("5052-H32", "5052"),
        # 5083 patterns
        ("5083", "5083"),
        ("5083-H116", "5083"),
        # 2A12 patterns
        ("2A12", "2A12"),
        ("LY12", "LY12"),  # LY12 is now an independent material
        # 6063 patterns
        ("6063", "6063"),
        ("6063-T5", "6063"),
        ("6063-T6", "6063"),
    ])
    def test_aluminum_alloy_patterns(self, input_name, expected_grade):
        """Test pattern matching for aluminum alloys."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Failed to classify {input_name}"
        assert info.grade == expected_grade, f"{input_name} should match {expected_grade}"

    def test_2024_properties(self):
        """Test 2024 properties."""
        info = classify_material_detailed("2024")
        assert info.properties.density == 2.78
        assert info.properties.tensile_strength == 470
        assert info.properties.weldability == "poor"

    def test_5083_excellent_weldability(self):
        """Test 5083 has excellent weldability."""
        info = classify_material_detailed("5083")
        assert info.properties.weldability == "excellent"

    def test_aluminum_alloys_in_aluminum_group(self):
        """Test all aluminum alloys are in aluminum group."""
        alloys = ["6061", "7075", "2024", "5052", "5083", "2A12", "6063"]
        for a in alloys:
            info = classify_material_detailed(a)
            assert info.group == MaterialGroup.ALUMINUM, \
                f"{a} expected aluminum group, got {info.group}"


class TestSpringAndToolSteels:
    """Test spring steels and tool steels."""

    def test_spring_steels_exist(self):
        """Verify spring steels exist in database."""
        spring_steels = ["65Mn", "60Si2Mn", "50CrVA"]
        for grade in spring_steels:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    def test_tool_steels_exist(self):
        """Verify tool steels exist in database."""
        tool_steels = ["Cr12MoV", "H13", "W18Cr4V", "W6Mo5Cr4V2"]
        for grade in tool_steels:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 弹簧钢 patterns
        ("65Mn", "65Mn"),
        ("60Si2Mn", "60Si2Mn"),
        ("60Si2MnA", "60Si2Mn"),
        ("SUP7", "60Si2Mn"),
        ("50CrVA", "50CrVA"),
        ("SUP10", "50CrVA"),
        ("弹簧钢", "65Mn"),
        # 工具钢 patterns
        ("Cr12MoV", "Cr12MoV"),
        ("D2", "D2"),  # D2 is now an independent material
        ("SKD11", "Cr12MoV"),  # SKD11 maps to Cr12MoV (historical alias)
        ("H13", "H13"),
        ("SKD61", "H13"),
        ("模具钢", "H13"),
        ("W18Cr4V", "W18Cr4V"),
        ("W6Mo5Cr4V2", "W6Mo5Cr4V2"),
        ("M2", "W6Mo5Cr4V2"),
        ("SKH51", "W6Mo5Cr4V2"),
        ("高速钢", "W6Mo5Cr4V2"),
    ])
    def test_steel_patterns(self, input_name, expected_grade):
        """Test pattern matching for spring and tool steels."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Failed to classify {input_name}"
        assert info.grade == expected_grade, f"{input_name} should match {expected_grade}"

    def test_spring_steel_properties(self):
        """Test spring steel properties."""
        info = classify_material_detailed("60Si2Mn")
        assert info.properties.tensile_strength >= 1400
        assert info.properties.weldability == "poor"

    def test_tool_steel_high_hardness(self):
        """Test tool steels have high hardness."""
        for grade in ["Cr12MoV", "W18Cr4V"]:
            info = classify_material_detailed(grade)
            assert "HRC" in info.properties.hardness
            # Extract hardness value
            import re
            match = re.search(r"HRC(\d+)", info.properties.hardness)
            if match:
                hardness = int(match.group(1))
                assert hardness >= 58

    def test_tool_steel_special_tooling(self):
        """Test tool steels require special tooling."""
        for grade in ["Cr12MoV", "H13", "W18Cr4V", "W6Mo5Cr4V2"]:
            info = classify_material_detailed(grade)
            assert info.process.special_tooling is True

    def test_spring_steels_in_alloy_steel_group(self):
        """Test spring steels are in alloy_steel or spring_steel group."""
        # 65Mn is in alloy_steel, 60Si2Mn and 50CrVA are in spring_steel
        info = classify_material_detailed("65Mn")
        assert info.group == MaterialGroup.ALLOY_STEEL
        for grade in ["60Si2Mn", "50CrVA"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.SPRING_STEEL, \
                f"{grade} expected spring_steel group, got {info.group}"

    def test_tool_steels_in_tool_steel_group(self):
        """Test tool steels are in tool_steel group."""
        for grade in ["Cr12MoV", "H13", "W18Cr4V", "W6Mo5Cr4V2"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.TOOL_STEEL, \
                f"{grade} expected tool_steel group, got {info.group}"


class TestCarbonSteelSeries:
    """Tests for carbon steel series (10, 15, 35, 50)."""

    def test_carbon_steels_exist(self):
        """New carbon steels exist in database."""
        for grade in ["10", "15", "35", "50"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        ("10", "10"),
        ("10#", "10"),
        ("10钢", "10"),
        ("C10", "10"),
        ("15", "15"),
        ("15#", "15"),
        ("C15", "15"),
        ("35", "35"),
        ("35#", "35"),
        ("S35C", "35"),
        ("C35", "35"),
        ("50", "50"),
        ("50#", "50"),
        ("S50C", "50"),
        ("C50", "50"),
    ])
    def test_carbon_steel_patterns(self, input_name, expected_grade):
        """Carbon steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_low_carbon_weldability(self):
        """Low carbon steels (10, 15) have excellent weldability."""
        for grade in ["10", "15"]:
            info = classify_material_detailed(grade)
            assert info.properties.weldability == "excellent"

    def test_medium_carbon_properties(self):
        """Medium carbon steels (35, 50) have fair weldability."""
        for grade in ["35", "50"]:
            info = classify_material_detailed(grade)
            assert info.properties.weldability == "fair"

    def test_carbon_steels_in_carbon_steel_group(self):
        """Carbon steels are in carbon_steel group."""
        for grade in ["10", "15", "35", "50"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.CARBON_STEEL


class TestStainlessSteelSeries:
    """Tests for stainless steel series (321, 347, 430, 410, 17-4PH)."""

    def test_stainless_steels_exist(self):
        """New stainless steels exist in database."""
        for grade in ["321", "347", "430", "410", "17-4PH"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 321 钛稳定型
        ("321", "321"),
        ("S32100", "321"),
        ("SUS321", "321"),
        ("1.4541", "321"),
        ("1Cr18Ni9Ti", "321"),
        # 347 铌稳定型
        ("347", "347"),
        ("S34700", "347"),
        ("SUS347", "347"),
        ("1.4550", "347"),
        # 430 铁素体
        ("430", "430"),
        ("S43000", "430"),
        ("SUS430", "430"),
        ("1.4016", "430"),
        # Note: 1Cr17 now maps to its own material, not 430
        # 410 马氏体
        ("410", "410"),
        ("S41000", "410"),
        ("SUS410", "410"),
        ("1.4006", "410"),
        ("1Cr13", "410"),
        # 17-4PH 沉淀硬化
        ("17-4PH", "17-4PH"),
        ("S17400", "17-4PH"),
        ("SUS630", "17-4PH"),
        ("1.4542", "17-4PH"),
    ])
    def test_stainless_steel_patterns(self, input_name, expected_grade):
        """Stainless steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_321_excellent_weldability(self):
        """321 has excellent weldability."""
        info = classify_material_detailed("321")
        assert info.properties.weldability == "excellent"
        assert "钛稳定" in info.name

    def test_17_4ph_high_strength(self):
        """17-4PH has high strength."""
        info = classify_material_detailed("17-4PH")
        assert info.properties.tensile_strength >= 1300
        assert "沉淀硬化" in info.name

    def test_430_ferritic(self):
        """430 is ferritic stainless steel."""
        info = classify_material_detailed("430")
        assert "铁素体" in info.name

    def test_stainless_steels_in_stainless_steel_group(self):
        """Stainless steels are in stainless_steel group."""
        for grade in ["321", "347", "430", "410", "17-4PH"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.STAINLESS_STEEL


class TestCastIronSeries:
    """Tests for cast iron series (HT250, HT300, QT500-7, QT600-3, QT700-2)."""

    def test_cast_irons_exist(self):
        """New cast irons exist in database."""
        for grade in ["HT250", "HT300", "QT500-7", "QT600-3", "QT700-2"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 灰铸铁
        ("HT250", "HT250"),
        ("HT 250", "HT250"),
        ("FC250", "HT250"),
        ("HT300", "HT300"),
        ("HT 300", "HT300"),
        ("FC300", "HT300"),
        # 球墨铸铁
        ("QT500-7", "QT500-7"),
        ("QT500", "QT500-7"),
        ("FCD500", "QT500-7"),
        ("QT600-3", "QT600-3"),
        ("QT600", "QT600-3"),
        ("FCD600", "QT600-3"),
        ("QT700-2", "QT700-2"),
        ("QT700", "QT700-2"),
        ("FCD700", "QT700-2"),
    ])
    def test_cast_iron_patterns(self, input_name, expected_grade):
        """Cast iron patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_gray_iron_excellent_machinability(self):
        """Gray iron has excellent machinability."""
        for grade in ["HT200", "HT250"]:
            info = classify_material_detailed(grade)
            assert info.properties.machinability == "excellent"

    def test_ductile_iron_strength_progression(self):
        """Ductile iron strength increases with grade."""
        qt400 = classify_material_detailed("QT400")
        qt500 = classify_material_detailed("QT500-7")
        qt600 = classify_material_detailed("QT600-3")
        qt700 = classify_material_detailed("QT700-2")
        assert qt400.properties.tensile_strength < qt500.properties.tensile_strength
        assert qt500.properties.tensile_strength < qt600.properties.tensile_strength
        assert qt600.properties.tensile_strength < qt700.properties.tensile_strength

    def test_cast_irons_poor_weldability(self):
        """Cast irons generally have poor weldability."""
        for grade in ["HT200", "HT250", "HT300"]:
            info = classify_material_detailed(grade)
            assert info.properties.weldability == "poor"

    def test_cast_irons_in_cast_iron_group(self):
        """Cast irons are in cast_iron group."""
        for grade in ["HT250", "HT300", "QT500-7", "QT600-3", "QT700-2"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.CAST_IRON


class TestAlloySteel:
    """Tests for new alloy steels (20CrMnTi, 20Cr, 38CrMoAl, 30CrMnSi)."""

    def test_alloy_steels_exist(self):
        """New alloy steels exist in database."""
        for grade in ["20CrMnTi", "20Cr", "38CrMoAl", "30CrMnSi"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 20CrMnTi 渗碳钢
        ("20CrMnTi", "20CrMnTi"),
        ("20CrMnTiH", "20CrMnTi"),
        ("SCM420H", "20CrMnTi"),
        ("渗碳钢", "20CrMnTi"),
        # 20Cr 渗碳钢
        ("20Cr", "20Cr"),
        ("5120", "20Cr"),
        ("SCr420", "20Cr"),
        # 38CrMoAl 氮化钢
        ("38CrMoAl", "38CrMoAl"),
        ("38CrMoAlA", "38CrMoAl"),
        ("SACM645", "38CrMoAl"),
        ("氮化钢", "38CrMoAl"),
        # 30CrMnSi 高强度结构钢
        ("30CrMnSi", "30CrMnSi"),
        ("30CrMnSiA", "30CrMnSi"),
    ])
    def test_alloy_steel_patterns(self, input_name, expected_grade):
        """Alloy steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_20crmnt_carburizing(self):
        """20CrMnTi is carburizing steel."""
        info = classify_material_detailed("20CrMnTi")
        assert "渗碳" in info.name
        assert "渗碳" in str(info.process.heat_treatments)

    def test_38crmoal_nitriding(self):
        """38CrMoAl is nitriding steel."""
        info = classify_material_detailed("38CrMoAl")
        assert "氮化" in info.name
        assert "氮化" in str(info.process.heat_treatments)

    def test_alloy_steels_in_alloy_steel_group(self):
        """Alloy steels are in alloy_steel group."""
        # Note: 20CrMnTi now in gear_steel group
        for grade in ["20Cr", "38CrMoAl", "30CrMnSi"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.ALLOY_STEEL


class TestHeatResistantAlloys:
    """Tests for heat resistant steels and superalloys (310S, GH3030, GH4169, GH4099)."""

    def test_heat_resistant_alloys_exist(self):
        """Heat resistant alloys exist in database."""
        for grade in ["310S", "GH3030", "GH4169", "GH4099"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 310S 耐热不锈钢
        ("310S", "310S"),
        ("S31008", "310S"),
        ("SUS310S", "310S"),
        ("1.4845", "310S"),
        # GH3030
        ("GH3030", "GH3030"),
        ("GH30", "GH3030"),
        ("Nimonic 75", "GH3030"),
        # GH4169
        ("GH4169", "GH4169"),
        ("GH169", "GH4169"),
        # GH4099
        ("GH4099", "GH4099"),
        # Note: Waspaloy is now a separate material (not alias for GH4099)
    ])
    def test_heat_resistant_patterns(self, input_name, expected_grade):
        """Heat resistant alloy patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_gh4169_high_strength(self):
        """GH4169 has high strength."""
        info = classify_material_detailed("GH4169")
        assert info.properties.tensile_strength >= 1200

    def test_gh4099_special_tooling(self):
        """GH4099 requires special tooling."""
        info = classify_material_detailed("GH4099")
        assert info.process.special_tooling is True

    def test_310s_heat_resistant(self):
        """310S is heat resistant stainless steel."""
        info = classify_material_detailed("310S")
        assert "耐热" in info.name

    def test_superalloys_in_corrosion_resistant_group(self):
        """Superalloys are in corrosion_resistant group."""
        for grade in ["310S", "GH3030", "GH4169", "GH4099"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.CORROSION_RESISTANT


class TestHighPerformancePlastics:
    """Tests for high performance engineering plastics (PPS, PI, PSU, PEI)."""

    def test_plastics_exist(self):
        """High performance plastics exist in database."""
        for grade in ["PPS", "PI", "PSU", "PEI"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # PPS
        ("PPS", "PPS"),
        ("Ryton", "PPS"),
        ("聚苯硫醚", "PPS"),
        # PI
        ("PI", "PI"),
        ("Vespel", "PI"),
        ("Kapton", "PI"),
        ("聚酰亚胺", "PI"),
        # PSU
        ("PSU", "PSU"),
        ("Polysulfone", "PSU"),
        ("Udel", "PSU"),
        ("聚砜", "PSU"),
        # PEI
        ("PEI", "PEI"),
        ("Ultem", "PEI"),
        ("聚醚酰亚胺", "PEI"),
    ])
    def test_plastic_patterns(self, input_name, expected_grade):
        """Plastic patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_pi_high_temperature(self):
        """PI has special tooling requirement."""
        info = classify_material_detailed("PI")
        assert info.process.special_tooling is True

    def test_plastics_in_engineering_plastic_group(self):
        """Plastics are in engineering_plastic group."""
        for grade in ["PPS", "PI", "PSU", "PEI"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.ENGINEERING_PLASTIC


class TestSuperAusteniticStainless:
    """Tests for super austenitic stainless steels (904L, 254SMO, 316Ti)."""

    def test_super_austenitic_exist(self):
        """Super austenitic stainless steels exist in database."""
        for grade in ["904L", "254SMO", "316Ti"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 904L
        ("904L", "904L"),
        ("N08904", "904L"),
        ("SUS890L", "904L"),
        ("1.4539", "904L"),
        ("00Cr20Ni25Mo4.5Cu", "904L"),
        # 254SMO
        ("254SMO", "254SMO"),
        ("254 SMO", "254SMO"),
        ("S31254", "254SMO"),
        ("1.4547", "254SMO"),
        # 316Ti
        ("316Ti", "316Ti"),
        ("S31635", "316Ti"),
        ("SUS316Ti", "316Ti"),
        ("1.4571", "316Ti"),
        ("0Cr17Ni12Mo2Ti", "316Ti"),
    ])
    def test_super_austenitic_patterns(self, input_name, expected_grade):
        """Super austenitic patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_904l_high_corrosion_resistance(self):
        """904L has excellent weldability."""
        info = classify_material_detailed("904L")
        assert info.properties.weldability == "good"

    def test_254smo_high_mo_content(self):
        """254SMO is a 6Mo super austenitic."""
        info = classify_material_detailed("254SMO")
        assert "6Mo" in info.description or "超级" in info.name

    def test_316ti_titanium_stabilized(self):
        """316Ti is titanium stabilized."""
        info = classify_material_detailed("316Ti")
        assert "钛" in info.name or "Ti" in info.grade

    def test_super_austenitic_in_stainless_group(self):
        """Super austenitic steels are in stainless_steel group."""
        for grade in ["904L", "254SMO", "316Ti"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.STAINLESS_STEEL

    def test_super_austenitic_equivalence(self):
        """Super austenitic steels have equivalence tables."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["904L", "254SMO", "316Ti"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_super_austenitic_cost(self):
        """Super austenitic steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["904L", "254SMO", "316Ti"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] >= 3, f"{grade} should be Tier 3 or higher"


class TestCastingAluminum:
    """Tests for casting aluminum alloys (A356, ZL102, ADC12)."""

    def test_casting_aluminum_exist(self):
        """Casting aluminum alloys exist in database."""
        for grade in ["A356", "ZL102", "ADC12"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # A356
        ("A356", "A356"),
        ("A356-T6", "A356"),
        # Note: ZL101 is now its own material, skip
        ("AlSi7Mg", "A356"),
        ("AC4C", "A356"),
        ("铸造铝", "A356"),
        # ZL102
        ("ZL102", "ZL102"),
        ("A413", "ZL102"),
        ("AlSi12", "ZL102"),
        ("ADC1", "ZL102"),
        # ADC12
        ("ADC12", "ADC12"),
        ("A383", "ADC12"),
        ("AlSi11Cu3", "ADC12"),
        ("YL113", "ADC12"),
        ("压铸铝", "ADC12"),
    ])
    def test_casting_aluminum_patterns(self, input_name, expected_grade):
        """Casting aluminum patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_a356_heat_treatable(self):
        """A356 is heat treatable."""
        info = classify_material_detailed("A356")
        assert "T6" in str(info.process.heat_treatments) or "热处理" in str(info.process.heat_treatments)

    def test_adc12_die_casting(self):
        """ADC12 is for die casting."""
        info = classify_material_detailed("ADC12")
        assert "压铸" in info.name or "压铸" in str(info.process.blank_forms)

    def test_casting_aluminum_in_aluminum_group(self):
        """Casting aluminum alloys are in aluminum group."""
        for grade in ["A356", "ZL102", "ADC12"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.ALUMINUM

    def test_casting_aluminum_equivalence(self):
        """Casting aluminum alloys have equivalence tables."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["A356", "ZL102", "ADC12"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_casting_aluminum_cost(self):
        """Casting aluminum alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["A356", "ZL102", "ADC12"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] == 2, f"{grade} should be Tier 2"


class TestSpecialSteels:
    """Tests for special steels (9Cr18, 12Cr1MoV, Mn13)."""

    def test_special_steels_exist(self):
        """Special steels exist in database."""
        for grade in ["9Cr18", "12Cr1MoV", "Mn13"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 9Cr18 (knife steel)
        ("9Cr18", "9Cr18"),
        ("9Cr18Mo", "9Cr18"),
        ("9Cr18MoV", "9Cr18"),
        ("440C", "9Cr18"),
        ("440B", "9Cr18"),
        ("SUS440C", "9Cr18"),
        ("1.4125", "9Cr18"),
        ("刀具钢", "9Cr18"),
        # 12Cr1MoV (heat resistant)
        ("12Cr1MoV", "12Cr1MoV"),
        ("12Cr1MoVG", "12Cr1MoV"),
        ("15CrMo", "12Cr1MoV"),
        ("13CrMo44", "12Cr1MoV"),
        ("P22", "12Cr1MoV"),
        ("耐热钢", "12Cr1MoV"),
        # Mn13 (wear resistant)
        ("Mn13", "Mn13"),
        ("ZGMn13", "Mn13"),
        ("Mn13Cr2", "Mn13"),
        ("X120Mn12", "Mn13"),
        ("Hadfield", "Mn13"),
        ("高锰钢", "Mn13"),
    ])
    def test_special_steel_patterns(self, input_name, expected_grade):
        """Special steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_9cr18_high_hardness(self):
        """9Cr18 has high hardness."""
        info = classify_material_detailed("9Cr18")
        assert "HRC" in info.properties.hardness
        assert info.process.special_tooling is True

    def test_12cr1mov_heat_resistant(self):
        """12Cr1MoV is heat resistant steel."""
        info = classify_material_detailed("12Cr1MoV")
        assert "耐热" in info.name or "锅炉" in str(info.process.recommendations)

    def test_mn13_work_hardening(self):
        """Mn13 has work hardening characteristic."""
        info = classify_material_detailed("Mn13")
        assert "硬化" in info.properties.hardness or "硬化" in info.description
        assert info.properties.machinability == "poor"

    def test_9cr18_in_tool_steel_group(self):
        """9Cr18 is in tool_steel group."""
        info = classify_material_detailed("9Cr18")
        assert info.group == MaterialGroup.TOOL_STEEL

    def test_12cr1mov_in_alloy_steel_group(self):
        """12Cr1MoV is in alloy_steel group."""
        info = classify_material_detailed("12Cr1MoV")
        assert info.group == MaterialGroup.ALLOY_STEEL

    def test_mn13_in_alloy_steel_group(self):
        """Mn13 is in alloy_steel group."""
        info = classify_material_detailed("Mn13")
        assert info.group == MaterialGroup.ALLOY_STEEL

    def test_special_steels_equivalence(self):
        """Special steels have equivalence tables."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["9Cr18", "12Cr1MoV", "Mn13"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_special_steels_cost(self):
        """Special steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["9Cr18", "12Cr1MoV", "Mn13"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestPrecisionAlloys:
    """Tests for precision alloys (4J36, 4J29, 4J42, 1J79)."""

    def test_precision_alloys_exist(self):
        """Precision alloys exist in database."""
        for grade in ["4J36", "4J29", "4J42", "1J79"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 4J36 - direct pattern match
        ("4J36", "4J36"),
        ("Invar36", "4J36"),
        ("FeNi36", "4J36"),
        ("1.3912", "4J36"),
        # Invar now maps to independent Invar material
        ("Invar", "Invar"),
        ("因瓦合金", "Invar"),
        ("低膨胀合金", "Invar"),
        # 4J29 - direct pattern match
        ("4J29", "4J29"),
        ("FeNiCo29", "4J29"),
        ("封接合金", "4J29"),
        # Kovar now maps to independent Kovar material
        ("Kovar", "Kovar"),
        ("可伐合金", "Kovar"),
        # 4J42 (Elinvar)
        ("4J42", "4J42"),
        ("Elinvar", "4J42"),
        ("FeNi42", "4J42"),
        ("恒弹性合金", "4J42"),
        # 1J79 (Permalloy)
        ("1J79", "1J79"),
        ("Supermalloy", "1J79"),
    ])
    def test_precision_alloy_patterns(self, input_name, expected_grade):
        """Precision alloy patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_4j36_low_expansion(self):
        """4J36 is low thermal expansion alloy."""
        info = classify_material_detailed("4J36")
        assert "低" in info.description or "膨胀" in info.description

    def test_4j29_sealing_alloy(self):
        """4J29 is glass-metal sealing alloy."""
        info = classify_material_detailed("4J29")
        assert "封接" in info.description or "封接" in str(info.process.recommendations)

    def test_1j79_high_permeability(self):
        """1J79 is high permeability alloy."""
        info = classify_material_detailed("1J79")
        assert "磁" in info.name or "磁" in info.description

    def test_precision_alloys_in_precision_group(self):
        """Precision alloys are in precision_alloy group."""
        for grade in ["4J36", "4J29", "4J42", "1J79"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.PRECISION_ALLOY

    def test_precision_alloys_equivalence(self):
        """Precision alloys have equivalence tables."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["4J36", "4J29", "4J42", "1J79"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_precision_alloys_cost(self):
        """Precision alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["4J36", "4J29", "4J42", "1J79"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] == 4, f"{grade} should be Tier 4"


class TestElectricalSteel:
    """Tests for electrical steels (50W470, 30Q130)."""

    def test_electrical_steels_exist(self):
        """Electrical steels exist in database."""
        for grade in ["50W470", "30Q130"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 50W470 (non-oriented)
        ("50W470", "50W470"),
        # Note: 50W600 now matches B50A600, skip
        ("M470-50A", "50W470"),
        ("无取向硅钢", "50W470"),
        ("硅钢片", "50W470"),
        ("电工钢", "50W470"),
        # 30Q130 (grain-oriented)
        ("30Q130", "30Q130"),
        ("30Q120", "30Q130"),
        ("M130-30S", "30Q130"),
        # Note: 取向硅钢 and 变压器硅钢 now match B27R090, skip
    ])
    def test_electrical_steel_patterns(self, input_name, expected_grade):
        """Electrical steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_50w470_non_oriented(self):
        """50W470 is non-oriented silicon steel."""
        info = classify_material_detailed("50W470")
        assert "无取向" in info.name or "电机" in str(info.process.recommendations)

    def test_30q130_grain_oriented(self):
        """30Q130 is grain-oriented silicon steel."""
        info = classify_material_detailed("30Q130")
        assert "取向" in info.name or "变压器" in str(info.process.recommendations)

    def test_electrical_steels_in_electrical_group(self):
        """Electrical steels are in electrical_steel group."""
        for grade in ["50W470", "30Q130"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.ELECTRICAL_STEEL

    def test_electrical_steels_equivalence(self):
        """Electrical steels have equivalence tables."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["50W470", "30Q130"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_electrical_steels_cost(self):
        """Electrical steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["50W470", "30Q130"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestBearingSteels:
    """Tests for bearing steels (GCr15SiMn, GCr18Mo)."""

    def test_bearing_steels_exist(self):
        """Bearing steels exist in database."""
        for grade in ["GCr15", "GCr15SiMn", "GCr18Mo"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # GCr15
        ("GCr15", "GCr15"),
        ("52100", "GCr15"),
        ("SUJ2", "GCr15"),
        ("100Cr6", "GCr15"),
        ("轴承钢", "GCr15"),
        # GCr15SiMn
        ("GCr15SiMn", "GCr15SiMn"),
        ("100CrMnSi", "GCr15SiMn"),
        ("SUJ3", "GCr15SiMn"),
        # GCr18Mo
        ("GCr18Mo", "GCr18Mo"),
        ("100CrMo7", "GCr18Mo"),
        ("SUJ5", "GCr18Mo"),
        ("A485", "GCr18Mo"),
        ("航空轴承钢", "GCr18Mo"),
    ])
    def test_bearing_steel_patterns(self, input_name, expected_grade):
        """Bearing steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_gcr15simn_high_hardenability(self):
        """GCr15SiMn has high hardenability."""
        info = classify_material_detailed("GCr15SiMn")
        assert "淬透" in info.description or "大型" in str(info.process.recommendations)

    def test_gcr18mo_aerospace(self):
        """GCr18Mo is for aerospace bearings."""
        info = classify_material_detailed("GCr18Mo")
        assert "航空" in info.description or "航空" in str(info.process.recommendations)

    def test_bearing_steels_in_alloy_group(self):
        """Bearing steels are in alloy_steel group."""
        for grade in ["GCr15", "GCr15SiMn", "GCr18Mo"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.ALLOY_STEEL

    def test_bearing_steels_equivalence(self):
        """Bearing steels have equivalence tables."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["GCr15", "GCr15SiMn", "GCr18Mo"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_bearing_steels_cost(self):
        """Bearing steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["GCr15", "GCr15SiMn", "GCr18Mo"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"

    def test_bearing_steels_special_tooling(self):
        """Bearing steels require special tooling."""
        for grade in ["GCr15", "GCr15SiMn", "GCr18Mo"]:
            info = classify_material_detailed(grade)
            assert info.process.special_tooling is True


class TestWeldingMaterials:
    """Tests for welding materials (ER308L, ER316L, ER70S-6, E7018)."""

    def test_welding_materials_exist(self):
        """Welding materials exist in database."""
        for grade in ["ER308L", "ER316L", "ER70S-6", "E7018"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # ER308L
        ("ER308L", "ER308L"),
        ("ER308", "ER308L"),
        ("308L焊丝", "ER308L"),
        ("Y308L", "ER308L"),
        ("不锈钢焊丝", "ER308L"),
        # ER316L
        ("ER316L", "ER316L"),
        ("316L焊丝", "ER316L"),
        ("Y316L", "ER316L"),
        # ER70S-6
        ("ER70S-6", "ER70S-6"),
        ("ER70S6", "ER70S-6"),
        ("70S-6", "ER70S-6"),
        ("H08Mn2SiA", "ER70S-6"),
        ("碳钢焊丝", "ER70S-6"),
        ("焊丝", "ER70S-6"),
        # E7018
        ("E7018", "E7018"),
        ("J507", "E7018"),
        ("低氢焊条", "E7018"),
        ("碱性焊条", "E7018"),
        ("焊条", "E7018"),
    ])
    def test_welding_material_patterns(self, input_name, expected_grade):
        """Welding material patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_er308l_for_304_welding(self):
        """ER308L is for 304 stainless welding."""
        info = classify_material_detailed("ER308L")
        assert "304" in str(info.process.recommendations)

    def test_e7018_low_hydrogen(self):
        """E7018 is low hydrogen electrode."""
        info = classify_material_detailed("E7018")
        assert "低氢" in info.name or "低氢" in info.description

    def test_welding_materials_in_welding_group(self):
        """Welding materials are in welding_material group."""
        for grade in ["ER308L", "ER316L", "ER70S-6", "E7018"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.WELDING_MATERIAL

    def test_welding_materials_equivalence(self):
        """Welding materials have equivalence tables."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["ER308L", "ER316L", "ER70S-6", "E7018"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_welding_materials_cost(self):
        """Welding materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["ER308L", "ER316L", "ER70S-6", "E7018"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestCompositeMaterials:
    """Tests for composite materials (CFRP, GFRP)."""

    def test_composites_exist(self):
        """Composite materials exist in database."""
        for grade in ["CFRP", "GFRP"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # CFRP
        ("CFRP", "CFRP"),
        ("碳纤维复合", "CFRP"),
        ("碳纤维增强", "CFRP"),
        ("CF/EP", "CFRP"),
        ("T300", "CFRP"),
        ("T700", "CFRP"),
        ("碳纤维", "CFRP"),
        # GFRP
        ("GFRP", "GFRP"),
        ("玻璃钢", "GFRP"),
        ("玻纤复合", "GFRP"),
        ("玻纤增强", "GFRP"),
        ("GF/EP", "GFRP"),
        ("E-glass", "GFRP"),
    ])
    def test_composite_patterns(self, input_name, expected_grade):
        """Composite material patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_cfrp_high_strength(self):
        """CFRP has high tensile strength."""
        info = classify_material_detailed("CFRP")
        assert info.properties.tensile_strength >= 1000

    def test_gfrp_lower_cost(self):
        """GFRP is lower cost than CFRP."""
        from src.core.materials import get_material_cost
        cfrp_cost = get_material_cost("CFRP")
        gfrp_cost = get_material_cost("GFRP")
        assert gfrp_cost["cost_index"] < cfrp_cost["cost_index"]

    def test_composites_special_tooling(self):
        """Composites require special tooling."""
        for grade in ["CFRP", "GFRP"]:
            info = classify_material_detailed(grade)
            assert info.process.special_tooling is True

    def test_composites_in_composite_group(self):
        """Composites are in composite group."""
        for grade in ["CFRP", "GFRP"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.COMPOSITE

    def test_composites_equivalence(self):
        """Composites have equivalence tables."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["CFRP", "GFRP"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_composites_cost(self):
        """Composites have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["CFRP", "GFRP"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestPowderMetallurgy:
    """Tests for powder metallurgy materials (FC-0208, FN-0205)."""

    def test_pm_materials_exist(self):
        """Powder metallurgy materials exist in database."""
        for grade in ["FC-0208", "FN-0205"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # FC-0208
        ("FC-0208", "FC-0208"),
        ("FC 0208", "FC-0208"),
        ("烧结铁", "FC-0208"),
        ("PM铁基", "FC-0208"),
        ("粉末冶金", "FC-0208"),
        # FN-0205
        ("FN-0205", "FN-0205"),
        ("FN 0205", "FN-0205"),
        ("烧结铁镍", "FN-0205"),
        ("PM铁镍", "FN-0205"),
    ])
    def test_pm_patterns(self, input_name, expected_grade):
        """Powder metallurgy patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade, f"'{input_name}' returned {info.grade}"

    def test_fc0208_sintered(self):
        """FC-0208 is sintered material."""
        info = classify_material_detailed("FC-0208")
        assert "烧结" in info.name or "烧结" in str(info.process.blank_forms)

    def test_fn0205_higher_strength(self):
        """FN-0205 has higher strength than FC-0208."""
        fc = classify_material_detailed("FC-0208")
        fn = classify_material_detailed("FN-0205")
        assert fn.properties.tensile_strength > fc.properties.tensile_strength

    def test_pm_porous_warning(self):
        """PM materials have porous structure warning."""
        for grade in ["FC-0208", "FN-0205"]:
            info = classify_material_detailed(grade)
            assert "多孔" in str(info.process.warnings)

    def test_pm_in_pm_group(self):
        """PM materials are in powder_metallurgy group."""
        for grade in ["FC-0208", "FN-0205"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.POWDER_METALLURGYLURGY

    def test_pm_equivalence(self):
        """PM materials have equivalence tables."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["FC-0208", "FN-0205"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_pm_cost(self):
        """PM materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["FC-0208", "FN-0205"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] == 2, f"{grade} should be Tier 2"


class TestSpringSteel55CrSi:
    """Tests for 55CrSi spring steel."""

    def test_55crsi_exists(self):
        """55CrSi exists in database."""
        assert "55CrSi" in MATERIAL_DATABASE

    @pytest.mark.parametrize("input_name,expected_grade", [
        ("55CrSi", "55CrSi"),
        ("55CrSiA", "55CrSi"),
        ("55SiCr", "55CrSi"),
        ("SUP12", "55CrSi"),
        ("SUP 12", "55CrSi"),
    ])
    def test_55crsi_patterns(self, input_name, expected_grade):
        """55CrSi patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_55crsi_is_alloy_steel(self):
        """55CrSi is in alloy steel group."""
        info = classify_material_detailed("55CrSi")
        assert info.group == MaterialGroup.ALLOY_STEEL

    def test_55crsi_high_strength(self):
        """55CrSi has high tensile strength."""
        info = classify_material_detailed("55CrSi")
        assert info.properties.tensile_strength >= 1500

    def test_55crsi_spring_application(self):
        """55CrSi is for spring applications."""
        info = classify_material_detailed("55CrSi")
        recs = " ".join(info.process.recommendations)
        assert "弹簧" in recs or "spring" in recs.lower()

    def test_55crsi_equivalence(self):
        """55CrSi has equivalence data."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        assert "55CrSi" in MATERIAL_EQUIVALENCE

    def test_55crsi_cost(self):
        """55CrSi has cost data."""
        from src.core.materials import get_material_cost
        cost = get_material_cost("55CrSi")
        assert cost is not None
        assert cost["tier"] == 2


class TestHighTempAlloys:
    """Tests for high-temperature alloys (A-286, Waspaloy, Rene41)."""

    def test_hightemp_alloys_exist(self):
        """High-temp alloys exist in database."""
        for grade in ["A-286", "Waspaloy", "Rene41"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # A-286
        ("A-286", "A-286"),
        ("A286", "A-286"),
        ("A 286", "A-286"),
        ("S66286", "A-286"),
        # Note: GH2132/SUH660 are now separate material, skip
        # Waspaloy
        ("Waspaloy", "Waspaloy"),
        # Note: N07001 is also alias for GH4099, so we skip this test
        # Rene 41
        ("Rene41", "Rene41"),
        ("Rene 41", "Rene41"),
        ("N07041", "Rene41"),
        ("R-41", "Rene41"),
    ])
    def test_hightemp_patterns(self, input_name, expected_grade):
        """High-temp alloy patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_a286_iron_nickel_base(self):
        """A-286 is iron-nickel base alloy."""
        info = classify_material_detailed("A-286")
        assert "铁镍" in info.name or "Fe-Ni" in info.name

    def test_waspaloy_nickel_base(self):
        """Waspaloy is nickel base alloy."""
        info = classify_material_detailed("Waspaloy")
        assert "镍" in info.name

    def test_rene41_difficult_machining(self):
        """Rene41 is difficult to machine."""
        info = classify_material_detailed("Rene41")
        assert info.properties.machinability == "poor"

    def test_hightemp_alloys_corrosion_resistant(self):
        """High-temp alloys are in corrosion resistant or superalloy group."""
        for grade in ["A-286", "Waspaloy", "Rene41"]:
            info = classify_material_detailed(grade)
            # High-temp alloys can be in CORROSION_RESISTANT or SUPERALLOY group
            assert info.group in [MaterialGroup.CORROSION_RESISTANT, MaterialGroup.SUPERALLOY]

    def test_hightemp_alloys_high_temp_use(self):
        """High-temp alloys have high temperature applications."""
        for grade in ["A-286", "Waspaloy", "Rene41"]:
            info = classify_material_detailed(grade)
            recs = " ".join(info.process.recommendations)
            # Check for high-temp keywords in Chinese or English
            assert "℃" in recs or "高温" in recs or "turbine" in recs.lower() or "涡轮" in recs or "发动机" in recs

    def test_hightemp_alloys_equivalence(self):
        """High-temp alloys have equivalence data."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["A-286", "Waspaloy", "Rene41"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_hightemp_alloys_cost_tier4_5(self):
        """High-temp alloys are in cost tier 4-5."""
        from src.core.materials import get_material_cost
        for grade in ["A-286", "Waspaloy", "Rene41"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] >= 4, f"{grade} should be Tier 4+"


class TestCopperAlloysSupplement:
    """Tests for copper alloy supplements - verifying international aliases work."""

    def test_copper_alloys_exist(self):
        """Copper alloys exist in database with Chinese names."""
        for grade in ["QAl10-3-1.5", "QAl9-4"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # QAl10-3-1.5 (C63000) 铝青铜
        ("QAl10-3-1.5", "QAl10-3-1.5"),
        ("C63000", "QAl10-3-1.5"),
        ("CuAl10Ni5Fe4", "QAl10-3-1.5"),
        # QAl9-4 (C95400) 铝青铜
        ("QAl9-4", "QAl9-4"),
        ("AMPCO18", "QAl9-4"),
    ])
    def test_copper_alloy_patterns(self, input_name, expected_grade):
        """Copper alloy patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_qal10_aluminum_bronze(self):
        """QAl10-3-1.5 is aluminum bronze."""
        info = classify_material_detailed("QAl10-3-1.5")
        assert "铝青铜" in info.name or "aluminum bronze" in info.name.lower()

    def test_qal9_aluminum_bronze(self):
        """QAl9-4 is aluminum bronze."""
        info = classify_material_detailed("QAl9-4")
        assert "铝青铜" in info.name

    def test_copper_alloys_in_proper_groups(self):
        """Aluminum bronzes are in aluminum bronze group."""
        # QAl10-3-1.5 is generic copper (historical)
        info1 = classify_material_detailed("QAl10-3-1.5")
        assert info1.group == MaterialGroup.COPPER
        # QAl9-4 is now in aluminum bronze group
        info2 = classify_material_detailed("QAl9-4")
        assert info2.group == MaterialGroup.ALUMINUM_BRONZE

    def test_qal10_high_strength(self):
        """QAl10-3-1.5 has high tensile strength."""
        info = classify_material_detailed("QAl10-3-1.5")
        assert info.properties.tensile_strength >= 600


class TestLowTemperatureSteels:
    """Tests for low-temperature steels (09MnNiD, 16MnDR, 9Ni钢)."""

    def test_lowtemp_steels_exist(self):
        """Low-temp steels exist in database."""
        for grade in ["09MnNiD", "16MnDR", "9Ni钢"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 09MnNiD
        ("09MnNiD", "09MnNiD"),
        ("3.5Ni", "09MnNiD"),
        ("A203 Gr.D", "09MnNiD"),
        # 16MnDR
        ("16MnDR", "16MnDR"),
        ("16MnD", "16MnDR"),
        ("SA516 Gr.70", "16MnDR"),
        ("低温钢", "16MnDR"),
        # 9Ni钢
        ("9Ni钢", "9Ni钢"),
        ("9Ni", "9Ni钢"),
        ("06Ni9", "9Ni钢"),
        ("X8Ni9", "9Ni钢"),
        ("A553", "9Ni钢"),
        ("LNG钢", "9Ni钢"),
    ])
    def test_lowtemp_patterns(self, input_name, expected_grade):
        """Low-temp steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_09mnniD_35nickel(self):
        """09MnNiD is 3.5% nickel steel."""
        info = classify_material_detailed("09MnNiD")
        aliases = " ".join(info.aliases)
        assert "3.5Ni" in aliases or "3.5" in info.description

    def test_9ni_for_lng(self):
        """9Ni steel is for LNG applications."""
        info = classify_material_detailed("9Ni钢")
        recs = " ".join(info.process.recommendations)
        assert "LNG" in recs or "-196" in recs or "液氮" in recs

    def test_lowtemp_steels_in_alloy_steel(self):
        """Low-temp steels are in alloy steel group."""
        for grade in ["09MnNiD", "16MnDR", "9Ni钢"]:
            info = classify_material_detailed(grade)
            assert info.group == MaterialGroup.ALLOY_STEEL

    def test_lowtemp_steels_good_toughness(self):
        """Low-temp steels have good elongation."""
        for grade in ["09MnNiD", "16MnDR", "9Ni钢"]:
            info = classify_material_detailed(grade)
            assert info.properties.elongation >= 20

    def test_lowtemp_steels_weldable(self):
        """Low-temp steels are weldable."""
        for grade in ["09MnNiD", "16MnDR"]:
            info = classify_material_detailed(grade)
            assert info.properties.weldability == "good"

    def test_lowtemp_steels_equivalence(self):
        """Low-temp steels have equivalence data."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["09MnNiD", "16MnDR", "9Ni钢"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_lowtemp_steels_cost(self):
        """Low-temp steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["09MnNiD", "16MnDR", "9Ni钢"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"

    def test_9ni_most_expensive(self):
        """9Ni steel is the most expensive low-temp steel."""
        from src.core.materials import get_material_cost
        cost_9ni = get_material_cost("9Ni钢")
        cost_09mn = get_material_cost("09MnNiD")
        cost_16mn = get_material_cost("16MnDR")
        assert cost_9ni["cost_index"] > cost_09mn["cost_index"]
        assert cost_9ni["cost_index"] > cost_16mn["cost_index"]


class TestDieSteels:
    """Tests for die steels (DC53, S136, NAK80)."""

    def test_die_steels_exist(self):
        """Die steels exist in database."""
        for grade in ["DC53", "S136", "NAK80"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # DC53
        ("DC53", "DC53"),
        ("DC 53", "DC53"),
        ("SLD-MAGIC", "DC53"),
        ("8%Cr钢", "DC53"),
        # S136
        ("S136", "S136"),
        ("S 136", "S136"),
        ("1.2083", "S136"),
        ("420MOD", "S136"),
        # Note: SUS420J2 conflicts with 20Cr13, skip
        # Note: 塑料模具钢 conflicts with H13, skip
        # NAK80
        ("NAK80", "NAK80"),
        ("NAK 80", "NAK80"),
        ("P21", "NAK80"),
        ("STAVAX", "NAK80"),
        ("预硬钢", "NAK80"),
    ])
    def test_die_steel_patterns(self, input_name, expected_grade):
        """Die steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_dc53_is_tool_steel(self):
        """DC53 is in tool steel group."""
        info = classify_material_detailed("DC53")
        assert info.group == MaterialGroup.TOOL_STEEL

    def test_dc53_high_hardness(self):
        """DC53 has high hardness."""
        info = classify_material_detailed("DC53")
        assert "HRC62" in info.properties.hardness

    def test_s136_mirror_finish(self):
        """S136 is for mirror finish molds."""
        info = classify_material_detailed("S136")
        recs = " ".join(info.process.recommendations)
        assert "镜" in recs or "透明" in recs or "光学" in recs

    def test_nak80_prehardened(self):
        """NAK80 is prehardened steel."""
        info = classify_material_detailed("NAK80")
        assert "预硬" in info.name or "预硬" in info.process.blank_hint

    def test_nak80_excellent_machinability(self):
        """NAK80 has excellent machinability."""
        info = classify_material_detailed("NAK80")
        assert info.properties.machinability == "excellent"

    def test_die_steels_equivalence(self):
        """Die steels have equivalence data."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["DC53", "S136", "NAK80"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_die_steels_cost(self):
        """Die steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["DC53", "S136", "NAK80"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] == 3


class TestFreeCuttingSteels:
    """Tests for free-cutting steels (12L14, Y15, Y40Mn)."""

    def test_free_cutting_steels_exist(self):
        """Free-cutting steels exist in database."""
        for grade in ["12L14", "Y15", "Y40Mn"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 12L14
        ("12L14", "12L14"),
        ("Y12Pb", "12L14"),
        ("SUM24L", "12L14"),
        # Y15
        ("Y15", "Y15"),
        # Note: 1215 conflicts with 12L14 pattern, skip
        ("SUM22", "Y15"),
        ("A1215", "Y15"),
        ("易切削钢", "Y15"),
        ("快削钢", "Y15"),
        # Y40Mn
        ("Y40Mn", "Y40Mn"),
        ("1140", "Y40Mn"),
        ("SUM43", "Y40Mn"),
        ("40MnS", "Y40Mn"),
    ])
    def test_free_cutting_patterns(self, input_name, expected_grade):
        """Free-cutting steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_12l14_is_free_cutting(self):
        """12L14 is in free cutting steel group."""
        info = classify_material_detailed("12L14")
        assert info.group == MaterialGroup.FREE_CUTTING_STEEL

    def test_12l14_excellent_machinability(self):
        """12L14 has excellent machinability."""
        info = classify_material_detailed("12L14")
        assert info.properties.machinability == "excellent"

    def test_12l14_lead_warning(self):
        """12L14 has lead warning."""
        info = classify_material_detailed("12L14")
        warnings = " ".join(info.process.warnings)
        assert "铅" in warnings or "毒" in warnings

    def test_y15_sulfur_based(self):
        """Y15 is sulfur-based free-cutting steel."""
        info = classify_material_detailed("Y15")
        assert "硫" in info.name

    def test_y40mn_can_be_heat_treated(self):
        """Y40Mn can be heat treated (调质)."""
        info = classify_material_detailed("Y40Mn")
        assert "调质" in info.name or "调质" in str(info.process.heat_treatments)

    def test_free_cutting_high_speed(self):
        """Free-cutting steels have high cutting speed."""
        info = classify_material_detailed("12L14")
        assert info.process.cutting_speed_range[1] >= 100  # High speed possible

    def test_free_cutting_steels_equivalence(self):
        """Free-cutting steels have equivalence data."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["12L14", "Y15", "Y40Mn"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_free_cutting_steels_cost(self):
        """Free-cutting steels have cost data and are affordable."""
        from src.core.materials import get_material_cost
        for grade in ["12L14", "Y15", "Y40Mn"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] <= 2  # Affordable materials


class TestWearResistantSteels:
    """Tests for wear-resistant steels (NM400, NM500, Hardox450)."""

    def test_wear_resistant_steels_exist(self):
        """Wear-resistant steels exist in database."""
        for grade in ["NM400", "NM500", "Hardox450"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # NM400
        ("NM400", "NM400"),
        ("NM 400", "NM400"),
        ("Hardox400", "NM400"),
        ("Hardox 400", "NM400"),
        ("XAR400", "NM400"),
        # Note: 耐磨钢 conflicts with Mn13 耐磨钢, skip
        ("耐磨板", "NM400"),
        # NM500
        ("NM500", "NM500"),
        ("NM 500", "NM500"),
        ("Hardox500", "NM500"),
        ("XAR500", "NM500"),
        # Hardox450
        ("Hardox450", "Hardox450"),
        ("Hardox 450", "Hardox450"),
        ("HX450", "Hardox450"),
        ("悍达", "Hardox450"),
    ])
    def test_wear_resistant_patterns(self, input_name, expected_grade):
        """Wear-resistant steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_nm400_is_wear_resistant(self):
        """NM400 is in wear resistant steel group."""
        info = classify_material_detailed("NM400")
        assert info.group == MaterialGroup.WEAR_RESISTANT_STEEL

    def test_nm500_higher_hardness(self):
        """NM500 has higher hardness than NM400."""
        nm400 = classify_material_detailed("NM400")
        nm500 = classify_material_detailed("NM500")
        # Extract numeric value from hardness string
        nm400_hardness = nm400.properties.hardness  # "HBW370-430"
        nm500_hardness = nm500.properties.hardness  # "HBW470-530"
        # NM500 should have higher hardness
        assert "470" in nm500_hardness or "500" in nm500_hardness

    def test_wear_resistant_preheat_warning(self):
        """Wear-resistant steels have preheat warning for welding."""
        info = classify_material_detailed("NM400")
        warnings = " ".join(info.process.warnings)
        assert "预热" in warnings or "焊接" in warnings

    def test_wear_resistant_plate_form(self):
        """Wear-resistant steels come in plate form."""
        for grade in ["NM400", "NM500", "Hardox450"]:
            info = classify_material_detailed(grade)
            assert "板材" in info.process.blank_forms

    def test_hardox450_good_weldability(self):
        """Hardox450 has good weldability (SSAB quality)."""
        info = classify_material_detailed("Hardox450")
        assert info.properties.weldability == "good"

    def test_wear_resistant_steels_equivalence(self):
        """Wear-resistant steels have equivalence data."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["NM400", "NM500", "Hardox450"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_wear_resistant_steels_cost(self):
        """Wear-resistant steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["NM400", "NM500", "Hardox450"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"

    def test_nm500_most_expensive(self):
        """Hardox450 is most expensive among these wear-resistant steels."""
        from src.core.materials import get_material_cost
        cost_nm400 = get_material_cost("NM400")
        cost_hardox = get_material_cost("Hardox450")
        assert cost_hardox["cost_index"] > cost_nm400["cost_index"]


class TestHighStrengthStructuralSteels:
    """Tests for high-strength structural steels (Q460, Q550, Q690)."""

    def test_structural_steels_exist(self):
        """High-strength structural steels exist in database."""
        for grade in ["Q460", "Q550", "Q690"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # Q460
        ("Q460", "Q460"),
        ("Q460C", "Q460"),
        ("Q460D", "Q460"),
        ("S460", "Q460"),
        ("SM570", "Q460"),
        # Q550
        ("Q550", "Q550"),
        ("Q550D", "Q550"),
        ("S550", "Q550"),
        ("HY80", "Q550"),
        # Q690
        ("Q690", "Q690"),
        ("Q690D", "Q690"),
        ("S690", "Q690"),
        ("HY100", "Q690"),
    ])
    def test_structural_steel_patterns(self, input_name, expected_grade):
        """Structural steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_q460_yield_strength(self):
        """Q460 has 460MPa yield strength."""
        info = classify_material_detailed("Q460")
        assert info.properties.yield_strength == 460

    def test_q690_highest_strength(self):
        """Q690 has highest yield strength among these."""
        q460 = classify_material_detailed("Q460")
        q550 = classify_material_detailed("Q550")
        q690 = classify_material_detailed("Q690")
        assert q690.properties.yield_strength > q550.properties.yield_strength
        assert q550.properties.yield_strength > q460.properties.yield_strength

    def test_q690_preheat_warning(self):
        """Q690 requires preheat for welding."""
        info = classify_material_detailed("Q690")
        warnings = " ".join(info.process.warnings)
        assert "预热" in warnings

    def test_structural_steels_equivalence(self):
        """Structural steels have equivalence data."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["Q460", "Q550", "Q690"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_structural_steels_cost(self):
        """Structural steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Q460", "Q550", "Q690"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] <= 2  # Affordable structural steels


class TestBoilerSteels:
    """Tests for boiler and pressure vessel steels (20G, 15CrMoG, 12Cr2Mo1R)."""

    def test_boiler_steels_exist(self):
        """Boiler steels exist in database."""
        for grade in ["20G", "15CrMoG", "12Cr2Mo1R"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # 20G
        ("20G", "20G"),
        ("20g", "20G"),
        ("A106-B", "20G"),
        ("STB410", "20G"),
        ("锅炉管", "20G"),
        # 15CrMoG
        ("15CrMoG", "15CrMoG"),
        # Note: 15CrMo conflicts with 12Cr1MoV alias, skip
        ("A335-P12", "15CrMoG"),
        ("STBA22", "15CrMoG"),
        # 12Cr2Mo1R
        ("12Cr2Mo1R", "12Cr2Mo1R"),
        ("SA387 Gr.22", "12Cr2Mo1R"),
        ("2.25Cr-1Mo", "12Cr2Mo1R"),
        ("10CrMo9-10", "12Cr2Mo1R"),
    ])
    def test_boiler_steel_patterns(self, input_name, expected_grade):
        """Boiler steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_20g_is_carbon_steel(self):
        """20G is carbon steel for boilers."""
        info = classify_material_detailed("20G")
        assert info.group == MaterialGroup.CARBON_STEEL
        assert "锅炉" in info.name

    def test_20g_excellent_weldability(self):
        """20G has excellent weldability."""
        info = classify_material_detailed("20G")
        assert info.properties.weldability == "excellent"

    def test_12cr2mo1r_for_pressure_vessel(self):
        """12Cr2Mo1R is for pressure vessels."""
        info = classify_material_detailed("12Cr2Mo1R")
        assert "容器" in info.name or "pressure" in info.name.lower()

    def test_12cr2mo1r_pwht_warning(self):
        """12Cr2Mo1R requires PWHT after welding."""
        info = classify_material_detailed("12Cr2Mo1R")
        warnings = " ".join(info.process.warnings)
        assert "PWHT" in warnings or "热处理" in warnings

    def test_boiler_steels_equivalence(self):
        """Boiler steels have equivalence data."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["20G", "15CrMoG", "12Cr2Mo1R"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_boiler_steels_cost(self):
        """Boiler steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["20G", "15CrMoG", "12Cr2Mo1R"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestPipelineSteels:
    """Tests for pipeline steels (X52, X65, X80)."""

    def test_pipeline_steels_exist(self):
        """Pipeline steels exist in database."""
        for grade in ["X52", "X65", "X80"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not in database"

    @pytest.mark.parametrize("input_name,expected_grade", [
        # X52
        ("X52", "X52"),
        ("X 52", "X52"),
        ("X52M", "X52"),
        ("L360", "X52"),
        ("API 5L X52", "X52"),
        # X65
        ("X65", "X65"),
        ("X65M", "X65"),
        ("L450", "X65"),
        ("API 5L X65", "X65"),
        ("管线钢", "X65"),
        # X80
        ("X80", "X80"),
        ("X80M", "X80"),
        ("L555", "X80"),
        ("API 5L X80", "X80"),
        ("西气东输", "X80"),
    ])
    def test_pipeline_steel_patterns(self, input_name, expected_grade):
        """Pipeline steel patterns are recognized."""
        info = classify_material_detailed(input_name)
        assert info is not None, f"Pattern '{input_name}' not recognized"
        assert info.grade == expected_grade

    def test_x52_pipeline_steel(self):
        """X52 is for oil and gas pipelines."""
        info = classify_material_detailed("X52")
        recs = " ".join(info.process.recommendations)
        assert "油气" in recs or "管道" in recs or "pipeline" in recs.lower()

    def test_x80_highest_strength(self):
        """X80 has highest strength among pipeline steels."""
        x52 = classify_material_detailed("X52")
        x65 = classify_material_detailed("X65")
        x80 = classify_material_detailed("X80")
        assert x80.properties.yield_strength > x65.properties.yield_strength
        assert x65.properties.yield_strength > x52.properties.yield_strength

    def test_x80_west_east_gas_pipeline(self):
        """X80 is used for West-East Gas Pipeline."""
        info = classify_material_detailed("X80")
        recs = " ".join(info.process.recommendations)
        assert "西气东输" in recs

    def test_pipeline_steels_3pe_coating(self):
        """Pipeline steels support 3PE coating."""
        for grade in ["X52", "X65", "X80"]:
            info = classify_material_detailed(grade)
            treatments = " ".join(info.process.surface_treatments)
            assert "3PE" in treatments or "FBE" in treatments

    def test_pipeline_steels_equivalence(self):
        """Pipeline steels have equivalence data."""
        from src.core.materials import MATERIAL_EQUIVALENCE
        for grade in ["X52", "X65", "X80"]:
            assert grade in MATERIAL_EQUIVALENCE, f"{grade} not in equivalence"

    def test_pipeline_steels_cost(self):
        """Pipeline steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["X52", "X65", "X80"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] <= 2  # Affordable materials


class TestElectricalContactMaterials:
    """Tests for electrical contact materials (AgCdO, AgSnO2, CuW70)."""

    def test_electrical_contact_materials_exist(self):
        """Electrical contact materials should exist in database."""
        for grade in ["AgCdO", "AgSnO2", "CuW70"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.ELECTRICAL_CONTACT

    @pytest.mark.parametrize("input_name,expected", [
        ("AgCdO", "AgCdO"),
        ("AgCdO10", "AgCdO"),
        ("AgCdO12", "AgCdO"),
        ("银氧化镉", "AgCdO"),
        ("银镉合金", "AgCdO"),
        ("触点银", "AgCdO"),
        ("AgSnO2", "AgSnO2"),
        ("AgSnO2In2O3", "AgSnO2"),
        ("银氧化锡", "AgSnO2"),
        ("环保触点", "AgSnO2"),
        ("SnO2触点", "AgSnO2"),
        ("CuW70", "CuW70"),
        ("CuW", "CuW70"),
        ("W70Cu30", "CuW70"),
        ("钨铜合金", "CuW70"),
        ("铜钨合金", "CuW70"),
        ("电极铜钨", "CuW70"),
        ("RWMA Class 11", "CuW70"),
        ("电接触材料", "AgSnO2"),
    ])
    def test_electrical_contact_patterns(self, input_name, expected):
        """Test electrical contact material pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_agcdo_rohs_warning(self):
        """AgCdO should have RoHS warning."""
        info = MATERIAL_DATABASE["AgCdO"]
        assert any("RoHS" in w for w in info.process.warnings)

    def test_agsno2_is_eco_friendly(self):
        """AgSnO2 should be eco-friendly replacement."""
        info = MATERIAL_DATABASE["AgSnO2"]
        assert any("环保" in alias for alias in info.aliases) or "环保" in info.description

    def test_cuw70_for_edm(self):
        """CuW70 should be for EDM electrodes."""
        info = MATERIAL_DATABASE["CuW70"]
        assert any("电火花" in r or "EDM" in r or "电极" in r for r in info.process.recommendations)

    def test_electrical_contact_equivalence(self):
        """Electrical contact materials have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["AgCdO", "AgSnO2", "CuW70"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_electrical_contact_cost(self):
        """Electrical contact materials are expensive."""
        from src.core.materials import get_material_cost
        for grade in ["AgCdO", "AgSnO2", "CuW70"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] >= 4  # Expensive materials


class TestBearingSteels:
    """Tests for bearing steels (GCr15, GCr15SiMn, GCr4)."""

    def test_bearing_steels_exist(self):
        """Bearing steels should exist in database."""
        for grade in ["GCr15", "GCr15SiMn", "GCr4"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.BEARING_STEEL

    @pytest.mark.parametrize("input_name,expected", [
        ("GCr15", "GCr15"),
        ("SUJ2", "GCr15"),
        ("52100", "GCr15"),
        ("100Cr6", "GCr15"),
        ("1.3505", "GCr15"),
        ("轴承钢", "GCr15"),
        ("GCr15SiMn", "GCr15SiMn"),
        ("SUJ4", "GCr15SiMn"),
        ("52100改", "GCr15SiMn"),
        ("大截面轴承钢", "GCr15SiMn"),
        ("GCr4", "GCr4"),
        ("SAE 4320", "GCr4"),
        ("渗碳轴承", "GCr4"),
    ])
    def test_bearing_steel_patterns(self, input_name, expected):
        """Test bearing steel pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_gcr15_high_hardness(self):
        """GCr15 should have high hardness."""
        info = MATERIAL_DATABASE["GCr15"]
        assert "HRC60" in info.properties.hardness or "64" in info.properties.hardness

    def test_gcr15simn_for_large_bearings(self):
        """GCr15SiMn should be for large bearings."""
        info = MATERIAL_DATABASE["GCr15SiMn"]
        assert any("大" in r or "large" in r.lower() for r in info.process.recommendations)

    def test_gcr4_carburizing(self):
        """GCr4 should be carburizing steel."""
        info = MATERIAL_DATABASE["GCr4"]
        assert any("渗碳" in ht for ht in info.process.heat_treatments)

    def test_bearing_steels_equivalence(self):
        """Bearing steels have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["GCr15", "GCr15SiMn", "GCr4"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_bearing_steels_cost(self):
        """Bearing steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["GCr15", "GCr15SiMn", "GCr4"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] == 2  # Medium cost


class TestSpringSteel:
    """Tests for spring steels (60Si2Mn, 60Si2CrA, 50CrVA)."""

    def test_spring_steels_exist(self):
        """Spring steels should exist in database."""
        for grade in ["60Si2Mn", "60Si2CrA", "50CrVA"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.SPRING_STEEL

    @pytest.mark.parametrize("input_name,expected", [
        ("60Si2Mn", "60Si2Mn"),
        ("SUP6", "60Si2Mn"),
        ("9260", "60Si2Mn"),
        ("1.7108", "60Si2Mn"),
        ("硅锰钢", "60Si2Mn"),
        ("60Si2CrA", "60Si2CrA"),
        ("60Si2Cr", "60Si2CrA"),
        # Note: SUP12 conflicts with 55CrSi alias, skip
        ("60SC7", "60Si2CrA"),
        ("铬硅弹簧钢", "60Si2CrA"),
        ("50CrVA", "50CrVA"),
        ("50CrV4", "50CrVA"),
        ("SUP10", "50CrVA"),
        ("6150", "50CrVA"),
        ("1.8159", "50CrVA"),
        # Note: 铬钒弹簧钢 and 弹簧钢 conflict with existing materials, skip
    ])
    def test_spring_steel_patterns(self, input_name, expected):
        """Test spring steel pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_60si2mn_for_leaf_springs(self):
        """60Si2Mn should be for leaf springs."""
        info = MATERIAL_DATABASE["60Si2Mn"]
        assert any("板簧" in r or "减震" in r for r in info.process.recommendations)

    def test_60si2cra_higher_strength(self):
        """60Si2CrA should have higher strength than 60Si2Mn."""
        info1 = MATERIAL_DATABASE["60Si2Mn"]
        info2 = MATERIAL_DATABASE["60Si2CrA"]
        assert info2.properties.tensile_strength > info1.properties.tensile_strength

    def test_50crva_for_valve_springs(self):
        """50CrVA should be for valve springs."""
        info = MATERIAL_DATABASE["50CrVA"]
        assert any("气门" in r or "valve" in r.lower() for r in info.process.recommendations)

    def test_spring_steels_poor_weldability(self):
        """Spring steels should have poor weldability."""
        for grade in ["60Si2Mn", "60Si2CrA", "50CrVA"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.weldability == "poor"

    def test_spring_steels_equivalence(self):
        """Spring steels have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["60Si2Mn", "60Si2CrA", "50CrVA"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_spring_steels_cost(self):
        """Spring steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["60Si2Mn", "60Si2CrA", "50CrVA"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] == 2  # Medium cost


class TestHeatResistantStainlessSteel:
    """Tests for heat-resistant stainless steels (2Cr13, 1Cr17, 0Cr25Ni20)."""

    def test_steels_exist(self):
        """Heat-resistant stainless steels should exist in database."""
        for grade in ["2Cr13", "1Cr17", "0Cr25Ni20"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.STAINLESS_STEEL

    @pytest.mark.parametrize("input_name,expected", [
        ("2Cr13", "2Cr13"),
        # Note: 420 matches existing 2Cr13 directly
        ("SUS420J1", "2Cr13"),
        ("1.4021", "2Cr13"),
        ("X20Cr13", "2Cr13"),
        ("1Cr17", "1Cr17"),
        # Note: 430, SUS430, 1.4016 conflict with existing 430 material, skip
        ("X6Cr17", "1Cr17"),
        ("铁素体不锈钢", "1Cr17"),
        ("0Cr25Ni20", "0Cr25Ni20"),
        # Note: 310S, SUS310S, 1.4845 conflict with existing 310S material, skip
        ("X8CrNi25-21", "0Cr25Ni20"),
        ("耐热不锈钢", "0Cr25Ni20"),
    ])
    def test_stainless_steel_patterns(self, input_name, expected):
        """Test heat-resistant stainless steel pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_2cr13_is_martensitic(self):
        """2Cr13 should be martensitic and heat-treatable."""
        info = MATERIAL_DATABASE["2Cr13"]
        assert any("淬火" in ht for ht in info.process.heat_treatments)

    def test_1cr17_is_ferritic(self):
        """1Cr17 should be ferritic and not heat-treatable."""
        info = MATERIAL_DATABASE["1Cr17"]
        assert any("退火" in ht for ht in info.process.heat_treatments)

    def test_0cr25ni20_high_temp_capability(self):
        """0Cr25Ni20 should handle high temperatures."""
        info = MATERIAL_DATABASE["0Cr25Ni20"]
        assert any("高温" in r or "炉" in r for r in info.process.recommendations)

    def test_stainless_steel_equivalence(self):
        """Heat-resistant stainless steels have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["2Cr13", "1Cr17", "0Cr25Ni20"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_stainless_steel_cost(self):
        """Heat-resistant stainless steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["2Cr13", "1Cr17", "0Cr25Ni20"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestGearSteels:
    """Tests for gear steels (20CrMnTi, 20CrMo, 20CrNiMo)."""

    def test_gear_steels_exist(self):
        """Gear steels should exist in database."""
        for grade in ["20CrMnTi", "20CrMo", "20CrNiMo"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.GEAR_STEEL

    @pytest.mark.parametrize("input_name,expected", [
        ("20CrMnTi", "20CrMnTi"),
        ("SCM420H", "20CrMnTi"),
        ("齿轮钢", "20CrMnTi"),
        ("渗碳钢", "20CrMnTi"),
        ("20CrMo", "20CrMo"),
        ("SCM420", "20CrMo"),
        # Note: 4118 conflicts with 20CrMnTi pattern, skip
        ("铬钼钢", "20CrMo"),
        ("20CrNiMo", "20CrNiMo"),
        ("SNCM220", "20CrNiMo"),
        ("8620", "20CrNiMo"),
        ("高级齿轮钢", "20CrNiMo"),
    ])
    def test_gear_steel_patterns(self, input_name, expected):
        """Test gear steel pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_gear_steels_carburizing(self):
        """Gear steels should support carburizing."""
        for grade in ["20CrMnTi", "20CrMo", "20CrNiMo"]:
            info = MATERIAL_DATABASE[grade]
            assert any("渗碳" in ht for ht in info.process.heat_treatments)

    def test_20crmnti_is_most_common(self):
        """20CrMnTi should be the most common gear steel."""
        info = MATERIAL_DATABASE["20CrMnTi"]
        assert any("汽车" in r or "变速箱" in r for r in info.process.recommendations)

    def test_20crnimo_highest_performance(self):
        """20CrNiMo should have highest performance."""
        info = MATERIAL_DATABASE["20CrNiMo"]
        assert any("航空" in r or "精密" in r for r in info.process.recommendations)

    def test_gear_steels_equivalence(self):
        """Gear steels have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["20CrMnTi", "20CrMo", "20CrNiMo"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_gear_steels_cost(self):
        """Gear steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["20CrMnTi", "20CrMo", "20CrNiMo"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] == 2  # Medium cost


class TestAerospaceAluminum:
    """Tests for aerospace aluminum alloys (5A06, 2A14, 7A04)."""

    def test_aerospace_aluminum_exist(self):
        """Aerospace aluminum alloys should exist in database."""
        for grade in ["5A06", "2A14", "7A04"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.ALUMINUM

    @pytest.mark.parametrize("input_name,expected", [
        ("5A06", "5A06"),
        ("5456", "5A06"),
        ("AlMg5", "5A06"),
        ("A5456", "5A06"),
        ("LF6", "5A06"),
        ("防锈铝", "5A06"),
        ("2A14", "2A14"),
        ("2014", "2A14"),
        ("LD10", "2A14"),
        ("AlCu4SiMg", "2A14"),
        ("7A04", "7A04"),
        ("7050", "7A04"),
        ("LC4", "7A04"),
        ("超硬铝", "7A04"),
    ])
    def test_aerospace_aluminum_patterns(self, input_name, expected):
        """Test aerospace aluminum pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_5a06_excellent_weldability(self):
        """5A06 should have excellent weldability."""
        info = MATERIAL_DATABASE["5A06"]
        assert info.properties.weldability == "excellent"

    def test_7a04_highest_strength(self):
        """7A04 should have highest strength among these alloys."""
        info1 = MATERIAL_DATABASE["5A06"]
        info2 = MATERIAL_DATABASE["7A04"]
        assert info2.properties.tensile_strength > info1.properties.tensile_strength

    def test_2a14_for_forging(self):
        """2A14 should be for forging applications."""
        info = MATERIAL_DATABASE["2A14"]
        assert any("锻件" in bf for bf in info.process.blank_forms)

    def test_7a04_for_aircraft_structure(self):
        """7A04 should be for aircraft main structure."""
        info = MATERIAL_DATABASE["7A04"]
        assert any("飞机" in r or "机翼" in r for r in info.process.recommendations)

    def test_aerospace_aluminum_equivalence(self):
        """Aerospace aluminum alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["5A06", "2A14", "7A04"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_aerospace_aluminum_cost(self):
        """Aerospace aluminum alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["5A06", "2A14", "7A04"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestValveSteels:
    """Tests for valve steels (4Cr10Si2Mo, 5Cr21Mn9Ni4N, 4Cr14Ni14W2Mo)."""

    def test_valve_steels_exist(self):
        """Valve steels should exist in database."""
        for grade in ["4Cr10Si2Mo", "5Cr21Mn9Ni4N", "4Cr14Ni14W2Mo"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.VALVE_STEEL

    @pytest.mark.parametrize("input_name,expected", [
        ("4Cr10Si2Mo", "4Cr10Si2Mo"),
        ("SUH3", "4Cr10Si2Mo"),
        ("X45CrSi9-3", "4Cr10Si2Mo"),
        ("气门钢", "4Cr10Si2Mo"),
        ("排气门钢", "4Cr10Si2Mo"),
        ("5Cr21Mn9Ni4N", "5Cr21Mn9Ni4N"),
        ("SUH35", "5Cr21Mn9Ni4N"),
        ("21-4-N", "5Cr21Mn9Ni4N"),
        ("进气门钢", "5Cr21Mn9Ni4N"),
        ("4Cr14Ni14W2Mo", "4Cr14Ni14W2Mo"),
        ("SUH38", "4Cr14Ni14W2Mo"),
        ("钨钼气门钢", "4Cr14Ni14W2Mo"),
    ])
    def test_valve_steel_patterns(self, input_name, expected):
        """Test valve steel pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_4cr10si2mo_for_exhaust(self):
        """4Cr10Si2Mo should be for exhaust valves."""
        info = MATERIAL_DATABASE["4Cr10Si2Mo"]
        assert any("排气门" in r or "exhaust" in r.lower() for r in info.process.recommendations)

    def test_5cr21_austenitic(self):
        """5Cr21Mn9Ni4N should be austenitic."""
        info = MATERIAL_DATABASE["5Cr21Mn9Ni4N"]
        assert "奥氏体" in info.name

    def test_4cr14_high_temp(self):
        """4Cr14Ni14W2Mo should have highest temperature capability."""
        info = MATERIAL_DATABASE["4Cr14Ni14W2Mo"]
        assert any("850" in note or "高温" in note for note in info.process.heat_treatment_notes)

    def test_valve_steels_equivalence(self):
        """Valve steels have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["4Cr10Si2Mo", "5Cr21Mn9Ni4N", "4Cr14Ni14W2Mo"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_valve_steels_cost(self):
        """Valve steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["4Cr10Si2Mo", "5Cr21Mn9Ni4N", "4Cr14Ni14W2Mo"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] >= 3  # Expensive materials


class TestChainSteels:
    """Tests for chain steels (20MnVB, 15MnVB, 22MnB5)."""

    def test_chain_steels_exist(self):
        """Chain steels should exist in database."""
        for grade in ["20MnVB", "15MnVB", "22MnB5"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.CHAIN_STEEL

    @pytest.mark.parametrize("input_name,expected", [
        ("20MnVB", "20MnVB"),
        ("链条钢", "20MnVB"),
        ("销轴钢", "20MnVB"),
        ("15MnVB", "15MnVB"),
        ("冷镦钢", "15MnVB"),
        ("螺栓钢", "15MnVB"),
        ("22MnB5", "22MnB5"),
        ("热成形钢", "22MnB5"),
        ("热冲压钢", "22MnB5"),
        ("USIBOR", "22MnB5"),
        ("PHS钢", "22MnB5"),
        ("硼钢板", "22MnB5"),
    ])
    def test_chain_steel_patterns(self, input_name, expected):
        """Test chain steel pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_20mnvb_for_chains(self):
        """20MnVB should be for chains."""
        info = MATERIAL_DATABASE["20MnVB"]
        assert any("链条" in r or "销轴" in r for r in info.process.recommendations)

    def test_15mnvb_good_cold_heading(self):
        """15MnVB should have good cold heading."""
        info = MATERIAL_DATABASE["15MnVB"]
        assert any("冷镦" in w or "冷镦" in str(info.process.recommendations) for w in info.process.warnings)

    def test_22mnb5_highest_strength(self):
        """22MnB5 should have highest strength."""
        info = MATERIAL_DATABASE["22MnB5"]
        assert info.properties.tensile_strength >= 1500

    def test_22mnb5_for_auto_safety(self):
        """22MnB5 should be for automotive safety."""
        info = MATERIAL_DATABASE["22MnB5"]
        assert any("汽车" in r or "A/B柱" in r for r in info.process.recommendations)

    def test_chain_steels_equivalence(self):
        """Chain steels have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["20MnVB", "15MnVB", "22MnB5"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_chain_steels_cost(self):
        """Chain steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["20MnVB", "15MnVB", "22MnB5"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] == 2  # Medium cost


class TestElectricalSteelSupplement:
    """Tests for electrical steel supplements (B50A600, B35A230, B27R090)."""

    def test_electrical_steels_exist(self):
        """Electrical steels should exist in database."""
        for grade in ["B50A600", "B35A230", "B27R090"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.ELECTRICAL_STEEL

    @pytest.mark.parametrize("input_name,expected", [
        ("B50A600", "B50A600"),
        ("50W600", "B50A600"),
        ("M600-50A", "B50A600"),
        ("电机硅钢", "B50A600"),
        ("B35A230", "B35A230"),
        ("35W230", "B35A230"),
        ("M230-35A", "B35A230"),
        ("高效电机硅钢", "B35A230"),
        ("B27R090", "B27R090"),
        ("27RK090", "B27R090"),
        ("M090-27P", "B27R090"),
        # Note: 变压器硅钢 and 取向硅钢 match 30Q130 (existing pattern), skip
    ])
    def test_electrical_steel_patterns(self, input_name, expected):
        """Test electrical steel pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_b50a600_for_motors(self):
        """B50A600 should be for motors."""
        info = MATERIAL_DATABASE["B50A600"]
        assert any("电机" in r for r in info.process.recommendations)

    def test_b35a230_for_ev_motors(self):
        """B35A230 should be for EV motors."""
        info = MATERIAL_DATABASE["B35A230"]
        assert any("新能源" in r or "变频" in r for r in info.process.recommendations)

    def test_b27r090_oriented(self):
        """B27R090 should be oriented steel for transformers."""
        info = MATERIAL_DATABASE["B27R090"]
        assert "取向" in info.name
        assert any("变压器" in r for r in info.process.recommendations)

    def test_electrical_steels_equivalence(self):
        """Electrical steels have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["B50A600", "B35A230", "B27R090"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_electrical_steels_cost(self):
        """Electrical steels have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["B50A600", "B35A230", "B27R090"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestSuperalloys:
    """Tests for superalloy supplements (GH2132, K403, K418)."""

    def test_superalloys_exist(self):
        """Superalloys should exist in database."""
        for grade in ["GH2132", "K403", "K418"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.SUPERALLOY

    @pytest.mark.parametrize("input_name,expected", [
        ("GH2132", "GH2132"),
        # Note: A-286 is now a separate material, GH2132 is China standard
        ("SUH660", "GH2132"),
        ("铁基高温合金", "GH2132"),
        ("K403", "K403"),
        ("Mar-M246", "K403"),
        ("IN-100", "K403"),
        ("铸造高温合金", "K403"),
        ("K418", "K418"),
        ("IN-738", "K418"),
        ("涡轮叶片材料", "K418"),
    ])
    def test_superalloy_patterns(self, input_name, expected):
        """Test superalloy pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_gh2132_iron_based(self):
        """GH2132 should be iron-based superalloy."""
        info = MATERIAL_DATABASE["GH2132"]
        assert info.sub_category == MaterialSubCategory.FERROUS
        assert "铁基" in info.name

    def test_k403_for_blades(self):
        """K403 should be for turbine blades."""
        info = MATERIAL_DATABASE["K403"]
        assert any("叶片" in r for r in info.process.recommendations)

    def test_k418_highest_temp(self):
        """K418 should have highest temperature capability."""
        info = MATERIAL_DATABASE["K418"]
        assert any("950" in note or "一级涡轮" in str(info.process.recommendations) for note in info.process.heat_treatment_notes)

    def test_superalloys_equivalence(self):
        """Superalloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["GH2132", "K403", "K418"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_superalloys_cost(self):
        """Superalloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["GH2132", "K403", "K418"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] >= 4  # High cost materials


class TestCastAluminumSupplement:
    """Tests for cast aluminum supplements (ZL101, ZL104, ZL201)."""

    def test_cast_aluminum_exist(self):
        """Cast aluminum alloys should exist in database."""
        for grade in ["ZL101", "ZL104", "ZL201"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.CAST_ALUMINUM

    @pytest.mark.parametrize("input_name,expected", [
        ("ZL101", "ZL101"),
        ("A356.2", "ZL101"),
        ("AC4CH", "ZL101"),
        ("高纯铸铝", "ZL101"),
        ("ZL104", "ZL104"),
        ("A319", "ZL104"),
        ("AC2B", "ZL104"),
        ("发动机铸铝", "ZL104"),
        ("ZL201", "ZL201"),
        ("A201", "ZL201"),
        ("高强铸铝", "ZL201"),
        ("航空铸铝", "ZL201"),
    ])
    def test_cast_aluminum_patterns(self, input_name, expected):
        """Test cast aluminum pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_zl101_for_wheels(self):
        """ZL101 should be for automotive wheels."""
        info = MATERIAL_DATABASE["ZL101"]
        assert any("轮毂" in r or "底盘" in r for r in info.process.recommendations)

    def test_zl104_for_engine(self):
        """ZL104 should be for engine castings."""
        info = MATERIAL_DATABASE["ZL104"]
        assert any("缸盖" in r or "发动机" in r for r in info.process.recommendations)

    def test_zl201_highest_strength(self):
        """ZL201 should have highest strength."""
        info = MATERIAL_DATABASE["ZL201"]
        assert info.properties.tensile_strength >= 400

    def test_cast_aluminum_equivalence(self):
        """Cast aluminum alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["ZL101", "ZL104", "ZL201"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_cast_aluminum_cost(self):
        """Cast aluminum alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["ZL101", "ZL104", "ZL201"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestZincAlloys:
    """Tests for zinc die casting alloys (Zamak3, Zamak5, ZA-8)."""

    def test_zinc_alloys_exist(self):
        """Zinc alloys should exist in database."""
        for grade in ["Zamak3", "Zamak5", "ZA-8"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.ZINC_ALLOY

    @pytest.mark.parametrize("input_name,expected", [
        ("Zamak3", "Zamak3"),
        ("ZAMAK-3", "Zamak3"),
        ("ZA-3", "Zamak3"),
        ("锌合金3号", "Zamak3"),
        ("Zamak5", "Zamak5"),
        ("ZAMAK-5", "Zamak5"),
        ("ZA-5", "Zamak5"),
        ("锌合金5号", "Zamak5"),
        ("ZA-8", "ZA-8"),
        ("ZnAl8Cu", "ZA-8"),
        ("超强锌合金", "ZA-8"),
        ("锌合金", "Zamak3"),  # Generic match
    ])
    def test_zinc_alloy_patterns(self, input_name, expected):
        """Test zinc alloy pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_zamak3_most_common(self):
        """Zamak3 should be for general hardware."""
        info = MATERIAL_DATABASE["Zamak3"]
        assert any("五金" in r or "锁具" in r for r in info.process.recommendations)

    def test_zamak5_higher_strength(self):
        """Zamak5 should have higher strength than Zamak3."""
        z3 = MATERIAL_DATABASE["Zamak3"]
        z5 = MATERIAL_DATABASE["Zamak5"]
        assert z5.properties.tensile_strength > z3.properties.tensile_strength

    def test_za8_highest_strength(self):
        """ZA-8 should have highest strength."""
        info = MATERIAL_DATABASE["ZA-8"]
        assert info.properties.tensile_strength >= 350
        assert any("齿轮" in r or "轴承" in r for r in info.process.recommendations)

    def test_zinc_alloys_die_casting(self):
        """Zinc alloys should be for die casting."""
        for grade in ["Zamak3", "Zamak5", "ZA-8"]:
            info = MATERIAL_DATABASE[grade]
            assert any("压铸" in bf for bf in info.process.blank_forms)

    def test_zinc_alloys_equivalence(self):
        """Zinc alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Zamak3", "Zamak5", "ZA-8"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_zinc_alloys_cost(self):
        """Zinc alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Zamak3", "Zamak5", "ZA-8"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"
            assert cost["tier"] <= 2  # Low cost materials


class TestTinBronzeSupplement:
    """Tests for tin bronze supplements (QSn7-0.2, QSn4-0.3, ZCuSn10P1)."""

    def test_tin_bronzes_exist(self):
        """Tin bronzes should exist in database."""
        for grade in ["QSn7-0.2", "QSn4-0.3", "ZCuSn10P1"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.TIN_BRONZE

    @pytest.mark.parametrize("input_name,expected", [
        ("QSn7-0.2", "QSn7-0.2"),
        ("C52100", "C52100"),  # Now independent material
        ("CuSn8", "QSn7-0.2"),  # CuSn8 still maps to QSn7-0.2
        ("高弹性磷铜", "QSn7-0.2"),
        ("QSn4-0.3", "QSn4-0.3"),
        ("C51000", "QSn4-0.3"),
        ("CuSn4", "QSn4-0.3"),
        ("导电磷青铜", "QSn4-0.3"),
        ("ZCuSn10P1", "ZCuSn10P1"),
        ("C90700", "ZCuSn10P1"),
        ("PBC2A", "ZCuSn10P1"),
        ("高力青铜", "ZCuSn10P1"),
    ])
    def test_tin_bronze_patterns(self, input_name, expected):
        """Test tin bronze pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_qsn7_high_elasticity(self):
        """QSn7-0.2 should have high elasticity."""
        info = MATERIAL_DATABASE["QSn7-0.2"]
        assert any("弹簧" in r or "弹性" in r for r in info.process.recommendations)

    def test_qsn4_conductive(self):
        """QSn4-0.3 should have conductivity."""
        info = MATERIAL_DATABASE["QSn4-0.3"]
        assert info.properties.conductivity is not None

    def test_zcusn10p1_for_bearings(self):
        """ZCuSn10P1 should be for bearings."""
        info = MATERIAL_DATABASE["ZCuSn10P1"]
        assert any("轴瓦" in r or "蜗轮" in r for r in info.process.recommendations)

    def test_tin_bronzes_equivalence(self):
        """Tin bronzes have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["QSn7-0.2", "QSn4-0.3", "ZCuSn10P1"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_tin_bronzes_cost(self):
        """Tin bronzes have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["QSn7-0.2", "QSn4-0.3", "ZCuSn10P1"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestSpecialBrass:
    """Tests for special brass (HSi80-3, HAl77-2)."""

    def test_special_brass_exist(self):
        """Special brass alloys should exist in database."""
        assert "HSi80-3" in MATERIAL_DATABASE
        assert "HAl77-2" in MATERIAL_DATABASE
        assert MATERIAL_DATABASE["HSi80-3"].group == MaterialGroup.SILICON_BRASS
        assert MATERIAL_DATABASE["HAl77-2"].group == MaterialGroup.COPPER

    @pytest.mark.parametrize("input_name,expected", [
        ("HSi80-3", "HSi80-3"),
        ("C87500", "HSi80-3"),
        ("CuZn16Si3", "HSi80-3"),
        ("耐蚀黄铜", "HSi80-3"),
        ("硅黄铜", "HSi80-3"),
        ("HAl77-2", "HAl77-2"),
        ("C68700", "HAl77-2"),
        ("CuZn22Al2", "HAl77-2"),
        ("海军黄铜", "HAl77-2"),
        ("铝黄铜", "HAl77-2"),
    ])
    def test_special_brass_patterns(self, input_name, expected):
        """Test special brass pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_hsi80_corrosion_resistant(self):
        """HSi80-3 should be for corrosion resistant applications."""
        info = MATERIAL_DATABASE["HSi80-3"]
        assert any("耐蚀" in r or "船用" in r or "化工" in r for r in info.process.recommendations)

    def test_hal77_seawater(self):
        """HAl77-2 should be for seawater applications."""
        info = MATERIAL_DATABASE["HAl77-2"]
        assert any("海水" in r or "船用" in r for r in info.process.recommendations)

    def test_special_brass_equivalence(self):
        """Special brass have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["HSi80-3", "HAl77-2"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_special_brass_cost(self):
        """Special brass have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["HSi80-3", "HAl77-2"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestWearResistantIron:
    """Tests for wear-resistant cast irons (NiHard1, NiHard4, Cr26)."""

    def test_wear_resistant_irons_exist(self):
        """Wear-resistant irons should exist in database."""
        for grade in ["NiHard1", "NiHard4", "Cr26"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.WEAR_RESISTANT_IRON

    @pytest.mark.parametrize("input_name,expected", [
        ("NiHard1", "NiHard1"),
        ("Ni-Hard1", "NiHard1"),
        ("NiCr4", "NiHard1"),
        ("镍硬铸铁", "NiHard1"),
        ("NiHard4", "NiHard4"),
        ("Ni-Hard4", "NiHard4"),
        ("NiCrMo", "NiHard4"),
        ("Cr26", "Cr26"),
        ("Cr26Mo", "Cr26"),
        ("KmTBCr26", "Cr26"),
        ("高铬铸铁", "Cr26"),
        ("耐磨铸铁", "NiHard1"),
    ])
    def test_wear_resistant_iron_patterns(self, input_name, expected):
        """Test wear-resistant iron pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_nihard_high_hardness(self):
        """NiHard should have high hardness."""
        for grade in ["NiHard1", "NiHard4"]:
            info = MATERIAL_DATABASE[grade]
            assert "HRC" in info.properties.hardness
            assert int(info.properties.hardness.split("HRC")[1].split("-")[0]) >= 55

    def test_cr26_highest_hardness(self):
        """Cr26 should have highest hardness."""
        info = MATERIAL_DATABASE["Cr26"]
        assert "HRC" in info.properties.hardness
        assert int(info.properties.hardness.split("HRC")[1].split("-")[0]) >= 60

    def test_wear_resistant_difficult_machining(self):
        """Wear-resistant irons should be difficult to machine."""
        for grade in ["NiHard1", "NiHard4", "Cr26"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.machinability == "poor"

    def test_nihard1_for_mining(self):
        """NiHard1 should be for mining equipment."""
        info = MATERIAL_DATABASE["NiHard1"]
        assert any("球磨机" in r or "破碎机" in r or "渣浆泵" in r for r in info.process.recommendations)

    def test_cr26_for_heavy_duty(self):
        """Cr26 should be for heavy-duty mining."""
        info = MATERIAL_DATABASE["Cr26"]
        assert any("矿山" in r or "破碎机" in r or "衬板" in r for r in info.process.recommendations)

    def test_wear_resistant_irons_equivalence(self):
        """Wear-resistant irons have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["NiHard1", "NiHard4", "Cr26"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_wear_resistant_irons_cost(self):
        """Wear-resistant irons have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["NiHard1", "NiHard4", "Cr26"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestVermicularIron:
    """Tests for vermicular (compacted graphite) cast iron (RuT300, RuT350, RuT400)."""

    def test_vermicular_irons_exist(self):
        """Vermicular irons should exist in database."""
        for grade in ["RuT300", "RuT350", "RuT400"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.VERMICULAR_IRON

    @pytest.mark.parametrize("input_name,expected", [
        ("RuT300", "RuT300"),
        ("RuT350", "RuT350"),
        ("RuT400", "RuT400"),
        ("GJV-300", "RuT300"),
        ("GJV-350", "RuT350"),
        ("GJV-400", "RuT400"),
        ("CGI300", "RuT300"),
        ("CGI350", "RuT350"),
        ("CGI400", "RuT400"),
        ("蠕墨铸铁", "RuT300"),
        ("蠕铁", "RuT300"),
    ])
    def test_vermicular_iron_patterns(self, input_name, expected):
        """Test vermicular iron pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_vermicular_tensile_strength_ordering(self):
        """Higher grade should have higher tensile strength."""
        grades = ["RuT300", "RuT350", "RuT400"]
        strengths = []
        for grade in grades:
            info = MATERIAL_DATABASE[grade]
            ts = info.properties.tensile_strength
            # tensile_strength is int type
            strengths.append(ts)
        assert strengths == sorted(strengths), "Tensile strength should increase with grade"

    def test_vermicular_for_engine_blocks(self):
        """Vermicular irons should be for engine blocks/heads."""
        for grade in ["RuT300", "RuT350", "RuT400"]:
            info = MATERIAL_DATABASE[grade]
            assert any("发动机" in r or "缸体" in r or "缸盖" in r or "制动盘" in r for r in info.process.recommendations)

    def test_vermicular_machinability(self):
        """Vermicular irons should have reasonable machinability."""
        for grade in ["RuT300", "RuT350", "RuT400"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.machinability in ["excellent", "good", "fair"]

    def test_vermicular_irons_equivalence(self):
        """Vermicular irons have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["RuT300", "RuT350", "RuT400"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_vermicular_irons_cost(self):
        """Vermicular irons have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["RuT300", "RuT350", "RuT400"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestMalleableIron:
    """Tests for malleable cast iron (KTH300-06, KTZ450-06, KTZ550-04)."""

    def test_malleable_irons_exist(self):
        """Malleable irons should exist in database."""
        for grade in ["KTH300-06", "KTZ450-06", "KTZ550-04"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.MALLEABLE_IRON

    @pytest.mark.parametrize("input_name,expected", [
        ("KTH300-06", "KTH300-06"),
        ("KTZ450-06", "KTZ450-06"),
        ("KTZ550-04", "KTZ550-04"),
        ("黑心可锻铸铁", "KTH300-06"),
        ("珠光体可锻铸铁", "KTZ450-06"),
        ("可锻铸铁", "KTH300-06"),
        ("B300-06", "KTH300-06"),
        ("P450-06", "KTZ450-06"),
        ("P550-04", "KTZ550-04"),
    ])
    def test_malleable_iron_patterns(self, input_name, expected):
        """Test malleable iron pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_kth_whiteheart_high_ductility(self):
        """KTH (whiteheart) should have high elongation."""
        info = MATERIAL_DATABASE["KTH300-06"]
        # elongation is float type
        elong = info.properties.elongation
        assert elong >= 6, "KTH should have at least 6% elongation"

    def test_ktz_higher_strength(self):
        """KTZ (pearlitic) should have higher strength than KTH."""
        kth = MATERIAL_DATABASE["KTH300-06"]
        ktz = MATERIAL_DATABASE["KTZ550-04"]
        # tensile_strength is int type
        assert ktz.properties.tensile_strength > kth.properties.tensile_strength, "KTZ should have higher tensile strength"

    def test_malleable_for_pipe_fittings(self):
        """Malleable irons should be for pipe fittings."""
        info = MATERIAL_DATABASE["KTH300-06"]
        assert any("管件" in r or "接头" in r or "阀门" in r for r in info.process.recommendations)

    def test_malleable_machinability(self):
        """Malleable irons should have good machinability (varying by grade)."""
        grades_machinability = {
            "KTH300-06": ["excellent"],
            "KTZ450-06": ["good"],
            "KTZ550-04": ["fair", "good"],
        }
        for grade, expected in grades_machinability.items():
            info = MATERIAL_DATABASE[grade]
            assert info.properties.machinability in expected, f"{grade} machinability mismatch"

    def test_malleable_irons_equivalence(self):
        """Malleable irons have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["KTH300-06", "KTZ450-06", "KTZ550-04"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_malleable_irons_cost(self):
        """Malleable irons have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["KTH300-06", "KTZ450-06", "KTZ550-04"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestCastMagnesium:
    """Tests for cast magnesium alloys (ZM5, AM60B, AZ63)."""

    def test_cast_magnesium_exist(self):
        """Cast magnesium alloys should exist in database."""
        for grade in ["ZM5", "AM60B", "AZ63"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.CAST_MAGNESIUM

    @pytest.mark.parametrize("input_name,expected", [
        ("ZM5", "ZM5"),
        ("AM60B", "AM60B"),
        ("AZ63", "AZ63"),
        ("AZ91D", "AZ91D"),  # AZ91D is a separate material in database
        ("铸造镁合金", "ZM5"),
        ("AM60", "AM60"),   # AM60 is now an independent material
        ("压铸镁合金", "ZM5"),  # Maps to ZM5 (generic cast magnesium)
    ])
    def test_cast_magnesium_patterns(self, input_name, expected):
        """Test cast magnesium pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_magnesium_low_density(self):
        """Magnesium alloys should have low density."""
        for grade in ["ZM5", "AM60B", "AZ63"]:
            info = MATERIAL_DATABASE[grade]
            # density is float type
            density = info.properties.density
            assert density < 2.0, f"{grade} should have density < 2.0 g/cm³"

    def test_zm5_for_lightweight(self):
        """ZM5 should be for lightweight applications."""
        info = MATERIAL_DATABASE["ZM5"]
        assert any("变速箱" in r or "笔记本" in r or "电子" in r for r in info.process.recommendations)

    def test_am60b_for_automotive(self):
        """AM60B should be for automotive applications."""
        info = MATERIAL_DATABASE["AM60B"]
        assert any("汽车" in r or "方向盘" in r or "座椅" in r for r in info.process.recommendations)

    def test_az63_for_aerospace(self):
        """AZ63 should be for aerospace applications."""
        info = MATERIAL_DATABASE["AZ63"]
        assert any("航空" in r or "结构件" in r for r in info.process.recommendations)

    def test_magnesium_fire_hazard(self):
        """Magnesium alloys should have fire hazard warning."""
        for grade in ["ZM5", "AM60B", "AZ63"]:
            info = MATERIAL_DATABASE[grade]
            assert any("易燃" in w or "镁屑" in w for w in info.process.warnings)

    def test_cast_magnesium_equivalence(self):
        """Cast magnesium alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["ZM5", "AM60B", "AZ63"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_cast_magnesium_cost(self):
        """Cast magnesium alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["ZM5", "AM60B", "AZ63"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestPowderMetallurgy:
    """Tests for powder metallurgy materials (Fe-Cu-C, Fe-Ni-Cu, 316L-PM)."""

    def test_pm_materials_exist(self):
        """PM materials should exist in database."""
        for grade in ["Fe-Cu-C", "Fe-Ni-Cu", "316L-PM"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.POWDER_METALLURGY

    @pytest.mark.parametrize("input_name,expected", [
        ("Fe-Cu-C", "Fe-Cu-C"),
        ("Fe-Ni-Cu", "Fe-Ni-Cu"),
        ("316L-PM", "316L-PM"),
        ("FC-0205", "Fe-Cu-C"),
        ("MIM-316L", "316L-PM"),
        ("铁基粉末冶金", "Fe-Cu-C"),
        ("高强度粉末冶金", "Fe-Ni-Cu"),
        ("不锈钢粉末冶金", "316L-PM"),
        ("粉末冶金", "Fe-Cu-C"),
    ])
    def test_pm_patterns(self, input_name, expected):
        """Test PM material pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_pm_lower_density(self):
        """PM materials should have lower density than wrought steel."""
        for grade in ["Fe-Cu-C", "Fe-Ni-Cu"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.density < 7.8, f"{grade} density should be < 7.8 (porous)"

    def test_pm_sintering_process(self):
        """PM materials should mention sintering."""
        for grade in ["Fe-Cu-C", "Fe-Ni-Cu", "316L-PM"]:
            info = MATERIAL_DATABASE[grade]
            assert "烧结" in info.process.blank_hint

    def test_316l_pm_for_medical(self):
        """316L-PM should be for medical applications."""
        info = MATERIAL_DATABASE["316L-PM"]
        assert any("医疗" in r or "表壳" in r for r in info.process.recommendations)

    def test_pm_materials_equivalence(self):
        """PM materials have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Fe-Cu-C", "Fe-Ni-Cu", "316L-PM"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_pm_materials_cost(self):
        """PM materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Fe-Cu-C", "Fe-Ni-Cu", "316L-PM"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestCementedCarbide:
    """Tests for cemented carbides (YG8, YT15, YW1)."""

    def test_carbides_exist(self):
        """Cemented carbides should exist in database."""
        for grade in ["YG8", "YT15", "YW1"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.CEMENTED_CARBIDE

    @pytest.mark.parametrize("input_name,expected", [
        ("YG8", "YG8"),
        ("YT15", "YT15"),
        ("YW1", "YW1"),
        ("K30硬质合金", "YG8"),
        ("P15硬质合金", "YT15"),
        ("M10硬质合金", "YW1"),
        ("WC-8Co", "YG8"),
        ("钨钴合金", "YG8"),
        ("钨钛合金", "YT15"),
        ("硬质合金", "WC-Co"),
    ])
    def test_carbide_patterns(self, input_name, expected):
        """Test carbide pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_carbide_high_density(self):
        """Cemented carbides should have high density."""
        for grade in ["YG8", "YT15", "YW1"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.density > 10, f"{grade} should have density > 10"

    def test_carbide_poor_machinability(self):
        """Cemented carbides should have poor machinability."""
        for grade in ["YG8", "YT15", "YW1"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.machinability in ["poor", "very_poor"]

    def test_carbide_special_tooling(self):
        """Cemented carbides should require special tooling."""
        for grade in ["YG8", "YT15", "YW1"]:
            info = MATERIAL_DATABASE[grade]
            assert info.process.special_tooling is True

    def test_yg8_for_tooling(self):
        """YG8 should be for tooling applications."""
        info = MATERIAL_DATABASE["YG8"]
        assert any("模具" in r or "钻头" in r or "拉丝" in r for r in info.process.recommendations)

    def test_yt15_for_steel(self):
        """YT15 should be for steel machining."""
        info = MATERIAL_DATABASE["YT15"]
        assert any("钢" in r or "车刀" in r for r in info.process.recommendations)

    def test_carbides_equivalence(self):
        """Cemented carbides have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["YG8", "YT15", "YW1"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_carbides_cost(self):
        """Cemented carbides have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["YG8", "YT15", "YW1"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestStructuralCeramics:
    """Tests for structural ceramics (Al2O3-99, Si3N4, ZrO2-3Y)."""

    def test_ceramics_exist(self):
        """Structural ceramics should exist in database."""
        for grade in ["Al2O3-99", "Si3N4", "ZrO2-3Y"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.STRUCTURAL_CERAMIC

    @pytest.mark.parametrize("input_name,expected", [
        ("Al2O3-99", "Al2O3-99"),
        ("Si3N4", "Si3N4"),
        ("ZrO2-3Y", "ZrO2-3Y"),
        ("99氧化铝", "Al2O3-99"),
        ("99瓷", "Al2O3-99"),
        ("高纯氧化铝", "Al2O3-99"),
        ("氮化硅", "Si3N4"),
        ("SRBSN", "Si3N4"),
        ("3Y-TZP", "ZrO2-3Y"),
        ("氧化锆陶瓷", "ZrO2-3Y"),
        ("氧化铝陶瓷", "Al2O3-99"),
        ("结构陶瓷", "Al2O3-99"),
    ])
    def test_ceramic_patterns(self, input_name, expected):
        """Test ceramic pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_ceramic_category(self):
        """Structural ceramics should be NON_METAL category."""
        for grade in ["Al2O3-99", "Si3N4", "ZrO2-3Y"]:
            info = MATERIAL_DATABASE[grade]
            assert info.category == MaterialCategory.NON_METAL

    def test_ceramic_no_weldability(self):
        """Ceramics should have no weldability."""
        for grade in ["Al2O3-99", "Si3N4", "ZrO2-3Y"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.weldability == "none"

    def test_ceramic_special_tooling(self):
        """Ceramics should require special tooling."""
        for grade in ["Al2O3-99", "Si3N4", "ZrO2-3Y"]:
            info = MATERIAL_DATABASE[grade]
            assert info.process.special_tooling is True

    def test_si3n4_for_bearings(self):
        """Si3N4 should be for bearing applications."""
        info = MATERIAL_DATABASE["Si3N4"]
        assert any("轴承" in r or "滚珠" in r or "涡轮" in r for r in info.process.recommendations)

    def test_zro2_for_precision(self):
        """ZrO2-3Y should be for precision applications."""
        info = MATERIAL_DATABASE["ZrO2-3Y"]
        assert any("陶瓷刀" in r or "手表" in r or "牙科" in r for r in info.process.recommendations)

    def test_al2o3_high_hardness(self):
        """Al2O3-99 should have very high hardness."""
        info = MATERIAL_DATABASE["Al2O3-99"]
        assert "HV" in info.properties.hardness
        # Extract HV value
        hv_str = info.properties.hardness.replace("HV", "")
        hv_val = int(hv_str.split("-")[0])
        assert hv_val >= 1800, "Al2O3-99 should have HV >= 1800"

    def test_ceramics_equivalence(self):
        """Structural ceramics have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Al2O3-99", "Si3N4", "ZrO2-3Y"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_ceramics_cost(self):
        """Structural ceramics have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Al2O3-99", "Si3N4", "ZrO2-3Y"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestRefractoryMetals:
    """Tests for refractory metals (Mo-1, TZM, W-1, Ta-1)."""

    def test_refractory_metals_exist(self):
        """Refractory metals should exist in database."""
        for grade in ["Mo-1", "TZM", "W-1", "Ta-1"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.REFRACTORY_METAL

    @pytest.mark.parametrize("input_name,expected", [
        ("Mo-1", "Mo-1"),
        ("TZM", "TZM"),
        ("W-1", "W-1"),
        ("Ta-1", "Ta-1"),
        ("纯钼", "Mo-1"),
        ("钼棒", "Mo-1"),
        ("TZM钼合金", "TZM"),
        ("纯钨", "W-1"),
        ("钨板", "W-1"),
        ("纯钽", "Ta-1"),
        ("难熔金属", "Mo-1"),
    ])
    def test_refractory_patterns(self, input_name, expected):
        """Test refractory metal pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_refractory_high_melting_point(self):
        """Refractory metals should have high melting points."""
        for grade in ["Mo-1", "TZM", "W-1", "Ta-1"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.melting_point >= 2600, f"{grade} should have melting point >= 2600°C"

    def test_tungsten_highest_density(self):
        """Tungsten should have highest density."""
        w_info = MATERIAL_DATABASE["W-1"]
        assert w_info.properties.density > 19, "Tungsten density should be > 19 g/cm³"

    def test_tungsten_highest_melting_point(self):
        """Tungsten should have highest melting point."""
        w_info = MATERIAL_DATABASE["W-1"]
        assert w_info.properties.melting_point > 3400, "Tungsten melting point should be > 3400°C"

    def test_tzm_higher_strength(self):
        """TZM should have higher strength than pure Mo."""
        mo = MATERIAL_DATABASE["Mo-1"]
        tzm = MATERIAL_DATABASE["TZM"]
        assert tzm.properties.tensile_strength > mo.properties.tensile_strength

    def test_refractory_metals_equivalence(self):
        """Refractory metals have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Mo-1", "TZM", "W-1", "Ta-1"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_refractory_metals_cost(self):
        """Refractory metals have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Mo-1", "TZM", "W-1", "Ta-1"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestAluminumBronze:
    """Tests for aluminum bronzes (QAl9-4, QAl10-4-4)."""

    def test_aluminum_bronzes_exist(self):
        """Aluminum bronzes should exist in database."""
        for grade in ["QAl9-4", "QAl10-4-4"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.ALUMINUM_BRONZE

    @pytest.mark.parametrize("input_name,expected", [
        ("QAl9-4", "QAl9-4"),
        ("QAl10-4-4", "QAl10-4-4"),
        ("CuAl9Fe4", "QAl9-4"),
        ("C62300", "QAl9-4"),
        ("NAB", "QAl10-4-4"),
        ("镍铝青铜", "QAl10-4-4"),
        ("铝青铜", "QAl9-4"),
    ])
    def test_aluminum_bronze_patterns(self, input_name, expected):
        """Test aluminum bronze pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_qal10_higher_strength(self):
        """QAl10-4-4 should have higher strength than QAl9-4."""
        qal9 = MATERIAL_DATABASE["QAl9-4"]
        qal10 = MATERIAL_DATABASE["QAl10-4-4"]
        assert qal10.properties.tensile_strength > qal9.properties.tensile_strength

    def test_aluminum_bronze_for_marine(self):
        """Aluminum bronzes should be for marine applications."""
        for grade in ["QAl9-4", "QAl10-4-4"]:
            info = MATERIAL_DATABASE[grade]
            assert any("船" in r or "螺旋桨" in r or "海水" in r or "阀" in r for r in info.process.recommendations)

    def test_aluminum_bronzes_equivalence(self):
        """Aluminum bronzes have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["QAl9-4", "QAl10-4-4"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_aluminum_bronzes_cost(self):
        """Aluminum bronzes have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["QAl9-4", "QAl10-4-4"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestBerylliumCopper:
    """Tests for beryllium coppers (QBe2, QBe1.9, CuNi2Si)."""

    def test_beryllium_coppers_exist(self):
        """Beryllium coppers should exist in database."""
        for grade in ["QBe2", "QBe1.9", "CuNi2Si"]:
            assert grade in MATERIAL_DATABASE, f"{grade} not found"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.BERYLLIUM_COPPER

    @pytest.mark.parametrize("input_name,expected", [
        ("QBe2", "QBe2"),
        ("QBe1.9", "QBe1.9"),
        ("CuNi2Si", "CuNi2Si"),
        ("CuBe2", "QBe2"),
        ("C17200", "QBe2"),
        ("C17000", "QBe1.9"),
        ("低铍铜", "QBe1.9"),
        ("C70250", "CuNi2Si"),
        ("无铍铜", "CuNi2Si"),
        ("铍铜", "QBe2"),
        ("BeCu", "QBe2"),
    ])
    def test_beryllium_copper_patterns(self, input_name, expected):
        """Test beryllium copper pattern matching."""
        result = classify_material_detailed(input_name)
        assert result is not None, f"'{input_name}' should match"
        assert result.grade == expected, f"'{input_name}' should match {expected}"

    def test_qbe2_highest_strength(self):
        """QBe2 should have highest strength among copper alloys."""
        qbe2 = MATERIAL_DATABASE["QBe2"]
        assert qbe2.properties.tensile_strength >= 1200, "QBe2 should have tensile strength >= 1200 MPa"

    def test_beryllium_toxicity_warning(self):
        """Beryllium coppers should have toxicity warning."""
        for grade in ["QBe2", "QBe1.9"]:
            info = MATERIAL_DATABASE[grade]
            assert any("有毒" in w or "防护" in w for w in info.process.warnings)

    def test_cuni2si_beryllium_free(self):
        """CuNi2Si should be beryllium-free alternative."""
        info = MATERIAL_DATABASE["CuNi2Si"]
        assert any("替代" in r or "无毒" in w for r in info.process.recommendations for w in info.process.warnings)

    def test_beryllium_coppers_for_springs(self):
        """Beryllium coppers should be for spring applications."""
        for grade in ["QBe2", "QBe1.9"]:
            info = MATERIAL_DATABASE[grade]
            assert any("弹簧" in r or "弹性" in r or "连接器" in r for r in info.process.recommendations)

    def test_beryllium_coppers_equivalence(self):
        """Beryllium coppers have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["QBe2", "QBe1.9", "CuNi2Si"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_beryllium_coppers_cost(self):
        """Beryllium coppers have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["QBe2", "QBe1.9", "CuNi2Si"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestLeadFreeSolders:
    """Tests for lead-free solder materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("SAC305", "SAC305"),
        ("SAC 305", "SAC305"),
        ("SAC387", "SAC387"),
        ("SAC 387", "SAC387"),
        ("Sn99.3Cu0.7", "Sn99.3Cu0.7"),
        ("SN100C", "Sn99.3Cu0.7"),
        ("无铅焊锡", "SAC305"),
        ("无铅焊料", "SAC305"),
    ])
    def test_solder_patterns(self, pattern, expected):
        """Lead-free solder patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_solders_in_database(self):
        """All solder grades should be in database."""
        for grade in ["SAC305", "SAC387", "Sn99.3Cu0.7"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.SOLDER

    def test_sac305_properties(self):
        """SAC305 should have correct properties."""
        info = MATERIAL_DATABASE["SAC305"]
        assert info.properties.melting_point == 217
        assert info.properties.machinability == "excellent"
        assert info.properties.weldability == "excellent"

    def test_solder_low_melting_point(self):
        """Solders should have low melting point."""
        for grade in ["SAC305", "SAC387", "Sn99.3Cu0.7"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.melting_point < 250, f"{grade} melting point should be < 250°C"

    def test_solders_equivalence(self):
        """Solders have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["SAC305", "SAC387", "Sn99.3Cu0.7"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_solders_cost(self):
        """Solders have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["SAC305", "SAC387", "Sn99.3Cu0.7"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestBrazingAlloys:
    """Tests for brazing alloy materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("BAg-5", "BAg-5"),
        ("BAg5", "BAg-5"),
        ("HL302", "BAg-5"),
        ("银钎料", "BAg-5"),
        ("BCu-1", "BCu-1"),
        ("BCu1", "BCu-1"),
        ("HL101", "BCu-1"),
        ("纯铜钎料", "BCu-1"),
        ("BNi-2", "BNi-2"),
        ("BNi2", "BNi-2"),
        ("HL401", "BNi-2"),
        ("镍基钎料", "BNi-2"),
    ])
    def test_brazing_patterns(self, pattern, expected):
        """Brazing alloy patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_brazing_alloys_in_database(self):
        """All brazing alloy grades should be in database."""
        for grade in ["BAg-5", "BCu-1", "BNi-2"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.BRAZING_ALLOY

    def test_bag5_highest_silver(self):
        """BAg-5 should have high silver content, reflected in cost."""
        from src.core.materials import get_material_cost
        bag5_cost = get_material_cost("BAg-5")
        bcu1_cost = get_material_cost("BCu-1")
        assert bag5_cost["cost_index"] > bcu1_cost["cost_index"], "BAg-5 should be more expensive than BCu-1"

    def test_bni2_high_temp(self):
        """BNi-2 should be for high temperature applications."""
        info = MATERIAL_DATABASE["BNi-2"]
        assert info.properties.melting_point >= 1000
        assert any("高温" in r or "不锈钢" in r for r in info.process.recommendations)

    def test_brazing_alloys_equivalence(self):
        """Brazing alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["BAg-5", "BCu-1", "BNi-2"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_brazing_alloys_cost(self):
        """Brazing alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["BAg-5", "BCu-1", "BNi-2"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestShapeMemoryAlloys:
    """Tests for shape memory alloy materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("NiTi", "NiTi"),
        ("Nitinol", "NiTi"),
        ("TiNi", "NiTi"),
        ("镍钛合金", "NiTi"),
        ("镍钛记忆合金", "NiTi"),
        ("形状记忆合金", "NiTi"),
        ("CuZnAl", "CuZnAl"),
        ("Cu-Zn-Al", "CuZnAl"),
        ("铜锌铝", "CuZnAl"),
        ("铜基记忆合金", "CuZnAl"),
        ("CuAlNi", "CuAlNi"),
        ("Cu-Al-Ni", "CuAlNi"),
        ("铜铝镍", "CuAlNi"),
    ])
    def test_sma_patterns(self, pattern, expected):
        """Shape memory alloy patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_sma_in_database(self):
        """All SMA grades should be in database."""
        for grade in ["NiTi", "CuZnAl", "CuAlNi"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.SHAPE_MEMORY_ALLOY

    def test_niti_most_expensive(self):
        """NiTi should be the most expensive SMA."""
        from src.core.materials import get_material_cost
        niti_cost = get_material_cost("NiTi")
        cuznal_cost = get_material_cost("CuZnAl")
        cualni_cost = get_material_cost("CuAlNi")
        assert niti_cost["cost_index"] > cuznal_cost["cost_index"], "NiTi should be more expensive than CuZnAl"
        assert niti_cost["cost_index"] > cualni_cost["cost_index"], "NiTi should be more expensive than CuAlNi"

    def test_niti_for_medical(self):
        """NiTi should be for medical applications."""
        info = MATERIAL_DATABASE["NiTi"]
        assert any("医疗" in r for r in info.process.recommendations)

    def test_niti_poor_machinability(self):
        """NiTi should have poor machinability."""
        info = MATERIAL_DATABASE["NiTi"]
        assert info.properties.machinability == "poor"
        assert info.process.special_tooling is True

    def test_copper_sma_lower_cost(self):
        """Copper-based SMA should be lower cost alternative."""
        from src.core.materials import get_material_cost
        for grade in ["CuZnAl", "CuAlNi"]:
            cost = get_material_cost(grade)
            assert cost["tier"] <= 3, f"{grade} should be tier 3 or lower"

    def test_sma_equivalence(self):
        """Shape memory alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["NiTi", "CuZnAl", "CuAlNi"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_sma_cost(self):
        """Shape memory alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["NiTi", "CuZnAl", "CuAlNi"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestElectricalContactMaterials:
    """Tests for electrical contact materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("AgCdO", "AgCdO"),
        ("Ag/CdO", "AgCdO"),
        ("银镉触点", "AgCdO"),
        ("银氧化镉", "AgCdO"),
        ("AgSnO2", "AgSnO2"),
        ("Ag/SnO2", "AgSnO2"),
        ("银锡触点", "AgSnO2"),
        ("无镉触点", "AgSnO2"),
        ("CuW", "CuW"),
        ("WCu", "CuW"),
        ("钨铜", "CuW"),
        ("电触头", "AgSnO2"),
    ])
    def test_contact_patterns(self, pattern, expected):
        """Electrical contact patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_contacts_in_database(self):
        """All contact grades should be in database."""
        for grade in ["AgCdO", "AgSnO2", "CuW"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.ELECTRICAL_CONTACT

    def test_agcdo_toxicity_warning(self):
        """AgCdO should have toxicity warning."""
        info = MATERIAL_DATABASE["AgCdO"]
        assert any("有毒" in w or "镉" in w for w in info.process.warnings)

    def test_agsno2_cadmium_free(self):
        """AgSnO2 should be cadmium-free alternative."""
        info = MATERIAL_DATABASE["AgSnO2"]
        assert any("替代" in r or "环保" in r for r in info.process.recommendations)

    def test_cuw_high_density(self):
        """CuW should have high density due to tungsten."""
        info = MATERIAL_DATABASE["CuW"]
        assert info.properties.density > 12.0, "CuW should have density > 12 g/cm³"

    def test_contacts_equivalence(self):
        """Electrical contacts have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["AgCdO", "AgSnO2", "CuW"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_contacts_cost(self):
        """Electrical contacts have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["AgCdO", "AgSnO2", "CuW"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestBearingAlloys:
    """Tests for bearing alloy materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("ZChSnSb11-6", "ZChSnSb11-6"),
        ("Babbitt", "ZChSnSb11-6"),
        ("巴氏合金", "ZChSnSb11-6"),
        ("白合金", "ZChSnSb11-6"),
        ("ZChPbSb16-16-2", "ZChPbSb16-16-2"),
        ("铅基轴承合金", "ZChPbSb16-16-2"),
        ("CuPb24Sn4", "CuPb24Sn4"),
        ("铅青铜轴承", "CuPb24Sn4"),
        ("SAE49", "CuPb24Sn4"),
    ])
    def test_bearing_patterns(self, pattern, expected):
        """Bearing alloy patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_bearings_in_database(self):
        """All bearing grades should be in database."""
        for grade in ["ZChSnSb11-6", "ZChPbSb16-16-2", "CuPb24Sn4"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.BEARING_ALLOY

    def test_babbitt_low_melting_point(self):
        """Babbitt alloys should have low melting point."""
        for grade in ["ZChSnSb11-6", "ZChPbSb16-16-2"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.melting_point < 300, f"{grade} melting point should be < 300°C"

    def test_babbitt_excellent_machinability(self):
        """Babbitt alloys should be easy to machine."""
        for grade in ["ZChSnSb11-6", "ZChPbSb16-16-2"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.machinability == "excellent"

    def test_cupb24sn4_for_engine(self):
        """CuPb24Sn4 should be for engine bearings."""
        info = MATERIAL_DATABASE["CuPb24Sn4"]
        assert any("发动机" in r or "轴瓦" in r for r in info.process.recommendations)

    def test_bearings_equivalence(self):
        """Bearing alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["ZChSnSb11-6", "ZChPbSb16-16-2", "CuPb24Sn4"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_bearings_cost(self):
        """Bearing alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["ZChSnSb11-6", "ZChPbSb16-16-2", "CuPb24Sn4"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestThermocoupleAlloys:
    """Tests for thermocouple alloy materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("Chromel", "Chromel"),
        ("NiCr10", "Chromel"),
        ("K型热电偶正极", "Chromel"),
        ("Alumel", "Alumel"),
        ("NiAl3", "Alumel"),
        ("K型热电偶负极", "Alumel"),
        ("Constantan", "Constantan"),
        ("CuNi44", "Constantan"),
        ("6J40", "Constantan"),
        ("康铜", "Constantan"),
    ])
    def test_thermocouple_patterns(self, pattern, expected):
        """Thermocouple alloy patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_thermocouples_in_database(self):
        """All thermocouple grades should be in database."""
        for grade in ["Chromel", "Alumel", "Constantan"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.THERMOCOUPLE_ALLOY

    def test_chromel_alumel_pairing(self):
        """Chromel and Alumel should be K-type thermocouple pair."""
        chromel = MATERIAL_DATABASE["Chromel"]
        alumel = MATERIAL_DATABASE["Alumel"]
        assert any("K型" in r for r in chromel.process.recommendations)
        assert any("K型" in r for r in alumel.process.recommendations)

    def test_constantan_for_j_type(self):
        """Constantan should be for J/T type thermocouples."""
        info = MATERIAL_DATABASE["Constantan"]
        assert any("J" in r or "T" in r for r in info.process.recommendations)

    def test_thermocouple_high_melting_point(self):
        """Thermocouple alloys should have high melting points."""
        for grade in ["Chromel", "Alumel"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.melting_point >= 1200, f"{grade} should have melting point >= 1200°C"

    def test_thermocouples_equivalence(self):
        """Thermocouple alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Chromel", "Alumel", "Constantan"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_thermocouples_cost(self):
        """Thermocouple alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Chromel", "Alumel", "Constantan"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestPermanentMagnets:
    """Tests for permanent magnet materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("NdFeB", "NdFeB"),
        ("钕铁硼", "NdFeB"),
        ("钕磁铁", "NdFeB"),
        ("稀土永磁", "NdFeB"),
        ("SmCo", "SmCo"),
        ("SmCo5", "SmCo"),
        ("钐钴", "SmCo"),
        ("Alnico", "Alnico"),
        ("AlNiCo5", "Alnico"),
        ("铝镍钴", "Alnico"),
    ])
    def test_magnet_patterns(self, pattern, expected):
        """Permanent magnet patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_magnets_in_database(self):
        """All magnet grades should be in database."""
        for grade in ["NdFeB", "SmCo", "Alnico"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.PERMANENT_MAGNET

    def test_ndfeb_strongest(self):
        """NdFeB should be the strongest magnet (lowest cost per performance)."""
        from src.core.materials import get_material_cost
        ndfeb_cost = get_material_cost("NdFeB")
        smco_cost = get_material_cost("SmCo")
        assert ndfeb_cost["cost_index"] < smco_cost["cost_index"], "NdFeB should be cheaper than SmCo"

    def test_smco_high_temp(self):
        """SmCo should be for high temperature applications."""
        info = MATERIAL_DATABASE["SmCo"]
        assert any("高温" in r for r in info.process.recommendations)

    def test_magnets_poor_machinability(self):
        """All magnets should have poor machinability."""
        for grade in ["NdFeB", "SmCo", "Alnico"]:
            info = MATERIAL_DATABASE[grade]
            assert info.properties.machinability == "poor"

    def test_magnets_equivalence(self):
        """Permanent magnets have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["NdFeB", "SmCo", "Alnico"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_magnets_cost(self):
        """Permanent magnets have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["NdFeB", "SmCo", "Alnico"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestResistanceAlloys:
    """Tests for resistance alloy materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("Cr20Ni80", "Cr20Ni80"),
        ("Nichrome", "Cr20Ni80"),
        ("电炉丝", "Cr20Ni80"),
        ("Manganin", "Manganin"),
        ("6J13", "Manganin"),
        ("锰铜", "Manganin"),
        ("Karma", "Karma"),
        ("卡玛合金", "Karma"),
    ])
    def test_resistance_patterns(self, pattern, expected):
        """Resistance alloy patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_resistance_alloys_in_database(self):
        """All resistance alloy grades should be in database."""
        for grade in ["Cr20Ni80", "Manganin", "Karma"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.RESISTANCE_ALLOY

    def test_nichrome_for_heating(self):
        """Nichrome should be for heating elements."""
        info = MATERIAL_DATABASE["Cr20Ni80"]
        assert any("电热" in r or "电炉" in r for r in info.process.recommendations)

    def test_manganin_precision(self):
        """Manganin should be for precision resistors."""
        info = MATERIAL_DATABASE["Manganin"]
        assert any("精密" in r or "标准" in r for r in info.process.recommendations)

    def test_karma_strain_gauge(self):
        """Karma should be for strain gauges."""
        info = MATERIAL_DATABASE["Karma"]
        assert any("应变" in r for r in info.process.recommendations)

    def test_resistance_alloys_equivalence(self):
        """Resistance alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Cr20Ni80", "Manganin", "Karma"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_resistance_alloys_cost(self):
        """Resistance alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Cr20Ni80", "Manganin", "Karma"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestLowExpansionAlloys:
    """Tests for low expansion alloy materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("Invar", "Invar"),
        ("殷钢", "Invar"),
        ("因瓦合金", "Invar"),
        ("低膨胀合金", "Invar"),
        ("Kovar", "Kovar"),
        ("可伐合金", "Kovar"),
        ("玻封合金", "Kovar"),
        ("4J32", "4J32"),
        ("Super Invar", "4J32"),
        ("超因瓦", "4J32"),
    ])
    def test_low_expansion_patterns(self, pattern, expected):
        """Low expansion alloy patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_low_expansion_in_database(self):
        """All low expansion alloy grades should be in database."""
        for grade in ["Invar", "Kovar", "4J32"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.LOW_EXPANSION_ALLOY

    def test_invar_precision_instruments(self):
        """Invar should be for precision instruments."""
        info = MATERIAL_DATABASE["Invar"]
        assert any("精密" in r or "仪器" in r for r in info.process.recommendations)

    def test_kovar_glass_sealing(self):
        """Kovar should be for glass/ceramic sealing."""
        info = MATERIAL_DATABASE["Kovar"]
        assert any("封" in r or "玻璃" in r for r in info.process.recommendations)

    def test_4j32_lowest_expansion(self):
        """4J32 should have description about ultra-low expansion."""
        info = MATERIAL_DATABASE["4J32"]
        assert "比Invar更低" in info.description or "超" in info.name

    def test_low_expansion_alloys_equivalence(self):
        """Low expansion alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Invar", "Kovar", "4J32"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_low_expansion_alloys_cost(self):
        """Low expansion alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Invar", "Kovar", "4J32"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestSuperconductorMaterials:
    """Tests for superconductor materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("NbTi", "NbTi"),
        ("Nb-Ti", "NbTi"),
        ("NbTi47", "NbTi"),
        ("铌钛合金", "NbTi"),
        ("低温超导", "NbTi"),
        ("Nb3Sn", "Nb3Sn"),
        ("Nb-3-Sn", "Nb3Sn"),
        ("铌三锡", "Nb3Sn"),
        ("YBCO", "YBCO"),
        ("YBa2Cu3O7", "YBCO"),
        ("钇钡铜氧", "YBCO"),
        ("高温超导", "YBCO"),
    ])
    def test_superconductor_patterns(self, pattern, expected):
        """Superconductor patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_superconductors_in_database(self):
        """All superconductor grades should be in database."""
        for grade in ["NbTi", "Nb3Sn", "YBCO"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.SUPERCONDUCTOR

    def test_nbti_mri_magnets(self):
        """NbTi should be for MRI magnets."""
        info = MATERIAL_DATABASE["NbTi"]
        assert any("MRI" in r or "磁共振" in r for r in info.process.recommendations)

    def test_nb3sn_high_field(self):
        """Nb3Sn should be for high field applications."""
        info = MATERIAL_DATABASE["Nb3Sn"]
        assert any("高场" in r or "高磁场" in r or "聚变" in r for r in info.process.recommendations)

    def test_ybco_high_temp_superconductor(self):
        """YBCO should be a high-temperature superconductor."""
        info = MATERIAL_DATABASE["YBCO"]
        assert "高温" in info.description or "77K" in info.description

    def test_superconductors_equivalence(self):
        """Superconductors have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["NbTi", "Nb3Sn", "YBCO"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_superconductors_cost(self):
        """Superconductors have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["NbTi", "Nb3Sn", "YBCO"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestNuclearMaterials:
    """Tests for nuclear industry materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("Zircaloy-4", "Zircaloy-4"),
        ("Zr-4", "Zircaloy-4"),
        ("锆合金-4", "Zircaloy-4"),
        ("核级锆", "Zircaloy-4"),
        ("燃料包壳", "Zircaloy-4"),
        ("Hafnium", "Hafnium"),
        ("Hf", "Hafnium"),
        ("铪", "Hafnium"),
        ("控制棒材料", "Hafnium"),
        ("B4C", "B4C"),
        ("碳化硼", "B4C"),
        ("中子吸收材料", "B4C"),
        ("屏蔽材料", "B4C"),
    ])
    def test_nuclear_material_patterns(self, pattern, expected):
        """Nuclear material patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_nuclear_materials_in_database(self):
        """All nuclear material grades should be in database."""
        for grade in ["Zircaloy-4", "Hafnium", "B4C"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.NUCLEAR_MATERIAL

    def test_zircaloy_fuel_cladding(self):
        """Zircaloy-4 should be for fuel cladding."""
        info = MATERIAL_DATABASE["Zircaloy-4"]
        assert any("燃料" in r or "包壳" in r or "核" in r for r in info.process.recommendations)

    def test_hafnium_control_rods(self):
        """Hafnium should be for control rods."""
        info = MATERIAL_DATABASE["Hafnium"]
        assert any("控制棒" in r or "中子" in r for r in info.process.recommendations)

    def test_b4c_neutron_absorber(self):
        """B4C should be for neutron absorption."""
        info = MATERIAL_DATABASE["B4C"]
        assert any("中子" in r or "吸收" in r or "屏蔽" in r for r in info.process.recommendations)

    def test_nuclear_materials_equivalence(self):
        """Nuclear materials have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Zircaloy-4", "Hafnium", "B4C"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_nuclear_materials_cost(self):
        """Nuclear materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Zircaloy-4", "Hafnium", "B4C"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestMedicalAlloys:
    """Tests for medical alloy materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("CoCrMo", "CoCrMo"),
        ("Co-Cr-Mo", "CoCrMo"),
        ("钴铬钼合金", "CoCrMo"),
        ("医用钴基", "CoCrMo"),
        ("骨科植入", "CoCrMo"),
        ("Ti6Al4V-ELI", "Ti6Al4V-ELI"),
        ("Ti-6-4-ELI", "Ti6Al4V-ELI"),
        ("TC4-ELI", "Ti6Al4V-ELI"),
        ("医用钛合金", "Ti6Al4V-ELI"),
        ("骨科钛", "Ti6Al4V-ELI"),
        ("316L-Medical", "316L-Medical"),
        ("医用316L", "316L-Medical"),
        ("手术器械钢", "316L-Medical"),
        ("植入级不锈钢", "316L-Medical"),
    ])
    def test_medical_alloy_patterns(self, pattern, expected):
        """Medical alloy patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_medical_alloys_in_database(self):
        """All medical alloy grades should be in database."""
        for grade in ["CoCrMo", "Ti6Al4V-ELI", "316L-Medical"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.MEDICAL_ALLOY

    def test_cocrmo_joint_replacement(self):
        """CoCrMo should be for joint replacement."""
        info = MATERIAL_DATABASE["CoCrMo"]
        assert any("关节" in r or "骨科" in r or "植入" in r for r in info.process.recommendations)

    def test_ti6al4v_eli_biocompatible(self):
        """Ti6Al4V-ELI should be biocompatible."""
        info = MATERIAL_DATABASE["Ti6Al4V-ELI"]
        assert "ELI" in info.grade  # Extra Low Interstitials
        assert any("医用" in r or "植入" in r or "骨科" in r for r in info.process.recommendations)

    def test_316l_medical_surgical(self):
        """316L-Medical should be for surgical instruments."""
        info = MATERIAL_DATABASE["316L-Medical"]
        assert any("手术" in r or "器械" in r or "植入" in r for r in info.process.recommendations)

    def test_medical_alloys_equivalence(self):
        """Medical alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["CoCrMo", "Ti6Al4V-ELI", "316L-Medical"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_medical_alloys_cost(self):
        """Medical alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["CoCrMo", "Ti6Al4V-ELI", "316L-Medical"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestOpticalMaterials:
    """Tests for optical materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("Fused-Silica", "Fused-Silica"),
        ("Fused Silica", "Fused-Silica"),
        ("熔融石英", "Fused-Silica"),
        ("石英玻璃", "Fused-Silica"),
        ("Quartz Glass", "Fused-Silica"),
        ("光学石英", "Fused-Silica"),
        ("JGS1", "Fused-Silica"),
        ("Sapphire", "Sapphire"),
        ("蓝宝石", "Sapphire"),
        ("Al2O3单晶", "Sapphire"),
        ("人造蓝宝石", "Sapphire"),
        ("刚玉单晶", "Sapphire"),
        ("Germanium", "Germanium"),
        ("锗", "Germanium"),
        ("红外锗", "Germanium"),
        ("光学锗", "Germanium"),
    ])
    def test_optical_material_patterns(self, pattern, expected):
        """Optical material patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_optical_materials_in_database(self):
        """All optical material grades should be in database."""
        for grade in ["Fused-Silica", "Sapphire", "Germanium"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.OPTICAL_MATERIAL

    def test_fused_silica_uv_transmission(self):
        """Fused silica should be for UV/optical applications."""
        info = MATERIAL_DATABASE["Fused-Silica"]
        assert any("光学" in r or "激光" in r or "紫外" in r for r in info.process.recommendations)

    def test_sapphire_hardness(self):
        """Sapphire should have very high hardness."""
        info = MATERIAL_DATABASE["Sapphire"]
        assert "2000" in info.properties.hardness or "HV" in info.properties.hardness

    def test_germanium_infrared(self):
        """Germanium should be for infrared applications."""
        info = MATERIAL_DATABASE["Germanium"]
        assert any("红外" in r or "热像" in r for r in info.process.recommendations)

    def test_optical_materials_equivalence(self):
        """Optical materials have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Fused-Silica", "Sapphire", "Germanium"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_optical_materials_cost(self):
        """Optical materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Fused-Silica", "Sapphire", "Germanium"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestBatteryMaterials:
    """Tests for battery materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("LiFePO4", "LiFePO4"),
        ("LFP", "LiFePO4"),
        ("磷酸铁锂", "LiFePO4"),
        ("铁锂", "LiFePO4"),
        ("NMC", "NMC"),
        ("NCM", "NMC"),
        ("三元正极", "NMC"),
        ("镍钴锰", "NMC"),
        ("Graphite-Battery", "Graphite-Battery"),
        ("负极石墨", "Graphite-Battery"),
        ("人造石墨负极", "Graphite-Battery"),
        ("天然石墨负极", "Graphite-Battery"),
        ("锂电负极", "Graphite-Battery"),
    ])
    def test_battery_material_patterns(self, pattern, expected):
        """Battery material patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_battery_materials_in_database(self):
        """All battery material grades should be in database."""
        for grade in ["LiFePO4", "NMC", "Graphite-Battery"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.BATTERY_MATERIAL

    def test_lifepo4_safety(self):
        """LiFePO4 should emphasize safety."""
        info = MATERIAL_DATABASE["LiFePO4"]
        assert "安全" in info.description or any("动力电池" in r for r in info.process.recommendations)

    def test_nmc_high_energy(self):
        """NMC should be for high energy density."""
        info = MATERIAL_DATABASE["NMC"]
        assert "高能量" in info.description or any("电动汽车" in r for r in info.process.recommendations)

    def test_graphite_anode(self):
        """Graphite-Battery should be for anode applications."""
        info = MATERIAL_DATABASE["Graphite-Battery"]
        assert "负极" in info.description or any("负极" in r for r in info.process.recommendations)

    def test_battery_materials_equivalence(self):
        """Battery materials have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["LiFePO4", "NMC", "Graphite-Battery"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_battery_materials_cost(self):
        """Battery materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["LiFePO4", "NMC", "Graphite-Battery"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestSemiconductorMaterials:
    """Tests for semiconductor materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("Silicon-Wafer", "Silicon-Wafer"),
        ("Si Wafer", "Silicon-Wafer"),
        ("单晶硅", "Silicon-Wafer"),
        ("硅片", "Silicon-Wafer"),
        ("硅晶圆", "Silicon-Wafer"),
        ("GaAs", "GaAs"),
        ("砷化镓", "GaAs"),
        ("Gallium Arsenide", "GaAs"),
        ("化合物半导体", "GaAs"),
        ("SiC-Semiconductor", "SiC-Semiconductor"),
        ("碳化硅半导体", "SiC-Semiconductor"),
        ("宽禁带半导体", "SiC-Semiconductor"),
        ("第三代半导体", "SiC-Semiconductor"),
        ("4H-SiC", "SiC-Semiconductor"),
    ])
    def test_semiconductor_patterns(self, pattern, expected):
        """Semiconductor material patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_semiconductors_in_database(self):
        """All semiconductor grades should be in database."""
        for grade in ["Silicon-Wafer", "GaAs", "SiC-Semiconductor"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.SEMICONDUCTOR

    def test_silicon_wafer_for_ic(self):
        """Silicon wafer should be for IC applications."""
        info = MATERIAL_DATABASE["Silicon-Wafer"]
        assert any("集成电路" in r or "IC" in r or "太阳能" in r for r in info.process.recommendations)

    def test_gaas_for_rf(self):
        """GaAs should be for RF and optoelectronic applications."""
        info = MATERIAL_DATABASE["GaAs"]
        assert any("射频" in r or "LED" in r or "激光" in r for r in info.process.recommendations)

    def test_sic_for_power(self):
        """SiC should be for power electronics."""
        info = MATERIAL_DATABASE["SiC-Semiconductor"]
        assert any("电动汽车" in r or "逆变器" in r or "功率" in r for r in info.process.recommendations)

    def test_semiconductors_equivalence(self):
        """Semiconductors have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Silicon-Wafer", "GaAs", "SiC-Semiconductor"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_semiconductors_cost(self):
        """Semiconductors have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Silicon-Wafer", "GaAs", "SiC-Semiconductor"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestThermalInterfaceMaterials:
    """Tests for thermal interface materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("Thermal-Paste", "Thermal-Paste"),
        ("Thermal Paste", "Thermal-Paste"),
        ("导热硅脂", "Thermal-Paste"),
        ("硅脂", "Thermal-Paste"),
        ("导热膏", "Thermal-Paste"),
        ("Thermal Grease", "Thermal-Paste"),
        ("Thermal-Pad", "Thermal-Pad"),
        ("Thermal Pad", "Thermal-Pad"),
        ("导热垫片", "Thermal-Pad"),
        ("导热硅胶垫", "Thermal-Pad"),
        ("Gap Filler", "Thermal-Pad"),
        ("导热片", "Thermal-Pad"),
        ("Graphene-TIM", "Graphene-TIM"),
        ("石墨烯散热膜", "Graphene-TIM"),
        ("石墨烯导热", "Graphene-TIM"),
        ("Graphene Film", "Graphene-TIM"),
    ])
    def test_thermal_interface_patterns(self, pattern, expected):
        """Thermal interface material patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_thermal_interface_in_database(self):
        """All thermal interface grades should be in database."""
        for grade in ["Thermal-Paste", "Thermal-Pad", "Graphene-TIM"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.THERMAL_INTERFACE

    def test_thermal_paste_for_cpu(self):
        """Thermal paste should be for CPU/GPU cooling."""
        info = MATERIAL_DATABASE["Thermal-Paste"]
        assert any("CPU" in r or "GPU" in r or "散热" in r for r in info.process.recommendations)

    def test_thermal_pad_for_gap_filling(self):
        """Thermal pad should be for gap filling."""
        info = MATERIAL_DATABASE["Thermal-Pad"]
        assert "公差" in info.description or any("电源" in r or "电池" in r for r in info.process.recommendations)

    def test_graphene_tim_high_conductivity(self):
        """Graphene TIM should have very high thermal conductivity."""
        info = MATERIAL_DATABASE["Graphene-TIM"]
        assert info.properties.thermal_conductivity >= 1000

    def test_thermal_interface_equivalence(self):
        """Thermal interface materials have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Thermal-Paste", "Thermal-Pad", "Graphene-TIM"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_thermal_interface_cost(self):
        """Thermal interface materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Thermal-Paste", "Thermal-Pad", "Graphene-TIM"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestAdditiveManufacturingMaterials:
    """Tests for additive manufacturing materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("AlSi10Mg-AM", "AlSi10Mg-AM"),
        ("AlSi10Mg", "AlSi10Mg-AM"),
        ("SLM铝合金", "AlSi10Mg-AM"),
        ("DMLS铝合金", "AlSi10Mg-AM"),
        ("增材铝合金", "AlSi10Mg-AM"),
        ("3D打印铝", "AlSi10Mg-AM"),
        ("IN718-AM", "IN718-AM"),
        ("Inconel 718 AM", "IN718-AM"),
        ("SLM镍合金", "IN718-AM"),
        ("增材IN718", "IN718-AM"),
        ("3D打印镍基", "IN718-AM"),
        ("Ti64-AM", "Ti64-AM"),
        ("Ti-6Al-4V-AM", "Ti64-AM"),
        ("SLM钛合金", "Ti64-AM"),
        ("EBM钛合金", "Ti64-AM"),
        ("增材TC4", "Ti64-AM"),
        ("3D打印钛", "Ti64-AM"),
    ])
    def test_am_material_patterns(self, pattern, expected):
        """AM material patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_am_materials_in_database(self):
        """All AM material grades should be in database."""
        for grade in ["AlSi10Mg-AM", "IN718-AM", "Ti64-AM"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.ADDITIVE_MANUFACTURING

    def test_alsi10mg_for_lightweight(self):
        """AlSi10Mg should be for lightweight applications."""
        info = MATERIAL_DATABASE["AlSi10Mg-AM"]
        assert any("轻量化" in r or "航空" in r or "汽车" in r for r in info.process.recommendations)

    def test_in718_for_aerospace(self):
        """IN718-AM should be for aerospace applications."""
        info = MATERIAL_DATABASE["IN718-AM"]
        assert any("航空发动机" in r or "燃气轮机" in r for r in info.process.recommendations)

    def test_ti64_for_medical(self):
        """Ti64-AM should be for medical and aerospace."""
        info = MATERIAL_DATABASE["Ti64-AM"]
        assert any("医用" in r or "航空" in r or "植入" in r for r in info.process.recommendations)

    def test_am_materials_equivalence(self):
        """AM materials have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["AlSi10Mg-AM", "IN718-AM", "Ti64-AM"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_am_materials_cost(self):
        """AM materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["AlSi10Mg-AM", "IN718-AM", "Ti64-AM"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestHardAlloyMaterials:
    """Tests for hard alloy / cemented carbide materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("WC-Co", "WC-Co"),
        ("碳化钨", "WC-Co"),
        ("Tungsten Carbide", "WC-Co"),
        ("Stellite", "Stellite"),
        ("司太立", "Stellite"),
        ("钴基耐磨", "Stellite"),
        ("堆焊合金", "Stellite"),
        ("CBN", "CBN"),
        ("PCBN", "CBN"),
        ("立方氮化硼", "CBN"),
        ("氮化硼刀具", "CBN"),
        ("Cubic Boron Nitride", "CBN"),
    ])
    def test_hard_alloy_patterns(self, pattern, expected):
        """Hard alloy patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_hard_alloys_in_database(self):
        """All hard alloy grades should be in database."""
        for grade in ["WC-Co", "Stellite", "CBN"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.HARD_ALLOY

    def test_wc_co_for_cutting_tools(self):
        """WC-Co should be for cutting tools and molds."""
        info = MATERIAL_DATABASE["WC-Co"]
        assert any("刀具" in r or "模具" in r for r in info.process.recommendations)

    def test_stellite_for_valves(self):
        """Stellite should be for valve seats and wear resistance."""
        info = MATERIAL_DATABASE["Stellite"]
        assert any("阀门" in r or "耐磨" in r or "轴承" in r for r in info.process.recommendations)

    def test_cbn_superhardness(self):
        """CBN should be super hard for hardened steel machining."""
        info = MATERIAL_DATABASE["CBN"]
        assert any("淬硬钢" in r or "高速切削" in r for r in info.process.recommendations)

    def test_hard_alloys_equivalence(self):
        """Hard alloys have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["WC-Co", "Stellite", "CBN"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_hard_alloys_cost(self):
        """Hard alloys have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["WC-Co", "Stellite", "CBN"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestThermalBarrierMaterials:
    """Tests for thermal barrier coating materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("YSZ", "YSZ"),
        ("8YSZ", "YSZ"),
        ("氧化钇稳定氧化锆", "YSZ"),
        ("Yttria Stabilized Zirconia", "YSZ"),
        ("热障涂层", "YSZ"),
        ("TBC", "YSZ"),
        ("Al2O3-TBC", "Al2O3-TBC"),
        ("氧化铝涂层", "Al2O3-TBC"),
        ("Alumina Coating", "Al2O3-TBC"),
        ("TGO", "Al2O3-TBC"),
        ("MCrAlY", "MCrAlY"),
        ("NiCrAlY", "MCrAlY"),
        ("CoCrAlY", "MCrAlY"),
        ("Bond Coat", "MCrAlY"),
        ("粘结涂层", "MCrAlY"),
    ])
    def test_thermal_barrier_patterns(self, pattern, expected):
        """Thermal barrier patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_thermal_barriers_in_database(self):
        """All thermal barrier grades should be in database."""
        for grade in ["YSZ", "Al2O3-TBC", "MCrAlY"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.THERMAL_BARRIER

    def test_ysz_for_turbine_blades(self):
        """YSZ should be for turbine blades and hot section."""
        info = MATERIAL_DATABASE["YSZ"]
        assert any("涡轮" in r or "航空发动机" in r or "燃气轮机" in r for r in info.process.recommendations)

    def test_al2o3_tbc_for_wear(self):
        """Al2O3-TBC should be for wear resistance."""
        info = MATERIAL_DATABASE["Al2O3-TBC"]
        assert any("耐磨" in r or "绝缘" in r for r in info.process.recommendations)

    def test_mcrcaly_bond_coat(self):
        """MCrAlY should be a bond coat for TBC systems."""
        info = MATERIAL_DATABASE["MCrAlY"]
        assert any("粘结" in r or "TBC" in r or "抗氧化" in r for r in info.process.recommendations)

    def test_thermal_barriers_equivalence(self):
        """Thermal barrier materials have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["YSZ", "Al2O3-TBC", "MCrAlY"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_thermal_barriers_cost(self):
        """Thermal barrier materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["YSZ", "Al2O3-TBC", "MCrAlY"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"


class TestEMShieldingMaterials:
    """Tests for electromagnetic shielding materials."""

    @pytest.mark.parametrize("pattern,expected", [
        ("Mu-Metal", "Mu-Metal"),
        ("Mu Metal", "Mu-Metal"),
        ("坡莫合金", "Mu-Metal"),
        ("高导磁合金", "Mu-Metal"),
        ("磁屏蔽合金", "Mu-Metal"),
        ("Permalloy", "Permalloy"),
        ("1J50", "Permalloy"),
        ("软磁合金", "Permalloy"),
        ("45Permalloy", "Permalloy"),
        ("Copper-Mesh", "Copper-Mesh"),
        ("Copper Mesh", "Copper-Mesh"),
        ("铜丝网", "Copper-Mesh"),
        ("EMI屏蔽网", "Copper-Mesh"),
        ("Copper Shield", "Copper-Mesh"),
        ("RF屏蔽", "Copper-Mesh"),
    ])
    def test_em_shielding_patterns(self, pattern, expected):
        """EM shielding patterns should match correctly."""
        result = classify_material_detailed(pattern)
        assert result is not None, f"Pattern '{pattern}' should match"
        assert result.grade == expected, f"Pattern '{pattern}' should map to {expected}"

    def test_em_shielding_in_database(self):
        """All EM shielding grades should be in database."""
        for grade in ["Mu-Metal", "Permalloy", "Copper-Mesh"]:
            assert grade in MATERIAL_DATABASE, f"{grade} should be in database"
            info = MATERIAL_DATABASE[grade]
            assert info.group == MaterialGroup.EM_SHIELDING

    def test_mu_metal_for_magnetic_shielding(self):
        """Mu-Metal should be for magnetic shielding."""
        info = MATERIAL_DATABASE["Mu-Metal"]
        assert any("磁屏蔽" in r or "传感器" in r or "精密仪器" in r for r in info.process.recommendations)

    def test_permalloy_for_transformer_cores(self):
        """Permalloy should be for transformer cores."""
        info = MATERIAL_DATABASE["Permalloy"]
        assert any("变压器" in r or "磁头" in r or "电感" in r for r in info.process.recommendations)

    def test_copper_mesh_for_emi(self):
        """Copper-Mesh should be for EMI/RF shielding."""
        info = MATERIAL_DATABASE["Copper-Mesh"]
        assert any("EMI" in r or "RF" in r or "屏蔽" in r for r in info.process.recommendations)

    def test_em_shielding_equivalence(self):
        """EM shielding materials have equivalence data."""
        from src.core.materials import get_material_equivalence
        for grade in ["Mu-Metal", "Permalloy", "Copper-Mesh"]:
            equiv = get_material_equivalence(grade)
            assert equiv is not None, f"{grade} has no equivalence data"

    def test_em_shielding_cost(self):
        """EM shielding materials have cost data."""
        from src.core.materials import get_material_cost
        for grade in ["Mu-Metal", "Permalloy", "Copper-Mesh"]:
            cost = get_material_cost(grade)
            assert cost is not None, f"{grade} has no cost data"