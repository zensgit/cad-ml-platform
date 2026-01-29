"""Tests for material API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from src.main import app
    return TestClient(app)


# API prefix
API_PREFIX = "/api/v1/materials"


class TestListMaterials:
    """Test GET /api/v1/materials endpoint."""

    def test_list_all_materials(self, client):
        """List all materials returns results."""
        response = client.get(f"{API_PREFIX}")
        assert response.status_code == 200

        data = response.json()
        assert "total" in data
        assert "materials" in data
        assert data["total"] > 30
        assert len(data["materials"]) == data["total"]

    def test_filter_by_category(self, client):
        """Filter by category works."""
        response = client.get(f"{API_PREFIX}?category=metal")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] > 0
        for mat in data["materials"]:
            assert mat["category"] == "metal"

    def test_filter_by_group(self, client):
        """Filter by group works."""
        response = client.get(f"{API_PREFIX}?group=stainless_steel")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] > 0
        for mat in data["materials"]:
            assert mat["group"] == "stainless_steel"

    def test_filter_returns_empty_for_invalid(self, client):
        """Invalid filter returns empty list."""
        response = client.get(f"{API_PREFIX}?category=invalid")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 0
        assert data["materials"] == []


class TestListMaterialGroups:
    """Test GET /api/v1/materials/groups endpoint."""

    def test_list_groups(self, client):
        """List groups returns all categories."""
        response = client.get(f"{API_PREFIX}/groups")
        assert response.status_code == 200

        data = response.json()
        assert "groups" in data
        assert "metal" in data["groups"]
        assert "non_metal" in data["groups"]
        assert "composite" in data["groups"]

    def test_metal_groups_not_empty(self, client):
        """Metal groups are populated."""
        response = client.get(f"{API_PREFIX}/groups")
        data = response.json()

        metal_groups = data["groups"]["metal"]
        assert len(metal_groups) > 5
        assert "stainless_steel" in metal_groups
        assert "carbon_steel" in metal_groups


class TestGetMaterial:
    """Test GET /api/v1/materials/{grade} endpoint."""

    def test_get_existing_material(self, client):
        """Get existing material returns full info."""
        response = client.get(f"{API_PREFIX}/S30408")
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is True
        assert data["grade"] == "S30408"
        assert data["name"] == "奥氏体不锈钢"
        assert data["category"] == "metal"
        assert data["group"] == "stainless_steel"
        assert data["properties"] is not None
        assert data["process"] is not None

    def test_get_material_by_alias(self, client):
        """Get material by alias works."""
        response = client.get(f"{API_PREFIX}/304")
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is True
        assert data["grade"] == "S30408"

    def test_get_material_by_pattern(self, client):
        """Get material by pattern works."""
        response = client.get(f"{API_PREFIX}/C-22")
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is True
        assert data["grade"] == "C22"

    def test_get_unknown_material(self, client):
        """Get unknown material returns not found."""
        response = client.get(f"{API_PREFIX}/UNKNOWN123")
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is False
        assert data["grade"] == "UNKNOWN123"

    def test_material_properties_populated(self, client):
        """Material properties are populated."""
        response = client.get(f"{API_PREFIX}/45")
        data = response.json()

        props = data["properties"]
        assert props["density"] is not None
        assert props["tensile_strength"] is not None
        assert props["machinability"] is not None

    def test_material_process_populated(self, client):
        """Material process recommendations are populated."""
        response = client.get(f"{API_PREFIX}/TC4")
        data = response.json()

        process = data["process"]
        assert len(process["blank_forms"]) > 0
        assert process["blank_hint"] != ""
        assert process["special_tooling"] is True
        assert len(process["warnings"]) > 0


class TestClassifyMaterial:
    """Test POST /api/v1/materials/classify endpoint."""

    def test_classify_known_material(self, client):
        """Classify known material returns info."""
        response = client.post(
            f"{API_PREFIX}/classify",
            json={"material": "304不锈钢"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is True
        assert data["grade"] == "S30408"
        assert data["group"] == "stainless_steel"

    def test_classify_unknown_material(self, client):
        """Classify unknown material returns not found."""
        response = client.post(
            f"{API_PREFIX}/classify",
            json={"material": "XYZ999"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is False
        assert data["input"] == "XYZ999"

    def test_classify_with_pattern(self, client):
        """Classify with pattern matching."""
        response = client.post(
            f"{API_PREFIX}/classify",
            json={"material": "Hastelloy C-276"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is True
        assert data["grade"] == "C276"


class TestBatchClassify:
    """Test POST /api/v1/materials/batch-classify endpoint."""

    def test_batch_classify(self, client):
        """Batch classify multiple materials."""
        response = client.post(
            f"{API_PREFIX}/batch-classify?materials=304&materials=C22&materials=UNKNOWN"
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 3

        # First should be found
        assert data[0]["found"] is True
        assert data[0]["grade"] == "S30408"

        # Second should be found
        assert data[1]["found"] is True
        assert data[1]["grade"] == "C22"

        # Third should not be found
        assert data[2]["found"] is False


class TestGetMaterialProcess:
    """Test GET /api/v1/materials/{grade}/process endpoint."""

    def test_get_process_for_existing(self, client):
        """Get process for existing material."""
        response = client.get(f"{API_PREFIX}/S30408/process")
        assert response.status_code == 200

        data = response.json()
        assert "固溶处理" in data["heat_treatments"]
        assert "钝化" in data["surface_treatments"]
        assert "淬火" in data["forbidden_heat_treatments"]

    def test_get_process_for_unknown(self, client):
        """Get process for unknown material returns 404."""
        response = client.get(f"{API_PREFIX}/UNKNOWN123/process")
        assert response.status_code == 404


class TestMaterialEquivalence:
    """Test GET /api/v1/materials/{grade}/equivalents endpoint."""

    def test_get_equivalents_for_stainless(self, client):
        """Get equivalents for 304 stainless steel."""
        response = client.get(f"{API_PREFIX}/304/equivalents")
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is True
        assert data["equivalents"]["CN"] == "S30408"
        assert data["equivalents"]["US"] == "304"
        assert data["equivalents"]["JP"] == "SUS304"
        assert data["equivalents"]["DE"] == "1.4301"

    def test_get_equivalents_from_japanese_standard(self, client):
        """Get equivalents using Japanese standard input."""
        response = client.get(f"{API_PREFIX}/SUS304/equivalents")
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is True
        assert data["equivalents"]["CN"] == "S30408"

    def test_get_equivalents_for_alloy_steel(self, client):
        """Get equivalents for 42CrMo alloy steel."""
        response = client.get(f"{API_PREFIX}/42CrMo/equivalents")
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is True
        assert data["equivalents"]["US"] == "4140"
        assert data["equivalents"]["JP"] == "SCM440"
        assert data["equivalents"]["DE"] == "42CrMo4"

    def test_get_equivalents_for_unknown(self, client):
        """Get equivalents for unknown material returns not found."""
        response = client.get(f"{API_PREFIX}/UNKNOWN_MATERIAL_XYZ/equivalents")
        assert response.status_code == 200

        data = response.json()
        assert data["found"] is False


class TestMaterialConvert:
    """Test GET /api/v1/materials/{grade}/convert/{target} endpoint."""

    def test_convert_japanese_to_chinese(self, client):
        """Convert Japanese S45C to Chinese standard."""
        response = client.get(f"{API_PREFIX}/S45C/convert/CN")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["result"] == "45"

    def test_convert_american_to_german(self, client):
        """Convert American 4140 to German standard."""
        response = client.get(f"{API_PREFIX}/4140/convert/DE")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["result"] == "42CrMo4"

    def test_convert_to_uns(self, client):
        """Convert to UNS number."""
        response = client.get(f"{API_PREFIX}/C276/convert/UNS")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["result"] == "N10276"

    def test_convert_unknown_material(self, client):
        """Convert unknown material returns error."""
        response = client.get(f"{API_PREFIX}/UNKNOWN123/convert/CN")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"]


class TestMaterialExport:
    """Test GET /api/v1/materials/export/* endpoints."""

    def test_export_materials_csv(self, client):
        """Export materials CSV returns valid CSV."""
        response = client.get(f"{API_PREFIX}/export/csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        assert "materials.csv" in response.headers["content-disposition"]

        # Check CSV content
        content = response.text
        lines = content.strip().split('\n')
        assert len(lines) > 40  # At least 40 materials + header
        assert "牌号" in lines[0]  # Header contains grade column
        assert "S30408" in content  # Contains common material

    def test_export_equivalence_csv(self, client):
        """Export equivalence CSV returns valid CSV."""
        response = client.get(f"{API_PREFIX}/export/equivalence-csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        assert "equivalence.csv" in response.headers["content-disposition"]

        # Check CSV content
        content = response.text
        lines = content.strip().split('\n')
        assert len(lines) > 20  # At least 20 equivalences + header
        assert "中国(CN)" in lines[0]  # Header contains CN column
        assert "美国(US)" in lines[0]  # Header contains US column


class TestMaterialSearch:
    """Test GET /api/v1/materials/search endpoint."""

    def test_search_exact_match(self, client):
        """Search with exact grade match."""
        response = client.get(f"{API_PREFIX}/search?q=S30408")
        assert response.status_code == 200

        data = response.json()
        assert data["query"] == "S30408"
        assert data["total"] == 1
        assert data["results"][0]["grade"] == "S30408"
        assert data["results"][0]["score"] == 1.0

    def test_search_alias(self, client):
        """Search with alias."""
        response = client.get(f"{API_PREFIX}/search?q=304")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert data["results"][0]["grade"] == "S30408"

    def test_search_chinese_name(self, client):
        """Search with Chinese name."""
        response = client.get(f"{API_PREFIX}/search?q=黄铜")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 1
        # 应该返回 H62
        grades = [r["grade"] for r in data["results"]]
        assert "H62" in grades

    def test_search_pinyin(self, client):
        """Search with pinyin."""
        response = client.get(f"{API_PREFIX}/search?q=huangtong")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 1
        grades = [r["grade"] for r in data["results"]]
        assert "H62" in grades or "H68" in grades

    def test_search_pinyin_abbrev(self, client):
        """Search with pinyin abbreviation."""
        response = client.get(f"{API_PREFIX}/search?q=bxg")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 1
        grades = [r["grade"] for r in data["results"]]
        assert "S30408" in grades or "S31603" in grades

    def test_search_fuzzy(self, client):
        """Search with fuzzy matching."""
        response = client.get(f"{API_PREFIX}/search?q=铝合")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 1
        # 应该返回铝合金相关材料
        groups = [r["group"] for r in data["results"]]
        assert "aluminum" in groups

    def test_search_with_category_filter(self, client):
        """Search with category filter."""
        response = client.get(f"{API_PREFIX}/search?q=steel&category=metal")
        assert response.status_code == 200

        data = response.json()
        for r in data["results"]:
            assert r["category"] == "metal"

    def test_search_with_limit(self, client):
        """Search with limit parameter."""
        response = client.get(f"{API_PREFIX}/search?q=钢&limit=3")
        assert response.status_code == 200

        data = response.json()
        assert len(data["results"]) <= 3

    def test_search_empty_query(self, client):
        """Search with empty query returns error."""
        response = client.get(f"{API_PREFIX}/search?q=")
        # FastAPI validation should return 422
        assert response.status_code == 422


class TestPropertySearch:
    """Test GET /api/v1/materials/search/properties endpoint."""

    def test_search_by_density(self, client):
        """Search by density range."""
        response = client.get(f"{API_PREFIX}/search/properties?density_min=2.5&density_max=3.0")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 1
        for r in data["results"]:
            assert 2.5 <= r["density"] <= 3.0

    def test_search_by_strength(self, client):
        """Search by tensile strength range."""
        response = client.get(f"{API_PREFIX}/search/properties?strength_min=1000&strength_max=1500")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 1
        for r in data["results"]:
            assert 1000 <= r["tensile_strength"] <= 1500

    def test_search_by_hardness_type(self, client):
        """Search by hardness type."""
        response = client.get(f"{API_PREFIX}/search/properties?hardness=HRC")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 1
        for r in data["results"]:
            assert "HRC" in r["hardness"]

    def test_search_by_machinability(self, client):
        """Search by machinability."""
        response = client.get(f"{API_PREFIX}/search/properties?machinability=excellent")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 1
        for r in data["results"]:
            assert r["machinability"] == "excellent"

    def test_search_combined_filters(self, client):
        """Search with combined filters."""
        response = client.get(
            f"{API_PREFIX}/search/properties?density_max=8.0&machinability=good&category=metal"
        )
        assert response.status_code == 200

        data = response.json()
        for r in data["results"]:
            assert r["density"] <= 8.0
            assert r["machinability"] == "good"
            assert r["category"] == "metal"

    def test_search_with_limit(self, client):
        """Search with limit parameter."""
        response = client.get(f"{API_PREFIX}/search/properties?limit=5")
        assert response.status_code == 200

        data = response.json()
        assert len(data["results"]) <= 5


class TestMaterialApplications:
    """Test GET /api/v1/materials/applications endpoint."""

    def test_list_applications(self, client):
        """List all applications."""
        response = client.get(f"{API_PREFIX}/applications")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] > 10
        assert len(data["applications"]) == data["total"]

        # 检查必需的用途
        codes = [a["code"] for a in data["applications"]]
        assert "structural" in codes
        assert "corrosion_resistant" in codes
        assert "electrical" in codes


class TestMaterialRecommendations:
    """Test GET /api/v1/materials/recommend/{application} endpoint."""

    def test_recommend_structural(self, client):
        """Recommend materials for structural use."""
        response = client.get(f"{API_PREFIX}/recommend/structural")
        assert response.status_code == 200

        data = response.json()
        assert data["application"] == "structural"
        assert data["total"] > 0

        # 结构件应推荐碳钢或合金钢
        groups = [r["group"] for r in data["recommendations"]]
        assert "carbon_steel" in groups or "alloy_steel" in groups

    def test_recommend_electrical(self, client):
        """Recommend materials for electrical use."""
        response = client.get(f"{API_PREFIX}/recommend/electrical")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] > 0

        # 导电件应推荐铜或铝
        groups = [r["group"] for r in data["recommendations"]]
        assert "copper" in groups or "aluminum" in groups

    def test_recommend_corrosion_resistant(self, client):
        """Recommend materials for corrosion resistant use."""
        response = client.get(f"{API_PREFIX}/recommend/corrosion_resistant")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] > 0

        # 耐腐蚀应推荐不锈钢或耐蚀合金
        groups = [r["group"] for r in data["recommendations"]]
        assert "stainless_steel" in groups or "corrosion_resistant" in groups

    def test_recommend_with_strength_filter(self, client):
        """Recommend with minimum strength filter."""
        response = client.get(f"{API_PREFIX}/recommend/structural?min_strength=600")
        assert response.status_code == 200

        data = response.json()
        for r in data["recommendations"]:
            assert r["properties"]["tensile_strength"] >= 600

    def test_recommend_with_density_filter(self, client):
        """Recommend with maximum density filter."""
        response = client.get(f"{API_PREFIX}/recommend/lightweight?max_density=5.0")
        assert response.status_code == 200

        data = response.json()
        for r in data["recommendations"]:
            assert r["properties"]["density"] <= 5.0

    def test_recommend_invalid_application(self, client):
        """Recommend for invalid application returns empty."""
        response = client.get(f"{API_PREFIX}/recommend/invalid_app")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 0

    def test_recommend_with_limit(self, client):
        """Recommend with limit parameter."""
        response = client.get(f"{API_PREFIX}/recommend/structural?limit=3")
        assert response.status_code == 200

        data = response.json()
        assert len(data["recommendations"]) <= 3


class TestMaterialAlternatives:
    """Test GET /api/v1/materials/{grade}/alternatives endpoint."""

    def test_get_alternatives_s30408(self, client):
        """Get alternatives for S30408."""
        response = client.get(f"{API_PREFIX}/S30408/alternatives")
        assert response.status_code == 200

        data = response.json()
        assert data["original_grade"] == "S30408"
        assert data["total"] > 0

        # 检查预定义替代
        grades = [a["grade"] for a in data["alternatives"]]
        assert "S31603" in grades

    def test_get_alternatives_cheaper(self, client):
        """Get cheaper alternatives."""
        response = client.get(f"{API_PREFIX}/7075/alternatives?preference=cheaper")
        assert response.status_code == 200

        data = response.json()
        assert data["preference"] == "cheaper"
        for a in data["alternatives"]:
            assert a["cost_factor"] < 1.0

    def test_get_alternatives_better(self, client):
        """Get better alternatives."""
        response = client.get(f"{API_PREFIX}/45/alternatives?preference=better")
        assert response.status_code == 200

        data = response.json()
        assert data["preference"] == "better"
        for a in data["alternatives"]:
            assert a["cost_factor"] > 1.0

    def test_get_alternatives_by_alias(self, client):
        """Get alternatives using alias."""
        response = client.get(f"{API_PREFIX}/304/alternatives")
        assert response.status_code == 200

        data = response.json()
        # 304 应该被解析为 S30408
        assert data["total"] > 0

    def test_get_alternatives_unknown_material(self, client):
        """Get alternatives for unknown material returns empty."""
        response = client.get(f"{API_PREFIX}/UNKNOWN_MATERIAL/alternatives")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 0


class TestMaterialCostTiers:
    """Test GET /api/v1/materials/cost/tiers endpoint."""

    def test_list_cost_tiers(self, client):
        """List all cost tiers."""
        response = client.get(f"{API_PREFIX}/cost/tiers")
        assert response.status_code == 200

        data = response.json()
        assert len(data["tiers"]) == 5

        # 检查等级顺序
        tiers = [t["tier"] for t in data["tiers"]]
        assert tiers == [1, 2, 3, 4, 5]

        # 检查必要字段
        for t in data["tiers"]:
            assert "name" in t
            assert "description" in t
            assert "price_range" in t


class TestMaterialCostSearch:
    """Test GET /api/v1/materials/cost/search endpoint."""

    def test_search_by_max_tier(self, client):
        """Search materials by max cost tier."""
        response = client.get(f"{API_PREFIX}/cost/search?max_tier=1")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] > 0
        for r in data["results"]:
            assert r["tier"] == 1

    def test_search_by_max_cost_index(self, client):
        """Search materials by max cost index."""
        response = client.get(f"{API_PREFIX}/cost/search?max_cost_index=2.0")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] > 0
        for r in data["results"]:
            assert r["cost_index"] <= 2.0

    def test_search_with_category_filter(self, client):
        """Search with category filter."""
        response = client.get(f"{API_PREFIX}/cost/search?max_tier=2&category=metal")
        assert response.status_code == 200

        data = response.json()
        for r in data["results"]:
            assert r["category"] == "metal"
            assert r["tier"] <= 2

    def test_search_with_limit(self, client):
        """Search with limit parameter."""
        response = client.get(f"{API_PREFIX}/cost/search?limit=5")
        assert response.status_code == 200

        data = response.json()
        assert len(data["results"]) <= 5


class TestMaterialCostCompare:
    """Test POST /api/v1/materials/cost/compare endpoint."""

    def test_compare_costs(self, client):
        """Compare costs of multiple materials."""
        response = client.post(
            f"{API_PREFIX}/cost/compare?grades=Q235B&grades=S30408&grades=TC4"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 3
        assert data["missing"] == []

        # 应该按成本排序
        costs = [c["cost_index"] for c in data["comparison"]]
        assert costs == sorted(costs)

        # Q235B 应该最便宜
        assert data["comparison"][0]["grade"] == "Q235B"
        assert data["comparison"][0]["relative_to_cheapest"] == 1.0

    def test_compare_with_alias(self, client):
        """Compare using alias names."""
        response = client.post(
            f"{API_PREFIX}/cost/compare?grades=304&grades=316L"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 2
        assert data["missing"] == []

    def test_compare_with_unknown_material(self, client):
        """Compare costs with unknown material returns missing list."""
        response = client.post(
            f"{API_PREFIX}/cost/compare?grades=Q235B&grades=UNKNOWN_XYZ"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert data["missing"] == ["UNKNOWN_XYZ"]


class TestMaterialCost:
    """Test GET /api/v1/materials/{grade}/cost endpoint."""

    def test_get_cost_q235b(self, client):
        """Get cost info for Q235B (baseline)."""
        response = client.get(f"{API_PREFIX}/Q235B/cost")
        assert response.status_code == 200

        data = response.json()
        assert data["grade"] == "Q235B"
        assert data["tier"] == 1
        assert data["cost_index"] == 1.0
        assert data["tier_name"] == "经济型"

    def test_get_cost_s30408(self, client):
        """Get cost info for S30408 (304 stainless)."""
        response = client.get(f"{API_PREFIX}/S30408/cost")
        assert response.status_code == 200

        data = response.json()
        assert data["grade"] == "S30408"
        assert data["tier"] == 2
        assert data["cost_index"] > 1.0

    def test_get_cost_tc4(self, client):
        """Get cost info for TC4 (titanium)."""
        response = client.get(f"{API_PREFIX}/TC4/cost")
        assert response.status_code == 200

        data = response.json()
        assert data["grade"] == "TC4"
        assert data["tier"] == 4
        assert data["cost_index"] >= 20.0

    def test_get_cost_by_alias(self, client):
        """Get cost using alias."""
        response = client.get(f"{API_PREFIX}/304/cost")
        assert response.status_code == 200

        data = response.json()
        assert data["grade"] == "S30408"

    def test_get_cost_unknown_material(self, client):
        """Get cost for unknown material returns 404."""
        response = client.get(f"{API_PREFIX}/UNKNOWN_MATERIAL_XYZ/cost")
        assert response.status_code == 404


class TestWeldCompatibility:
    """Test GET /api/v1/materials/compatibility/weld endpoint."""

    def test_weld_same_carbon_steel(self, client):
        """Weld compatibility for same material group (carbon steel)."""
        response = client.get(
            f"{API_PREFIX}/compatibility/weld?material1=Q235B&material2=45"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["compatible"] is True
        assert data["rating"] == "excellent"
        assert data["rating_cn"] == "优秀"
        assert data["material1"]["grade"] == "Q235B"
        assert data["material2"]["grade"] == "45"

    def test_weld_same_stainless_steel(self, client):
        """Weld compatibility for stainless steel."""
        response = client.get(
            f"{API_PREFIX}/compatibility/weld?material1=S30408&material2=S31603"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["compatible"] is True
        assert data["rating"] in ["excellent", "good"]
        assert "method" in data
        assert data["method"] != ""

    def test_weld_carbon_to_stainless(self, client):
        """Weld compatibility for carbon steel to stainless steel."""
        response = client.get(
            f"{API_PREFIX}/compatibility/weld?material1=Q235B&material2=S30408"
        )
        assert response.status_code == 200

        data = response.json()
        # 异种钢焊接，兼容等级应为 fair 或以上
        assert data["rating"] in ["fair", "good", "poor"]
        assert "notes" in data

    def test_weld_aluminum_to_steel(self, client):
        """Weld compatibility for aluminum to steel (difficult)."""
        response = client.get(
            f"{API_PREFIX}/compatibility/weld?material1=6061&material2=Q235B"
        )
        assert response.status_code == 200

        data = response.json()
        # 铝和钢焊接困难
        assert data["rating"] in ["poor", "incompatible", "not_recommended"]
        assert data["compatible"] is False

    def test_weld_with_alias(self, client):
        """Weld compatibility using alias."""
        response = client.get(
            f"{API_PREFIX}/compatibility/weld?material1=304&material2=316L"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["compatible"] is True
        assert data["material1"]["name"] != "304"  # Should be resolved to full name

    def test_weld_unknown_material(self, client):
        """Weld compatibility with unknown material."""
        response = client.get(
            f"{API_PREFIX}/compatibility/weld?material1=UNKNOWN_XYZ&material2=Q235B"
        )
        assert response.status_code == 200

        data = response.json()
        # 未知材料应返回 unknown 状态
        assert data["rating"] == "unknown"


class TestGalvanicCorrosion:
    """Test GET /api/v1/materials/compatibility/galvanic endpoint."""

    def test_galvanic_same_material(self, client):
        """Galvanic corrosion for same material group."""
        response = client.get(
            f"{API_PREFIX}/compatibility/galvanic?material1=S30408&material2=S31603"
        )
        assert response.status_code == 200

        data = response.json()
        # 相同材料组电位差小，风险低
        assert data["risk"] in ["negligible", "low", "safe"]
        assert "risk_cn" in data

    def test_galvanic_aluminum_to_stainless(self, client):
        """Galvanic corrosion for aluminum to stainless steel."""
        response = client.get(
            f"{API_PREFIX}/compatibility/galvanic?material1=6061&material2=S30408"
        )
        assert response.status_code == 200

        data = response.json()
        # 铝和不锈钢电位差大，风险较高
        assert data["risk"] in ["moderate", "high", "severe"]
        assert data["potential_difference"] is not None
        assert abs(data["potential_difference"]) > 0.3

        # 铝是阳极
        assert data["anode"] is not None
        assert "6061" in data["anode"]["grade"]

    def test_galvanic_copper_to_steel(self, client):
        """Galvanic corrosion for copper to steel."""
        response = client.get(
            f"{API_PREFIX}/compatibility/galvanic?material1=H62&material2=Q235B"
        )
        assert response.status_code == 200

        data = response.json()
        # 铜和钢有一定电位差
        assert data["risk"] in ["low", "moderate", "medium", "high"]

        # 钢是阳极（更活泼）
        if data["anode"]:
            assert "Q235B" in data["anode"]["grade"]

    def test_galvanic_titanium_to_aluminum(self, client):
        """Galvanic corrosion for titanium to aluminum (high risk)."""
        response = client.get(
            f"{API_PREFIX}/compatibility/galvanic?material1=TC4&material2=6061"
        )
        assert response.status_code == 200

        data = response.json()
        # 钛和铝电位差很大，高风险
        assert data["risk"] in ["high", "severe"]
        assert data["recommendation"] is not None

    def test_galvanic_with_recommendation(self, client):
        """Galvanic corrosion returns recommendations."""
        response = client.get(
            f"{API_PREFIX}/compatibility/galvanic?material1=7075&material2=S30408"
        )
        assert response.status_code == 200

        data = response.json()
        # 高风险组合应有建议
        if data["risk"] in ["moderate", "high", "severe"]:
            assert data["recommendation"] is not None
            assert len(data["recommendation"]) > 0

    def test_galvanic_unknown_material(self, client):
        """Galvanic corrosion with unknown material."""
        response = client.get(
            f"{API_PREFIX}/compatibility/galvanic?material1=UNKNOWN_XYZ&material2=Q235B"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["risk"] == "unknown"


class TestHeatTreatmentCompatibility:
    """Test GET /api/v1/materials/compatibility/heat-treatment endpoint."""

    def test_heat_treatment_carbon_steel_quench(self, client):
        """Heat treatment compatibility for carbon steel quenching."""
        response = client.get(
            f"{API_PREFIX}/compatibility/heat-treatment?grade=45&treatment=淬火"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["compatible"] is True
        assert data["status"] in ["recommended", "allowed"]
        assert data["grade"] == "45"
        assert data["treatment"] == "淬火"

    def test_heat_treatment_stainless_solution(self, client):
        """Heat treatment compatibility for stainless steel solution treatment."""
        response = client.get(
            f"{API_PREFIX}/compatibility/heat-treatment?grade=S30408&treatment=固溶处理"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["compatible"] is True
        assert data["status"] == "recommended"

    def test_heat_treatment_stainless_forbidden_quench(self, client):
        """Heat treatment: stainless steel cannot be quenched normally."""
        response = client.get(
            f"{API_PREFIX}/compatibility/heat-treatment?grade=S30408&treatment=淬火"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["compatible"] is False
        assert data["status"] in ["forbidden", "not_recommended"]
        assert "reason" in data

    def test_heat_treatment_aluminum_aging(self, client):
        """Heat treatment compatibility for aluminum aging."""
        response = client.get(
            f"{API_PREFIX}/compatibility/heat-treatment?grade=7075&treatment=时效处理"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["compatible"] is True

    def test_heat_treatment_returns_lists(self, client):
        """Heat treatment returns recommended and forbidden lists."""
        response = client.get(
            f"{API_PREFIX}/compatibility/heat-treatment?grade=42CrMo&treatment=调质"
        )
        assert response.status_code == 200

        data = response.json()
        assert "recommended_treatments" in data
        assert "forbidden_treatments" in data
        assert isinstance(data["recommended_treatments"], list)
        assert isinstance(data["forbidden_treatments"], list)

    def test_heat_treatment_with_alias(self, client):
        """Heat treatment compatibility using alias."""
        response = client.get(
            f"{API_PREFIX}/compatibility/heat-treatment?grade=304&treatment=固溶处理"
        )
        assert response.status_code == 200

        data = response.json()
        # 304 应该被解析为 S30408
        assert data["name"] != "304"

    def test_heat_treatment_unknown_material(self, client):
        """Heat treatment with unknown material."""
        response = client.get(
            f"{API_PREFIX}/compatibility/heat-treatment?grade=UNKNOWN_XYZ&treatment=淬火"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["compatible"] is False
        assert data["status"] == "unknown"


class TestFullCompatibility:
    """Test GET /api/v1/materials/compatibility/full endpoint."""

    def test_full_compatibility_same_group(self, client):
        """Full compatibility for same material group."""
        response = client.get(
            f"{API_PREFIX}/compatibility/full?material1=Q235B&material2=45"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["overall"] in ["excellent", "good", "compatible"]
        assert data["overall_cn"] in ["优秀", "良好", "兼容"]
        assert "weld_compatibility" in data
        assert "galvanic_corrosion" in data
        assert isinstance(data["issues"], list)
        assert isinstance(data["recommendations"], list)

    def test_full_compatibility_different_groups(self, client):
        """Full compatibility for different material groups."""
        response = client.get(
            f"{API_PREFIX}/compatibility/full?material1=S30408&material2=6061"
        )
        assert response.status_code == 200

        data = response.json()
        # 不锈钢和铝有兼容性问题
        assert data["overall"] in ["fair", "poor", "incompatible"]
        # 应该有问题列表
        assert len(data["issues"]) > 0 or data["overall"] in ["good", "excellent"]

    def test_full_compatibility_includes_weld(self, client):
        """Full compatibility includes weld details."""
        response = client.get(
            f"{API_PREFIX}/compatibility/full?material1=Q235B&material2=S30408"
        )
        assert response.status_code == 200

        data = response.json()
        weld = data["weld_compatibility"]
        assert "compatible" in weld
        assert "rating" in weld
        assert "method" in weld

    def test_full_compatibility_includes_galvanic(self, client):
        """Full compatibility includes galvanic details."""
        response = client.get(
            f"{API_PREFIX}/compatibility/full?material1=H62&material2=6061"
        )
        assert response.status_code == 200

        data = response.json()
        galvanic = data["galvanic_corrosion"]
        assert "risk" in galvanic

    def test_full_compatibility_recommendations(self, client):
        """Full compatibility provides recommendations for problematic pairs."""
        response = client.get(
            f"{API_PREFIX}/compatibility/full?material1=TC4&material2=6061"
        )
        assert response.status_code == 200

        data = response.json()
        # 钛和铝组合问题多，应有建议
        if data["overall"] in ["poor", "incompatible"]:
            assert len(data["recommendations"]) > 0

    def test_full_compatibility_with_alias(self, client):
        """Full compatibility using alias."""
        response = client.get(
            f"{API_PREFIX}/compatibility/full?material1=304&material2=316L"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["overall"] in ["excellent", "good", "compatible"]

    def test_full_compatibility_unknown_material(self, client):
        """Full compatibility with unknown material."""
        response = client.get(
            f"{API_PREFIX}/compatibility/full?material1=UNKNOWN_XYZ&material2=Q235B"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["overall"] in ["unknown", "caution"]
