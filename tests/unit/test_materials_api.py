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
