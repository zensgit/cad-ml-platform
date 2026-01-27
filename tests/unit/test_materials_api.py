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
