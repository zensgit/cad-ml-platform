"""Tests: downgrade migration chain (v4→v3→v2) and status counting.

Goals (TODO):
1. Seed vectors tagged with v4.
2. Migrate to v3 then to v2 using /api/v1/vectors/migrate.
3. Assert per-item statuses include "downgraded" and metrics vector_migrate_total increments.
4. Use summary endpoint to verify aggregated counts.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


@pytest.mark.skip(reason="TODO: implement vector seeding and downgrade chain assertions")
def test_vector_migrate_downgrade_chain_counts() -> None:
    # Placeholder call – real test will prepare IDs and perform migrations.
    response = client.get("/api/v1/vectors/migrate/summary")
    assert response.status_code == 200

