import pytest


def test_preview_endpoint_registered():
    from src.main import app

    paths = {route.path for route in app.router.routes}
    assert "/api/v1/vectors/migrate/preview" in paths
