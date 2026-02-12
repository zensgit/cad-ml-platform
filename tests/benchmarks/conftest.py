"""Benchmark test configuration."""

import pytest


def pytest_configure(config):
    """Register benchmark marker."""
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as a benchmark test",
    )


def pytest_collection_modifyitems(config, items):
    """Skip benchmark tests by default unless --benchmark flag is passed."""
    if not config.getoption("--benchmark", default=False):
        skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmark)


def pytest_addoption(parser):
    """Add --benchmark command line option."""
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="run benchmark tests",
    )
