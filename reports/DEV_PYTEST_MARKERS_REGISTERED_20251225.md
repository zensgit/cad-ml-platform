# Pytest markers registration check

- Date: 2025-12-25
- Change: Registered `performance` and `slow` markers in `pytest.ini`
- Command: `.venv/bin/python -m pytest tests/test_provider_timeout_simulation.py::test_timeout_scenario_report -q`
- Result: PASS (1 passed in 8.88s)
- Notes: No unknown-mark warnings during collection.
