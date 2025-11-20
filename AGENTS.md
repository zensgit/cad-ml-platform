# Repository Guidelines

## Project Structure & Module Organization
- `src/main.py` FastAPI entry; Prometheus at `/metrics`, health at `/health`.
- `src/api/v1/` API routers (`analyze`, `similarity`, `classify`); register in `src/api/__init__.py`.
- `src/core/` Core logic (analyzers, adapters, models); `src/utils/` shared utils.
- `deployments/docker/` Compose files and images; `docs/` design notes; `examples/` demos.
- `tests/` Pytest suites (add `unit/`, `integration/` as needed).

## Build, Test, and Development Commands
- Setup env: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run API (dev): `uvicorn src.main:app --reload` (docs at `/docs`).
- Make targets: `make run`, `make test`, `make format`, `make lint`, `make type-check`.
- Docker compose: `docker-compose -f deployments/docker/docker-compose.yml up -d`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, mandatory type hints for new code.
- Format/lint/type: `black src tests`, `isort --profile black`, `flake8 --max-line-length=100`, `mypy src`.
- Naming: `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE` constants.
- FastAPI: place new routes under `src/api/v1/…` and register in `src/api/__init__.py`; prefer dependency injection and Pydantic models.
- Logging via `src/utils/logging`; avoid `print` and global state.

## Testing Guidelines
- Use `pytest` (+ `pytest-asyncio` for async). Name files `tests/test_*.py` and functions `test_*`.
- Run: `pytest tests -v --cov=src --cov-report=term-missing`.
- Cover happy paths and failures (e.g., empty file, unsupported format). Mock Redis/external calls for determinism.
- Target ≥80% coverage on changed lines; add golden cases for CAD parsing when possible.

## Commit & Pull Request Guidelines
- Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`). Example: `feat: add /api/v1/ocr/extract endpoint`.
- PR checklist: clear description, linked issue, test plan, API diffs (routes/models), sample cURL, and screenshots/logs if relevant.
- CI must pass (lint/type/tests). Run `make pre-commit` locally before pushing.

## Security & Configuration Tips
- Never commit secrets. Configure via env (e.g., `REDIS_URL`, API keys); document new vars in `README.md`.
- Review CORS/Trusted Hosts in settings before exposing services; prefer least privilege for service accounts.
