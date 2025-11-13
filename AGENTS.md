# Repository Guidelines

## Project Structure & Module Organization
- `src/` FastAPI app and core logic (`src/main.py`, `src/api/v1/…`).
- `tests/` Python tests (create `unit/`, `integration/` subfolders as needed).
- `deployments/docker/` Dockerfile and `docker-compose.yml`.
- `clients/` Example SDKs (e.g., `clients/python/cad_ml_client.py`).
- `docs/` Design plans and integration notes; `scripts/` helper scripts.
- `config/`, `knowledge_base/` Configuration and domain data.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run API (dev): `uvicorn src.main:app --reload`
- Compose stack: `docker-compose -f deployments/docker/docker-compose.yml up -d`
  - Exposes API, Redis, Prometheus; metrics at `/metrics`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, add type hints in new/changed code.
- Format/lint/type-check: `black src tests`, `flake8 src tests`, `mypy src`.
- Names: `snake_case` (functions/files), `PascalCase` (classes), `UPPER_SNAKE` (constants).
- FastAPI: place routes under `src/api/v1/` and register in `src/api/__init__.py`.
- Use `src.utils.logging` for logs; avoid `print`.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, coverage via `pytest-cov`.
- Naming: files `tests/test_*.py`; functions `test_*`.
- Run tests: `pytest tests/`
- Coverage: `pytest --cov=src --cov-report=term-missing`
- Aim for ≥80% coverage on changed lines; cover async paths.

## Commit & Pull Request Guidelines
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
  - Example: `fix: handle Redis readiness failure in /ready`
- PRs must include: description, linked issues, test plan, API changes (routes/models), and cURL examples for new endpoints.
- CI must pass; no lint/type errors; code formatted.

## Security & Configuration Tips
- Do not commit secrets; configure via env (e.g., `REDIS_URL`, provider API keys). Pass API key header `X-API-Key` to protected routes.
- Update CORS/hosts in `src/core/config.py` carefully and document changes.
