# DEV_DEFAULT_HEALTH_POST_RESET_20251225

## Scope

- Verify default post-reset health after returning to standard compose config.

## Validation

- `GET /health` -> 200
- `GET /api/v1/dedup/2d/health` -> 503 (expected when dedupcad-vision not running)

## Notes

- `DEDUPCAD_VISION_URL` default remains `http://host.docker.internal:58001` in compose.
