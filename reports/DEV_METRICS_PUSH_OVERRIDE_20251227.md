# Metrics Pushgateway URL Override

- Date: 2025-12-27
- Scope: Makefile metrics-push target

## Change
- Allow override via `PUSHGATEWAY_URL` (default remains http://localhost:9091).

## Notes
- Use `PUSHGATEWAY_URL=http://<host>:<port> make metrics-push` to target the correct Pushgateway.
