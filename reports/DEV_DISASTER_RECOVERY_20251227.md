# Disaster Recovery Verification (Redis Crash)

- Date: 2025-12-27
- Scope: Simulate Redis crash and verify data recovery

## Commands
- docker exec cad-ml-redis redis-cli SET cad_ml_disaster_test <ts>
- sleep 2
- docker kill cad-ml-redis
- docker start cad-ml-redis
- docker exec cad-ml-redis redis-cli GET cad_ml_disaster_test
- curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health

## Result
- PASS

## Notes
- Test key persisted across SIGKILL + restart (AOF path)
- API health returned 200 after Redis restart
