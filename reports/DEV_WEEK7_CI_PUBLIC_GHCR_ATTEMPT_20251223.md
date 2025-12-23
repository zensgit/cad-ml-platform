# CI Rerun After GHCR Public Switch Attempt (Week 7)

## Summary
- Removed `GHCR_TOKEN` secret after user indicated the package was public.
- Reran CI; e2e-smoke still failed to pull `dedupcad-vision` with `manifest unknown`.

## Run
- Workflow: CI
- Run URL: https://github.com/zensgit/cad-ml-platform/actions/runs/20461012669
- Conclusion: failed (e2e-smoke)

## Failure Detail
- Step: `Start dedupcad-vision (real image)`
- Error: `docker: Error response from daemon: manifest unknown`

## Observations
- GitHub API still reports package visibility as `private`:
  - `GET /user/packages/container/dedupcad-vision` -> `"visibility":"private"`

## Next Actions
- Confirm `dedupcad-vision` package visibility is public in GitHub Packages UI.
- If it remains private, re-add a `GHCR_TOKEN` secret or keep package private and use PAT-based auth.
