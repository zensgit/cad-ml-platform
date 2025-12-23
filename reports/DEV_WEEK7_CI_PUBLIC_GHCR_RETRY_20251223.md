# CI Rerun After Public Visibility Confirmation (Week 7)

## Summary
- Reran CI after user confirmed GHCR package was set to public.
- e2e-smoke still failed to pull `dedupcad-vision` with `manifest unknown`.

## Run
- Workflow: CI
- Run URL: https://github.com/zensgit/cad-ml-platform/actions/runs/20461971541
- Conclusion: failure (e2e-smoke)

## Failure Detail
- Step: `Start dedupcad-vision (real image)`
- Error: `docker: Error response from daemon: manifest unknown`

## Public Access Check
- Unauthenticated registry probe returned `401`:
  - `curl -I https://ghcr.io/v2/zensgit/dedupcad-vision/manifests/sha256:9f7f...` -> `HTTP/2 401`
- GitHub API still reports visibility as `private`:
  - `GET /user/packages/container/dedupcad-vision`

## Next Actions
- Recheck the package visibility in GH Packages UI and ensure it is set to Public.
- If visibility is correct but still private via API, consider re-adding `GHCR_TOKEN` temporarily to unblock CI.
