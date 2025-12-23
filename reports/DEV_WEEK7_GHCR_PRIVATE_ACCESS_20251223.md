# GHCR Private Image Access (Week 7)

## Summary
- CI e2e-smoke failed to pull the private `dedupcad-vision` image (manifest unknown).
- Added a GHCR token fallback to allow CI to authenticate when the package is private.
- Stored `GHCR_TOKEN` repository secret for GHCR pulls.

## Code Changes
- `.github/workflows/ci.yml`
  - Use `${{ secrets.GHCR_TOKEN || github.token }}` for `docker login`.

## Actions Taken
- Repository secret set: `GHCR_TOKEN` (read/write packages token) for GHCR pull access.

## Notes
- GHCR package `dedupcad-vision` is currently private; GITHUB_TOKEN cannot access user-owned private packages.
- If you prefer not to store a PAT in secrets, make the package public or move it under an org with repo access.
