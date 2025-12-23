# DEV_WEEK6_GHCR_IMAGE_PINNED_20251223

## Objective
- Build and publish the dedupcad-vision image to GHCR.
- Pin CI default image to the immutable digest and update docs.

## Changes
- Updated CI default image to GHCR digest.
- Updated README and CI optimization doc to reference the pinned digest.

## Image build/push
- Tag: ghcr.io/zensgit/dedupcad-vision:1.1.0
- Digest: ghcr.io/zensgit/dedupcad-vision@sha256:41cd67e8f7aeeb2a96b5fa3c49797af79ee4dda4df9885640a1385826cbbe5ce

## Tests / Verification
- Container health check:
  - docker run ... -p 58002:8000 ghcr.io/zensgit/dedupcad-vision@sha256:41cd67e8f7aeeb2a96b5fa3c49797af79ee4dda4df9885640a1385826cbbe5ce
  - curl http://localhost:58002/health -> OK

## Notes
- One initial curl attempt returned "connection reset by peer" before the service became ready; subsequent check passed.
