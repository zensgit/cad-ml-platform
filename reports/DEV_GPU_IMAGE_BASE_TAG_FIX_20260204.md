# DEV_GPU_IMAGE_BASE_TAG_FIX_20260204

## Summary
- Updated GPU base image tag to a valid NVIDIA CUDA runtime tag.
- Fixes GHCR multi-architecture GPU build failure caused by missing tag.

## Change
- `Dockerfile`: `nvidia/cuda:11.8-runtime-ubuntu22.04` → `nvidia/cuda:11.8.0-runtime-ubuntu22.04`

## Verification
- CI Multi-Architecture Docker Build and Security Scan are expected to pass once the GPU image builds successfully.
