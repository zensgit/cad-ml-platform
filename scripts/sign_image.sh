#!/bin/bash
# Script to sign Docker images using Cosign

IMAGE_NAME="${1:-cad-ml-platform:latest}"
KEY_PATH="${COSIGN_KEY_PATH:-cosign.key}"

if ! command -v cosign &> /dev/null; then
    echo "⚠️  Cosign is not installed. Skipping signing."
    echo "To install: brew install cosign"
    exit 0
fi

if [ ! -f "$KEY_PATH" ] && [ -z "$COSIGN_PRIVATE_KEY" ]; then
    echo "⚠️  No signing key found at $KEY_PATH or COSIGN_PRIVATE_KEY env var. Skipping signing."
    exit 0
fi

echo "🔐 Signing image: $IMAGE_NAME"

# If using env var for key
if [ -n "$COSIGN_PRIVATE_KEY" ]; then
    cosign sign --yes --key env://COSIGN_PRIVATE_KEY "$IMAGE_NAME"
else
    cosign sign --yes --key "$KEY_PATH" "$IMAGE_NAME"
fi

if [ $? -eq 0 ]; then
    echo "✅ Image signed successfully."
else
    echo "❌ Failed to sign image."
    exit 1
fi
