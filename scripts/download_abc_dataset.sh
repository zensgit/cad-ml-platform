#!/bin/bash

# Script to download the ABC Dataset (Step files)
# Reference: https://deep-geometry.github.io/abc-dataset/

set -e

OUTPUT_DIR="data/abc_dataset"
BASE_URL="https://deep-geometry.github.io/abc-dataset/data/abc_0000_obj_v00.7z"
# Note: The official links are hosted on various mirrors. 
# For stability in this script, we will use a representative link structure.
# Real usage often requires parsing their file list or using their provided torrents.

# ABC Dataset is split into chunks (0000 to 0099).
# Each chunk is a 7z archive containing STEP/OBJ files.

mkdir -p $OUTPUT_DIR

echo "========================================================"
echo "   ABC Dataset Downloader"
echo "========================================================"
echo "This script helps download chunks of the ABC Dataset."
echo "Each chunk is large (~500MB - 1GB compressed)."
echo ""
echo "Usage:"
echo "  ./download_abc_dataset.sh [chunk_id] [format]"
echo ""
echo "  chunk_id: 0000 to 0099 (default: 0000)"
echo "  format:   step (default) | obj | all"
echo ""

CHUNK_ID=${1:-"0000"}
FORMAT=${2:-"step"}

# Construct URL (This is an example URL pattern, official URLs change)
# We use a known mirror or direct link pattern if available.
# Since direct deep-linking often expires or changes, we advise using the torrent 
# or the official python downloader. 
# Here we simulate the process or point to the main landing page instructions 
# if a direct curl is unreliable.

echo "⚠️  NOTE: Direct HTTP download links for ABC often expire."
echo "⚠️  The most reliable way is via BitTorrent."
echo ""
echo "Downloading metadata for Chunk $CHUNK_ID..."

# Placeholder for actual curl command if a stable http mirror exists
# For now, we will create the directory structure and show the torrent link
# which is the recommended way.

TORRENT_URL="https://deep-geometry.github.io/abc-dataset/data/abc_v00.torrent"

echo "Downloading Torrent file..."
curl -L -o "$OUTPUT_DIR/abc_v00.torrent" "$TORRENT_URL"

echo ""
echo "✅ Torrent file downloaded to: $OUTPUT_DIR/abc_v00.torrent"
echo ""
echo "INSTRUCTIONS:"
echo "1. Use a BitTorrent client (Transmission, qBittorrent, etc.) to open the file."
echo "2. Select 'abc_$CHUNK_ID' folder to download specific chunks."
echo "3. Extract the 'step' files into '$OUTPUT_DIR/raw'."
echo ""
echo "Why Torrent? The dataset is >1TB. HTTP downloads are slow and fragile for this size."
