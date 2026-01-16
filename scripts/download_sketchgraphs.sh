#!/bin/bash

# Script to download SketchGraphs dataset (2D Geometric Constraints)
# Reference: https://github.com/PrincetonLIPS/SketchGraphs

set -e

OUTPUT_DIR="data/sketchgraphs"
mkdir -p $OUTPUT_DIR

echo "========================================================"
echo "   SketchGraphs Dataset Downloader"
echo "========================================================"
echo "Downloading minimal test set (sequence data)..."

# SketchGraphs provides data in .tar.xz format hosted on Princeton servers.
# We download the 'sequence' validation set which is smaller and good for starting.

URL="https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_validation.tar.xz"
FILE_NAME="sg_t16_validation.tar.xz"

echo "Downloading $URL..."
curl -L -o "$OUTPUT_DIR/$FILE_NAME" "$URL"

echo "Extracting..."
tar -xf "$OUTPUT_DIR/$FILE_NAME" -C "$OUTPUT_DIR"

echo "✅ Download complete."
echo "Data location: $OUTPUT_DIR"
echo "This data contains JSON-like sequences of construction steps (Lines, Arcs, Constraints)."
