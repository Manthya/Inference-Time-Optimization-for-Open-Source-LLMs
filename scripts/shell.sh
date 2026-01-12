#!/bin/bash
# =============================================================================
# Open Interactive Shell in Docker Container
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

IMAGE_NAME="${IMAGE_NAME:-inference-optimization}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=============================================="
echo "  Interactive Development Shell"
echo "=============================================="
echo ""

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
    echo "Image not found. Building first..."
    ./scripts/build.sh
fi

# Run interactive shell with code mounted
docker run -it --rm \
    --gpus all \
    --shm-size 16g \
    -v "$PROJECT_DIR/src:/app/src" \
    -v "$PROJECT_DIR/configs:/app/configs" \
    -v "$PROJECT_DIR/results:/app/results" \
    -v "$PROJECT_DIR/scripts:/app/scripts" \
    -v "$PROJECT_DIR/run_experiment.py:/app/run_experiment.py" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=0 \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    bash
