#!/bin/bash
# =============================================================================
# Run Experiment in Docker
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

IMAGE_NAME="${IMAGE_NAME:-inference-optimization}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Parse arguments
COMMAND="${1:-run}"
shift || true

echo "=============================================="
echo "  Running Inference Optimization Experiment"
echo "=============================================="
echo ""
echo "Command: $COMMAND"
echo ""

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
    echo "Image not found. Building first..."
    ./scripts/build.sh
fi

# Create results directory
mkdir -p "$PROJECT_DIR/results/plots" "$PROJECT_DIR/results/logs"

# Run
docker run --rm \
    --gpus all \
    --shm-size 16g \
    -v "$PROJECT_DIR/results:/app/results" \
    -v "$PROJECT_DIR/configs:/app/configs:ro" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=0 \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$COMMAND" "$@"

echo ""
echo "=============================================="
echo "  Experiment Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $PROJECT_DIR/results/"
