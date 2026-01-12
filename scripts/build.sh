#!/bin/bash
# =============================================================================
# Build Docker Image
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo "  Building Docker Image"
echo "=============================================="
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# Check NVIDIA Docker runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "WARNING: NVIDIA Docker runtime may not be available"
    echo "         GPU support might not work"
    echo ""
fi

# Build options
IMAGE_NAME="${IMAGE_NAME:-inference-optimization}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Model: ${MODEL_NAME}"
echo ""

# Build
echo "Building image (this may take 30-60 minutes for model download)..."
echo ""

docker build \
    --build-arg MODEL_NAME="${MODEL_NAME}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile \
    .

echo ""
echo "=============================================="
echo "  Build Complete!"
echo "=============================================="
echo ""
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run the experiment:"
echo "  docker-compose up"
echo ""
echo "Or with docker run:"
echo "  docker run --gpus all -v \$(pwd)/results:/app/results ${IMAGE_NAME}:${IMAGE_TAG} run"
echo ""
