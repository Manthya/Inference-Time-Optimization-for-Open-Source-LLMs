#!/bin/bash
# =============================================================================
# Docker Entrypoint for Inference Optimization Experiment
# =============================================================================
set -e

# Print banner
echo "=============================================="
echo "  Inference-Time Optimization Experiment"
echo "=============================================="
echo ""

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  No GPU detected!"
echo ""

# Set environment
export PYTHONPATH=/app:$PYTHONPATH
export HF_HOME=/app/models
export TRANSFORMERS_CACHE=/app/models

# Handle commands
case "$1" in
    run)
        echo "Running full experiment..."
        echo ""
        python /app/run_experiment.py "${@:2}"
        ;;
    
    test)
        echo "Running quick test (reduced samples)..."
        echo ""
        python /app/run_experiment.py --config /app/configs/test_config.yaml "${@:2}"
        ;;
    
    stage0)
        echo "Running Stage 0 only (Baseline)..."
        echo ""
        python /app/run_experiment.py --stages stage_0 "${@:2}"
        ;;
    
    stage1)
        echo "Running Stage 1 only (vLLM)..."
        echo ""
        python /app/run_experiment.py --stages stage_1 "${@:2}"
        ;;
    
    stage2)
        echo "Running Stage 2 only (vLLM + Fusion)..."
        echo ""
        python /app/run_experiment.py --stages stage_2 "${@:2}"
        ;;
    
    verify)
        echo "Verifying setup..."
        echo ""
        python /app/test_setup.py
        ;;
    
    bash)
        echo "Starting interactive shell..."
        echo ""
        exec /bin/bash
        ;;
    
    *)
        echo "Usage: docker run <image> [command]"
        echo ""
        echo "Commands:"
        echo "  run      - Run full experiment (all stages)"
        echo "  test     - Run quick test with reduced samples"
        echo "  stage0   - Run Stage 0 only (HF Baseline)"
        echo "  stage1   - Run Stage 1 only (vLLM)"
        echo "  stage2   - Run Stage 2 only (vLLM + Fusion)"
        echo "  verify   - Verify setup and dependencies"
        echo "  bash     - Start interactive shell"
        echo ""
        echo "Examples:"
        echo "  docker-compose run experiment run"
        echo "  docker-compose run experiment test"
        echo "  docker-compose run experiment bash"
        echo ""
        
        if [ -n "$1" ]; then
            echo "Unknown command: $1"
            exit 1
        fi
        ;;
esac
