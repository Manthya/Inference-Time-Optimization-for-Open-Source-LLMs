# Inference-Time Optimization for Open-Source LLMs

A production-ready experiment framework to measure inference performance improvements through runtime and kernel-level optimizations, without retraining or changing model behavior.

## 🎯 Objective

Measure how much inference performance can be improved **purely via runtime optimizations**:
- Same model weights (Qwen2.5-7B-Instruct)
- Same prompts & decoding (greedy, temperature=0)
- Same hardware
- Only execution changes

## 📊 Experiment Stages

| Stage | Description | Expected Speedup |
|-------|-------------|------------------|
| **Stage 0** | Vanilla HuggingFace Transformers | 1× (baseline) |
| **Stage 1** | vLLM (FlashAttention, PagedAttention, CUDA Graphs) | **4-6×** |
| **Stage 2** | vLLM + Kernel Fusion (RMSNorm+RoPE, FFN) | **5-8×** |

## 🐳 Quick Start (Docker)

### Prerequisites
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU (A100 80GB recommended)
- ~50GB disk space for model + image

### Build & Run

```bash
# Build image (downloads model + datasets - ~30-60 min first time)
./scripts/build.sh

# Run full experiment
./scripts/run.sh run

# Run quick test (reduced samples)
./scripts/run.sh test

# Interactive shell
./scripts/shell.sh
```

### Using Docker Compose

```bash
# Build
docker-compose build

# Run experiment
docker-compose run experiment run

# Run test
docker-compose run experiment test

# Development mode (mounts code)
docker-compose run dev bash
```

## 📁 Project Structure

```
.
├── Dockerfile              # Multi-stage build with model pre-download
├── docker-compose.yml      # Service definitions
├── docker/
│   ├── download_model.py   # Model download script
│   ├── download_datasets.py # Dataset download script
│   └── entrypoint.sh       # Container entrypoint
├── scripts/
│   ├── build.sh            # Build Docker image
│   ├── run.sh              # Run experiment
│   └── shell.sh            # Interactive shell
├── configs/
│   ├── experiment_config.yaml  # Full experiment config
│   └── test_config.yaml        # Quick test config
├── src/
│   ├── stages/             # Inference implementations
│   │   ├── stage_0_baseline.py
│   │   ├── stage_1_vllm.py
│   │   └── stage_2_fused.py
│   ├── evaluation/         # Dataset loaders
│   └── metrics/            # Metrics & visualization
├── run_experiment.py       # Main runner
└── results/                # Output directory (mounted)
```

## 🔧 Configuration

Edit `configs/experiment_config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  precision: "bfloat16"

decoding:
  temperature: 0.0  # Deterministic (CRITICAL)

stages:
  stage_0:
    enabled: true   # Enable/disable stages
    batch_size: 1
  stage_1:
    enabled: true
    batch_size: 8
```

## 📈 Expected Results

### Throughput
```
Stage 0 (Baseline):  ~20 tokens/s   (1.0×)
Stage 1 (vLLM):     ~100 tokens/s   (5.0×)
Stage 2 (Fused):    ~130 tokens/s   (6.5×)
```

### Latency Reduction
```
Stage 0 → Stage 1:  -80% latency
Stage 1 → Stage 2:  -15% latency
```

## 📊 Output

Results saved to `./results/`:
- `*_metrics.json` - Detailed per-stage metrics
- `*_comparison.csv` - Stage comparison table
- `*_summary.txt` - Human-readable summary
- `plots/` - Throughput, latency, speedup visualizations

## 🧪 Evaluation Datasets

| Dataset | Purpose | Samples |
|---------|---------|---------|
| GSM8K | Math/numerical stability | 100 |
| HumanEval | Code generation | 50 |
| MMLU | General reasoning | 50 |
| Synthetic | Long-context stress | 20 |

## 🔬 What This Proves

1. **Runtime systems dominate inference speed** (not model architecture)
2. **vLLM captures 80% of gains** (production-ready OSS)
3. **Kernel fusion adds 10-30%** (diminishing returns)
4. **No retraining required** for these optimizations

## 🛠️ Development

### Run Locally (without Docker)

```bash
# Create venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python run_experiment.py --stages stage_1 stage_2
```

### Custom Model

```bash
# Build with different model
MODEL_NAME="meta-llama/Llama-2-7b-hf" ./scripts/build.sh
```

## ⚙️ Requirements

- **GPU**: NVIDIA A100 80GB (or equivalent)
- **CUDA**: 12.1+
- **Docker**: 20.10+ with nvidia-container-toolkit
- **Disk**: ~50GB for image

## 📝 License

MIT License

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - Production inference engine
- [Triton](https://github.com/openai/triton) - Kernel development
- [HuggingFace](https://huggingface.co) - Model hub & transformers
