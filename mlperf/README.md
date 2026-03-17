# MLPerf Inference — NVIDIA B300 SXM6

MLPerf Inference offline scenario benchmarks for **ResNet-50** and **BERT-Large** on
the NVIDIA B300 SXM6 (Blackwell, SM 10.3), using synthetic data for throughput-only
measurement. No ImageNet or SQuAD download required.

---

## Files

| File | Description |
|---|---|
| `mlperf_b300_resnet50.py` | ResNet-50 FP16 offline SUT — DataParallel, synthetic ImageNet |
| `mlperf_b300_bert.py` | BERT-Large FP16 offline SUT — DataParallel, synthetic SQuAD |
| `run_mlperf_b300.sh` | Runner script — conda `s2s` env, 4-GPU (CUDA_VISIBLE_DEVICES) |
| `run_mlperf_b300_docker.sh` | Runner script — NGC Docker container (`mlperf-b300-ngc:latest`) |
| `Dockerfile.mlperf` | NGC 25.03 base + mlperf_loadgen built from source |

---

## Quick Start

### Option A — Conda (s2s env)

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=4 bash run_mlperf_b300.sh

# 4 GPUs (DataParallel)
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_mlperf_b300.sh
```

### Option B — NGC Docker container

```bash
# Build image once (NGC 25.03 + mlperf_loadgen from source)
docker build -f Dockerfile.mlperf -t mlperf-b300-ngc:latest .

# Run on GPU 4 (default)
bash run_mlperf_b300_docker.sh

# Run on a specific GPU
CUDA_GPU=0 bash run_mlperf_b300_docker.sh
```

---

## Prerequisites

### mlperf_loadgen

The `mlperf_loadgen` Python package must be installed. It is **not available on PyPI**
for sm_103 — build from source:

```bash
git clone --depth=1 https://github.com/mlcommons/inference.git /tmp/mlperf-inference
cd /tmp/mlperf-inference/loadgen
CFLAGS="-std=c++14" pip install --no-cache-dir .
```

Or use the provided `Dockerfile.mlperf` which does this automatically inside the container.

### Python dependencies

```bash
pip install torch torchvision transformers numpy
```

---

## Design

Both benchmarks use the **MLPerf Offline scenario** with `PerformanceOnly` mode:

- Loadgen issues one large query with all samples upfront
- SUT processes them in batches using `torch.nn.DataParallel` across all visible GPUs
- `QuerySamplesComplete` is called once all batches finish
- Loadgen computes QPS from wall-clock time

**Key implementation details:**

| Detail | ResNet-50 | BERT-Large |
|---|---|---|
| Model | ResNet-50 V2 (IMAGENET1K_V2) | BertModel (hidden=1024, layers=24) |
| Precision | FP16 | FP16 |
| Batch size (4 GPU) | 256 (64 per GPU) | 128 (32 per GPU) |
| Sequence length | N/A | 384 (MLPerf standard) |
| Dataset | 5,000 synthetic images (CPU) | 2,000 synthetic token sequences (CPU) |
| Response pointer | `buf.ctypes.data` | `buf.ctypes.data` |

> **Note on `buf.ctypes.data`:** mlperf_loadgen's `QuerySampleResponse` requires a raw
> C pointer. `memoryview.__array_interface__` is broken in Python 3.9+; `buf.ctypes.data`
> is the correct portable replacement.

---

## Multi-GPU Scaling

Both scripts auto-detect `CUDA_VISIBLE_DEVICES` and scale batch size accordingly:

```python
GPUS       = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
N_GPUS     = len(GPUS)
BATCH_SIZE = 64 * N_GPUS   # ResNet-50: 64 per GPU
```

`torch.nn.DataParallel` is used for inference — the dataset is kept on CPU and scattered
to each GPU per batch, so no GPU memory is pre-allocated for the full dataset.

---

## Expected Results (4× B300 SXM6, synthetic data)

> Results will be filled in once the benchmark run completes.

| Model | Precision | GPUs | Batch | QPS (measured) | Result |
|---|---|---|---|---|---|
| ResNet-50 | FP16 | 4× B300 | 256 | **405.6** | VALID |
| BERT-Large | FP16 | 4× B300 | 128 | **1524.8** | VALID |

See [`../report/b300_benchmark_report.md`](../report/b300_benchmark_report.md) for
full results once the run completes.

---

## Logs

Benchmark logs are written to `logs/` (not tracked in git — too large):

```
logs/
├── resnet50/
│   ├── mlperf_log_summary.txt    ← QPS result here
│   ├── mlperf_log_detail.txt
│   └── mlperf_log_accuracy.json
└── bert/
    ├── mlperf_log_summary.txt
    ├── mlperf_log_detail.txt
    └── mlperf_log_accuracy.json
```

To run and capture logs:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_mlperf_b300.sh \
    2>&1 | tee logs/mlperf_4gpu_run.log
```
