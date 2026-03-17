#!/usr/bin/env python3
"""
MLPerf Inference — ResNet-50 Offline Scenario on B300
Synthetic data throughput benchmark (no ImageNet download required).
Reports QPS matching MLPerf offline format.
"""
import os, sys, time, array, threading, queue
import ctypes
import numpy as np
import torch
import torchvision.models as models
import mlperf_loadgen as lg

# ── Config ──────────────────────────────────────────────────────────────────
GPUS        = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
N_GPUS      = len(GPUS)
DEVICE      = "cuda:0"                  # logical device 0 after CUDA_VISIBLE_DEVICES
BATCH_SIZE  = 64 * N_GPUS              # scale batch with GPU count
N_SAMPLES   = 5000
IMG_SHAPE   = (3, 224, 224)
SCENARIO    = lg.TestScenario.Offline
MODE        = lg.TestMode.PerformanceOnly
TARGET_QPS  = 3000 * N_GPUS   # conservative; actual QPS measured by loadgen from wall time

# ── Model (DataParallel across all visible GPUs) ──────────────────────────────
print(f"Loading ResNet-50 on {N_GPUS} GPU(s): {GPUS}", flush=True)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = model.half().eval()
if N_GPUS > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(N_GPUS)))
model = model.to(DEVICE)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Warm up
with torch.no_grad():
    dummy = torch.randn(BATCH_SIZE, *IMG_SHAPE, device=DEVICE, dtype=torch.float16)
    for _ in range(5):
        _ = model(dummy)
torch.cuda.synchronize()
print("Warmup done.", flush=True)

# ── Synthetic dataset (kept on CPU to allow DataParallel scatter) ─────────────
print(f"Generating {N_SAMPLES} synthetic samples...", flush=True)
dataset = torch.randn(N_SAMPLES, *IMG_SHAPE, dtype=torch.float16)

# ── Issue / Response queues ───────────────────────────────────────────────────
response_queue = queue.Queue()

def process_queries(query_samples):
    indices = [qs.index for qs in query_samples]
    responses = []
    output_buf = []
    with torch.no_grad():
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            batch = dataset[batch_idx].to(DEVICE)
            out = model(batch)
            pred = out.argmax(dim=1).cpu().numpy().astype(np.int32)
            for j, qs in enumerate(query_samples[i:i+len(batch_idx)]):
                buf = np.array([pred[j]], dtype=np.int32)
                output_buf.append(buf)
                responses.append(lg.QuerySampleResponse(
                    qs.id, buf.ctypes.data, buf.nbytes))
    torch.cuda.synchronize()
    lg.QuerySamplesComplete(responses)

def flush_queries():
    pass

def load_query_samples(sample_list):
    pass

def unload_query_samples(sample_list):
    pass

# ── LoadGen settings ──────────────────────────────────────────────────────────
settings = lg.TestSettings()
settings.scenario = SCENARIO
settings.mode = MODE
settings.offline_expected_qps = TARGET_QPS
settings.min_duration_ms = 60000   # 60s run
settings.min_query_count = 1024

sut = lg.ConstructSUT(process_queries, flush_queries)
qsl = lg.ConstructQSL(N_SAMPLES, min(N_SAMPLES, 1024),
                       load_query_samples, unload_query_samples)

log_settings = lg.LogSettings()
log_settings.enable_trace = False
log_output = lg.LogOutputSettings()
log_output.outdir = "/weka/scratch/rdesouz4/b300/mlperf/logs/resnet50"
log_output.copy_summary_to_stdout = True
os.makedirs(log_output.outdir, exist_ok=True)
log_settings.log_output = log_output

print(f"\n{'='*60}")
print("MLPerf Inference — ResNet-50 Offline — NVIDIA B300 SXM6")
print(f"{'='*60}")
print(f"  GPUs:       {N_GPUS}x {torch.cuda.get_device_name(0)}")
print(f"  Precision:  FP16")
print(f"  Batch size: {BATCH_SIZE} ({BATCH_SIZE//N_GPUS} per GPU)")
print(f"  Scenario:   Offline")
print(f"  Mode:       Performance Only (synthetic data)")
print(f"{'='*60}\n", flush=True)

t0 = time.time()
lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
elapsed = time.time() - t0

lg.DestroyQSL(qsl)
lg.DestroySUT(sut)

# Parse result
log_file = os.path.join(log_output.outdir, "mlperf_log_summary.txt")
if os.path.exists(log_file):
    with open(log_file) as f:
        for line in f:
            if "Samples per second" in line or "QPS" in line or "Result" in line:
                print(line.rstrip())
