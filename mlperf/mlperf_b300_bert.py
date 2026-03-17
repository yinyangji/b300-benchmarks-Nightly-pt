#!/usr/bin/env python3
"""
MLPerf Inference — BERT-Large Offline Scenario on B300
Synthetic data throughput benchmark (no SQuAD download required).
Reports QPS matching MLPerf offline format.
"""
import os, sys, time, ctypes
import numpy as np
import torch
from transformers import BertModel, BertConfig
import mlperf_loadgen as lg

# ── Config ───────────────────────────────────────────────────────────────────
GPUS        = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
N_GPUS      = len(GPUS)
DEVICE      = "cuda:0"
BATCH_SIZE  = 32 * N_GPUS
SEQ_LEN     = 384          # MLPerf standard sequence length for BERT
N_SAMPLES   = 2000
SCENARIO    = lg.TestScenario.Offline
MODE        = lg.TestMode.PerformanceOnly
TARGET_QPS  = 800 * N_GPUS   # conservative; actual QPS measured by loadgen from wall time

# ── Model: BERT-Large (DataParallel) ─────────────────────────────────────────
print(f"Loading BERT-Large on {N_GPUS} GPU(s): {GPUS}", flush=True)
config = BertConfig(
    hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
    intermediate_size=4096, max_position_embeddings=512,
)
model = BertModel(config).half().eval()
if N_GPUS > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(N_GPUS)))
model = model.to(DEVICE)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Warm up
with torch.no_grad():
    ids  = torch.randint(0, 30522, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, device=DEVICE, dtype=torch.long)
    for _ in range(3):
        _ = model(input_ids=ids, attention_mask=mask)
torch.cuda.synchronize()
print("Warmup done.", flush=True)

# ── Synthetic dataset (CPU, scattered to GPUs by DataParallel) ────────────────
print(f"Generating {N_SAMPLES} synthetic BERT samples...", flush=True)
input_ids      = torch.randint(0, 30522, (N_SAMPLES, SEQ_LEN))
attention_mask = torch.ones(N_SAMPLES, SEQ_LEN, dtype=torch.long)

# ── SUT callbacks ─────────────────────────────────────────────────────────────
def process_queries(query_samples):
    indices = [qs.index for qs in query_samples]
    responses = []
    output_buf = []  # keep references alive until QuerySamplesComplete
    with torch.no_grad():
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            ids  = input_ids[batch_idx]
            mask = attention_mask[batch_idx]
            out  = model(input_ids=ids, attention_mask=mask)
            pred = out.pooler_output[:, :1].cpu().numpy().astype(np.float16)
            for j, qs in enumerate(query_samples[i:i+len(batch_idx)]):
                buf = np.array(pred[j], dtype=np.float16)
                output_buf.append(buf)
                responses.append(lg.QuerySampleResponse(
                    qs.id, buf.ctypes.data, buf.nbytes))
    torch.cuda.synchronize()
    lg.QuerySamplesComplete(responses)

def flush_queries(): pass
def load_query_samples(sample_list): pass
def unload_query_samples(sample_list): pass

# ── LoadGen ───────────────────────────────────────────────────────────────────
settings = lg.TestSettings()
settings.scenario = SCENARIO
settings.mode = MODE
settings.offline_expected_qps = TARGET_QPS
settings.min_duration_ms = 60000
settings.min_query_count = 512

sut = lg.ConstructSUT(process_queries, flush_queries)
qsl = lg.ConstructQSL(N_SAMPLES, min(N_SAMPLES, 512),
                       load_query_samples, unload_query_samples)

log_settings = lg.LogSettings()
log_settings.enable_trace = False
log_output = lg.LogOutputSettings()
log_output.outdir = "/weka/scratch/rdesouz4/b300/mlperf/logs/bert"
log_output.copy_summary_to_stdout = True
os.makedirs(log_output.outdir, exist_ok=True)
log_settings.log_output = log_output

print(f"\n{'='*60}")
print("MLPerf Inference — BERT-Large Offline — NVIDIA B300 SXM6")
print(f"{'='*60}")
print(f"  Device:     {torch.cuda.get_device_name(0)}")
print(f"  Precision:  FP16")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Seq length: {SEQ_LEN}")
print(f"  Scenario:   Offline")
print(f"  Mode:       Performance Only (synthetic data)")
print(f"{'='*60}\n", flush=True)

lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

lg.DestroyQSL(qsl)
lg.DestroySUT(sut)

log_file = os.path.join(log_output.outdir, "mlperf_log_summary.txt")
if os.path.exists(log_file):
    with open(log_file) as f:
        for line in f:
            if "Samples per second" in line or "QPS" in line or "Result" in line:
                print(line.rstrip())
