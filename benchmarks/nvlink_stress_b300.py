#!/usr/bin/env python3
"""
NVLink 5 Stress Benchmark — NVIDIA B300 SXM6 (4 GPU)
=====================================================
Pushes NVLink 5 as hard as possible across all 4 GPUs using every
collective operation and simultaneous bidirectional P2P transfers.

NVLink 5 theoretical (B300 SXM6, 18 lanes × 100 GB/s):
  Unidirectional per GPU : 900 GB/s
  Bidirectional per GPU  : 1800 GB/s

Bus bandwidth formulas (n = world_size):
  All-Reduce   : bytes * 2*(n-1)/n / time
  All-to-All   : bytes * (n-1)/n   / time
  Reduce-Scatter: bytes * (n-1)/n  / time
  All-Gather   : bytes * (n-1)/n   / time
  Broadcast    : bytes              / time  (root only)
  P2P bidir    : bytes * 2          / time  (simultaneous send+recv)

Run:
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \\
      benchmarks/nvlink_stress_b300.py

  Or via the runner script:
  bash benchmarks/run_nvlink_stress.sh
"""

import os, sys, time, json, socket
import torch
import torch.distributed as dist

# ── NCCL env (NCCL tuning for NVLink 5 / Blackwell) ──────────────────────────
CONDA_ENV = os.environ.get("CONDA_ENV", "/scratch/rdesouz4/envs/s2s")
_nccl_lib = f"{CONDA_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib"
if os.path.isdir(_nccl_lib):
    os.environ["LD_LIBRARY_PATH"] = f"{_nccl_lib}:{os.environ.get('LD_LIBRARY_PATH','')}"

os.environ.setdefault("NCCL_BUFFSIZE",      "16777216")   # 16 MB ring buffer
os.environ.setdefault("NCCL_MAX_NCHANNELS", "32")         # saturate NVLink lanes
os.environ.setdefault("NCCL_ALGO",          "Ring")       # ring optimal for 4-GPU NVLink
os.environ.setdefault("NCCL_DEBUG",         "WARN")

# ── Sizes to sweep (bytes) ─────────────────────────────────────────────────────
SIZES_MB = [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096]
SIZES    = [s * 1024 * 1024 for s in SIZES_MB]

WARMUP_ITERS = 5
BENCH_ITERS  = 20
DTYPE        = torch.bfloat16   # primary training dtype
DTYPE_BYTES  = 2

NVLINK5_UNI_BW = 900.0   # GB/s — NVLink 5 unidirectional per GPU (theoretical)


# ── Utility ───────────────────────────────────────────────────────────────────

def sep(char="=", width=70):
    if dist.get_rank() == 0:
        print(char * width, flush=True)

def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)

def sync_time(iters, fn, *args, **kwargs):
    """Run fn() iters times, measure wall time with barrier synchronisation."""
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / iters   # seconds per iteration

def busbw(nbytes, t_sec, factor):
    """Bus bandwidth in GB/s: nbytes * factor / t_sec / 1e9."""
    return nbytes * factor / t_sec / 1e9

def n_elems(nbytes):
    return nbytes // DTYPE_BYTES


# ── Collective benchmarks ─────────────────────────────────────────────────────

def bench_allreduce(rank, world, device, nbytes):
    t = torch.zeros(n_elems(nbytes), dtype=DTYPE, device=device)
    # warmup
    for _ in range(WARMUP_ITERS):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    # bench
    t_sec = sync_time(BENCH_ITERS, dist.all_reduce, t, op=dist.ReduceOp.SUM)
    factor = 2.0 * (world - 1) / world
    return busbw(nbytes, t_sec, factor)

def bench_alltoall(rank, world, device, nbytes):
    per_rank = n_elems(nbytes) // world
    inp = torch.zeros(world * per_rank, dtype=DTYPE, device=device)
    out = torch.zeros(world * per_rank, dtype=DTYPE, device=device)
    inp_list = list(inp.chunk(world))
    out_list = list(out.chunk(world))
    for _ in range(WARMUP_ITERS):
        dist.all_to_all(out_list, inp_list)
    torch.cuda.synchronize()
    t_sec = sync_time(BENCH_ITERS,
                      lambda: dist.all_to_all(out_list, inp_list))
    factor = (world - 1) / world
    return busbw(nbytes, t_sec, factor)

def bench_reduce_scatter(rank, world, device, nbytes):
    inp = torch.zeros(n_elems(nbytes), dtype=DTYPE, device=device)
    out = torch.zeros(n_elems(nbytes) // world, dtype=DTYPE, device=device)
    out_list = [out]
    inp_list = list(inp.chunk(world))
    for _ in range(WARMUP_ITERS):
        dist.reduce_scatter(out, inp_list)
    torch.cuda.synchronize()
    t_sec = sync_time(BENCH_ITERS,
                      lambda: dist.reduce_scatter(out, inp_list))
    factor = (world - 1) / world
    return busbw(nbytes, t_sec, factor)

def bench_allgather(rank, world, device, nbytes):
    per_rank = n_elems(nbytes) // world
    inp = torch.zeros(per_rank, dtype=DTYPE, device=device)
    out = [torch.zeros(per_rank, dtype=DTYPE, device=device) for _ in range(world)]
    for _ in range(WARMUP_ITERS):
        dist.all_gather(out, inp)
    torch.cuda.synchronize()
    t_sec = sync_time(BENCH_ITERS,
                      lambda: dist.all_gather(out, inp))
    factor = (world - 1) / world
    return busbw(nbytes, t_sec, factor)

def bench_broadcast(rank, world, device, nbytes):
    t = torch.zeros(n_elems(nbytes), dtype=DTYPE, device=device)
    for _ in range(WARMUP_ITERS):
        dist.broadcast(t, src=0)
    torch.cuda.synchronize()
    t_sec = sync_time(BENCH_ITERS,
                      lambda: dist.broadcast(t, src=0))
    # broadcast: only root sends; factor = 1.0
    return busbw(nbytes, t_sec, 1.0)

def bench_p2p_bidir(rank, world, device, nbytes):
    """
    Simultaneous bidirectional P2P via batch_isend_irecv.
    Each rank sends to (rank+1)%world and recvs from (rank-1)%world.
    All 4 links active at the same time.
    """
    peer_send = (rank + 1) % world
    peer_recv = (rank - 1) % world
    send_t = torch.zeros(n_elems(nbytes), dtype=DTYPE, device=device)
    recv_t = torch.zeros(n_elems(nbytes), dtype=DTYPE, device=device)

    def _step():
        ops = [
            dist.P2POp(dist.isend, send_t, peer_send),
            dist.P2POp(dist.irecv, recv_t, peer_recv),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    for _ in range(WARMUP_ITERS):
        _step()
    torch.cuda.synchronize()
    t_sec = sync_time(BENCH_ITERS, _step)
    # both directions simultaneously: factor = 2.0 (send + recv per rank)
    return busbw(nbytes, t_sec, 2.0)

def bench_stress(rank, world, device, nbytes, duration_sec=10):
    """
    Sustained NVLink stress: alternate all-reduce and all-to-all for
    `duration_sec` seconds. Reports average bus bandwidth over the run.
    """
    t_ar  = torch.zeros(n_elems(nbytes), dtype=DTYPE, device=device)
    per_r = n_elems(nbytes) // world
    inp   = torch.zeros(world * per_r, dtype=DTYPE, device=device)
    out_chunks = list(torch.zeros(world * per_r, dtype=DTYPE, device=device).chunk(world))
    inp_chunks = list(inp.chunk(world))

    iters = 0
    dist.barrier()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    while (time.perf_counter() - t_start) < duration_sec:
        dist.all_reduce(t_ar, op=dist.ReduceOp.SUM)
        dist.all_to_all(out_chunks, inp_chunks)
        iters += 2

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t_start

    # average across all-reduce (factor 1.5) and all-to-all (factor 0.75) → ~1.125
    avg_factor = (2.0*(world-1)/world + (world-1)/world) / 2.0
    bw = nbytes * iters * avg_factor / elapsed / 1e9
    return bw, iters, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Init dist ─────────────────────────────────────────────────────────────
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    # Use existing MASTER_PORT if set (torchrun sets it)
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(port)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world      = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    gpu_name = torch.cuda.get_device_name(device)
    nccl_ver = torch.cuda.nccl.version()

    sep()
    log(f"  NVLink 5 Stress Benchmark — NVIDIA B300 SXM6")
    log(f"  GPUs     : {world}× {gpu_name}")
    log(f"  NCCL     : {'.'.join(str(v) for v in nccl_ver)}")
    log(f"  dtype    : {DTYPE} ({DTYPE_BYTES} bytes/elem)")
    log(f"  NVLink 5 theoretical (unidirectional/GPU): {NVLINK5_UNI_BW} GB/s")
    sep()

    results = {}

    # ── Per-size collective sweep ──────────────────────────────────────────────
    collectives = [
        ("All-Reduce",     bench_allreduce),
        ("All-to-All",     bench_alltoall),
        ("Reduce-Scatter", bench_reduce_scatter),
        ("All-Gather",     bench_allgather),
        ("Broadcast",      bench_broadcast),
        ("P2P-BiDir",      bench_p2p_bidir),
    ]

    for name, fn in collectives:
        sep("-")
        log(f"  {name}  ({BENCH_ITERS} iters/size, {WARMUP_ITERS} warmup)")
        log(f"  {'Size (MB)':>10}  {'Bus BW (GB/s)':>14}  {'% NVLink5':>10}")
        sep("-")
        rows = []
        for sz, sz_mb in zip(SIZES, SIZES_MB):
            try:
                bw = fn(rank, world, device, sz)
                pct = bw / NVLINK5_UNI_BW * 100.0
                log(f"  {sz_mb:>10}  {bw:>14.1f}  {pct:>9.1f}%")
                rows.append({"size_mb": sz_mb, "bw_gbs": round(bw,1), "pct_nvlink5": round(pct,1)})
            except Exception as e:
                log(f"  {sz_mb:>10}  ERROR: {e}")
                rows.append({"size_mb": sz_mb, "bw_gbs": None, "error": str(e)})
        results[name] = rows

    # ── Sustained stress test (1 GB, 10 s) ────────────────────────────────────
    sep()
    log("  Sustained Stress Test — 1 GB, 10 seconds (All-Reduce + All-to-All)")
    sep()
    bw, iters, elapsed = bench_stress(rank, world, device, SIZES[-2])  # 2 GB
    log(f"  Iterations : {iters}")
    log(f"  Elapsed    : {elapsed:.1f} s")
    log(f"  Avg bus BW : {bw:.1f} GB/s  ({bw/NVLINK5_UNI_BW*100:.1f}% NVLink 5)")
    results["Stress-Sustained-2GB"] = {"bw_gbs": round(bw,1),
                                        "pct_nvlink5": round(bw/NVLINK5_UNI_BW*100,1),
                                        "iters": iters, "elapsed_s": round(elapsed,1)}

    # ── Peak summary ──────────────────────────────────────────────────────────
    sep()
    log("  PEAK BUS BANDWIDTH SUMMARY (4× B300, NVLink 5)")
    sep()
    log(f"  {'Collective':<20}  {'Peak (GB/s)':>12}  {'% NVLink5 Uni':>14}")
    sep("-")
    for name, rows in results.items():
        if name == "Stress-Sustained-2GB":
            bw_peak = rows["bw_gbs"]
            pct     = rows["pct_nvlink5"]
        else:
            valid = [r["bw_gbs"] for r in rows if r.get("bw_gbs") is not None]
            if not valid:
                continue
            bw_peak = max(valid)
            pct     = bw_peak / NVLINK5_UNI_BW * 100.0
        log(f"  {name:<20}  {bw_peak:>12.1f}  {pct:>13.1f}%")
    sep()
    log(f"  Reference: nccl-tests all_reduce_perf (4 GPU, 1 GB) = 654 GB/s (72.7%)")
    log(f"  NVLink 5 theoretical unidirectional/GPU              = {NVLINK5_UNI_BW} GB/s")
    sep()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if rank == 0:
        os.makedirs("results", exist_ok=True)
        out_path = "results/nvlink_stress_b300.json"
        with open(out_path, "w") as f:
            json.dump({
                "gpu":        gpu_name,
                "world_size": world,
                "nccl":       '.'.join(str(v) for v in nccl_ver),
                "dtype":      str(DTYPE),
                "nvlink5_theoretical_gbs": NVLINK5_UNI_BW,
                "results":    results,
            }, f, indent=2)
        print(f"\nResults saved to {out_path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
