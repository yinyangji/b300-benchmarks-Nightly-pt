#!/bin/bash
# ============================================================
# GROMACS 2024 Water-Box Benchmark — runs INSIDE the container
# Uses gmx benchmark (self-contained, no external input needed)
# Called by run_gromacs_b300.sh via docker run
# ============================================================
set -uo pipefail

GMX="${GMX:-/usr/local/gromacs/bin/gmx}"
OUT_DIR="${OUT_DIR:-/workspace/logs/gromacs}"
OMP_THREADS="${OMP_THREADS:-16}"
NSTEPS="${NSTEPS:-50000}"
RESETSTEP="${RESETSTEP:-10000}"

sep() { printf '=%.0s' {1..60}; echo; }

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

sep
echo "  GROMACS 2024 — Water-Box GPU Benchmark"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  GROMACS: $(${GMX} --version 2>&1 | grep 'GROMACS version' | head -1)"
echo "  System: ~96000 atoms (SPC water, PME electrostatics)"
sep
echo ""

# ── gmx benchmark — self-generating water box ─────────────────────────────────
# Creates a water-box TPR, runs mdrun internally, reports ns/day
echo ">>> gmx benchmark (${NSTEPS} steps, nb+pme+bonded on GPU)..."
${GMX} benchmark \
    -ntmpi 1 \
    -ntomp "${OMP_THREADS}" \
    -nb gpu \
    -pme gpu \
    -bonded gpu \
    -nsteps "${NSTEPS}" \
    -resetstep "${RESETSTEP}" \
    -noconfout \
    -g "${OUT_DIR}/gromacs_b300_run.log" \
    2>&1 | tee "${OUT_DIR}/gromacs_benchmark_stdout.log"

# ── Parse result ──────────────────────────────────────────────────────────────
echo ""
sep
echo "  RESULTS"
sep

PERF_LINE=$(grep "^Performance:" "${OUT_DIR}/gromacs_b300_run.log" 2>/dev/null | tail -1)
if [ -z "${PERF_LINE}" ]; then
    # Try stdout log as fallback
    PERF_LINE=$(grep "^Performance:" "${OUT_DIR}/gromacs_benchmark_stdout.log" 2>/dev/null | tail -1)
fi

NS_DAY=$(echo "${PERF_LINE}" | awk '{print $2}')
HR_NS=$(echo "${PERF_LINE}" | awk '{print $3}')

printf "  %-22s %s ns/day  (%s hours/ns)\n" "B300 SXM6 (measured):" "${NS_DAY}" "${HR_NS}"
echo ""
echo "  Reference (single GPU, GROMACS 2024, water box ~96k atoms):"
printf "  %-22s %s\n" "H100 SXM5:"  "~500 ns/day"
printf "  %-22s %s\n" "B200 SXM:"   "~650 ns/day (estimated)"
printf "  %-22s %s\n" "B300 SXM6:"  "${NS_DAY} ns/day  ← this run"
sep
echo ""

# Save JSON
python3 -c "
import json
data = {
    'ns_day': '${NS_DAY}',
    'hours_ns': '${HR_NS}',
    'nsteps': ${NSTEPS},
    'resetstep': ${RESETSTEP},
    'system': 'Water box ~96k atoms (gmx benchmark)',
    'gpu': '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)',
    'offload': 'nb+pme+bonded GPU'
}
with open('${OUT_DIR}/gromacs_b300_result.json', 'w') as f:
    json.dump(data, f, indent=2)
print('Result saved to ${OUT_DIR}/gromacs_b300_result.json')
"
