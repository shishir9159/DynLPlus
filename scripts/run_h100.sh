#!/usr/bin/env bash
# DynLP+ experiments on one H100 (80 GB). Usage:
#   bash scripts/run_h100.sh setup      # uv sync (with the right CuPy extra) + tests
#   bash scripts/run_h100.sh smoke      # 2-minute end-to-end check
#   bash scripts/run_h100.sh e1 ... e6  # individual experiments (see RUN_H100.md)
#   bash scripts/run_h100.sh all        # e1..e5 in order (several hours)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
# CuPy extra matching the driver: cu12 for CUDA 12.x drivers, cu13 for CUDA 13.x.
# nvidia-smi prints "CUDA Version: 12.8" (most drivers) or "CUDA UMD Version: 13.4" (newer ones).
CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: [0-9]+' | grep -oE '[0-9]+$' | head -1 || true)
CU=${CU:-$([ "${CUDA_MAJOR:-12}" -ge 13 ] && echo cu13 || echo cu12)}
# Only `setup` changes the environment; every run uses it as-is (--no-sync), so the
# cu12/cu13 CuPy builds (which share the `cupy` package directory) are never mixed.
PY="uv run --no-sync python"
OUT=${OUT:-results/h100}
mkdir -p "$OUT" data
B="$PY -m dynlp.bench --backend cupy"
ALL="itlp,itlp-warm,dynlp,dynlp-knowninit,dynlp+push,dynlp+pcg,dynlp+amg,dynlp+auto"
# pure push is Jacobi-speed on large batches; it is measured where it matters (small batches)
BIG="itlp,itlp-warm,dynlp,dynlp-knowninit,dynlp+pcg,dynlp+amg,dynlp+auto"
log() { echo; echo "=== $* ($(date '+%F %T'))"; }

setup() {
  log "setup (uv, extra=$CU, driver CUDA $CUDA_MAJOR)"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv sync --extra "$CU" --reinstall-package "cupy-cuda${CU#cu}x"
  $PY -c "import cupy as c; p=c.cuda.runtime.getDeviceProperties(0); print(p['name'].decode(), 'cc', p['major'], p['minor'])"
  uv run --no-sync pytest -q
}

smoke() {
  log "smoke: 100K SBM, all solvers"
  $B --dataset sbm --n 100000 --deg 10 --batches 3 --solvers "$ALL" --out "$OUT/smoke.csv" --tag smoke
}

e1() {  # kernel mapping: sub-warp vs DynLP's block-per-row
  log "E1 kernels"
  $PY -m dynlp.kernelbench --dataset sbm --n 20000000 --deg 10 --cols 1,2,8,32 \
      --fracs 0.001,0.01,0.1,1.0 --groups 1,2,4,8,16,32,128,256 --out "$OUT/e1_kernels_sbm20m.csv"
  $PY -m dynlp.kernelbench --dataset er --n 50000000 --deg 5 --cols 1,2 \
      --fracs 0.001,0.01,0.1,1.0 --groups 1,2,4,8,16,32,128,256 --out "$OUT/e1_kernels_er50m.csv"
}

e2() {  # the paper's single-batch setting on its 50M random graph: iteration counts
  log "E2 ER 50M single batch"
  $B --dataset er --n 50000000 --deg 5 --init-frac 1.0 --batches 0 \
     --solvers itlp,dynlp,dynlp+pcg,dynlp+amg --out "$OUT/e2_er50m_single.csv" --tag e2
}

e3() {  # streaming, binary, all solvers
  log "E3 streaming SBM 10M"
  $B --dataset sbm --n 10000000 --deg 10 --batches 10 --solvers "$BIG" --out "$OUT/e3_sbm10m.csv" --tag e3
  log "E3 streaming SBM 30M (no ItLP: from-scratch is the slow baseline already shown at 10M)"
  $B --dataset sbm --n 30000000 --deg 10 --batches 10 \
     --solvers itlp-warm,dynlp,dynlp+pcg,dynlp+amg,dynlp+auto --out "$OUT/e3_sbm30m.csv" --tag e3
  log "E3 small local batches (the push path): 100 batches over the last 10%"
  $B --dataset sbm --n 10000000 --deg 10 --init-frac 0.9 --batches 100 --del-frac 0.1 \
     --solvers dynlp,dynlp+push,dynlp+amg,dynlp+auto --out "$OUT/e3_sbm10m_small.csv" --tag e3small
}

e4() {  # accuracy/time trade-off: sweep delta (DynLP/ItLP) and tol (DynLP+)
  log "E4 accuracy sweep, SBM 5M"
  for d in 1e-2 1e-3 1e-4 1e-5 1e-6; do
    $B --dataset sbm --n 5000000 --deg 10 --batches 10 --solvers itlp-warm,dynlp --delta $d \
       --out "$OUT/e4_sweep.csv" --tag "delta=$d"
  done
  for t in 1e-2 1e-3 1e-4; do
    $B --dataset sbm --n 5000000 --deg 10 --batches 10 --solvers dynlp+pcg,dynlp+amg,dynlp+auto --tol $t \
       --out "$OUT/e4_sweep.csv" --tag "tol=$t"
  done
  # float64 lets the certificate go below the float32 floor
  $B --dataset sbm --n 5000000 --deg 10 --batches 10 --solvers dynlp+amg,dynlp+auto --tol 1e-6 --dtype float64 \
     --out "$OUT/e4_sweep.csv" --tag "tol=1e-6,f64"
}

e5() {  # multi-class
  log "E5 multi-class"
  for K in 4 16 64; do
    $B --dataset sbm --n 2000000 --deg 10 --classes $K --batches 10 \
       --solvers itlp-warm,dynlp,dynlp+pcg,dynlp+amg,dynlp+auto --out "$OUT/e5_sbm_K$K.csv" --tag "K=$K"
  done
  $B --dataset ogbn-arxiv --data-dir data --batches 10 --init-frac 0.3 \
     --solvers itlp,itlp-warm,dynlp,dynlp+pcg,dynlp+amg,dynlp+auto --out "$OUT/e5_arxiv.csv" --tag arxiv
  $B --dataset ogbn-products --data-dir data --batches 10 \
     --solvers itlp-warm,dynlp,dynlp+pcg,dynlp+amg,dynlp+auto --out "$OUT/e5_products.csv" --tag products
}

e6() {  # profiles for the frontier kernels and the PCG/AMG loop
  log "E6 profiling"
  nsys profile -o "$OUT/e6_nsys" --force-overwrite true --trace=cuda,nvtx \
    $B --dataset sbm --n 5000000 --deg 10 --batches 3 --solvers dynlp,dynlp+auto --no-reference --no-warmup
  ncu --set full -k regex:frontier_jacobi --launch-count 5 -o "$OUT/e6_ncu" -f \
    $B --dataset sbm --n 5000000 --deg 10 --batches 1 --solvers dynlp --no-reference --no-warmup
}

e7() {  # resident state (IncrementalStream + dynlp+inc): per-batch cost against the size of the change
  for nb in 10 100 400; do  # 10M graph; the last 5% arrives in nb batches (50K, 5K, 1.25K vertices each)
    log "E7 rebuild, $nb batches"
    $B --dataset sbm --n 10000000 --deg 10 --init-frac 0.95 --batches $nb \
       --solvers dynlp,dynlp+auto --no-reference --out "$OUT/e7_rebuild.csv" --tag "batches=$nb"
    log "E7 resident, $nb batches"
    $B --incremental --dataset sbm --n 10000000 --deg 10 --init-frac 0.95 --batches $nb \
       --solvers dynlp,dynlp+auto,dynlp+inc --no-reference --out "$OUT/e7_resident.csv" --tag "batches=$nb"
  done
  log "E7 correctness: dynlp+inc against the exact solution"
  $B --incremental --dataset sbm --n 1000000 --deg 10 --init-frac 0.95 --batches 20 \
     --solvers dynlp+inc --out "$OUT/e7_check.csv" --tag check
}

figs() {
  log "figures"
  for f in "$OUT"/e3_*.csv "$OUT"/e5_*.csv "$OUT"/smoke.csv; do
    [ -f "$f" ] && $PY scripts/plot_results.py "$f" --outdir "$OUT/fig"
  done
  [ -f "$OUT/e4_sweep.csv" ] && $PY scripts/plot_results.py "$OUT/e4_sweep.csv" --pareto --outdir "$OUT/fig"
  for f in "$OUT"/e1_*.csv; do [ -f "$f" ] && $PY scripts/plot_results.py --kernels "$f" --outdir "$OUT/fig"; done
  true
}

case "${1:-}" in
  setup) setup ;; smoke) smoke ;; e1) e1 ;; e2) e2 ;; e3) e3 ;; e4) e4 ;; e5) e5 ;; e6) e6 ;; e7) e7 ;;
  figs) figs ;;
  all) smoke; e1; e2; e3; e4; e5; e7; figs ;;
  *) echo "usage: $0 {setup|smoke|e1|e2|e3|e4|e5|e6|e7|figs|all}"; exit 1 ;;
esac
