# DynLP+ experiments on one H100 (80 GB). `just` lists recipes; OUT=... just e3
set shell := ["bash", "-euo", "pipefail", "-c"]

export PATH := env("HOME", "~") + "/.local/bin:" + env("PATH")
OUT := env("OUT", "results/h100")
# CuPy extra matching the driver: nvidia-smi prints "CUDA Version: 12.8" or "CUDA UMD Version: 13.4"
_cu := `v=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: [0-9]+' | grep -oE '[0-9]+$' | head -1 || true); [ "${v:-12}" -ge 13 ] && echo cu13 || echo cu12`
CU := env("CU", _cu)
py := "uv run --no-sync python"
b := py + " -m dynlp.bench --backend cupy"
kb := py + " -m dynlp.kernelbench --fracs 0.001,0.01,0.1,1.0 --groups 1,2,4,8,16,32,128,256,cols"
sbm := "--dataset sbm --deg 10"
all_s := "itlp,itlp-warm,dynlp,dynlp-knowninit,dynlp+push,dynlp+pcg,dynlp+amg,dynlp+auto"
big_s := "itlp,itlp-warm,dynlp,dynlp-knowninit,dynlp+pcg,dynlp+amg,dynlp+auto"

default:
    @just --list

# uv sync with the matching CuPy build, then tests
setup:
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync --extra {{CU}} --reinstall-package cupy-cuda{{trim_start_match(CU, "cu")}}x
    {{py}} -c "import cupy as c; p=c.cuda.runtime.getDeviceProperties(0); print(p['name'].decode(), 'cc', p['major'], p['minor'])"
    uv run --no-sync pytest -q

test *args:
    uv run pytest -q {{args}}

[private]
dirs:
    mkdir -p {{OUT}} data

# 2-minute end-to-end check
smoke: dirs
    {{b}} {{sbm}} --n 100000 --batches 3 --solvers {{all_s}} --out {{OUT}}/smoke.csv --tag smoke

# kernel mapping: sub-warp vs block-per-row
e1: dirs
    {{kb}} {{sbm}} --n 20000000 --cols 1,2,8,32 --out {{OUT}}/e1_kernels_sbm20m.csv
    {{kb}} --dataset er --n 50000000 --deg 5 --cols 1,2 --out {{OUT}}/e1_kernels_er50m.csv
    {{kb}} {{sbm}} --n 5000000 --cols 16,32,64 --fracs 0.01,1.0 --groups 8,32,cols --out {{OUT}}/e1_kernels_classes.csv
    for r in none rcm; do {{kb}} --dataset ogbn-products --cols 1,47 --fracs 0.01,1.0 --groups 8,cols --reorder $r --out {{OUT}}/e1_kernels_products_$r.csv; done

# paper's single batch on the 50M random graph
e2: dirs
    {{b}} --dataset er --n 50000000 --deg 5 --init-frac 1.0 --batches 0 --solvers itlp,dynlp,dynlp+pcg,dynlp+amg --out {{OUT}}/e2_er50m_single.csv --tag e2

# streaming, binary; then small local batches (push path)
e3: dirs
    {{b}} {{sbm}} --n 10000000 --batches 10 --solvers {{big_s}} --out {{OUT}}/e3_sbm10m.csv --tag e3
    {{b}} {{sbm}} --n 30000000 --batches 10 --solvers itlp-warm,dynlp,dynlp+pcg,dynlp+amg,dynlp+auto --out {{OUT}}/e3_sbm30m.csv --tag e3
    {{b}} {{sbm}} --n 10000000 --init-frac 0.9 --batches 100 --del-frac 0.1 --solvers dynlp,dynlp+push,dynlp+amg,dynlp+auto --out {{OUT}}/e3_sbm10m_small.csv --tag e3small

# accuracy/time: sweep delta (DynLP/ItLP) and tol (DynLP+); float64 below the float32 floor
e4: dirs
    for d in 1e-2 1e-3 1e-4 1e-5 1e-6; do {{b}} {{sbm}} --n 5000000 --batches 10 --solvers itlp-warm,dynlp --delta $d --out {{OUT}}/e4_sweep.csv --tag delta=$d; done
    for t in 1e-2 1e-3 1e-4; do {{b}} {{sbm}} --n 5000000 --batches 10 --solvers dynlp+pcg,dynlp+amg,dynlp+auto --tol $t --out {{OUT}}/e4_sweep.csv --tag tol=$t; done
    {{b}} {{sbm}} --n 5000000 --batches 10 --solvers dynlp+amg,dynlp+auto --tol 1e-6 --dtype float64 --out {{OUT}}/e4_sweep.csv --tag tol=1e-6,f64

# multi-class: SBM K=4/16/64, ogbn-arxiv, ogbn-products
e5: dirs
    for K in 4 16 64; do {{b}} {{sbm}} --n 2000000 --classes $K --batches 10 --solvers itlp-warm,dynlp,dynlp+pcg,dynlp+amg,dynlp+auto --out {{OUT}}/e5_sbm_K$K.csv --tag K=$K; done
    {{b}} --dataset ogbn-arxiv --data-dir data --batches 10 --init-frac 0.3 --solvers itlp,itlp-warm,dynlp,dynlp+pcg,dynlp+amg,dynlp+auto --out {{OUT}}/e5_arxiv.csv --tag arxiv
    {{b}} --dataset ogbn-products --data-dir data --batches 10 --solvers itlp-warm,dynlp,dynlp+pcg,dynlp+amg,dynlp+auto --out {{OUT}}/e5_products.csv --tag products

# nsys/ncu profiles of the frontier kernels and the PCG/AMG loop
e6: dirs
    nsys profile -o {{OUT}}/e6_nsys --force-overwrite true --trace=cuda,nvtx {{b}} {{sbm}} --n 5000000 --batches 3 --solvers dynlp,dynlp+auto --no-reference --no-warmup
    ncu --set full -k regex:frontier_jacobi --launch-count 5 -o {{OUT}}/e6_ncu -f {{b}} {{sbm}} --n 5000000 --batches 1 --solvers dynlp --no-reference --no-warmup

# resident state: per-batch cost against the size of the change, then correctness
e7: dirs
    for nb in 10 100 400; do \
      {{b}} {{sbm}} --n 10000000 --init-frac 0.95 --batches $nb --solvers dynlp,dynlp+auto --no-reference --out {{OUT}}/e7_rebuild.csv --tag batches=$nb; \
      {{b}} --incremental {{sbm}} --n 10000000 --init-frac 0.95 --batches $nb --solvers dynlp,dynlp+auto,dynlp+inc --no-reference --out {{OUT}}/e7_resident.csv --tag batches=$nb; \
    done
    {{b}} --incremental {{sbm}} --n 1000000 --init-frac 0.95 --batches 20 --solvers dynlp+inc --out {{OUT}}/e7_check.csv --tag check

figs:
    for f in {{OUT}}/e3_*.csv {{OUT}}/e5_*.csv {{OUT}}/smoke.csv; do [ ! -f $f ] || {{py}} scripts/plot_results.py $f --outdir {{OUT}}/fig; done
    [ ! -f {{OUT}}/e4_sweep.csv ] || {{py}} scripts/plot_results.py {{OUT}}/e4_sweep.csv --pareto --outdir {{OUT}}/fig
    for f in {{OUT}}/e1_*.csv; do [ ! -f $f ] || {{py}} scripts/plot_results.py --kernels $f --outdir {{OUT}}/fig; done

# e1..e5, e7 and figures (several hours)
all: smoke e1 e2 e3 e4 e5 e7 figs

# download OGB data up front, so rented GPU time isn't spent on it
data: dirs
    {{py}} -c "from dynlp import backend, graphs; backend.set_backend('numpy'); [graphs.ogbn(d) for d in ('ogbn-arxiv', 'ogbn-products')]"

# rented GPU: setup, then stages by priority until the time box runs out; resumable; ends with a tarball
rent hours="6" stages="smoke e2 e4 e3 e7 e5 e1 e6":
    #!/usr/bin/env bash
    set -uo pipefail
    mkdir -p {{OUT}}/.done && log={{OUT}}/stages.log && end=$(( $(date +%s) + {{hours}} * 3600 ))
    [ -f {{OUT}}/.done/setup ] || { just setup && just data && touch {{OUT}}/.done/setup; } || exit 1
    for s in {{stages}}; do
      [ -f {{OUT}}/.done/$s ] && { echo "skip $s (done)"; continue; }
      [ $(date +%s) -lt $end ] || { echo "time box reached; left: $s ..."; break; }
      rm -f {{OUT}}/${s}_*.csv {{OUT}}/$s.csv; t0=$(date +%s)
      just $s; rc=$?
      echo "$s $(date -d @$t0 '+%F %T') $(( $(date +%s) - t0 ))s exit=$rc" | tee -a $log
      [ $rc -eq 0 ] && touch {{OUT}}/.done/$s
    done
    just figs; just pack

# same as rent, detached from the SSH session; follow with `just status`
rent-bg hours="6":
    mkdir -p {{OUT}} && nohup just rent {{hours}} > {{OUT}}/rent.log 2>&1 &
    @echo "running in the background: tail -f {{OUT}}/rent.log"

# what finished, how long it took, and the last log lines
status:
    @echo "done: $(ls {{OUT}}/.done 2>/dev/null | xargs)"; cat {{OUT}}/stages.log 2>/dev/null; tail -n 5 {{OUT}}/rent.log 2>/dev/null || true

# results/h100.tar.gz: CSVs, configs, figures and logs (no profiles); copy it back before the box goes away
pack:
    tar czf results/h100.tar.gz --exclude='*.nsys-rep' --exclude='*.ncu-rep' {{OUT}}
    @echo "scp $(whoami)@$(hostname -I 2>/dev/null | cut -d' ' -f1):$(pwd)/results/h100.tar.gz ."

# submit a stage to SLURM (add --partition/--account for your site)
slurm stage="all":
    mkdir -p results
    sbatch -J dynlp-plus --gres=gpu:h100:1 -c 16 --mem=128G -t 08:00:00 -o results/slurm-%j.out --wrap "nvidia-smi && just {{stage}}"
