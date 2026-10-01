# DynLP+ method comparison on one GPU (H100). `just` lists recipes; each run ends in results/<dir>/matrix.md
set shell := ["bash", "-euo", "pipefail", "-c"]

export PATH := env("HOME", "~") + "/.local/bin:" + env("PATH")
# CuPy extra matching the driver: nvidia-smi prints "CUDA Version: 12.8" or "CUDA UMD Version: 13.4"
_cu := `v=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: [0-9]+' | grep -oE '[0-9]+$' | head -1 || true); [ "${v:-12}" -ge 13 ] && echo cu13 || echo cu12`
CU := env("CU", _cu)
export DATA_DIR := env("DATA_DIR", "data")
imdb := "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text"
py := "uv run --no-sync python"
b := py + " -m dynlp.bench"
kb := py + " -m dynlp.kernelbench"
methods := "itlp,dynlp,dynlp-fh,dynlp+pcg,dynlp+amg,dynlp+auto"
resident := "dynlp,dynlp-fh,dynlp+auto,dynlp+inc"
# run NAME CMD...: log to $o/NAME.log, time it, show the summary
_run := 'run() { local n=$1 s=$(date +%s); shift; "$@" > $o/$n.log 2>&1; local rc=$?; echo "$n $(( $(date +%s) - s ))s exit=$rc" | tee -a $o/stages.log; tail -n 10 $o/$n.log; }'

default:
    @just --list

# uv sync with the matching CuPy build, then tests
setup:
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync --extra {{CU}} --reinstall-package cupy-cuda{{trim_start_match(CU, "cu")}}x
    {{py}} -c "import cupy as c; p=c.cuda.runtime.getDeviceProperties(0); print(p['name'].decode(), 'cc', p['major'], p['minor'])"
    uv run --no-sync pytest -q

[private]
ready:
    [ -f .venv/.ready ] || { just setup && touch .venv/.ready; }

test *args:
    uv run --no-sync pytest -q {{args}}

# the paper's IMDB reviews (Maas et al. 2011; Hugging Face copy, ~41 MB) -> $DATA_DIR/imdb-{train,test}.parquet
download:
    mkdir -p $DATA_DIR
    for s in train test; do f=$DATA_DIR/imdb-$s.parquet; [ -f $f ] || { echo "downloading $f"; curl -fsSL --retry 3 -o $f.part {{imdb}}/$s-00000-of-00001.parquet && mv $f.part $f; }; done

# graphs: IMDB (TF-IDF, cosine 5-NN) and the generated synth2 / synth10 -> $DATA_DIR/*.npz (default ./data)
data: ready download
    {{py}} -c "from dynlp import graphs; [print(graphs.build_cached(d)) for d in ('imdb', 'synth2', 'synth10')]"

# one command on a fresh box: setup (once), download, graphs, then ~10 min of runs -> results/sanity/matrix.md
sanity: data
    #!/usr/bin/env bash
    set -uo pipefail
    o=results/sanity && rm -rf $o && mkdir -p $o && T0=$(date +%s) && {{_run}}
    for d in imdb synth2 synth10; do run $d {{b}} --dataset $d --batches 5 --solvers {{methods}} --out $o/$d.csv; done
    run imdb-resident {{b}} --incremental --dataset imdb --init-frac 0.9 --batches 20 --solvers {{resident}} --out $o/imdb_resident.csv --tag imdb-resident
    run sbm {{b}} --dataset sbm --n 1000000 --batches 5 --solvers {{methods}} --out $o/sbm1m.csv
    run kernels {{kb}} --n 500000 --cols 2,32 --reps 5 --out $o/kernels.csv
    just matrix $o
    echo "total $(( $(date +%s) - T0 ))s -> $o/matrix.md"

# the full method matrix (about 1-2 h on an H100) -> results/compare/matrix.md
compare: data
    #!/usr/bin/env bash
    set -uo pipefail
    o=results/compare && mkdir -p $o && rm -f $o/*.csv $o/stages.log && {{_run}}
    run imdb-single {{b}} --dataset imdb --init-frac 1.0 --batches 0 --solvers {{methods}} --out $o/imdb_single.csv --tag imdb-single
    for d in imdb synth2 synth10; do run $d {{b}} --dataset $d --batches 10 --solvers {{methods}} --out $o/$d.csv; done
    run imdb-resident {{b}} --incremental --dataset imdb --init-frac 0.9 --batches 100 --solvers {{resident}} --out $o/imdb_resident.csv --tag imdb-resident
    run er50m {{b}} --dataset er --n 50000000 --deg 5 --init-frac 1.0 --batches 0 --solvers itlp,dynlp,dynlp+pcg,dynlp+amg --out $o/er50m.csv --tag er50m-single
    run sbm10m {{b}} --dataset sbm --n 10000000 --batches 10 --solvers {{methods}} --out $o/sbm10m.csv --tag sbm10m
    run sbm-k16 {{b}} --dataset sbm --n 2000000 --classes 16 --batches 10 --solvers {{methods}} --out $o/sbm2m_k16.csv --tag sbm2m-K16
    run sbm-resident {{b}} --incremental --dataset sbm --n 10000000 --init-frac 0.95 --batches 100 --solvers {{resident}} --no-reference --out $o/sbm10m_resident.csv --tag sbm10m-resident
    just matrix $o

# kernel mappings vs cuSPARSE and our row-major SpMM, all rows, 2..64 columns
kernels: ready
    mkdir -p results/kernels
    {{kb}} --dataset sbm --n 20000000 --cols 2,16,32,64 --out results/kernels/sbm20m.csv
    {{kb}} --dataset er --n 50000000 --deg 5 --cols 2,16 --out results/kernels/er50m.csv

# nsys timeline and ncu report of the frontier kernel
profile: ready
    mkdir -p results/profile
    nsys profile -o results/profile/nsys --force-overwrite true --trace=cuda,nvtx {{b}} --dataset sbm --n 5000000 --batches 3 --solvers dynlp,dynlp+auto --no-reference --no-warmup
    ncu --set full -k regex:frontier_jacobi --launch-count 5 -o results/profile/ncu -f {{b}} --dataset sbm --n 5000000 --batches 1 --solvers dynlp --no-reference --no-warmup

# matrix.md / matrix.csv and figures for a results directory
matrix dir="results/sanity":
    {{py}} scripts/matrix.py {{dir}}
    {{py}} scripts/plot_results.py {{dir}} > /dev/null

# results/<name>.tar.gz of a results directory (no profiles)
pack dir="results/compare":
    tar czf {{dir}}.tar.gz --exclude='*.nsys-rep' --exclude='*.ncu-rep' {{dir}} && echo {{dir}}.tar.gz
