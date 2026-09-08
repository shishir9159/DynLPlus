# intra-gpu-reduction

Two H100 experiments, one harness.

1. **MXFP4 emulated on FP8 tensor cores** — the path that works. Bit-exact FP4
   arithmetic at FP8 speed, because a 32-deep FP4 dot product needs 13 bits and
   the Hopper FP8 accumulator has ~14.
2. **The packed dual-batch GEMM** — the reduction hypothesis. Can one fp32
   accumulator carry two independent FP4 GEMMs, halving accumulator registers
   and split-K traffic? The harness measures where it holds and where it breaks.

```
fp4.py                    E2M1 grid, MXFP4 quantize/pack, exact references
mxfp4_gemm.py             Triton: block-32 scaled FP8 GEMM (+ packed 2-per-byte)
dual_gemm.py              Triton: packed vs separate accumulators, incl. split-K
train_step.py             one Linear layer, fwd + bwd, vanilla vs the technique
bench.py                  driver: correctness + throughput
check_isa.py              which tensor-core datapath the compiler actually chose
pyproject.toml            uv project; triton is gated to linux

cuda/fp4_unpack.cuh       FP4 -> FP8 via two PRMT (exhaustively verified on CPU)
cuda/mxfp4_mma_gemm.cu    CUDA reference for the block-scaled mainloop
cuda/test_unpack.c        host-only exhaustive test, no GPU needed
```

## Run

```bash
uv sync                                # one time: venv + torch + triton
uv run train_step.py                   # fwd + bwd, vanilla vs technique, per pass
uv run bench.py                        # both experiments, 4096^3
uv run bench.py --only dual            # just the packing hypothesis
uv run check_isa.py                    # verify wgmma vs mma.sync selection
```

`uv run` syncs the environment first, so `uv sync` is optional. To pin a specific
CUDA build of torch instead of the default Linux wheels:

```bash
uv sync --index https://download.pytorch.org/whl/cu128
```

Triton ships Linux wheels only, so it is marked `sys_platform == 'linux'` in
`pyproject.toml`. `uv sync` still works on a Windows or macOS dev box — you get
torch and can lint and read, but not run the kernels. Lint with
`uvx ruff check .`.

The CUDA side is a plain Makefile, no uv involved:

```bash
cd cuda && make && ./mxfp4 --check && ./mxfp4 4096 4096 4096
cd cuda && make sass                   # SASS opcode histogram
cd cuda && make test_unpack            # host-only, no GPU needed
```

Needs a Hopper GPU, PyTorch with `float8_e4m3fn`, and Triton 3.x (`tl.join`,
`tl.float8e4nv`). The CUDA build needs `-arch=sm_90a`; on Windows pass
`make CCBIN="<path to MSVC Hostx64/x64>"`.

## Experiment 1: why the emulation is free

Every E2M1 value `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` is exactly representable
in E4M3, so FP4 → FP8 is a 16-entry table lookup, not a rounding step:

```
code 0..7  ->  0x00 0x30 0x38 0x3C 0x40 0x44 0x48 0x4C     (bit 3 = sign)
```

FP4 products are multiples of ¼ bounded by 36, so a block of `n` of them is
bounded by `144n` quarter-units:

| block | max sum | bits | on the ~14-bit Hopper FP8 accumulator |
|------:|--------:|-----:|--------------------------------------|
| 16 (NVFP4) | 2304 | 12 | exact |
| 32 (MXFP4) | 4608 | 13 | exact |
| 64 | 9216 | 14 | exact |
| 128 | 18432 | 15 | lossy |

Worst case, no data assumptions. So the mainloop steps K in blocks of 32 and
promotes to an fp32 CUDA-core accumulator at each boundary — which is where the
block scales have to be applied anyway. The promotion that makes the format work
is the same promotion that makes the arithmetic exact. It costs ~3 FFMA per 32
MACs, about 3% of the tensor-core work.

`bench.py` asserts this: a 128-deep GEMM with unit scales must match the exact
integer reference **bit for bit**, not within a tolerance.

## Experiment 2: the packing hypothesis

`acc = Σ(A1·B1) + 2^s·Σ(A2·B2)`, one accumulator, split on the host. Values ride
the integer grid `q = 2·value` so both results are exact integers.

Two independent limits, both measured:

- **Encoding.** The `2^s` offset lives in the operand. bf16 has the exponent
  range for it; e4m3 saturates at 448, so at most `2^5` per side — `s ≤ 10`,
  already short of the `s = 14` that K=32 needs. `bench.py` reports this as
  `fp8 encodable: NO (>448)` rather than silently returning garbage.
- **Accumulator width.** The slot layout needs `2·ceil(log2(144K)) + 1` bits:
  27 at K=32, 31 at K=128. True fp32 gives 24, the Hopper FP8 datapath ~14.

Offline simulation (exact BigInt vs a W-bit accumulator, 5000 random trials)
says what to expect on-device:

```
K     s    need   W=14     W=24     W=31
8     12   23     2.3%     100.0%   100.0%
16    13   25     0.7%     100.0%   100.0%
32    14   27     0.2%     100.0%   100.0%
128   16   31     0.1%      17.1%   100.0%
```

So the scheme is exact only for small total K on a true fp32 accumulator, and
never on the FP8 path. The throughput half of the experiment is the part still
worth running: same MAC count in every row, the only difference is accumulator
registers and store/atomic traffic.

## Experiment 3: fwd + bwd, where the reduction actually is

`train_step.py` runs one Linear layer with two microbatches and times the three
directions separately, because they are three different problems:

| pass | relationship between the microbatches | what helps |
|---|---|---|
| fwd | two results, **shared W** | concat along M — one GEMM, taller |
| dgrad | two results, **shared W** | concat along M |
| wgrad | two results that are **summed** | concat along K — the sum lands in one accumulator, the add disappears |

Only wgrad is a reduction. fwd and dgrad were never two problems: sharing W means
they are one GEMM with a taller M, which `concat` gets for free and exactly.

The fwd/dgrad ladder is all Triton with an fp32 accumulator and fp32 output, so
the only variable is the technique — 2 launches, fused 2-accumulator, packed
1-accumulator, concat, and `preadd`. `preadd` is the interesting one: because W
is shared, `(X1 + 2^s·X2) @ W^T = Y1 + 2^s·Y2` in one GEMM of the **original**
size, so it is the only variant that removes MACs rather than moving them. It is
also the first to break — the packed operand needs `bits(12) + s + 1` significand
bits (19 at K=32, more at real widths) against 8 in bf16 and 11 in fp16. The
harness prints that budget and then measures it.

wgrad is timed on cuBLAS for both variants, since there the question is purely op
count — 2 GEMMs plus an explicit add versus 1 GEMM whose accumulator performs the
sum — and routing it through Triton would add a transposed-operand penalty that
has nothing to do with the technique.

## Verified here, not assumed

- `cuda/test_unpack.c` checks the PRMT conversion exhaustively over all 16⁴
  nibble quadruples and confirms each E4M3 bit pattern decodes to the right E2M1
  value. **0 mismatches.**
- `cuda/mxfp4_mma_gemm.cu` compiles clean for sm_90a: 172 registers, 0 spills,
  36864 B smem. Its SASS shows something worth knowing:

  ```
  32 x HMMA.16816.F32           <- FP16 tensor core, k=16
  48 x F2FP.F16.E4M3.UNPACK_B   <- FP8 -> FP16 conversion
   8 x LDGSTS.E.BYPASS.128      <- cp.async
  ```

  There are 16 `mma.sync` per k-step in that kernel, so ptxas turned each
  `m16n8k32.e4m3` into **two FP16 MMAs plus conversions**. On Hopper the
  warp-level FP8 `mma.sync` path is emulated on the FP16 datapath; only
  `wgmma.mma_async` reaches the native FP8 tensor core and the 1979 TFLOP/s
  rate. That is why CUTLASS, DeepGEMM and Triton all use wgmma on sm_90, and why
  the Triton kernel here is the fast path while the CUDA file is the readable
  one. `check_isa.py` confirms which you got.

## Ceiling

H100 SXM dense: bf16 989, fp8 1979 TFLOP/s (the datasheet's 3958 is with 2:4
sparsity). Emulated FP4 tops out at the fp8 number — about 22% of a B200's ~9
PFLOP/s dense NVFP4. On Hopper, FP4 buys memory, bandwidth and comms; it does
not buy FLOPs.

## Next steps

- Port the mainloop to `wgmma` (64-bit SMEM matrix descriptors + swizzled shared
  layout). The structure — one MXFP4 block per iteration, scale-multiply into a
  separate fp32 accumulator — does not change. DeepGEMM's Hopper FP8 kernel is
  ~90% of this already; swap its per-128 scale granularity for per-32.
- Replace `cp.async` with TMA.
- The `_unpack_to_e4m3` helper in `mxfp4_gemm.py` uses portable arithmetic; the
  `prmt` version in `cuda/fp4_unpack.cuh` is ~2x cheaper and can be dropped in
  via `tl.inline_asm_elementwise`.
