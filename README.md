# blackwell_gemm

Full write-up of the techniques and per-step speedups:
**https://www.paulwillchan.com/articles/outperforming-cublas-b200**

## Setup

The benchmarking and profiling is all done via Modal. 
Sign up at [modal.com](https://modal.com/), get credentials, and:

```bash
export MODAL_TOKEN_ID=ak-...
export MODAL_TOKEN_SECRET=as-...
```

## Running benchmarks

```bash
modal run main.py --action benchmark_all
```

Spawns one B200 container per (shape, version) pair in parallel each using cold-L2 timing (see
`bench_core.flush_time_fn`). Saves to `src/benchmark/perf_all.json` and
plots `perf_all.jpg`.

### Single-kernel profiling

```bash
# ncu --set full report 
modal run main.py --action ncu_full --shape 8192,8192,8192 --version v8

# Lightweight ncu profile
modal run main.py --action profile --shape 8192,8192,8192 --version v8

# Per-warp intra-kernel profiler (Chrome trace JSON)
modal run main.py --action intra_profile --shape 8192,8192,8192 --version v8
```

Outputs land in `src/benchmark/`. Open the intra-kernel `.json.gz` with
[ui.perfetto.dev](https://ui.perfetto.dev).
