"""Modal-free core benchmarking/profiling logic.

Both main.py (Modal) and main_local.py (local B200) call into this module.
Anything Modal-specific lives in those entrypoints, not here.
"""
from pathlib import Path
import gzip
import json
import shutil
import subprocess


def _find_tool(name: str) -> str:
    """Locate an Nsight tool, falling back to the CUDA install dir if not on PATH."""
    p = shutil.which(name)
    if p:
        return p
    fallback = f"/usr/local/cuda/bin/{name}"
    if Path(fallback).exists():
        return fallback
    raise FileNotFoundError(
        f"{name} not found on PATH or at {fallback}. Install Nsight or add it to PATH.")


def build_module(source_dir: Path):
    import torch
    import torch.utils.cpp_extension

    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")

    cu_sources = [p for p in source_dir.glob("kernels/*.cu")]
    return torch.utils.cpp_extension.load(
        "module",
        sources=list(source_dir.glob("*.cpp")) + cu_sources,
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )


def flush_time_fn(fn):
    """Like default_time_fn but evicts L2 (256 MB zero-fill) before each timed
    iteration. Matches Triton do_bench(fast_flush=True): cold-L2 single-call
    latency. Use with kernels where steady-state warm-cache throughput would
    overstate real-world perf (e.g., GEMMs in pipelines that don't reuse A/B).
    """
    import torch
    cache = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device="cuda")  # 256 MB
    stream = torch.cuda.current_stream()
    WARMUP, ITERS = 20, 100
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        cache.zero_()                  # queued on stream BEFORE start event
        starts[i].record(stream)
        fn()
        ends[i].record(stream)
    torch.cuda.synchronize()
    samples_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {"median_ms": samples_ms[len(samples_ms) // 2]}


def run_benchmark(module, shapes, versions, *, time_fn=None, sample_hook=None):
    """Run correctness + timing for each (shape, version).

    time_fn(callable) -> tflops_dict_for_one_kernel  — defaults to a CUDA-event
        loop with NO L2 flush (warm-cache throughput). Override from main_local.py
        to add stability sampling / pynvml checks.
    sample_hook(stage, shape, version) — optional callback for pre/post GPU state checks.
    """
    import torch

    def default_time_fn(fn):
        stream = torch.cuda.current_stream()
        # Matched roughly to do_bench(warmup=50ms, rep=500ms) at 8K scale —
        # short enough to stay under the B200 power cap's integration window
        # so measurements capture boost clocks instead of throttled steady-state.
        WARMUP, ITERS = 20, 100
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            starts[i].record(stream)
            fn()
            ends[i].record(stream)
        torch.cuda.synchronize()
        samples_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        return {"median_ms": samples_ms[len(samples_ms) // 2]}

    if time_fn is None:
        time_fn = default_time_fn

    results = {}

    for shape in shapes:
        M, N, K = map(int, shape.split(","))
        print(f"\n{M=}, {N=}, {K=}")

        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T

        output_ref = torch.matmul(A, B)
        for version in versions:
            f = getattr(module, f"matmul_{version}")
            out = f(A, B)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, output_ref, atol=1e-2, rtol=1e-2)

        all_kernels = [(v, getattr(module, f"matmul_{v}")) for v in reversed(versions)]
        all_kernels.append(("cuBLAS", lambda A, B: torch.matmul(A, B)))

        flops = 2 * M * N * K
        per_kernel_tflops = {}
        per_kernel_stats = {}
        NUM_ROUNDS = 1
        for r in range(NUM_ROUNDS):
            print(f"  Round {r+1}: {[name for name, _ in all_kernels]}")
            for name, f in all_kernels:
                if sample_hook:
                    sample_hook("pre", shape, name)
                stats = time_fn(lambda: f(A, B))
                if sample_hook:
                    sample_hook("post", shape, name)
                latency_ms = stats.get("median_ms") or stats["mean_ms"]
                tflops = flops / latency_ms / 1e9
                per_kernel_tflops.setdefault(name, []).append(tflops)
                per_kernel_stats.setdefault(name, []).append(stats)

        # Median across rounds (more representative than max)
        median_tflops = {
            name: sorted(vals)[len(vals) // 2]
            for name, vals in per_kernel_tflops.items()
        }

        results[shape] = {"best_tflops": median_tflops, "stats": per_kernel_stats}
        for name, tflops in median_tflops.items():
            latency_ms = flops / tflops / 1e9
            print(f"  {name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS")

    return results


def run_profile(source_dir: Path, shape: str, version: str) -> bytes:
    """ncu profile of a single kernel launch. Returns .ncu-rep bytes.

    Pass version='cuBLAS' to profile torch.matmul instead of a custom kernel.
    """
    import torch
    is_cublas = version.lower() == "cublas"

    M, N, K = map(int, shape.split(","))

    if is_cublas:
        # Run a quick warmup in this process so cuBLAS picks its preferred algo,
        # but the actual measured launch happens in the ncu driver child.
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
        for _ in range(5):
            torch.matmul(A, B)
        torch.cuda.synchronize()
    else:
        module = build_module(source_dir)
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
        f = getattr(module, f"matmul_{version}")
        for _ in range(5):
            f(A, B)
        torch.cuda.synchronize()

    driver = Path("/tmp/ncu_driver.py")
    if is_cublas:
        driver.write_text(f"""\
import torch
M, N, K = {M}, {N}, {K}
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
# warmup so cuBLAS algo selection is stable before the captured launch
for _ in range(5):
    torch.matmul(A, B)
torch.cuda.synchronize()
torch.matmul(A, B)
torch.cuda.synchronize()
""")
    else:
        driver.write_text(f"""\
import torch
import torch.utils.cpp_extension
from pathlib import Path

source_dir = Path("{source_dir}")
module = torch.utils.cpp_extension.load(
    "module",
    sources=list(source_dir.glob("*.cpp")) + list(source_dir.glob("kernels/*.cu")),
    extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_100a,code=sm_100a"],
    extra_ldflags=["-lcuda"],
    verbose=False,
)

M, N, K = {M}, {N}, {K}
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
module.matmul_{version}(A, B)
torch.cuda.synchronize()
""")

    out_file = f"/tmp/profile_{version}_{M}x{N}x{K}.ncu-rep"
    cmd = [_find_tool("ncu"), "-f", "--set", "full", "--target-processes", "all", "-o", out_file]
    if is_cublas:
        # cuBLAS bf16 gemm on B200 dispatches to nvjet_sm100_* kernels.
        # Filter on that name and take the last one (the timed launch after 5 warmups).
        cmd += ["--kernel-name", "regex:nvjet_sm100.*",
                "--launch-skip", "5", "--launch-count", "1"]
    cmd += ["python3", str(driver)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"ncu failed with code {result.returncode}")
    rep_path = Path(out_file)
    print(f"Profile size: {rep_path.stat().st_size / 1024:.0f} KB")
    return rep_path.read_bytes()


def run_ncu_full(source_dir: Path, shape: str = "8192,8192,8192",
                 version: str = "v8") -> bytes:
    """ncu --set full with source-counter attribution for the v8 kernel.

    Runs 15 launches and captures the 11th (--launch-skip 10 --launch-count 1).
    Returns .ncu-rep bytes.
    """
    import torch

    M, N, K = map(int, shape.split(","))

    # Prebuild the extension so the ncu child launches a warm-compiled module.
    module = build_module(source_dir)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
    f = getattr(module, f"matmul_{version}")
    for _ in range(5):
        f(A, B)
    torch.cuda.synchronize()
    del A, B, module
    torch.cuda.empty_cache()

    driver = Path("/tmp/ncu_full_driver.py")
    driver.write_text(f"""\
import torch
import torch.utils.cpp_extension
from pathlib import Path

source_dir = Path("{source_dir}")
module = torch.utils.cpp_extension.load(
    "module",
    sources=list(source_dir.glob("*.cpp")) + list(source_dir.glob("kernels/*.cu")),
    extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_100a,code=sm_100a"],
    extra_ldflags=["-lcuda"],
    verbose=False,
)

M, N, K = {M}, {N}, {K}
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
for _ in range(15):
    module.matmul_{version}(A, B)
torch.cuda.synchronize()
""")

    out_stem = f"/tmp/{version}_ncu_full"
    cmd = [
        _find_tool("ncu"), "-f",
        "--set", "full",
        "--import-source", "on",
        "--source-folders", str(source_dir),
        "--kernel-name", "regex:clc_hilbert_gemm",
        "--launch-skip", "10", "--launch-count", "1",
        "-o", out_stem,
        "python3", str(driver),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"ncu failed with code {result.returncode}")
    rep_path = Path(out_stem + ".ncu-rep")
    print(f"Report size: {rep_path.stat().st_size / 1024:.0f} KB")
    return rep_path.read_bytes()


def run_profile_variants(source_dir: Path, shape: str, version: str,
                         variants: list) -> dict:
    """Run ncu for each variant (env-var configuration), export CSV metrics.

    variants: list of {"name": str, "env": {"VAR": "value", ...}}
    Returns: {variant_name: csv_bytes}
    """
    import os
    import torch

    M, N, K = map(int, shape.split(","))

    # Prebuild the module once so the ncu driver doesn't have to compile 4x.
    module = build_module(source_dir)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
    f = getattr(module, f"matmul_{version}")
    for _ in range(5):
        f(A, B)
    torch.cuda.synchronize()
    del A, B, module
    torch.cuda.empty_cache()

    driver = Path("/tmp/ncu_driver_variants.py")
    driver.write_text(f"""\
import torch
import torch.utils.cpp_extension
from pathlib import Path

source_dir = Path("{source_dir}")
module = torch.utils.cpp_extension.load(
    "module",
    sources=list(source_dir.glob("*.cpp")) + list(source_dir.glob("kernels/*.cu")),
    extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_100a,code=sm_100a"],
    extra_ldflags=["-lcuda"],
    verbose=False,
)

M, N, K = {M}, {N}, {K}
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
# warmup (not captured): --launch-skip=5 --launch-count=1 picks the 6th launch
for _ in range(5):
    module.matmul_{version}(A, B)
torch.cuda.synchronize()
module.matmul_{version}(A, B)
torch.cuda.synchronize()
""")

    ncu = _find_tool("ncu")
    results = {}
    for v in variants:
        name = v["name"]
        env_overrides = v.get("env", {})
        rep = f"/tmp/profile_{version}_{name}_{M}x{N}x{K}.ncu-rep"

        capture_env = os.environ.copy()
        capture_env.update({k: str(val) for k, val in env_overrides.items()})

        cap_cmd = [ncu, "-f", "--set", "full", "--target-processes", "all",
                   "--launch-skip", "5", "--launch-count", "1",
                   "-o", rep, "python3", str(driver)]
        print(f"[{name}] env={env_overrides}")
        print(f"[{name}] {' '.join(cap_cmd)}")
        cap = subprocess.run(cap_cmd, capture_output=True, text=True,
                             env=capture_env, timeout=600)
        if cap.returncode != 0:
            print(f"STDOUT:\n{cap.stdout[-2000:]}")
            print(f"STDERR:\n{cap.stderr[-2000:]}")
            raise RuntimeError(f"ncu capture failed for variant '{name}' (code {cap.returncode})")

        # Export CSV (raw metric values, one row per kernel launch)
        exp_cmd = [ncu, "--import", rep, "--csv", "--page", "raw"]
        exp = subprocess.run(exp_cmd, capture_output=True, text=True, timeout=120)
        if exp.returncode != 0:
            print(f"STDERR:\n{exp.stderr[-2000:]}")
            raise RuntimeError(f"ncu CSV export failed for variant '{name}'")

        results[name] = exp.stdout.encode("utf-8")
        print(f"[{name}] csv size: {len(results[name]) / 1024:.0f} KB")

    return results


def run_trace(source_dir: Path, shape: str, version: str) -> bytes:
    """nsys trace of 10 launches. Returns .nsys-rep bytes."""
    import torch
    module = build_module(source_dir)

    M, N, K = map(int, shape.split(","))
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
    f = getattr(module, f"matmul_{version}")

    for _ in range(5):
        f(A, B)
    torch.cuda.synchronize()

    driver = Path("/tmp/nsys_driver.py")
    driver.write_text(f"""\
import torch
import torch.utils.cpp_extension
from pathlib import Path

source_dir = Path("{source_dir}")
module = torch.utils.cpp_extension.load(
    "module",
    sources=list(source_dir.glob("*.cpp")) + list(source_dir.glob("kernels/*.cu")),
    extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_100a,code=sm_100a"],
    extra_ldflags=["-lcuda"],
    verbose=False,
)

M, N, K = {M}, {N}, {K}
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T

for _ in range(3):
    module.matmul_{version}(A, B)
torch.cuda.synchronize()

torch.cuda.nvtx.range_push("benchmark")
for _ in range(10):
    module.matmul_{version}(A, B)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
""")

    out_file = f"/tmp/trace_{version}_{M}x{N}x{K}"
    cmd = [_find_tool("nsys"), "profile", "--trace=cuda,nvtx", "--cuda-memory-usage=true",
           "--force-overwrite=true", "-o", out_file, "python3", str(driver)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"nsys failed with code {result.returncode}")
    rep_path = Path(out_file + ".nsys-rep")
    print(f"Trace size: {rep_path.stat().st_size / 1024:.0f} KB")
    return rep_path.read_bytes()


def run_intra_profile(module, shape: str, version: str) -> bytes:
    """Per-warp intra-kernel profiler. Returns gzipped Chrome trace JSON."""
    import torch

    TAGS = ["SETUP", "ISSUE_TMA", "ISSUE_MMA", "WAIT_TMA", "WAIT_MMA",
            "WAIT_MAINLOOP", "WAIT_EPILOGUE", "EPILOGUE"]

    M, N, K = map(int, shape.split(","))
    print(f"{M=}, {N=}, {K=}")

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
    f = getattr(module, f"profile_matmul_{version}")

    SM_COUNT = 148
    WARPS_PER_TB = 6
    NUM_BLOCKS = SM_COUNT * WARPS_PER_TB
    NUM_ENTRIES = 10_000

    profiler_buf = torch.zeros(NUM_BLOCKS, 1 + NUM_ENTRIES * 4, dtype=torch.int64, device="cuda")

    for _ in range(5):
        f(A, B, profiler_buf, NUM_ENTRIES)

    torch.cuda.synchronize()
    profiler_buf.zero_()
    f(A, B, profiler_buf, NUM_ENTRIES)
    torch.cuda.synchronize()

    profiler_cpu = profiler_buf.cpu()
    events = []
    CTA_GROUP_SIZE = 2
    WARP_NAMES = {4: "TMA", 5: "MMA"}

    seen_pids = {}
    seen_tids = set()
    tid_meta = {}

    for bid in range(NUM_BLOCKS):
        count = profiler_cpu[bid, 0].item()
        if count == 0:
            continue

        block_idx = bid // WARPS_PER_TB
        warp_id = bid % WARPS_PER_TB
        cta_rank = block_idx % CTA_GROUP_SIZE

        data = profiler_cpu[bid, 1:1 + count * 4].tolist()
        sm_id = int(data[0])

        for i in range(count):
            tag, start, duration = int(data[i * 4 + 1]), data[i * 4 + 2], data[i * 4 + 3]
            events.append(dict(name=TAGS[tag], ph="X", ts=start, dur=duration,
                               pid=sm_id, tid=bid))

        if sm_id not in seen_pids:
            seen_pids[sm_id] = cta_rank
        if (sm_id, bid) not in seen_tids:
            seen_tids.add((sm_id, bid))
            role = WARP_NAMES.get(warp_id, "")
            name = f"warp{warp_id} - {role}" if role else f"warp{warp_id}"
            tid_meta[(sm_id, bid)] = (warp_id, name)

    if not events:
        raise RuntimeError("No profiler events captured")

    offset = min(evt["ts"] for evt in events)
    for evt in events:
        evt["ts"] -= offset

    for sm_id, cta_rank in seen_pids.items():
        events.append(dict(name="process_name", ph="M", pid=sm_id,
                           args=dict(name=f"SM{sm_id} / CTA{cta_rank}")))
        events.append(dict(name="process_sort_index", ph="M", pid=sm_id,
                           args=dict(sort_index=sm_id)))
    for (pid, tid), (warp_id, warp_name) in tid_meta.items():
        events.append(dict(name="thread_name", ph="M", pid=pid, tid=tid,
                           args=dict(name=warp_name)))
        events.append(dict(name="thread_sort_index", ph="M", pid=pid, tid=tid,
                           args=dict(sort_index=warp_id)))

    trace = dict(traceEvents=events)
    return gzip.compress(json.dumps(trace).encode("utf-8"))


def plot_results(all_results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = list(all_results.keys())
    sizes = [s.split(",")[0] for s in shapes]

    first = all_results[shapes[0]]
    if isinstance(first, dict) and "best_tflops" in first:
        get = lambda s: all_results[s]["best_tflops"]
    else:
        get = lambda s: all_results[s]
    kernels = list(get(shapes[0]).keys())

    x = range(len(shapes))
    fig, ax = plt.subplots(figsize=(10, 6))

    for kernel in kernels:
        tflops = [get(s).get(kernel, 0) for s in shapes]
        ax.plot(x, tflops, marker="o", label=kernel)
        for xi, val in zip(x, tflops):
            ax.text(xi, val + 0.5, f"{val:.0f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Matrix Size (M=N=K)")
    ax.set_ylabel("TFLOPS")
    ax.set_title("GEMM Kernel Performance (B200)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
