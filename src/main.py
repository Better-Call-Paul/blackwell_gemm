# run this script with
#   modal run main.py --action benchmark

import modal

from pathlib import Path

current_dir = Path(__file__).parent
remote_dir = Path("/my_extension")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .run_commands("apt-get update && apt-get install -y nsight-systems-2025.3")
    .uv_pip_install("ninja")
    .uv_pip_install("numpy")
    .uv_pip_install("matplotlib")
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .add_local_dir(current_dir, remote_path=remote_dir)
)
app = modal.App("sm100-matmul", image=image)


@app.function(gpu="B200", timeout=600)
def benchmark(shapes: list[str], versions: list[str]):
    import sys; sys.path.insert(0, str(remote_dir))
    import bench_core
    module = bench_core.build_module(remote_dir)
    results = bench_core.run_benchmark(module, shapes, versions)
    return {s: r["best_tflops"] for s, r in results.items()}


@app.function(gpu="B200", timeout=600)
def profile(shape: str, version: str):
    import sys; sys.path.insert(0, str(remote_dir))
    import bench_core
    return bench_core.run_profile(remote_dir, shape, version)


@app.function(gpu="B200", timeout=1800)
def profile_variants(shape: str, version: str, variants: list):
    import sys; sys.path.insert(0, str(remote_dir))
    import bench_core
    return bench_core.run_profile_variants(remote_dir, shape, version, variants)


@app.function(gpu="B200", timeout=1200)
def ncu_full(shape: str, version: str):
    import sys; sys.path.insert(0, str(remote_dir))
    import bench_core
    return bench_core.run_ncu_full(remote_dir, shape, version)


@app.function(gpu="B200", timeout=600)
def trace(shape: str, version: str):
    import sys; sys.path.insert(0, str(remote_dir))
    import bench_core
    return bench_core.run_trace(remote_dir, shape, version)


@app.function(gpu="B200")
def intra_profile(shape: str, version: str):
    import sys; sys.path.insert(0, str(remote_dir))
    import bench_core
    module = bench_core.build_module(remote_dir)
    return bench_core.run_intra_profile(module, shape, version)


@app.function(gpu="B200", timeout=600)
def benchmark_one(shape: str, version: str):
    """Run a single (shape, version) benchmark in its own container.

    Returns just the best_tflops dict, e.g. {"v5": 123.4, "cuBLAS": 456.7}.
    Each call rebuilds the extension from scratch — compile cost is paid per
    job, but the isolation means a crash/OOM in one kernel doesn't take the
    rest down, and all 44 (4 shapes × 11 versions) runs go in parallel.
    """
    import sys; sys.path.insert(0, str(remote_dir))
    import bench_core
    module = bench_core.build_module(remote_dir)
    results = bench_core.run_benchmark(module, [shape], [version])
    return results[shape]["best_tflops"]


@app.function(gpu="B200", timeout=600)
def benchmark_one_flush(shape: str, version: str):
    """Same as benchmark_one but evicts L2 before each timed iteration.

    Uses bench_core.flush_time_fn — reports cold-L2 per-call TFLOPS instead of
    warm-cache steady-state. Useful when comparing kernels whose A/B reuse
    patterns differ or when matching Triton do_bench(fast_flush=True) numbers.
    """
    import sys; sys.path.insert(0, str(remote_dir))
    import bench_core
    module = bench_core.build_module(remote_dir)
    results = bench_core.run_benchmark(
        module, [shape], [version], time_fn=bench_core.flush_time_fn
    )
    return results[shape]["best_tflops"]


@app.local_entrypoint()
def main(
    action: str,
    shape: str = "8192,8192,8192",
    version: str = "v8",
    shapes: str = "2048,2048,2048;4096,4096,4096;8192,8192,8192;16384,16384,16384",
    versions: str = "v1_3d,v2,v3,v4,v5,v6,v7,v8",
    name: str = "",
):
    import sys; sys.path.insert(0, str(current_dir))
    import bench_core

    shapes_list = [s.strip() for s in shapes.split(";") if s.strip()]
    versions_list = [v.strip() for v in versions.split(",") if v.strip()]

    if action == "benchmark":
        all_results = benchmark.remote(shapes_list, versions_list)

        import json, datetime
        out_dir = current_dir / "benchmark"
        out_dir.mkdir(exist_ok=True)
        json_path = out_dir / "perf.json"

        if json_path.exists():
            with open(json_path) as f:
                entries = json.load(f)
        else:
            entries = []

        run_name = name.strip()
        if not run_name:
            try:
                run_name = input("Enter a name for this run (blank for default): ").strip()
            except EOFError:
                run_name = ""
        if not run_name:
            run_name = str(len(entries) + 1)

        entries.append({
            "name": run_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "shapes": shapes_list,
            "versions": versions_list,
            "results": all_results,
        })

        with open(json_path, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"Saved run '{run_name}' to {json_path}")

        bench_core.plot_results(all_results, out_dir / "perf.jpg")

    elif action == "benchmark_all":
        import json, datetime

        print(f"Spawning {len(shapes_list) * len(versions_list)} L2-flushed jobs on B200 in parallel...")
        handles = {(s, v): benchmark_one_flush.spawn(s, v) for s in shapes_list for v in versions_list}

        all_results = {s: {} for s in shapes_list}
        for (s, v), handle in handles.items():
            try:
                best = handle.get()
                all_results[s][v] = best.get(v, 0.0)
                cublas = best.get("cuBLAS", 0.0)
                if cublas > all_results[s].get("cuBLAS", 0.0):
                    all_results[s]["cuBLAS"] = cublas
                print(f"  {s} {v}: {best}")
            except Exception as e:
                print(f"  {s} {v}: FAILED — {e}")
                all_results[s][v] = 0.0

        out_dir = current_dir / "benchmark"
        out_dir.mkdir(exist_ok=True)
        json_path = out_dir / "perf_all.json"

        if json_path.exists():
            with open(json_path) as f:
                entries = json.load(f)
        else:
            entries = []

        run_name = name.strip()
        if not run_name:
            try:
                run_name = input("Enter a name for this run (blank for default): ").strip()
            except EOFError:
                run_name = ""
        if not run_name:
            run_name = str(len(entries) + 1)

        entries.append({
            "name": run_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "shapes": shapes_list,
            "versions": versions_list,
            "results": all_results,
            "l2_flush": True,
        })

        with open(json_path, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"Saved run '{run_name}' to {json_path}")

        bench_core.plot_results(all_results, out_dir / "perf_all.jpg")

    elif action == "profile":
        rep_bytes = profile.remote(shape, version)
        out_path = current_dir / "benchmark" / f"profile_{version}_{shape.replace(',','x')}.ncu-rep"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_bytes(rep_bytes)
        print(f"Saved profile to {out_path}")
        print(f"Open with: ncu-ui {out_path}")

    elif action == "intra_profile":
        trace_bytes = intra_profile.remote(shape, version)
        out_path = current_dir / "benchmark" / f"intra_profile_{version}_{shape.replace(',','x')}.json.gz"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_bytes(trace_bytes)
        print(f"Saved intra-kernel profile to {out_path}")
        print(f"Open with: chrome://tracing or https://ui.perfetto.dev")

    elif action == "ncu_full":
        target_shape = "8192,8192,8192"
        target_version = "v7"
        rep_bytes = ncu_full.remote(target_shape, target_version)
        out_path = current_dir / "benchmark" / f"{target_version}_ncu_full.ncu-rep"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_bytes(rep_bytes)
        print(f"Saved ncu --set full report to {out_path}")
        print(f"Open with: ncu-ui {out_path}")

    elif action == "trace":
        rep_bytes = trace.remote(shape, version)
        out_path = current_dir / "benchmark" / f"trace_{version}_{shape.replace(',','x')}.nsys-rep"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_bytes(rep_bytes)
        print(f"Saved trace to {out_path}")
        print(f"Open with: nsys-ui {out_path}")
