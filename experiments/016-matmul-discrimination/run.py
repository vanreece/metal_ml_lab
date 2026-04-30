# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
#   "numpy>=1.26",
# ]
# ///
"""
Experiment 016: matmul shape-discrimination on M4 Max.

Mirrors 014b/015's protocol (5 s cell target, 5 000 trial cap,
chunked sample-buffer resolve, 1 s inter-cell idle, IOReport sidecar
at 250 ms) on naive fp32 matmul across a dense shape grid.

Shape grid (4 sweeps, 34 unique shapes, ~190 cells total):
  A: square diagonal M=N=K ∈ {8..2048} (17 shapes)
  B: K-sweep at M=N=128, K ∈ {2..4096} (7 shapes)
  C: K-sweep at M=N=512, K ∈ {2..4096} (7 shapes)
  D: memory-bound probes, narrow-output × big-reduction (3 shapes)

Each shape gets an N_AMP grid clipped so the largest cell's trial
wall-clock stays under ~80 ms (rough estimate at 5 % of fp32 peak).
"""
from __future__ import annotations

import csv
import ctypes
import datetime as dt
import platform
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import Metal
import numpy as np
import objc


KERNEL_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

constant int M_CONST = {m};
constant int N_CONST = {n};
constant int K_CONST = {k};
constant int N_AMP = {n_amp};

kernel void matmul_amplified(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float       *C [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {{
    if (tid.y >= (uint)M_CONST || tid.x >= (uint)N_CONST) return;
    float acc = 0.0f;
    for (int n_i = 0; n_i < N_AMP; n_i++) {{
        for (int k = 0; k < K_CONST; k++) {{
            acc += A[tid.y * K_CONST + k] * B[k * N_CONST + tid.x];
        }}
    }}
    C[tid.y * N_CONST + tid.x] = acc;
}}
"""

# Shape grid
SQUARE_SIZES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                384, 512, 768, 1024, 1536, 2048]
KSWEEP_128_K = [2, 4, 16, 64, 256, 1024, 4096]
KSWEEP_512_K = [2, 4, 16, 64, 256, 1024, 4096]
MEMBOUND_SHAPES = [
    (8, 4096, 4096),
    (4, 8192, 4096),
    (2, 8192, 4096),
]


def all_shapes():
    shapes = []
    for s in SQUARE_SIZES:
        shapes.append(("square", s, s, s))
    for k in KSWEEP_128_K:
        shapes.append(("ksweep_128", 128, 128, k))
    for k in KSWEEP_512_K:
        shapes.append(("ksweep_512", 512, 512, k))
    for (m, n, k) in MEMBOUND_SHAPES:
        shapes.append(("membound", m, n, k))
    return shapes


# Amp grid
FULL_AMP_GRID = [1, 2, 4, 8, 16, 64, 256, 1024]
TRIAL_BUDGET_S = 0.080
ASSUMED_FP32_RATE = 1.4e12  # ~5 % of M4 Max fp32 peak (28 TFLOPs)


def amp_grid_for_shape(m, n, k):
    flops_per_matmul = 2 * m * n * k
    t_one_matmul_s = flops_per_matmul / ASSUMED_FP32_RATE
    n_amp_max_estimated = TRIAL_BUDGET_S / max(t_one_matmul_s, 1e-12)
    grid = [na for na in FULL_AMP_GRID if na <= n_amp_max_estimated]
    if len(grid) < 4:
        grid = FULL_AMP_GRID[:4]  # always at least 4 levels for slope
    return grid


CELL_DURATION_S = 5.0
INTER_CELL_S = 1.0
BASELINE_S = 2.0
TAIL_S = 2.0
MAX_TRIALS_PER_CELL = 5000
RESOLVE_CHUNK_TRIALS = 2000

TG_M = 16
TG_N = 16

INIT_SEED = 16092604
TELEMETRY_INTERVAL_MS = 250

DONT_SAMPLE = Metal.MTLCounterDontSample

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
IOREPORT_SCRIPT = PROJECT_ROOT / "notes" / "ioreport.py"


def set_user_interactive_qos() -> str:
    QOS_CLASS_USER_INTERACTIVE = 0x21
    libsystem = ctypes.CDLL("/usr/lib/libSystem.dylib")
    rc = libsystem.pthread_set_qos_class_self_np(
        ctypes.c_int(QOS_CLASS_USER_INTERACTIVE), ctypes.c_int(0)
    )
    return f"pthread_set_qos_class_self_np -> {rc}"


def power_source() -> str:
    out = subprocess.run(
        ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip().splitlines()[0] if out.stdout else "<unavailable>"


def find_timestamp_counter_set(device):
    for cs in device.counterSets() or []:
        if str(cs.name()) == "timestamp":
            return cs
    raise RuntimeError("No 'timestamp' counter set on this device")


def build_pipeline(device, m, n, k, n_amp):
    options = Metal.MTLCompileOptions.alloc().init()
    src = KERNEL_TEMPLATE.format(m=m, n=n, k=k, n_amp=n_amp)
    library, err = device.newLibraryWithSource_options_error_(src, options, None)
    if library is None:
        raise RuntimeError(
            f"compile failed (m={m},n={n},k={k},n_amp={n_amp}): {err}"
        )
    fn = library.newFunctionWithName_("matmul_amplified")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(
            f"pipeline create failed (m={m},n={n},k={k},n_amp={n_amp}): {err}"
        )
    return pipeline


def make_buffer(device, num_floats, init_array=None):
    nbytes = num_floats * 4
    if init_array is not None:
        buf = device.newBufferWithBytes_length_options_(
            init_array.tobytes(), nbytes, Metal.MTLResourceStorageModeShared,
        )
    else:
        buf = device.newBufferWithLength_options_(
            nbytes, Metal.MTLResourceStorageModeShared,
        )
    if buf is None:
        raise RuntimeError(f"buffer alloc {nbytes}B failed")
    return buf


def make_sample_buffer(device, counter_set, sample_count, label):
    desc = Metal.MTLCounterSampleBufferDescriptor.alloc().init()
    desc.setCounterSet_(counter_set)
    desc.setSampleCount_(sample_count)
    desc.setStorageMode_(Metal.MTLStorageModeShared)
    desc.setLabel_(label)
    buf, err = device.newCounterSampleBufferWithDescriptor_error_(desc, None)
    if buf is None:
        raise RuntimeError(f"sample buffer alloc failed ({label}): {err}")
    return buf


def resolve_range(sample_buffer, start_idx, count):
    data = sample_buffer.resolveCounterRange_((start_idx, count))
    if data is None or data.length() < 8 * count:
        raise RuntimeError(
            f"resolveCounterRange({start_idx},{count}) returned {data}"
        )
    raw = bytes(data)
    return list(struct.unpack(f"<{count}Q", raw[:count * 8]))


def trial(queue, pipeline, buffers, m, n, sample_buffer, slot_pair):
    a, b, c = buffers
    start_idx, end_idx = slot_pair
    pass_desc = Metal.MTLComputePassDescriptor.computePassDescriptor()
    att = pass_desc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
    att.setSampleBuffer_(sample_buffer)
    att.setStartOfEncoderSampleIndex_(start_idx)
    att.setEndOfEncoderSampleIndex_(end_idx)

    cb = queue.commandBuffer()
    monotonic_ns = time.monotonic_ns()
    encoder = cb.computeCommandEncoderWithDescriptor_(pass_desc)
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(a, 0, 0)
    encoder.setBuffer_offset_atIndex_(b, 0, 1)
    encoder.setBuffer_offset_atIndex_(c, 0, 2)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n, m, 1),       # x = N (col), y = M (row)
        Metal.MTLSizeMake(TG_N, TG_M, 1),
    )
    encoder.endEncoding()
    cpu_t0 = time.perf_counter_ns()
    cb.commit()
    cb.waitUntilCompleted()
    cpu_total_ns = time.perf_counter_ns() - cpu_t0
    return monotonic_ns, cpu_total_ns


def run_cell(queue, pipeline, buffers, m, n, sample_buffer):
    full_trials = []
    deadline = time.monotonic() + CELL_DURATION_S
    while (time.monotonic() < deadline
           and len(full_trials) < MAX_TRIALS_PER_CELL):
        chunk_target = min(
            RESOLVE_CHUNK_TRIALS,
            MAX_TRIALS_PER_CELL - len(full_trials),
        )
        chunk = []
        for i in range(chunk_target):
            slot = (2 * i, 2 * i + 1)
            mn, ct = trial(queue, pipeline, buffers, m, n, sample_buffer, slot)
            chunk.append((mn, ct))
            if time.monotonic() >= deadline:
                break
        n_chunk = len(chunk)
        if n_chunk == 0:
            break
        timestamps = resolve_range(sample_buffer, 0, 2 * n_chunk)
        for i, (mn, ct) in enumerate(chunk):
            gs = timestamps[2 * i]
            ge = timestamps[2 * i + 1]
            full_trials.append({
                "monotonic_ns": mn,
                "cpu_total_ns": ct,
                "gpu_t_start_raw": gs,
                "gpu_t_end_raw": ge,
                "gpu_delta_raw": ge - gs,
            })
    return full_trials


def percentile(sorted_xs, q):
    if not sorted_xs:
        return 0
    idx = int(round(q * (len(sorted_xs) - 1)))
    return sorted_xs[idx]


def cpu_matmul_check(device, queue, m, n, k, a_buffer, b_buffer, c_buffer):
    """Sanity check: compile a single (m,n,k,N_AMP=1) pipeline, run
    matmul once, compare to numpy reference if buffer readback works.
    Returns None if readback API path fails (best-effort)."""
    pipeline = build_pipeline(device, m, n, k, 1)
    cb = queue.commandBuffer()
    encoder = cb.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(a_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(b_buffer, 0, 1)
    encoder.setBuffer_offset_atIndex_(c_buffer, 0, 2)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n, m, 1),
        Metal.MTLSizeMake(TG_N, TG_M, 1),
    )
    encoder.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    if cb.error() is not None:
        raise RuntimeError(f"sanity-check dispatch failed: {cb.error()}")

    try:
        a_np = np.frombuffer(
            a_buffer.contents().as_buffer(m * k * 4), dtype=np.float32
        ).reshape(m, k).copy()
        b_np = np.frombuffer(
            b_buffer.contents().as_buffer(k * n * 4), dtype=np.float32
        ).reshape(k, n).copy()
        c_gpu = np.frombuffer(
            c_buffer.contents().as_buffer(m * n * 4), dtype=np.float32
        ).reshape(m, n).copy()
    except Exception as e:
        print(f"  (buffer readback unsupported in this PyObjC binding: "
              f"{type(e).__name__}: {e}; skipping numpy compare)")
        return None
    c_cpu = a_np @ b_np
    return float(np.max(np.abs(c_gpu - c_cpu)))


def main():
    if not IOREPORT_SCRIPT.exists():
        print(f"ERROR: notes/ioreport.py not found at {IOREPORT_SCRIPT}",
              file=sys.stderr)
        return 2

    RAW_DIR.mkdir(exist_ok=True)
    qos_result = set_user_interactive_qos()
    pwr = power_source()

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("MTLCreateSystemDefaultDevice returned nil")
    arch = device.architecture().name() if (
        hasattr(device, "architecture") and device.architecture()
    ) else "<unavailable>"
    queue = device.newCommandQueue()
    counter_set = find_timestamp_counter_set(device)

    print("=" * 78)
    print("Experiment 016: matmul shape-discrimination (naive fp32)")
    print("=" * 78)
    print(f"device: {device.name()}  arch: {arch}")
    print(f"OS:     {platform.platform()}")
    print(f"power:  {pwr}")
    print(f"qos:    {qos_result}")

    shapes = all_shapes()
    plan = []
    for (sweep, m, n, k) in shapes:
        for n_amp in amp_grid_for_shape(m, n, k):
            plan.append((sweep, m, n, k, n_amp))
    print(f"\nplan: {len(shapes)} shapes, {len(plan)} cells")

    # Determine max buffer sizes across all shapes (allocate once,
    # reuse — kernel only accesses M*K, K*N, M*N elements based on
    # each pipeline's compile-time constants).
    max_a = max(m * k for (_, m, _, k) in shapes)
    max_b = max(k * n for (_, _, n, k) in shapes)
    max_c = max(m * n for (_, m, n, _) in shapes)
    print(f"\nbuffer sizes: A={max_a*4/(1<<20):.1f}MiB  "
          f"B={max_b*4/(1<<20):.1f}MiB  C={max_c*4/(1<<20):.1f}MiB")

    rng = np.random.default_rng(INIT_SEED)
    print(f"initializing A ({max_a:,} floats) and B ({max_b:,} floats) "
          f"with seed {INIT_SEED}...")
    t0 = time.monotonic()
    a_init = rng.standard_normal(max_a).astype(np.float32)
    b_init = rng.standard_normal(max_b).astype(np.float32)
    print(f"  initialized in {time.monotonic() - t0:.1f}s")

    a_buffer = make_buffer(device, max_a, a_init)
    b_buffer = make_buffer(device, max_b, b_init)
    c_buffer = make_buffer(device, max_c)

    # Sanity check on a mid-sized square shape.
    print("\nsanity check: M=N=K=128, N_AMP=1...")
    err = cpu_matmul_check(device, queue, 128, 128, 128,
                           a_buffer, b_buffer, c_buffer)
    if err is None:
        print(f"  numpy compare skipped; dispatch did not error")
    else:
        print(f"  max abs error vs numpy: {err:.4e}")
        if err > 1e-2:
            print(f"  ERROR: tolerance exceeded; aborting", file=sys.stderr)
            return 4

    # Compile pipelines for all (shape, n_amp) pairs upfront.
    print(f"\ncompiling {len(plan)} pipelines...")
    t0 = time.monotonic()
    pipelines = {}
    for (sweep, m, n, k, n_amp) in plan:
        key = (m, n, k, n_amp)
        if key in pipelines:
            continue
        pipelines[key] = build_pipeline(device, m, n, k, n_amp)
    t1 = time.monotonic()
    print(f"  done in {t1 - t0:.1f}s "
          f"({(t1 - t0) / len(pipelines) * 1000:.1f}ms avg)")

    sb_slots = 2 * RESOLVE_CHUNK_TRIALS
    # One sample buffer reused across all cells. Allocating per cell
    # leaks counter resources at scale (247 cells exhausted the
    # MTLCounterErrorDomain pool around cell 32 in the first run).
    sample_buffer = make_sample_buffer(
        device, counter_set, sb_slots, "exp016-shared"
    )
    print(f"sample-buffer slots: {sb_slots} "
          f"({sb_slots * 8} B; resolves every {RESOLVE_CHUNK_TRIALS} trials; "
          f"shared across all cells)")

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    energy_csv = RAW_DIR / f"{ts}.csv"
    states_csv = RAW_DIR / f"{ts}-states.csv"
    cells_csv = RAW_DIR / f"{ts}-cells.csv"
    trials_csv = RAW_DIR / f"{ts}-trials.csv"
    meta_path = RAW_DIR / f"{ts}-meta.txt"
    iolog = RAW_DIR / f"{ts}-ioreport-stdout.log"

    ioreport_proc = subprocess.Popen(
        ["uv", "run", str(IOREPORT_SCRIPT),
         "--include-states",
         "--interval-ms", str(TELEMETRY_INTERVAL_MS),
         "--csv", str(energy_csv)],
        stdout=open(iolog, "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.5)
    if ioreport_proc.poll() is not None:
        print(f"ERROR: ioreport.py exited rc={ioreport_proc.returncode}",
              file=sys.stderr)
        return 3
    print(f"ioreport.py: pid={ioreport_proc.pid}")

    cells_f = open(cells_csv, "w", newline="")
    cells_w = csv.writer(cells_f)
    cells_w.writerow([
        "cell_idx", "sweep", "m", "n", "k", "n_amp",
        "monotonic_ns_start", "monotonic_ns_end",
        "trial_count", "p10", "p50", "p90", "p99", "min", "max",
    ])
    trials_f = open(trials_csv, "w", newline="")
    trials_fields = [
        "cell_idx", "sweep", "m", "n", "k", "n_amp", "trial_idx",
        "monotonic_ns",
        "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
        "cpu_total_ns",
    ]
    trials_w = csv.DictWriter(trials_f, fieldnames=trials_fields)
    trials_w.writeheader()

    run_start_ns = time.monotonic_ns()
    cell_idx = 0
    try:
        print()
        print(f"=== phase 0: baseline {BASELINE_S:.1f}s ===")
        time.sleep(BASELINE_S)

        print(f"\n=== phase 1: {len(plan)} cells ===")
        for (sweep, m, n, k, n_amp) in plan:
            pipeline = pipelines[(m, n, k, n_amp)]

            cell_start_ns = time.monotonic_ns()
            full_trials = run_cell(
                queue, pipeline, (a_buffer, b_buffer, c_buffer),
                m, n, sample_buffer,
            )
            cell_end_ns = time.monotonic_ns()
            trial_count = len(full_trials)

            deltas = []
            for i, t in enumerate(full_trials):
                deltas.append(t["gpu_delta_raw"])
                trials_w.writerow({
                    "cell_idx": cell_idx,
                    "sweep": sweep,
                    "m": m, "n": n, "k": k, "n_amp": n_amp,
                    "trial_idx": i,
                    "monotonic_ns": t["monotonic_ns"],
                    "gpu_t_start_raw": t["gpu_t_start_raw"],
                    "gpu_t_end_raw": t["gpu_t_end_raw"],
                    "gpu_delta_raw": t["gpu_delta_raw"],
                    "cpu_total_ns": t["cpu_total_ns"],
                })
            trials_f.flush()

            sd = sorted(deltas)
            p10 = percentile(sd, 0.10)
            p50 = percentile(sd, 0.50)
            p90 = percentile(sd, 0.90)
            p99 = percentile(sd, 0.99)
            cells_w.writerow([
                cell_idx, sweep, m, n, k, n_amp,
                cell_start_ns, cell_end_ns, trial_count,
                p10, p50, p90, p99, min(deltas), max(deltas),
            ])
            cells_f.flush()

            cap_marker = " (CAP)" if trial_count >= MAX_TRIALS_PER_CELL else ""
            print(f"  cell {cell_idx:>3}: {sweep:<11} "
                  f"{m:>4}x{n:>4}x{k:<5} N={n_amp:>4}  "
                  f"trials={trial_count:>5}{cap_marker}  "
                  f"p50={p50:>10} p90={p90:>10}  "
                  f"({(cell_end_ns - cell_start_ns) / 1e9:.2f}s)")

            cell_idx += 1
            time.sleep(INTER_CELL_S)

        print(f"\n=== phase 2: tail {TAIL_S:.1f}s ===")
        time.sleep(TAIL_S)
    finally:
        cells_f.close()
        trials_f.close()
        run_end_ns = time.monotonic_ns()
        print()
        print("stopping ioreport.py (SIGINT)")
        ioreport_proc.send_signal(signal.SIGINT)
        try:
            ioreport_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            ioreport_proc.terminate()
            ioreport_proc.wait(timeout=3.0)
        print(f"ioreport.py rc={ioreport_proc.returncode}")

    elapsed_s = (run_end_ns - run_start_ns) / 1e9

    amp_summary = []
    for (sweep, m, n, k) in shapes:
        amp_summary.append(
            f"  {sweep:<11} m={m:<5} n={n:<5} k={k:<5} "
            f"amp_grid={amp_grid_for_shape(m, n, k)}"
        )

    meta_lines = [
        "experiment: 016-matmul-discrimination",
        f"timestamp: {ts}",
        f"device: {device.name()}",
        f"architecture: {arch}",
        f"os: {platform.platform()}",
        f"python: {sys.version.splitlines()[0]}",
        f"pyobjc: {objc.__version__}",
        f"numpy: {np.__version__}",
        f"qos: {qos_result}",
        f"power: {pwr}",
        f"trials_csv: {trials_csv.name}",
        f"cells_csv: {cells_csv.name}",
        f"energy_csv: {energy_csv.name}",
        f"states_csv: {states_csv.name}",
        f"telemetry_interval_ms: {TELEMETRY_INTERVAL_MS}",
        f"baseline_s: {BASELINE_S}",
        f"tail_s: {TAIL_S}",
        f"cell_duration_s: {CELL_DURATION_S}",
        f"inter_cell_s: {INTER_CELL_S}",
        f"max_trials_per_cell: {MAX_TRIALS_PER_CELL}",
        f"resolve_chunk_trials: {RESOLVE_CHUNK_TRIALS}",
        f"threadgroup: {TG_M}x{TG_N}",
        f"init_seed: {INIT_SEED}",
        f"max_a_floats: {max_a}",
        f"max_b_floats: {max_b}",
        f"max_c_floats: {max_c}",
        f"trial_budget_s: {TRIAL_BUDGET_S}",
        f"assumed_fp32_rate: {ASSUMED_FP32_RATE}",
        f"sanity_check_max_abs_err: {err if err is not None else '<readback unsupported>'}",
        f"shape_count: {len(shapes)}",
        f"cell_count: {len(plan)}",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {elapsed_s:.2f}",
        "shapes_with_amp_grids:",
    ] + amp_summary
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"wrote {meta_path.name}")
    print()
    print(f"Now run: uv run experiments/016-matmul-discrimination/analysis.py "
          f"--prefix {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
