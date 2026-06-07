# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
#   "pyobjc-framework-MetalPerformanceShaders>=10.0",
#   "numpy>=1.26",
# ]
# ///
"""
Experiment 017: MPS GEMM ceiling on M4 Max.

Mirrors 016's protocol (5 s cell target, 5 000 trial cap, 1 s
inter-cell idle, IOReport sidecar at 250 ms) but with two changes:

  - Kernel: MPSMatrixMultiplication (Apple's optimized fp32 GEMM).
  - Timing: cb.GPUStartTime() / cb.GPUEndTime() per trial instead
    of MTLCounterSampleBuffer-attached encoder boundaries. MPS owns
    its own compute encoder internally, so the sample-buffer
    instrument 014b/015/016 use is not available. See README pre-
    reg amendment 2026-05-05.

Shape grid is identical to 016's 34-shape grid:
  A: square diagonal M=N=K ∈ {8..2048} (17 shapes)
  B: K-sweep at M=N=128, K ∈ {2..4096} (7 shapes)
  C: K-sweep at M=N=512, K ∈ {2..4096} (7 shapes)
  D: memory-bound probes (3 shapes)

One cell per shape (no N_AMP grid; MPS doesn't expose an internal
loop knob, and the pre-reg explicitly skips amplification).
"""
from __future__ import annotations

import csv
import ctypes
import datetime as dt
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import Metal
import MetalPerformanceShaders as MPS
import numpy as np
import objc


# Shape grid — identical to 016 -----------------------------------
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


# Cell protocol ----------------------------------------------------
WARMUP_DISPATCHES = 5
CELL_DURATION_S = 5.0
INTER_CELL_S = 1.0
BASELINE_S = 2.0
TAIL_S = 2.0
MAX_TRIALS_PER_CELL = 5000

INIT_SEED = 16092604
TELEMETRY_INTERVAL_MS = 250

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


def make_mps_matrix(buffer, rows, cols):
    """Wrap an MTLBuffer as an fp32 MPSMatrix view of (rows, cols)
    with rowBytes = cols * 4 (tightly packed)."""
    desc = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        rows, cols, cols * 4, MPS.MPSDataTypeFloat32
    )
    mat = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(buffer, desc)
    if mat is None:
        raise RuntimeError(f"MPSMatrix init failed for {rows}x{cols}")
    return mat


def make_mps_kernel(device, m, n, k):
    """Build an MPSMatrixMultiplication kernel for given (M, N, K)."""
    mm = MPS.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
        device, False, False, m, n, k, 1.0, 0.0
    )
    if mm is None:
        raise RuntimeError(f"MPSMatrixMultiplication init failed for {m}x{n}x{k}")
    return mm


def trial(queue, mm, a_mat, b_mat, c_mat):
    """Encode one MPS GEMM dispatch, wait for completion, return
    (monotonic_ns_at_commit, gpu_start_ns, gpu_end_ns, cpu_total_ns)."""
    cb = queue.commandBuffer()
    monotonic_ns = time.monotonic_ns()
    mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
        cb, a_mat, b_mat, c_mat
    )
    cpu_t0 = time.perf_counter_ns()
    cb.commit()
    cb.waitUntilCompleted()
    cpu_total_ns = time.perf_counter_ns() - cpu_t0
    if cb.error() is not None:
        raise RuntimeError(f"command buffer error: {cb.error()}")
    # GPUStartTime / GPUEndTime return CFTimeInterval (seconds, double).
    gpu_start_s = cb.GPUStartTime()
    gpu_end_s = cb.GPUEndTime()
    gpu_start_ns = int(round(gpu_start_s * 1e9))
    gpu_end_ns = int(round(gpu_end_s * 1e9))
    return monotonic_ns, gpu_start_ns, gpu_end_ns, cpu_total_ns


def run_cell(queue, mm, a_mat, b_mat, c_mat):
    """Run a single shape's cell: warmup + timed trials until either
    5 s wall-clock or 5 000 trials."""
    # Warmup
    for _ in range(WARMUP_DISPATCHES):
        cb = queue.commandBuffer()
        mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
            cb, a_mat, b_mat, c_mat
        )
        cb.commit()
        cb.waitUntilCompleted()
        if cb.error() is not None:
            raise RuntimeError(f"warmup cb error: {cb.error()}")

    full_trials = []
    deadline = time.monotonic() + CELL_DURATION_S
    while (time.monotonic() < deadline
           and len(full_trials) < MAX_TRIALS_PER_CELL):
        mn, gs, ge, ct = trial(queue, mm, a_mat, b_mat, c_mat)
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


def mps_sanity_check(device, queue, m, n, k, a_buffer, b_buffer, c_buffer):
    """One MPS GEMM at (m,n,k) compared to numpy. Returns max abs
    error or None if buffer readback fails."""
    a_mat = make_mps_matrix(a_buffer, m, k)
    b_mat = make_mps_matrix(b_buffer, k, n)
    c_mat = make_mps_matrix(c_buffer, m, n)
    mm = make_mps_kernel(device, m, n, k)

    cb = queue.commandBuffer()
    mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
        cb, a_mat, b_mat, c_mat
    )
    cb.commit()
    cb.waitUntilCompleted()
    if cb.error() is not None:
        raise RuntimeError(f"sanity-check cb error: {cb.error()}")

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

    print("=" * 78)
    print("Experiment 017: MPS GEMM ceiling (fp32, no transpose)")
    print("=" * 78)
    print(f"device: {device.name()}  arch: {arch}")
    print(f"OS:     {platform.platform()}")
    print(f"power:  {pwr}")
    print(f"qos:    {qos_result}")

    shapes = all_shapes()
    print(f"\nplan: {len(shapes)} shapes, {len(shapes)} cells "
          f"(no N_AMP grid for MPS)")

    # Allocate max-sized A, B, C once (reused across all shapes).
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

    # Sanity check at M=N=K=128.
    print("\nsanity check: M=N=K=128, MPS vs numpy...")
    err = mps_sanity_check(device, queue, 128, 128, 128,
                           a_buffer, b_buffer, c_buffer)
    if err is None:
        print("  numpy compare skipped; MPS dispatch did not error")
    else:
        print(f"  max abs error vs numpy: {err:.4e}")
        if err > 1e-2:
            print(f"  ERROR: tolerance exceeded; aborting", file=sys.stderr)
            return 4

    # Build MPS kernels for each unique (m, n, k).
    print(f"\nbuilding {len(shapes)} MPS kernels...")
    t0 = time.monotonic()
    mps_kernels = {}
    for (sweep, m, n, k) in shapes:
        key = (m, n, k)
        if key in mps_kernels:
            continue
        mps_kernels[key] = make_mps_kernel(device, m, n, k)
    print(f"  done in {time.monotonic() - t0:.2f}s")

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
        "cell_idx", "sweep", "m", "n", "k",
        "monotonic_ns_start", "monotonic_ns_end",
        "trial_count", "p10", "p50", "p90", "p99", "min", "max",
    ])
    trials_f = open(trials_csv, "w", newline="")
    trials_fields = [
        "cell_idx", "sweep", "m", "n", "k", "trial_idx",
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

        print(f"\n=== phase 1: {len(shapes)} cells ===")
        for (sweep, m, n, k) in shapes:
            mm = mps_kernels[(m, n, k)]
            a_mat = make_mps_matrix(a_buffer, m, k)
            b_mat = make_mps_matrix(b_buffer, k, n)
            c_mat = make_mps_matrix(c_buffer, m, n)

            cell_start_ns = time.monotonic_ns()
            full_trials = run_cell(queue, mm, a_mat, b_mat, c_mat)
            cell_end_ns = time.monotonic_ns()
            trial_count = len(full_trials)

            deltas = []
            for i, t in enumerate(full_trials):
                deltas.append(t["gpu_delta_raw"])
                trials_w.writerow({
                    "cell_idx": cell_idx,
                    "sweep": sweep,
                    "m": m, "n": n, "k": k,
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
                cell_idx, sweep, m, n, k,
                cell_start_ns, cell_end_ns, trial_count,
                p10, p50, p90, p99, min(deltas), max(deltas),
            ])
            cells_f.flush()

            cap_marker = " (CAP)" if trial_count >= MAX_TRIALS_PER_CELL else ""
            print(f"  cell {cell_idx:>3}: {sweep:<11} "
                  f"{m:>4}x{n:>4}x{k:<5}  "
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

    meta_lines = [
        "experiment: 017-mps-gemm-ceiling",
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
        f"warmup_dispatches: {WARMUP_DISPATCHES}",
        f"init_seed: {INIT_SEED}",
        f"max_a_floats: {max_a}",
        f"max_b_floats: {max_b}",
        f"max_c_floats: {max_c}",
        f"sanity_check_max_abs_err: {err if err is not None else '<readback unsupported>'}",
        f"shape_count: {len(shapes)}",
        f"cell_count: {len(shapes)}",
        f"timing_source: cb.GPUStartTime/GPUEndTime (see README pre-reg amendment 2026-05-05)",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {elapsed_s:.2f}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"wrote {meta_path.name}")
    print()
    print(f"Now run: uv run experiments/017-mps-gemm-ceiling/analysis.py "
          f"--prefix {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
