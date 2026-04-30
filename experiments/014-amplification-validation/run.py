# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 014: Loop amplification + two-point timing validation.

Sweeps amplification factor N across two methods (internal-loop,
back-to-back) on a carry-dependent fma_loop base unit. Each cell runs
50 timed trials at sleep_0 with stage-boundary MTLCounterSampleBuffer
timing. ioreport.py runs as a sidecar so per-cell GPUPH/PWRCTRL
residency can be reconstructed in analysis.

Cells run sequentially in this order: internal-loop ascending N, then
back-to-back ascending N. No cooldown between cells (per pre-reg);
cross-cell contamination is part of the data.
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
import objc


# Carry-dependent fma_loop. Outer loop = amplification (N), inner =
# one base unit. N is a compile-time constant so we pre-compile a
# pipeline per N. The data dependency on `y` prevents the compiler
# from unrolling-and-fusing the chain.
KERNEL_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

constant int FMA_PER_ITER = 64;
constant int N_AMP = {n_amp};

kernel void fma_loop_amplified(device float *out [[buffer(0)]],
                               uint tid [[thread_position_in_grid]]) {{
    float x = float(tid) * 0.0001f + 1.0f;
    float y = 1.0f;
    for (int n = 0; n < N_AMP; n++) {{
        for (int i = 0; i < FMA_PER_ITER; i++) {{
            y = fma(y, x, x);
        }}
    }}
    out[tid] = y;
}}
"""

N_LEVELS = [1, 2, 4, 8, 16, 64, 256, 1024]
METHODS = ["internal-loop", "back-to-back"]
TRIALS_PER_CELL = 50

THREADS = 32
GROUP = 32
OUT_BUFFER_BYTES = 4 * 1024

BASELINE_S = 2.0
TAIL_S = 2.0
TELEMETRY_INTERVAL_MS = 250

# MTLCounterDontSample = NSUIntegerMax. Used to skip a sample on
# encoders that aren't the first or last in a back-to-back chain.
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


def build_pipeline(device, n_amp: int):
    options = Metal.MTLCompileOptions.alloc().init()
    src = KERNEL_TEMPLATE.format(n_amp=n_amp)
    library, err = device.newLibraryWithSource_options_error_(src, options, None)
    if library is None:
        raise RuntimeError(f"compile failed (N_AMP={n_amp}): {err}")
    fn = library.newFunctionWithName_("fma_loop_amplified")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(f"pipeline create failed (N_AMP={n_amp}): {err}")
    return pipeline


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


def resolve_pair(sample_buffer, start_idx):
    data = sample_buffer.resolveCounterRange_((start_idx, 2))
    if data is None or data.length() < 16:
        raise RuntimeError(f"resolveCounterRange({start_idx},2) returned {data}")
    raw = bytes(data)
    return (
        int.from_bytes(raw[0:8], "little", signed=False),
        int.from_bytes(raw[8:16], "little", signed=False),
    )


def trial_internal_loop(queue, pipeline, out_buffer, sample_buffer, slot_pair):
    """Single dispatch of the amplified-N pipeline. Sample at start
    and end of the sole encoder."""
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
    encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(THREADS, 1, 1),
        Metal.MTLSizeMake(GROUP, 1, 1),
    )
    encoder.endEncoding()
    cpu_t0 = time.perf_counter_ns()
    cb.commit()
    cb.waitUntilCompleted()
    cpu_total_ns = time.perf_counter_ns() - cpu_t0
    return monotonic_ns, cpu_total_ns


def trial_back_to_back(queue, pipeline, out_buffer, sample_buffer,
                       slot_pair, n_dispatches):
    """N back-to-back dispatches of the base-unit pipeline in one cb.
    Sample only on first and last encoders; middle encoders run
    untimed. When N==1 the first and last encoder are the same."""
    start_idx, end_idx = slot_pair
    cb = queue.commandBuffer()
    monotonic_ns = time.monotonic_ns()

    if n_dispatches == 1:
        pass_desc = Metal.MTLComputePassDescriptor.computePassDescriptor()
        att = pass_desc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
        att.setSampleBuffer_(sample_buffer)
        att.setStartOfEncoderSampleIndex_(start_idx)
        att.setEndOfEncoderSampleIndex_(end_idx)
        encoder = cb.computeCommandEncoderWithDescriptor_(pass_desc)
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(THREADS, 1, 1),
            Metal.MTLSizeMake(GROUP, 1, 1),
        )
        encoder.endEncoding()
    else:
        for i in range(n_dispatches):
            if i == 0:
                pass_desc = Metal.MTLComputePassDescriptor.computePassDescriptor()
                att = pass_desc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
                att.setSampleBuffer_(sample_buffer)
                att.setStartOfEncoderSampleIndex_(start_idx)
                att.setEndOfEncoderSampleIndex_(DONT_SAMPLE)
                encoder = cb.computeCommandEncoderWithDescriptor_(pass_desc)
            elif i == n_dispatches - 1:
                pass_desc = Metal.MTLComputePassDescriptor.computePassDescriptor()
                att = pass_desc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
                att.setSampleBuffer_(sample_buffer)
                att.setStartOfEncoderSampleIndex_(DONT_SAMPLE)
                att.setEndOfEncoderSampleIndex_(end_idx)
                encoder = cb.computeCommandEncoderWithDescriptor_(pass_desc)
            else:
                encoder = cb.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(THREADS, 1, 1),
                Metal.MTLSizeMake(GROUP, 1, 1),
            )
            encoder.endEncoding()

    cpu_t0 = time.perf_counter_ns()
    cb.commit()
    cb.waitUntilCompleted()
    cpu_total_ns = time.perf_counter_ns() - cpu_t0
    return monotonic_ns, cpu_total_ns


def run_cell(method, n_amp, queue, pipeline, out_buffer, sample_buffer,
             trials_writer, cell_idx):
    rows = []
    for trial_idx in range(TRIALS_PER_CELL):
        slot = (2 * trial_idx, 2 * trial_idx + 1)
        if method == "internal-loop":
            monotonic_ns, cpu_total_ns = trial_internal_loop(
                queue, pipeline, out_buffer, sample_buffer, slot
            )
        else:
            monotonic_ns, cpu_total_ns = trial_back_to_back(
                queue, pipeline, out_buffer, sample_buffer, slot, n_amp
            )
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot[0])
        rows.append({
            "cell_idx": cell_idx,
            "method": method,
            "n_amp": n_amp,
            "trial_idx": trial_idx,
            "monotonic_ns": monotonic_ns,
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            "cpu_total_ns": cpu_total_ns,
        })

    for r in rows:
        trials_writer.writerow(r)
    deltas = [r["gpu_delta_raw"] for r in rows]
    sd = sorted(deltas)
    p50 = sd[len(sd) // 2]
    p10 = sd[len(sd) // 10]
    p90 = sd[(len(sd) * 9) // 10]
    return {"p10": p10, "p50": p50, "p90": p90,
            "min": min(deltas), "max": max(deltas)}


def main() -> int:
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
    out_buffer = device.newBufferWithLength_options_(
        OUT_BUFFER_BYTES, Metal.MTLResourceStorageModeShared
    )

    print("=" * 78)
    print("Experiment 014: amplification + two-point timing validation")
    print("=" * 78)
    print(f"device: {device.name()}  arch: {arch}")
    print(f"OS:     {platform.platform()}")
    print(f"power:  {pwr}")
    print(f"qos:    {qos_result}")

    # Pre-compile pipelines. internal-loop needs one per N; back-to-back
    # only needs N_AMP=1 (we amplify by encoding repeatedly).
    pipelines = {}
    for n in N_LEVELS:
        pipelines[n] = build_pipeline(device, n)
    print(f"compiled internal-loop pipelines: {sorted(pipelines.keys())}")
    base_pipeline = pipelines[1]
    print("back-to-back uses N_AMP=1 base-unit pipeline")

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
        "cell_idx", "method", "n_amp",
        "monotonic_ns_start", "monotonic_ns_end",
        "trial_count", "p10", "p50", "p90", "min", "max",
    ])
    trials_f = open(trials_csv, "w", newline="")
    trials_fields = [
        "cell_idx", "method", "n_amp", "trial_idx",
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

        plan = [(m, n) for m in METHODS for n in N_LEVELS]
        print(f"\n=== phase 1: {len(plan)} cells "
              f"({len(METHODS)} methods x {len(N_LEVELS)} N values) ===")
        for method, n in plan:
            # Per-cell sample buffer with 2 slots per trial.
            label = f"exp014-{method}-N{n}"
            sb = make_sample_buffer(
                device, counter_set, 2 * TRIALS_PER_CELL, label
            )
            pipeline = pipelines[n] if method == "internal-loop" else base_pipeline

            start_ns = time.monotonic_ns()
            stats = run_cell(
                method, n, queue, pipeline, out_buffer, sb,
                trials_w, cell_idx,
            )
            trials_f.flush()
            end_ns = time.monotonic_ns()

            cells_w.writerow([
                cell_idx, method, n, start_ns, end_ns,
                TRIALS_PER_CELL,
                stats["p10"], stats["p50"], stats["p90"],
                stats["min"], stats["max"],
            ])
            cells_f.flush()
            print(f"  cell {cell_idx:>2}: {method:<14} N={n:>5}  "
                  f"p10={stats['p10']:>9} p50={stats['p50']:>9} "
                  f"p90={stats['p90']:>9}  "
                  f"({(end_ns - start_ns) / 1e6:.1f} ms cell)")
            cell_idx += 1

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
        "experiment: 014-amplification-validation",
        f"timestamp: {ts}",
        f"device: {device.name()}",
        f"architecture: {arch}",
        f"os: {platform.platform()}",
        f"python: {sys.version.splitlines()[0]}",
        f"pyobjc: {objc.__version__}",
        f"qos: {qos_result}",
        f"power: {pwr}",
        f"trials_csv: {trials_csv.name}",
        f"cells_csv: {cells_csv.name}",
        f"energy_csv: {energy_csv.name}",
        f"states_csv: {states_csv.name}",
        f"telemetry_interval_ms: {TELEMETRY_INTERVAL_MS}",
        f"baseline_s: {BASELINE_S}",
        f"tail_s: {TAIL_S}",
        f"n_levels: {N_LEVELS}",
        f"methods: {METHODS}",
        f"trials_per_cell: {TRIALS_PER_CELL}",
        f"threads: {THREADS}  group: {GROUP}",
        "fma_per_iter: 64  (compile-time constant in kernel)",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {elapsed_s:.2f}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"wrote {meta_path.name}")
    print()
    print(f"Now run: uv run experiments/014-amplification-validation/analysis.py "
          f"--prefix {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
