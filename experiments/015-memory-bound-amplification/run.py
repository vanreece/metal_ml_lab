# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
#   "numpy>=1.26",
# ]
# ///
"""
Experiment 015: pointer-chase amplification on M4 Max.

Mirrors 014b's protocol (5 s cell target, 5 000 trial cap, chunked
sample-buffer resolve, 1 s inter-cell idle, IOReport sidecar at
250 ms) with the FMA carry chain replaced by a pointer-chase
chain. A 128 MB random-permutation lookup table forces each
chained load to miss SLC, so per-amp-step cost reflects DRAM
round-trip latency rather than ALU throughput.

Same N grid as 014/014b: {1, 2, 4, 8, 16, 64, 256, 1024} × {internal-
loop, back-to-back}. Per-iter constant CHASE_PER_ITER = 64 mirrors
014's FMA_PER_ITER = 64 so slopes are directly comparable across
kernels.
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

constant int CHASE_PER_ITER = 64;
constant int N_AMP = {n_amp};

kernel void chase_amplified(
    device const uint *table [[buffer(0)]],
    device uint       *out   [[buffer(1)]],
    uint tid                  [[thread_position_in_grid]]
) {{
    uint addr = tid;
    for (int n = 0; n < N_AMP; n++) {{
        for (int i = 0; i < CHASE_PER_ITER; i++) {{
            addr = table[addr];
        }}
    }}
    out[tid] = addr;
}}
"""

N_LEVELS = [1, 2, 4, 8, 16, 64, 256, 1024]
METHODS = ["internal-loop", "back-to-back"]

CELL_DURATION_S = 5.0
INTER_CELL_S = 1.0
BASELINE_S = 2.0
TAIL_S = 2.0
MAX_TRIALS_PER_CELL = 5000
RESOLVE_CHUNK_TRIALS = 2000

THREADS = 32
GROUP = 32

# 128 MB table = 32 M uint32 entries. Beyond Apple silicon SLC
# (24-48 MB typical), so each chained load steady-states on DRAM
# round-trip latency.
TABLE_ENTRIES = 32_000_000
TABLE_BYTES = TABLE_ENTRIES * 4
PERMUTATION_SEED = 14092604

OUT_BUFFER_BYTES = 4 * 1024
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


def build_pipeline(device, n_amp: int):
    options = Metal.MTLCompileOptions.alloc().init()
    src = KERNEL_TEMPLATE.format(n_amp=n_amp)
    library, err = device.newLibraryWithSource_options_error_(src, options, None)
    if library is None:
        raise RuntimeError(f"compile failed (N_AMP={n_amp}): {err}")
    fn = library.newFunctionWithName_("chase_amplified")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(f"pipeline create failed (N_AMP={n_amp}): {err}")
    return pipeline


def build_chase_table(device):
    print(f"generating {TABLE_ENTRIES:,}-entry random permutation "
          f"(seed={PERMUTATION_SEED}, ~{TABLE_BYTES // (1024*1024)} MiB)...")
    t0 = time.monotonic()
    rng = np.random.default_rng(PERMUTATION_SEED)
    perm = rng.permutation(TABLE_ENTRIES).astype(np.uint32)
    t1 = time.monotonic()
    print(f"  permutation built in {t1 - t0:.2f}s, "
          f"uploading to MTLBuffer...")
    buf = device.newBufferWithBytes_length_options_(
        perm.tobytes(), TABLE_BYTES, Metal.MTLResourceStorageModeShared,
    )
    if buf is None:
        raise RuntimeError("table buffer alloc failed")
    t2 = time.monotonic()
    print(f"  uploaded in {t2 - t1:.2f}s")
    # Spot-check: every entry should be in [0, TABLE_ENTRIES) and
    # the permutation is a bijection. (Quick sanity, not full
    # validation.)
    sample = perm[:1024]
    assert sample.max() < TABLE_ENTRIES
    assert sample.min() >= 0
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


def trial_internal_loop(queue, pipeline, table_buffer, out_buffer,
                        sample_buffer, slot_pair):
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
    encoder.setBuffer_offset_atIndex_(table_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(out_buffer, 0, 1)
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


def trial_back_to_back(queue, pipeline, table_buffer, out_buffer,
                       sample_buffer, slot_pair, n_dispatches):
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
        encoder.setBuffer_offset_atIndex_(table_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(out_buffer, 0, 1)
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
            encoder.setBuffer_offset_atIndex_(table_buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(out_buffer, 0, 1)
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


def run_cell(method, n_amp, queue, pipeline, table_buffer, out_buffer,
             sample_buffer):
    full_trials = []
    cell_start_ns = time.monotonic_ns()
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
            if method == "internal-loop":
                mn, ct = trial_internal_loop(
                    queue, pipeline, table_buffer, out_buffer,
                    sample_buffer, slot,
                )
            else:
                mn, ct = trial_back_to_back(
                    queue, pipeline, table_buffer, out_buffer,
                    sample_buffer, slot, n_amp,
                )
            chunk.append((mn, ct))
            if time.monotonic() >= deadline:
                break
        n = len(chunk)
        if n == 0:
            break
        timestamps = resolve_range(sample_buffer, 0, 2 * n)
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
    cell_end_ns = time.monotonic_ns()
    return cell_start_ns, cell_end_ns, full_trials


def percentile(sorted_xs, q):
    if not sorted_xs:
        return 0
    idx = int(round(q * (len(sorted_xs) - 1)))
    return sorted_xs[idx]


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

    print("=" * 78)
    print("Experiment 015: pointer-chase amplification (memory-latency-bound)")
    print("=" * 78)
    print(f"device: {device.name()}  arch: {arch}")
    print(f"OS:     {platform.platform()}")
    print(f"power:  {pwr}")
    print(f"qos:    {qos_result}")

    table_buffer = build_chase_table(device)
    out_buffer = device.newBufferWithLength_options_(
        OUT_BUFFER_BYTES, Metal.MTLResourceStorageModeShared
    )

    pipelines = {}
    for n in N_LEVELS:
        pipelines[n] = build_pipeline(device, n)
    print(f"compiled internal-loop pipelines: {sorted(pipelines.keys())}")
    base_pipeline = pipelines[1]
    print("back-to-back uses N_AMP=1 base-unit pipeline")

    sb_slots = 2 * RESOLVE_CHUNK_TRIALS
    print(f"sample-buffer slots per cell: {sb_slots} "
          f"({sb_slots * 8} B; resolves every {RESOLVE_CHUNK_TRIALS} trials)")

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
        "trial_count", "p10", "p50", "p90", "p99", "min", "max",
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
            label = f"exp015-{method}-N{n}"
            sb = make_sample_buffer(device, counter_set, sb_slots, label)
            pipeline = pipelines[n] if method == "internal-loop" else base_pipeline

            start_ns, end_ns, full_trials = run_cell(
                method, n, queue, pipeline, table_buffer, out_buffer, sb,
            )
            trial_count = len(full_trials)

            deltas = []
            for i, t in enumerate(full_trials):
                deltas.append(t["gpu_delta_raw"])
                trials_w.writerow({
                    "cell_idx": cell_idx,
                    "method": method,
                    "n_amp": n,
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
                cell_idx, method, n, start_ns, end_ns, trial_count,
                p10, p50, p90, p99, min(deltas), max(deltas),
            ])
            cells_f.flush()

            cap_marker = " (CAP)" if trial_count >= MAX_TRIALS_PER_CELL else ""
            print(f"  cell {cell_idx:>2}: {method:<14} N={n:>5}  "
                  f"trials={trial_count:>5}{cap_marker}  "
                  f"p50={p50:>9} p90={p90:>9} p99={p99:>9}  "
                  f"({(end_ns - start_ns) / 1e9:.2f}s cell)")

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
        "experiment: 015-memory-bound-amplification",
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
        f"n_levels: {N_LEVELS}",
        f"methods: {METHODS}",
        f"threads: {THREADS}  group: {GROUP}",
        f"chase_per_iter: 64  (compile-time constant in kernel)",
        f"table_entries: {TABLE_ENTRIES}",
        f"table_bytes: {TABLE_BYTES}",
        f"permutation_seed: {PERMUTATION_SEED}",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {elapsed_s:.2f}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"wrote {meta_path.name}")
    print()
    print(f"Now run: uv run experiments/015-memory-bound-amplification/analysis.py "
          f"--prefix {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
