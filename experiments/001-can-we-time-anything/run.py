# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 001: Can we time anything at all from Python?

Pre-registered question: Can we get GPU-side timing of a Metal compute
dispatch from a Python process, using MTLCounterSampleBuffer with the
MTLCommonCounterSetTimestamp set, via PyObjC?

Per CLAUDE.md / experiment README:
- No warmup, no retries, no averaging, no exception swallowing.
- Raw values to CSV under raw/ with a timestamp.
- If something fails, let it fail loudly. The failure is the result.
"""
from __future__ import annotations

import csv
import ctypes
import datetime as dt
import platform
import subprocess
import sys
import time
from pathlib import Path

import Metal
import objc


KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void write_tid(device uint *out [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
"""

# 1 SIMD width on Apple Silicon = 32 threads. Keep work near-zero so
# dispatch overhead is the dominant signal.
THREADS_PER_GRID = 32
THREADS_PER_GROUP = 32
NUM_DISPATCHES = 100

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"


# Try to elevate to user-interactive QoS so we match the pre-registered
# conditions. If this fails we record the failure and continue.
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
    sets = device.counterSets()
    if sets is None:
        raise RuntimeError("device.counterSets() returned None")
    for cs in sets:
        if str(cs.name()) == "timestamp":
            return cs
    names = [str(s.name()) for s in sets]
    raise RuntimeError(f"No 'timestamp' counter set. Available: {names}")


def build_pipeline(device):
    options = Metal.MTLCompileOptions.alloc().init()
    library, err = device.newLibraryWithSource_options_error_(
        KERNEL_SOURCE, options, None
    )
    if library is None:
        raise RuntimeError(f"Kernel compile failed: {err}")
    fn = library.newFunctionWithName_("write_tid")
    if fn is None:
        raise RuntimeError("Could not find 'write_tid' in compiled library")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(f"Pipeline creation failed: {err}")
    return pipeline


def make_sample_buffer(device, counter_set, sample_count):
    desc = Metal.MTLCounterSampleBufferDescriptor.alloc().init()
    desc.setCounterSet_(counter_set)
    desc.setSampleCount_(sample_count)
    desc.setStorageMode_(Metal.MTLStorageModeShared)
    desc.setLabel_("exp001-stage-boundary")
    buf, err = device.newCounterSampleBufferWithDescriptor_error_(desc, None)
    if buf is None:
        raise RuntimeError(f"Counter sample buffer alloc failed: {err}")
    return buf


# Allocating a fresh sample buffer per dispatch failed after ~30 iterations
# with MTLCounterErrorDomain Code=0 "Cannot allocate sample buffer". Use one
# shared buffer with 2 slots per dispatch instead — the documented pattern.
def dispatch_once(queue, pipeline, out_buffer, sample_buffer, slot_pair):
    start_idx, end_idx = slot_pair
    pass_desc = Metal.MTLComputePassDescriptor.computePassDescriptor()
    att = pass_desc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
    att.setSampleBuffer_(sample_buffer)
    att.setStartOfEncoderSampleIndex_(start_idx)
    att.setEndOfEncoderSampleIndex_(end_idx)

    cb = queue.commandBuffer()

    cpu_t0 = time.perf_counter_ns()
    encoder = cb.computeCommandEncoderWithDescriptor_(pass_desc)
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(THREADS_PER_GRID, 1, 1),
        Metal.MTLSizeMake(THREADS_PER_GROUP, 1, 1),
    )
    encoder.endEncoding()
    cpu_t1 = time.perf_counter_ns()
    cb.commit()
    cpu_t2 = time.perf_counter_ns()
    cb.waitUntilCompleted()
    cpu_t3 = time.perf_counter_ns()

    return {
        "cpu_encode_ns": cpu_t1 - cpu_t0,
        "cpu_commit_ns": cpu_t2 - cpu_t1,
        "cpu_wait_ns": cpu_t3 - cpu_t2,
        "cpu_total_ns": cpu_t3 - cpu_t0,
    }


def resolve_pair(sample_buffer, start_idx):
    data = sample_buffer.resolveCounterRange_((start_idx, 2))
    if data is None or data.length() < 16:
        raise RuntimeError(f"resolveCounterRange({start_idx},2) returned {data}")
    raw = bytes(data)
    return (
        int.from_bytes(raw[0:8], "little", signed=False),
        int.from_bytes(raw[8:16], "little", signed=False),
    )


def run_condition(name, sleep_s, queue, pipeline, out_buffer, sample_buffer):
    rows = []
    for i in range(NUM_DISPATCHES):
        if sleep_s > 0 and i > 0:
            time.sleep(sleep_s)
        slot_pair = (2 * i, 2 * i + 1)
        cpu = dispatch_once(queue, pipeline, out_buffer, sample_buffer, slot_pair)
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot_pair[0])
        r = {
            "sample_idx": i,
            "condition": name,
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            **cpu,
        }
        rows.append(r)
        if i % 10 == 0 or i == NUM_DISPATCHES - 1:
            print(
                f"  [{name}] {i+1:3d}/{NUM_DISPATCHES}  "
                f"gpu_delta_raw={r['gpu_delta_raw']:>10d}  "
                f"cpu_wait_ns={r['cpu_wait_ns']:>10d}"
            )
    return rows


def write_csv(path, rows):
    fieldnames = [
        "sample_idx", "condition",
        "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
        "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def percentile(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def summarize(name, rows):
    d = [r["gpu_delta_raw"] for r in rows]
    w = [r["cpu_wait_ns"] for r in rows]
    print(f"\n  {name} (N={len(d)})")
    print(f"    gpu_delta_raw: min={min(d)}  med={percentile(d,50):.0f}  "
          f"p95={percentile(d,95):.0f}  max={max(d)}")
    print(f"    cpu_wait_ns:   min={min(w)}  med={percentile(w,50):.0f}  "
          f"p95={percentile(w,95):.0f}  max={max(w)}")


def main():
    RAW_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    print("=" * 72)
    print("Experiment 001: Can we time anything from Python via PyObjC?")
    print("=" * 72)

    qos_result = set_user_interactive_qos()
    print(f"QoS: {qos_result}")
    print(f"Power: {power_source()}")

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("MTLCreateSystemDefaultDevice returned nil")
    queue = device.newCommandQueue()
    print(f"Device: {device.name()}  registryID={device.registryID()}")
    print()

    sampling_points = [
        ("atStageBoundary", "MTLCounterSamplingPointAtStageBoundary"),
        ("atDrawBoundary", "MTLCounterSamplingPointAtDrawBoundary"),
        ("atBlitBoundary", "MTLCounterSamplingPointAtBlitBoundary"),
        ("atDispatchBoundary", "MTLCounterSamplingPointAtDispatchBoundary"),
        ("atTileDispatchBoundary", "MTLCounterSamplingPointAtTileDispatchBoundary"),
    ]
    print("supportsCounterSampling:")
    for label, attr in sampling_points:
        if not hasattr(Metal, attr):
            print(f"  {label:24s}  <constant {attr} not in PyObjC>")
            continue
        v = device.supportsCounterSampling_(getattr(Metal, attr))
        print(f"  {label:24s}  {v}")
    print()

    counter_set = find_timestamp_counter_set(device)
    print(f"timestamp counter set name={counter_set.name()}  "
          f"counters={[str(c.name()) for c in counter_set.counters()]}")

    pipeline = build_pipeline(device)
    out_buffer = device.newBufferWithLength_options_(
        4 * THREADS_PER_GRID, Metal.MTLResourceStorageModeShared
    )
    print(f"pipeline.threadExecutionWidth={pipeline.threadExecutionWidth()}  "
          f"maxTotalThreadsPerThreadgroup={pipeline.maxTotalThreadsPerThreadgroup()}")
    print()

    cpu_ts_a, gpu_ts_a = device.sampleTimestamps_gpuTimestamp_(None, None)
    print(f"timestamp correlation (start): cpu={cpu_ts_a}  gpu={gpu_ts_a}")

    print("\nCondition A: 100 dispatches back-to-back")
    sample_buffer_a = make_sample_buffer(device, counter_set, 2 * NUM_DISPATCHES)
    rows_a = run_condition("backtoback", 0.0, queue, pipeline,
                           out_buffer, sample_buffer_a)
    csv_a = RAW_DIR / f"{ts}-backtoback.csv"
    write_csv(csv_a, rows_a)
    print(f"  wrote {csv_a.name}")
    summarize("backtoback", rows_a)

    print("\nCondition B: 100 dispatches with 1s sleep between (~100s)")
    sample_buffer_b = make_sample_buffer(device, counter_set, 2 * NUM_DISPATCHES)
    rows_b = run_condition("spaced1s", 1.0, queue, pipeline,
                           out_buffer, sample_buffer_b)
    csv_b = RAW_DIR / f"{ts}-spaced1s.csv"
    write_csv(csv_b, rows_b)
    print(f"  wrote {csv_b.name}")
    summarize("spaced1s", rows_b)

    cpu_ts_b, gpu_ts_b = device.sampleTimestamps_gpuTimestamp_(None, None)
    print(f"\ntimestamp correlation (end):   cpu={cpu_ts_b}  gpu={gpu_ts_b}")
    cpu_dt = cpu_ts_b - cpu_ts_a
    gpu_dt = gpu_ts_b - gpu_ts_a
    print(f"elapsed cpu_ticks={cpu_dt}  gpu_ticks={gpu_dt}  "
          f"ratio_gpu/cpu={(gpu_dt / cpu_dt) if cpu_dt else float('nan'):.6f}")

    meta = RAW_DIR / f"{ts}-meta.txt"
    meta.write_text(
        "experiment: 001-can-we-time-anything\n"
        f"timestamp: {ts}\n"
        f"device: {device.name()}\n"
        f"registry_id: {device.registryID()}\n"
        f"architecture: {device.architecture().name() if hasattr(device, 'architecture') and device.architecture() else '<unavailable>'}\n"
        f"chip: {platform.processor()}  machine: {platform.machine()}\n"
        f"os: {platform.platform()}\n"
        f"python: {sys.version.splitlines()[0]}\n"
        f"pyobjc: {objc.__version__}\n"
        f"qos: {qos_result}\n"
        f"power: {power_source()}\n"
        f"num_dispatches: {NUM_DISPATCHES}\n"
        f"threads_per_grid: {THREADS_PER_GRID}\n"
        f"threads_per_threadgroup: {THREADS_PER_GROUP}\n"
        f"thread_execution_width: {pipeline.threadExecutionWidth()}\n"
        "kernel: write_tid (out[tid] = tid)\n"
        "sampling_point: MTLCounterSamplingPointAtStageBoundary\n"
        "counter_set: timestamp\n"
        f"correlation_start: cpu={cpu_ts_a} gpu={gpu_ts_a}\n"
        f"correlation_end:   cpu={cpu_ts_b} gpu={gpu_ts_b}\n"
    )
    print(f"\nwrote {meta.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
