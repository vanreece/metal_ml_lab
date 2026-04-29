# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 009 inner attempt: one fresh-process attempt at the
M4 Max sub-floor reproduction.

Per attempt:
  1. Fresh MTLDevice / queue / pipelines / sample buffer.
  2. Cooldown (sleep 5 s).
  3. Calibration probe: 10 sleep_0 dispatches of write_tid 32t (timed).
  4. 84 trials of: 20 fma_loop K=1024 untimed warmup dispatches, then
     1 write_tid 32t timed dispatch. sleep_0 between trials.
  5. Append rows to the CSV given as --output-csv.

The outer driver (run.py) launches this script as a subprocess once per
attempt. Subprocess separation gives a fresh MTLDevice while the chip
itself retains thermal / DVFS state across attempts.

Per CLAUDE.md conventions: no warmup beyond explicit K, no retries,
no averaging in live output.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import os
import sys
import time
from pathlib import Path

import Metal
import objc


WRITE_TID_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void write_tid(device uint *out [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
"""

FMA_LOOP_SOURCE = """
#include <metal_stdlib>
using namespace metal;

constant int FMA_ITERS = 1024;

kernel void fma_loop(device float *out [[buffer(0)]],
                     uint tid [[thread_position_in_grid]]) {
    float x = float(tid) * 0.001f + 0.000001f;
    float y = 1.0f;
    for (int i = 0; i < FMA_ITERS; i++) {
        y = fma(y, x, x);
    }
    out[tid] = y;
}
"""

# Measured kernel = same write_tid 32t as 001-003.
MEASURED_THREADS = 32
MEASURED_GROUP = 32

# fma_loop warmup kernel = 32 threads, FMA_ITERS=1024 (matches exp 003).
WARMUP_THREADS = 32
WARMUP_GROUP = 32

# Recipe constants.
K = 20                  # warmup dispatches per trial
N_TRIALS = 84           # measured trials per attempt
CAL_BURST = 10          # calibration probe dispatches per attempt
COOLDOWN_S = 5.0        # sleep before calibration probe

# Sample buffer must hold (CAL_BURST + N_TRIALS) start/end pairs.
SAMPLE_BUFFER_SLOTS = 2 * (CAL_BURST + N_TRIALS)
OUT_BUFFER_BYTES = 4 * 1024  # writes from up to 1024-thread fma_loop, comfortably


def set_user_interactive_qos() -> str:
    QOS_CLASS_USER_INTERACTIVE = 0x21
    libsystem = ctypes.CDLL("/usr/lib/libSystem.dylib")
    rc = libsystem.pthread_set_qos_class_self_np(
        ctypes.c_int(QOS_CLASS_USER_INTERACTIVE), ctypes.c_int(0)
    )
    return f"pthread_set_qos_class_self_np -> {rc}"


def find_timestamp_counter_set(device):
    sets = device.counterSets()
    if sets is None:
        raise RuntimeError("device.counterSets() returned None")
    for cs in sets:
        if str(cs.name()) == "timestamp":
            return cs
    raise RuntimeError(
        f"No 'timestamp' counter set; have {[str(s.name()) for s in sets]}"
    )


def build_pipeline(device, source: str, fn_name: str):
    options = Metal.MTLCompileOptions.alloc().init()
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None:
        raise RuntimeError(f"Kernel compile failed for {fn_name}: {err}")
    fn = library.newFunctionWithName_(fn_name)
    if fn is None:
        raise RuntimeError(f"Function {fn_name} not in library")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(f"Pipeline creation failed for {fn_name}: {err}")
    return pipeline


def make_sample_buffer(device, counter_set, sample_count, label):
    desc = Metal.MTLCounterSampleBufferDescriptor.alloc().init()
    desc.setCounterSet_(counter_set)
    desc.setSampleCount_(sample_count)
    desc.setStorageMode_(Metal.MTLStorageModeShared)
    desc.setLabel_(label)
    buf, err = device.newCounterSampleBufferWithDescriptor_error_(desc, None)
    if buf is None:
        raise RuntimeError(f"Counter sample buffer alloc failed: {err}")
    return buf


def dispatch_untimed(queue, pipeline, out_buffer, threads, group):
    cb = queue.commandBuffer()
    encoder = cb.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(threads, 1, 1),
        Metal.MTLSizeMake(group, 1, 1),
    )
    encoder.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()


def dispatch_timed(queue, pipeline, out_buffer, threads, group,
                   sample_buffer, slot_pair):
    start_idx, end_idx = slot_pair
    pass_desc = Metal.MTLComputePassDescriptor.computePassDescriptor()
    att = pass_desc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
    att.setSampleBuffer_(sample_buffer)
    att.setStartOfEncoderSampleIndex_(start_idx)
    att.setEndOfEncoderSampleIndex_(end_idx)

    cb = queue.commandBuffer()
    wall_clock_ns = time.monotonic_ns()
    cpu_t0 = time.perf_counter_ns()
    encoder = cb.computeCommandEncoderWithDescriptor_(pass_desc)
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(threads, 1, 1),
        Metal.MTLSizeMake(group, 1, 1),
    )
    encoder.endEncoding()
    cpu_t1 = time.perf_counter_ns()
    cb.commit()
    cpu_t2 = time.perf_counter_ns()
    cb.waitUntilCompleted()
    cpu_t3 = time.perf_counter_ns()

    return {
        "wall_clock_ns": wall_clock_ns,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt-idx", type=int, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--append", action="store_true",
                    help="append to output CSV (driver passes after attempt 0)")
    args = ap.parse_args()

    qos_result = set_user_interactive_qos()
    pid = os.getpid()
    print(f"[attempt {args.attempt_idx}] pid={pid} qos={qos_result}")

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("MTLCreateSystemDefaultDevice returned nil")
    queue = device.newCommandQueue()
    counter_set = find_timestamp_counter_set(device)
    write_tid_pipe = build_pipeline(device, WRITE_TID_SOURCE, "write_tid")
    fma_loop_pipe = build_pipeline(device, FMA_LOOP_SOURCE, "fma_loop")
    out_buffer = device.newBufferWithLength_options_(
        OUT_BUFFER_BYTES, Metal.MTLResourceStorageModeShared
    )
    sample_buffer = make_sample_buffer(
        device, counter_set, SAMPLE_BUFFER_SLOTS, f"exp009-attempt-{args.attempt_idx}"
    )
    arch = device.architecture().name() if (
        hasattr(device, "architecture") and device.architecture()
    ) else "<unavailable>"
    print(f"[attempt {args.attempt_idx}] device={device.name()} arch={arch}")

    # Cooldown so the chip can drop out of any prior elevated state.
    print(f"[attempt {args.attempt_idx}] cooldown {COOLDOWN_S:.1f}s ...")
    time.sleep(COOLDOWN_S)

    # Calibration probe (slots 0..2*CAL_BURST-1)
    cal_rows = []
    for i in range(CAL_BURST):
        slot = (2 * i, 2 * i + 1)
        cpu = dispatch_timed(
            queue, write_tid_pipe, out_buffer,
            MEASURED_THREADS, MEASURED_GROUP, sample_buffer, slot,
        )
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot[0])
        cal_rows.append({
            "attempt_idx": args.attempt_idx,
            "phase": "calibration",
            "idx_within_phase": i,
            "attempt_pid": pid,
            "wall_clock_ns": cpu["wall_clock_ns"],
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            "cpu_encode_ns": cpu["cpu_encode_ns"],
            "cpu_commit_ns": cpu["cpu_commit_ns"],
            "cpu_wait_ns": cpu["cpu_wait_ns"],
            "cpu_total_ns": cpu["cpu_total_ns"],
        })
    cal_first = cal_rows[0]["gpu_delta_raw"]
    rest = sorted(r["gpu_delta_raw"] for r in cal_rows[1:])
    cal_p50 = rest[len(rest) // 2]
    print(f"[attempt {args.attempt_idx}] cal_first={cal_first} "
          f"cal_p50_rest={cal_p50}")

    # Measured trials (slots 2*CAL_BURST .. 2*(CAL_BURST + N_TRIALS) - 1)
    meas_rows = []
    measured_slot_offset = 2 * CAL_BURST
    for trial_idx in range(N_TRIALS):
        for _ in range(K):
            dispatch_untimed(queue, fma_loop_pipe, out_buffer,
                             WARMUP_THREADS, WARMUP_GROUP)
        slot = (
            measured_slot_offset + 2 * trial_idx,
            measured_slot_offset + 2 * trial_idx + 1,
        )
        cpu = dispatch_timed(
            queue, write_tid_pipe, out_buffer,
            MEASURED_THREADS, MEASURED_GROUP, sample_buffer, slot,
        )
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot[0])
        meas_rows.append({
            "attempt_idx": args.attempt_idx,
            "phase": "measured",
            "idx_within_phase": trial_idx,
            "attempt_pid": pid,
            "wall_clock_ns": cpu["wall_clock_ns"],
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            "cpu_encode_ns": cpu["cpu_encode_ns"],
            "cpu_commit_ns": cpu["cpu_commit_ns"],
            "cpu_wait_ns": cpu["cpu_wait_ns"],
            "cpu_total_ns": cpu["cpu_total_ns"],
        })
    deltas = [r["gpu_delta_raw"] for r in meas_rows]
    sorted_d = sorted(deltas)
    p50 = sorted_d[len(sorted_d) // 2]
    sub_floor = [i for i, d in enumerate(deltas) if d < 5500]
    first_sub = sub_floor[0] if sub_floor else -1
    print(f"[attempt {args.attempt_idx}] N={len(deltas)} min={min(deltas)} "
          f"p50={p50} max={max(deltas)} below_5500={len(sub_floor)} "
          f"first_sub_floor_idx={first_sub}")

    # Write CSV
    fieldnames = [
        "attempt_idx", "phase", "idx_within_phase", "attempt_pid",
        "wall_clock_ns",
        "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
        "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
    ]
    mode = "a" if args.append else "w"
    write_header = not args.append
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in cal_rows + meas_rows:
            w.writerow(r)
    print(f"[attempt {args.attempt_idx}] wrote "
          f"{len(cal_rows) + len(meas_rows)} rows to {args.output_csv.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
