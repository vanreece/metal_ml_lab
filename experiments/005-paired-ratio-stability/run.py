# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 005: Paired-kernel ratio stability.

Pre-registered. See README.md.

Per CLAUDE.md and the experiment README:
- No averaging in live output, no swallowed exceptions, no retries.
- Raw values to CSV under raw/ with a timestamp prefix.
- 9 conditions per sweep:
    ref_alone, T1_alone, T1_paired, T2_alone, T2_paired,
    T3_alone, T3_paired, T4_alone, T4_paired
- 3 sweeps with 30s between-sweep cooldown.
- 2s per-condition cooldown, sleep_0 within a condition.
- K=1 untimed warmup of the trial kernel before each measured trial.
  No warmup of the reference (per README: ref's first dispatch in
  the paired condition sees a "fresh" pipeline of the kind it'll
  see in real use).
- Paired condition: ONE MTLCommandBuffer with TWO sequential
  MTLComputeCommandEncoders, each with its own
  MTLComputePassDescriptor + sampleBufferAttachments. Four
  timestamps per measured trial. Compute ratio = trial_delta /
  ref_delta and gap_ns = trial_t_start - ref_t_end.
- caffeinate sidecar holds display awake throughout.
- powermetrics gated on EXP005_NO_POWERMETRICS env var (default
  attempts sudo -n; set to non-empty to skip).
"""
from __future__ import annotations

import csv
import ctypes
import datetime as dt
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import Metal
import objc


# Kernel sources

WRITE_TID_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void write_tid(device uint *out [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
"""


def fma_loop_source(iters: int) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;
constant int FMA_ITERS = {iters};
kernel void fma_loop(device float *out [[buffer(0)]],
                     uint tid [[thread_position_in_grid]]) {{
    float x = float(tid) * 0.001f + 0.000001f;
    float y = 1.0f;
    for (int i = 0; i < FMA_ITERS; i++) {{
        y = fma(y, x, x);
    }}
    out[tid] = y;
}}
"""


# Reference kernel (fixed)
REF_FMA_ITERS = 1024
REF_THREADS = 32

# Trial kernels
# Each trial is (kind, params): kind in {"fma_loop", "write_tid"}
# params is (fma_iters, threads) for fma_loop, or (threads,) for write_tid
TRIALS = {
    "T1": ("fma_loop", {"fma_iters": 512,  "threads": 32}),
    "T2": ("fma_loop", {"fma_iters": 4096, "threads": 32}),
    "T3": ("write_tid", {"threads": 524288}),
    "T4": ("write_tid", {"threads": 1048576}),
}

THREADGROUP_SIZE = 32
N_PER_COMBO = 300
N_SWEEPS = 3
PER_COMBO_COOLDOWN_S = 2.0
BETWEEN_SWEEP_COOLDOWN_S = 30.0

# Sample buffer slot budget per condition:
#   alone: 2 slots * 300 = 600
#   paired: 4 slots * 300 = 1200
# Largest single combo therefore needs 1200 slots. Allocate exactly
# that and reuse across combos (writing over previous data; no need
# to retain).
SAMPLE_BUFFER_SLOTS = 4 * N_PER_COMBO

# Output buffer: largest dispatch is write_tid 1M threads writing uint
# = 4 MiB. fma_loop writes only 32 floats = 128 B.
OUT_BUFFER_BYTES = 4 * 1048576

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"


# Environment / sidecars

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


def display_powerstate() -> str:
    out = subprocess.run(
        ["pmset", "-g", "powerstate", "IODisplayWrangler"],
        capture_output=True, text=True, check=False,
    )
    return out.stdout.strip() if out.stdout else "<unavailable>"


def pmset_assertions() -> str:
    out = subprocess.run(
        ["pmset", "-g", "assertions"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.stdout else "<unavailable>"


def start_caffeinate() -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        ["caffeinate", "-d", "-i", "-m"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    if proc.poll() is not None:
        return proc, f"caffeinate exited immediately rc={proc.returncode}"
    return proc, f"started pid={proc.pid}"


def stop_caffeinate(proc: subprocess.Popen) -> str:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    return f"stopped rc={proc.returncode}"


def maybe_start_powermetrics(out_path: Path) -> tuple[subprocess.Popen | None, str]:
    if os.environ.get("EXP005_NO_POWERMETRICS"):
        return None, "skipped (EXP005_NO_POWERMETRICS set)"
    check = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if check.returncode != 0:
        return None, "sudo -n failed; powermetrics skipped"
    f = open(out_path, "w")
    cmd = ["sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-i", "200"]
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.5)
    if proc.poll() is not None:
        return None, f"powermetrics exited immediately rc={proc.returncode}"
    return proc, f"started pid={proc.pid} -> {out_path.name}"


def stop_powermetrics(proc: subprocess.Popen | None) -> str:
    if proc is None:
        return "no sidecar"
    subprocess.run(["sudo", "-n", "killall", "-INT", "powermetrics"],
                   capture_output=True)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        subprocess.run(["sudo", "-n", "killall", "-KILL", "powermetrics"],
                       capture_output=True)
        proc.wait(timeout=2)
    return f"stopped rc={proc.returncode}"


# Metal setup

def find_timestamp_counter_set(device):
    sets = device.counterSets()
    if sets is None:
        raise RuntimeError("device.counterSets() returned None")
    for cs in sets:
        if str(cs.name()) == "timestamp":
            return cs
    raise RuntimeError(f"No 'timestamp' counter set; have {[str(s.name()) for s in sets]}")


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


# Dispatching

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


def dispatch_alone_timed(queue, pipeline, out_buffer, threads, group,
                          sample_buffer, slot_pair):
    """Single timed dispatch: one compute pass, two timestamps."""
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


def dispatch_paired_timed(queue, ref_pipeline, ref_threads, ref_group,
                           trial_pipeline, trial_threads, trial_group,
                           out_buffer, sample_buffer, slot_quad):
    """Paired timed dispatch: ONE command buffer, TWO compute passes
    (ref, then trial), each pass with its own timestamp pair.
    Returns CPU timings spanning the entire encode->commit->wait
    cycle for the whole command buffer.
    """
    ref_start, ref_end, trial_start, trial_end = slot_quad

    pd_ref = Metal.MTLComputePassDescriptor.computePassDescriptor()
    att_ref = pd_ref.sampleBufferAttachments().objectAtIndexedSubscript_(0)
    att_ref.setSampleBuffer_(sample_buffer)
    att_ref.setStartOfEncoderSampleIndex_(ref_start)
    att_ref.setEndOfEncoderSampleIndex_(ref_end)

    pd_trial = Metal.MTLComputePassDescriptor.computePassDescriptor()
    att_trial = pd_trial.sampleBufferAttachments().objectAtIndexedSubscript_(0)
    att_trial.setSampleBuffer_(sample_buffer)
    att_trial.setStartOfEncoderSampleIndex_(trial_start)
    att_trial.setEndOfEncoderSampleIndex_(trial_end)

    cb = queue.commandBuffer()

    wall_clock_ns = time.monotonic_ns()
    cpu_t0 = time.perf_counter_ns()
    enc_ref = cb.computeCommandEncoderWithDescriptor_(pd_ref)
    enc_ref.setComputePipelineState_(ref_pipeline)
    enc_ref.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    enc_ref.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(ref_threads, 1, 1),
        Metal.MTLSizeMake(ref_group, 1, 1),
    )
    enc_ref.endEncoding()
    enc_trial = cb.computeCommandEncoderWithDescriptor_(pd_trial)
    enc_trial.setComputePipelineState_(trial_pipeline)
    enc_trial.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    enc_trial.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(trial_threads, 1, 1),
        Metal.MTLSizeMake(trial_group, 1, 1),
    )
    enc_trial.endEncoding()
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


# Combo runners

def run_alone_combo(queue, trial_pipeline, trial_threads, trial_group,
                    out_buffer, sample_buffer, n_trials,
                    do_warmup: bool):
    """N trials of (optional K=1 untimed warmup, 1 alone-timed dispatch)."""
    rows = []
    for trial_idx in range(n_trials):
        if do_warmup:
            dispatch_untimed(queue, trial_pipeline, out_buffer,
                             trial_threads, trial_group)
        slot_pair = (2 * trial_idx, 2 * trial_idx + 1)
        cpu = dispatch_alone_timed(
            queue, trial_pipeline, out_buffer,
            trial_threads, trial_group, sample_buffer, slot_pair,
        )
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot_pair[0])
        rows.append({
            "trial_idx_within_combo": trial_idx,
            "wall_clock_ns": cpu["wall_clock_ns"],
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            "cpu_encode_ns": cpu["cpu_encode_ns"],
            "cpu_commit_ns": cpu["cpu_commit_ns"],
            "cpu_wait_ns": cpu["cpu_wait_ns"],
            "cpu_total_ns": cpu["cpu_total_ns"],
        })
    return rows


def run_paired_combo(queue, ref_pipeline,
                     trial_pipeline, trial_threads, trial_group,
                     out_buffer, sample_buffer, n_trials):
    """N trials of (K=1 untimed warmup of trial only, then paired
    [ref, trial] in one command buffer). The reference is fma_loop
    iters=1024 at 32t; pipeline + thread params for ref are passed
    as ref_pipeline (threads/group are the fixed REF_THREADS /
    THREADGROUP_SIZE constants)."""
    rows = []
    for trial_idx in range(n_trials):
        # K=1 untimed warmup of TRIAL only (not ref) per README design
        dispatch_untimed(queue, trial_pipeline, out_buffer,
                         trial_threads, trial_group)
        slot_quad = (
            4 * trial_idx,
            4 * trial_idx + 1,
            4 * trial_idx + 2,
            4 * trial_idx + 3,
        )
        cpu = dispatch_paired_timed(
            queue,
            ref_pipeline, REF_THREADS, THREADGROUP_SIZE,
            trial_pipeline, trial_threads, trial_group,
            out_buffer, sample_buffer, slot_quad,
        )
        ref_start, ref_end = resolve_pair(sample_buffer, slot_quad[0])
        trial_start, trial_end = resolve_pair(sample_buffer, slot_quad[2])
        rows.append({
            "trial_idx_within_combo": trial_idx,
            "wall_clock_ns": cpu["wall_clock_ns"],
            "ref_t_start_raw": ref_start,
            "ref_t_end_raw": ref_end,
            "ref_delta_raw": ref_end - ref_start,
            "trial_t_start_raw": trial_start,
            "trial_t_end_raw": trial_end,
            "trial_delta_raw": trial_end - trial_start,
            "gap_ns": trial_start - ref_end,
            "ratio": (trial_end - trial_start) / (ref_end - ref_start)
                     if (ref_end - ref_start) > 0 else float("nan"),
            "cpu_encode_ns": cpu["cpu_encode_ns"],
            "cpu_commit_ns": cpu["cpu_commit_ns"],
            "cpu_wait_ns": cpu["cpu_wait_ns"],
            "cpu_total_ns": cpu["cpu_total_ns"],
        })
    return rows


# Summary helpers

def percentile(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def stddev(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def robust_cv(xs):
    p50 = percentile(xs, 50)
    p25 = percentile(xs, 25)
    p75 = percentile(xs, 75)
    iqr = p75 - p25
    return (iqr / 1.349) / p50 if p50 else float("nan")


def summarize(deltas):
    p50 = percentile(deltas, 50)
    sd = stddev(deltas)
    return {
        "n": len(deltas),
        "min": min(deltas),
        "p05": percentile(deltas, 5),
        "p50": p50,
        "p95": percentile(deltas, 95),
        "p99": percentile(deltas, 99),
        "max": max(deltas),
        "stddev": sd,
        "naive_cv": (sd / p50) if p50 else float("nan"),
        "robust_cv": robust_cv(deltas),
    }


# CSV writers

ALONE_FIELDS = [
    "condition", "sweep_idx", "trial_idx_within_combo",
    "wall_clock_ns",
    "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
    "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
]
PAIRED_FIELDS = [
    "condition", "sweep_idx", "trial_idx_within_combo",
    "wall_clock_ns",
    "ref_t_start_raw", "ref_t_end_raw", "ref_delta_raw",
    "trial_t_start_raw", "trial_t_end_raw", "trial_delta_raw",
    "gap_ns", "ratio",
    "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
]


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


# Trial dispatch helper

def trial_params(trial_label):
    kind, p = TRIALS[trial_label]
    if kind == "fma_loop":
        return ("fma_loop", p["fma_iters"], p["threads"])
    elif kind == "write_tid":
        return ("write_tid", None, p["threads"])
    else:
        raise ValueError(kind)


# Main

def main():
    RAW_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    print("=" * 78)
    print("Experiment 005: paired-kernel ratio stability")
    print("=" * 78)

    qos_result = set_user_interactive_qos()
    print(f"QoS:   {qos_result}")
    print(f"Power: {power_source()}")
    display_state_start = display_powerstate()
    print(f"Display powerstate (start):\n  {display_state_start}")
    assertions_start = pmset_assertions()
    print("pmset assertions (start, first 10 lines):")
    for line in assertions_start.splitlines()[:10]:
        print(f"  {line}")

    caffeinate_proc, caffeinate_status = start_caffeinate()
    print(f"caffeinate: {caffeinate_status}")

    pm_path = RAW_DIR / f"{ts}-powermetrics.txt"
    pm_proc, pm_status = maybe_start_powermetrics(pm_path)
    print(f"powermetrics: {pm_status}")
    pm_started_wall_ns = time.monotonic_ns()

    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("MTLCreateSystemDefaultDevice returned nil")
        queue = device.newCommandQueue()
        counter_set = find_timestamp_counter_set(device)
        out_buffer = device.newBufferWithLength_options_(
            OUT_BUFFER_BYTES, Metal.MTLResourceStorageModeShared
        )
        sample_buffer = make_sample_buffer(
            device, counter_set, SAMPLE_BUFFER_SLOTS, "exp005-shared"
        )

        # Build pipelines
        write_tid_pipe = build_pipeline(device, WRITE_TID_SOURCE, "write_tid")
        # We need fma_loop at 512, 1024 (ref), 4096
        fma_levels_needed = sorted({REF_FMA_ITERS}
                                   | {p["fma_iters"]
                                      for k, p in TRIALS.values()
                                      if k == "fma_loop"})
        fma_pipes = {}
        print(f"\nCompiling fma_loop variants: {fma_levels_needed}")
        for iters in fma_levels_needed:
            fma_pipes[iters] = build_pipeline(
                device, fma_loop_source(iters), "fma_loop"
            )
        ref_pipeline = fma_pipes[REF_FMA_ITERS]

        print(f"  write_tid TEW={write_tid_pipe.threadExecutionWidth()} "
              f"maxTPG={write_tid_pipe.maxTotalThreadsPerThreadgroup()}")
        print(f"  ref kernel: fma_loop iters={REF_FMA_ITERS} "
              f"threads={REF_THREADS}")
        print(f"  trials:")
        for label, (kind, p) in TRIALS.items():
            print(f"    {label}: {kind} {p}")
        print(f"  N_per_combo={N_PER_COMBO}  N_sweeps={N_SWEEPS}")
        print(f"  per-combo cooldown={PER_COMBO_COOLDOWN_S}s  "
              f"between-sweep cooldown={BETWEEN_SWEEP_COOLDOWN_S}s")
        print(f"  sample_buffer slots={SAMPLE_BUFFER_SLOTS}, "
              f"out_buffer bytes={OUT_BUFFER_BYTES}")

        cpu_ts_start, gpu_ts_start = device.sampleTimestamps_gpuTimestamp_(
            None, None
        )
        print(f"timestamp correlation (start): cpu={cpu_ts_start} "
              f"gpu={gpu_ts_start}")

        run_start_wall = time.monotonic_ns()
        all_alone = []
        all_paired = []

        # Dispatch helper for a trial label -> (pipeline, threads, group)
        def trial_pipeline_for(label):
            kind, fma_iters, threads = trial_params(label)
            if kind == "fma_loop":
                return fma_pipes[fma_iters], threads, THREADGROUP_SIZE
            elif kind == "write_tid":
                return write_tid_pipe, threads, THREADGROUP_SIZE
            else:
                raise ValueError(kind)

        for sweep_idx in range(N_SWEEPS):
            print(f"\n{'=' * 78}\nSweep {sweep_idx+1}/{N_SWEEPS}\n{'=' * 78}")

            # ref_alone (with K=1 untimed warmup of ref)
            time.sleep(PER_COMBO_COOLDOWN_S)
            print(f"\n  ref_alone: fma_loop iters={REF_FMA_ITERS} "
                  f"threads={REF_THREADS}")
            rows = run_alone_combo(
                queue, ref_pipeline, REF_THREADS, THREADGROUP_SIZE,
                out_buffer, sample_buffer, N_PER_COMBO, do_warmup=True,
            )
            for r in rows:
                r.update({"condition": "ref_alone", "sweep_idx": sweep_idx})
            all_alone.extend(rows)
            s = summarize([r["gpu_delta_raw"] for r in rows])
            print(f"    N={s['n']} p50={s['p50']:.0f} p95={s['p95']:.0f} "
                  f"robust_cv={s['robust_cv']:.4f} "
                  f"naive_cv={s['naive_cv']:.4f}")

            # For each trial: alone, then paired
            for label in ("T1", "T2", "T3", "T4"):
                pipe, threads, group = trial_pipeline_for(label)

                # alone
                time.sleep(PER_COMBO_COOLDOWN_S)
                print(f"\n  {label}_alone: {trial_params(label)}")
                rows = run_alone_combo(
                    queue, pipe, threads, group,
                    out_buffer, sample_buffer, N_PER_COMBO, do_warmup=True,
                )
                for r in rows:
                    r.update({"condition": f"{label}_alone",
                              "sweep_idx": sweep_idx})
                all_alone.extend(rows)
                s = summarize([r["gpu_delta_raw"] for r in rows])
                print(f"    N={s['n']} p50={s['p50']:.0f} "
                      f"p95={s['p95']:.0f} "
                      f"robust_cv={s['robust_cv']:.4f} "
                      f"naive_cv={s['naive_cv']:.4f}")

                # paired
                time.sleep(PER_COMBO_COOLDOWN_S)
                print(f"  {label}_paired: ref={REF_FMA_ITERS}@{REF_THREADS}t "
                      f"+ trial={trial_params(label)}")
                rows = run_paired_combo(
                    queue, ref_pipeline,
                    pipe, threads, group,
                    out_buffer, sample_buffer, N_PER_COMBO,
                )
                for r in rows:
                    r.update({"condition": f"{label}_paired",
                              "sweep_idx": sweep_idx})
                all_paired.extend(rows)
                ratios = [r["ratio"] for r in rows]
                trial_deltas = [r["trial_delta_raw"] for r in rows]
                ref_deltas = [r["ref_delta_raw"] for r in rows]
                gaps = [r["gap_ns"] for r in rows]
                s_ratio = summarize(ratios)
                s_trial = summarize(trial_deltas)
                s_ref = summarize(ref_deltas)
                s_gap = summarize(gaps)
                print(f"    ref:   N={s_ref['n']} p50={s_ref['p50']:.0f} "
                      f"robust_cv={s_ref['robust_cv']:.4f}")
                print(f"    trial: N={s_trial['n']} "
                      f"p50={s_trial['p50']:.0f} "
                      f"robust_cv={s_trial['robust_cv']:.4f}")
                print(f"    ratio: N={s_ratio['n']} "
                      f"p50={s_ratio['p50']:.4f} "
                      f"robust_cv={s_ratio['robust_cv']:.4f}")
                print(f"    gap_ns: p50={s_gap['p50']:.0f} "
                      f"p95={s_gap['p95']:.0f} max={s_gap['max']}")

            if sweep_idx < N_SWEEPS - 1:
                print(f"\n  Between-sweep cooldown "
                      f"{BETWEEN_SWEEP_COOLDOWN_S}s ...")
                time.sleep(BETWEEN_SWEEP_COOLDOWN_S)

        run_end_wall = time.monotonic_ns()
        cpu_ts_end, gpu_ts_end = device.sampleTimestamps_gpuTimestamp_(
            None, None
        )

        # Write CSVs
        alone_csv = RAW_DIR / f"{ts}-alone.csv"
        paired_csv = RAW_DIR / f"{ts}-paired.csv"
        write_csv(alone_csv, ALONE_FIELDS, all_alone)
        write_csv(paired_csv, PAIRED_FIELDS, all_paired)
        print(f"\nwrote {alone_csv.name} ({len(all_alone)} rows)")
        print(f"wrote {paired_csv.name} ({len(all_paired)} rows)")

        cpu_dt = cpu_ts_end - cpu_ts_start
        gpu_dt = gpu_ts_end - gpu_ts_start
        ratio_clock = (gpu_dt / cpu_dt) if cpu_dt else float("nan")
        elapsed_s = (run_end_wall - run_start_wall) / 1e9
        print(f"\ntimestamp correlation (end):   cpu={cpu_ts_end} "
              f"gpu={gpu_ts_end}")
        print(f"elapsed cpu_ticks={cpu_dt}  gpu_ticks={gpu_dt}  "
              f"ratio_gpu/cpu={ratio_clock:.6f}")
        print(f"experiment wall-clock: {elapsed_s:.1f}s")

        # Per-condition per-sweep summary
        meta_lines = [
            "experiment: 005-paired-ratio-stability",
            f"timestamp: {ts}",
            f"device: {device.name()}",
            f"registry_id: {device.registryID()}",
        ]
        if hasattr(device, "architecture") and device.architecture():
            meta_lines.append(f"architecture: {device.architecture().name()}")
        meta_lines.extend([
            f"chip: {platform.processor()}  machine: {platform.machine()}",
            f"os: {platform.platform()}",
            f"python: {sys.version.splitlines()[0]}",
            f"pyobjc: {objc.__version__}",
            f"qos: {qos_result}",
            f"power: {power_source()}",
            f"powermetrics: {pm_status}",
            f"powermetrics_started_wall_ns: {pm_started_wall_ns}",
            f"caffeinate: {caffeinate_status}",
            f"display_powerstate_start: {display_state_start}",
            f"display_powerstate_end:   {display_powerstate()}",
            "pmset_assertions_start: |",
            *[f"  {line}" for line in assertions_start.splitlines()],
            f"experiment_started_wall_ns: {run_start_wall}",
            f"experiment_ended_wall_ns:   {run_end_wall}",
            f"ref_kernel: fma_loop iters={REF_FMA_ITERS} threads={REF_THREADS}",
            f"trials: {TRIALS}",
            f"threadgroup_size: {THREADGROUP_SIZE}",
            f"n_per_combo: {N_PER_COMBO}",
            f"n_sweeps: {N_SWEEPS}",
            f"per_combo_cooldown_s: {PER_COMBO_COOLDOWN_S}",
            f"between_sweep_cooldown_s: {BETWEEN_SWEEP_COOLDOWN_S}",
            "warmup_recipe: K=1 untimed dispatch of trial kernel before "
            "each measurement (ref not warmed up)",
            "sampling_point: MTLCounterSamplingPointAtStageBoundary",
            "paired_pattern: ONE MTLCommandBuffer with TWO sequential "
            "MTLComputeCommandEncoders, each with own pass descriptor + "
            "timestamp pair. Order: [ref, trial].",
            f"correlation_start: cpu={cpu_ts_start} gpu={gpu_ts_start}",
            f"correlation_end:   cpu={cpu_ts_end} gpu={gpu_ts_end}",
            f"experiment_wall_clock_s: {elapsed_s:.2f}",
            "",
            "alone summary per (condition, sweep) "
            "(min | p50 | p95 | robust_cv | naive_cv):",
        ])
        alone_grouped = defaultdict(list)
        for r in all_alone:
            alone_grouped[(r["condition"], r["sweep_idx"])].append(
                r["gpu_delta_raw"]
            )
        for (cond, sweep), deltas in sorted(alone_grouped.items()):
            s = summarize(deltas)
            meta_lines.append(
                f"  {cond:12s} sweep={sweep} "
                f"min={s['min']:>9d} p50={s['p50']:>10.0f} "
                f"p95={s['p95']:>10.0f} "
                f"robust_cv={s['robust_cv']:.4f} naive_cv={s['naive_cv']:.4f}"
            )
        meta_lines.append("")
        meta_lines.append(
            "paired summary per (condition, sweep): "
            "ref / trial / ratio / gap"
        )
        paired_grouped = defaultdict(lambda: {"ref": [], "trial": [],
                                              "ratio": [], "gap": []})
        for r in all_paired:
            key = (r["condition"], r["sweep_idx"])
            paired_grouped[key]["ref"].append(r["ref_delta_raw"])
            paired_grouped[key]["trial"].append(r["trial_delta_raw"])
            paired_grouped[key]["ratio"].append(r["ratio"])
            paired_grouped[key]["gap"].append(r["gap_ns"])
        for (cond, sweep), groups in sorted(paired_grouped.items()):
            sr = summarize(groups["ref"])
            st = summarize(groups["trial"])
            sx = summarize(groups["ratio"])
            sg = summarize(groups["gap"])
            meta_lines.append(
                f"  {cond:12s} sweep={sweep}"
            )
            meta_lines.append(
                f"    ref:    p50={sr['p50']:>10.0f} robust_cv={sr['robust_cv']:.4f}"
            )
            meta_lines.append(
                f"    trial:  p50={st['p50']:>10.0f} robust_cv={st['robust_cv']:.4f}"
            )
            meta_lines.append(
                f"    ratio:  p50={sx['p50']:>8.4f} robust_cv={sx['robust_cv']:.4f}"
            )
            meta_lines.append(
                f"    gap_ns: p50={sg['p50']:>8.0f} p95={sg['p95']:>8.0f} max={sg['max']}"
            )
        meta_path = RAW_DIR / f"{ts}-meta.txt"
        meta_path.write_text("\n".join(meta_lines) + "\n")
        print(f"wrote {meta_path.name}")
    finally:
        pm_stop_status = stop_powermetrics(pm_proc)
        print(f"\npowermetrics: {pm_stop_status}")
        caffeinate_stop_status = stop_caffeinate(caffeinate_proc)
        print(f"caffeinate: {caffeinate_stop_status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
