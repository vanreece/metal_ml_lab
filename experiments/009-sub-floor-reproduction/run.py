# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 009 outer driver: M4 Max sub-floor reproduction.

Loops 5 attempts, each as a fresh subprocess of attempt.py. ioreport.py
runs as a sibling subprocess across the whole driver so we have power
telemetry covering all attempts. caffeinate keeps the display awake.

Per CLAUDE.md / 003-008 conventions: no warmup, no retries, no averaging
in live output. If anything fails, raise loudly and preserve partial
data.
"""
from __future__ import annotations

import datetime as dt
import platform
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import Metal
import objc


N_ATTEMPTS = 5
INTER_ATTEMPT_SLEEP_S = 2.0   # short; attempt.py also has its own 5s cooldown
IOREPORT_INTERVAL_MS = 500

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"
ATTEMPT_SCRIPT = EXPERIMENT_DIR / "attempt.py"
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
IOREPORT_SCRIPT = PROJECT_ROOT / "notes" / "ioreport.py"


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


def start_caffeinate():
    proc = subprocess.Popen(
        ["caffeinate", "-d", "-i", "-m"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    if proc.poll() is not None:
        return proc, f"caffeinate exited immediately rc={proc.returncode}"
    return proc, f"started pid={proc.pid}"


def stop_caffeinate(proc):
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    return f"stopped rc={proc.returncode}"


def percentile(xs, p):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def main():
    RAW_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    attempts_csv = RAW_DIR / f"{ts}-attempts.csv"
    ioreport_csv = RAW_DIR / f"{ts}-ioreport.csv"
    ioreport_log = RAW_DIR / f"{ts}-ioreport-stdout.log"
    meta_path = RAW_DIR / f"{ts}-meta.txt"

    print("=" * 78)
    print("Experiment 009: M4 Max sub-floor reproduction")
    print("=" * 78)

    # Probe device once for the metadata; attempt.py creates its own.
    device = Metal.MTLCreateSystemDefaultDevice()
    arch = device.architecture().name() if (
        hasattr(device, "architecture") and device.architecture()
    ) else "<unavailable>"
    print(f"device:  {device.name()}  arch: {arch}")
    print(f"OS:      {platform.platform()}")
    print(f"power:   {power_source()}")
    print(f"display: {display_powerstate()}")
    print(f"attempts CSV: {attempts_csv.name}")
    print(f"ioreport CSV: {ioreport_csv.name}")
    # Drop the device reference; attempt.py owns the working device.
    del device

    if not ATTEMPT_SCRIPT.exists():
        print(f"ERROR: attempt.py not found at {ATTEMPT_SCRIPT}", file=sys.stderr)
        return 2
    if not IOREPORT_SCRIPT.exists():
        print(f"ERROR: ioreport.py not found at {IOREPORT_SCRIPT}", file=sys.stderr)
        return 2

    caffeinate_proc, caffeinate_status = start_caffeinate()
    print(f"caffeinate: {caffeinate_status}")

    ioreport_proc = subprocess.Popen(
        ["uv", "run", str(IOREPORT_SCRIPT),
         "--interval-ms", str(IOREPORT_INTERVAL_MS),
         "--csv", str(ioreport_csv)],
        stdout=open(ioreport_log, "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.5)
    if ioreport_proc.poll() is not None:
        print(f"ERROR: ioreport.py exited rc={ioreport_proc.returncode}; "
              f"see {ioreport_log}", file=sys.stderr)
        stop_caffeinate(caffeinate_proc)
        return 3
    if not ioreport_csv.exists():
        print(f"ERROR: ioreport.py did not create CSV {ioreport_csv}",
              file=sys.stderr)
        ioreport_proc.terminate()
        stop_caffeinate(caffeinate_proc)
        return 3
    print(f"ioreport.py: started pid={ioreport_proc.pid}")

    attempt_pids = []
    attempt_rcs = []
    run_start_ns = time.monotonic_ns()
    try:
        for attempt_idx in range(N_ATTEMPTS):
            print()
            print(f"--- attempt {attempt_idx} ---")
            cmd = [
                "uv", "run", str(ATTEMPT_SCRIPT),
                "--attempt-idx", str(attempt_idx),
                "--output-csv", str(attempts_csv),
            ]
            if attempt_idx > 0:
                cmd.append("--append")
            t0 = time.monotonic()
            proc = subprocess.run(cmd, check=False)
            t1 = time.monotonic()
            attempt_pids.append("<see attempt log>")  # subprocess.run -> we lose pid; attempt.py prints it
            attempt_rcs.append(proc.returncode)
            print(f"[driver] attempt {attempt_idx} exited rc={proc.returncode} "
                  f"in {t1 - t0:.1f}s")
            if proc.returncode != 0:
                print(f"[driver] ABORT: attempt {attempt_idx} failed", file=sys.stderr)
                break
            if attempt_idx < N_ATTEMPTS - 1:
                time.sleep(INTER_ATTEMPT_SLEEP_S)
    finally:
        run_end_ns = time.monotonic_ns()
        print()
        print("[driver] stopping ioreport.py (SIGINT)")
        ioreport_proc.send_signal(signal.SIGINT)
        try:
            ioreport_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            print("[driver] ioreport.py did not exit in 5s, terminating")
            ioreport_proc.terminate()
            ioreport_proc.wait(timeout=3.0)
        print(f"[driver] ioreport.py rc={ioreport_proc.returncode}")
        print(f"[driver] {stop_caffeinate(caffeinate_proc)}")

    # Per-attempt summary from the combined CSV
    summary_lines = []
    if attempts_csv.exists():
        import csv as _csv
        per_attempt: dict[int, list[int]] = {}
        per_attempt_cal: dict[int, list[int]] = {}
        with attempts_csv.open() as f:
            r = _csv.DictReader(f)
            for row in r:
                a = int(row["attempt_idx"])
                d = int(row["gpu_delta_raw"])
                if row["phase"] == "measured":
                    per_attempt.setdefault(a, []).append(d)
                else:
                    per_attempt_cal.setdefault(a, []).append(d)
        summary_lines.append(
            "attempt | cal_first | cal_p50_rest | meas_min | meas_p50 | "
            "meas_p95 | meas_max | meas_cv | below_5500 | first_sub_idx"
        )
        for a in sorted(per_attempt):
            ds = per_attempt[a]
            cs = per_attempt_cal.get(a, [])
            cal_first = cs[0] if cs else "-"
            cal_p50 = percentile(cs[1:], 50) if len(cs) > 1 else float("nan")
            p50 = percentile(ds, 50)
            p95 = percentile(ds, 95)
            stdev = statistics.stdev(ds) if len(ds) > 1 else float("nan")
            cv = (stdev / p50) if p50 else float("nan")
            below = sum(1 for d in ds if d < 5500)
            first_sub = next((i for i, d in enumerate(ds) if d < 5500), -1)
            summary_lines.append(
                f"      {a} | {cal_first:>9} | {cal_p50:>12.0f} | "
                f"{min(ds):>8} | {p50:>8.0f} | {p95:>8.0f} | {max(ds):>8} | "
                f"{cv:>7.4f} | {below:>10} | {first_sub:>13}"
            )

    # Verdict
    if attempts_csv.exists():
        attempts_with_subfloor = sum(
            1 for ds in per_attempt.values()
            if sum(1 for d in ds if d < 5500) >= 10
        )
        if attempts_with_subfloor >= 3:
            verdict = f"STRONG REPRODUCE ({attempts_with_subfloor}/{N_ATTEMPTS})"
        elif attempts_with_subfloor >= 1:
            verdict = f"PARTIAL REPRODUCE ({attempts_with_subfloor}/{N_ATTEMPTS})"
        else:
            verdict = f"NOT REPRODUCED (0/{N_ATTEMPTS})"
    else:
        verdict = "NO DATA (CSV missing)"

    elapsed_s = (run_end_ns - run_start_ns) / 1e9

    meta_lines = [
        "experiment: 009-sub-floor-reproduction",
        f"timestamp: {ts}",
        f"os: {platform.platform()}",
        f"python: {sys.version.splitlines()[0]}",
        f"pyobjc: {objc.__version__}",
        f"power: {power_source()}",
        f"display_powerstate_end: {display_powerstate()}",
        f"caffeinate: {caffeinate_status}",
        f"ioreport_pid: {ioreport_proc.pid}  rc={ioreport_proc.returncode}",
        f"ioreport_csv: {ioreport_csv.name}",
        f"attempts_csv: {attempts_csv.name}",
        f"attempt_rcs: {attempt_rcs}",
        f"n_attempts: {N_ATTEMPTS}",
        f"inter_attempt_sleep_s: {INTER_ATTEMPT_SLEEP_S}",
        f"ioreport_interval_ms: {IOREPORT_INTERVAL_MS}",
        f"recipe: write_tid 32t measured, fma_loop K=20 sleep_0 warmup, "
        f"FMA_ITERS=1024",
        f"per_attempt_n_trials: 84  per_attempt_calibration: 10  "
        f"per_attempt_cooldown_s: 5.0",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {elapsed_s:.2f}",
        "",
        "per-attempt summary (gpu_delta_raw in ns):",
        *summary_lines,
        "",
        f"verdict: {verdict}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print()
    print("\n".join(summary_lines))
    print()
    print(f"verdict: {verdict}")
    print(f"wrote {meta_path.name}")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
