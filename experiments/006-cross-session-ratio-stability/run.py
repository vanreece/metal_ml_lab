# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 006: cross-session ratio stability orchestrator.

Runs experiment 005's run.py twice in two separate Python processes,
with a 30-minute idle gap between them. Records a linkage file so the
006 analysis script knows which two 005 timestamp prefixes to compare.

The actual measurement data goes into 005's raw/ directory (one
{tsA}-alone.csv, one {tsA}-paired.csv, one {tsB}-alone.csv, one
{tsB}-paired.csv). 006/raw/ holds only the linkage file and this
script's stdout log.

No GPU work happens in this process (only subprocesses). No
modification to 005's run.py.
"""
from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


SESSION_GAP_S = 1800  # 30 min, per 006/README.md pre-registration

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"

EXP005_DIR = EXPERIMENT_DIR.parent / "005-paired-ratio-stability"
EXP005_RAW = EXP005_DIR / "raw"
EXP005_RUN_PY = EXP005_DIR / "run.py"


def newest_alone_prefix() -> str | None:
    """Return the timestamp prefix of the newest *-alone.csv in 005/raw/, or None."""
    csvs = sorted(EXP005_RAW.glob("*-alone.csv"))
    if not csvs:
        return None
    # filename: 20260428T115757-alone.csv -> prefix 20260428T115757
    return csvs[-1].name.split("-alone.csv")[0]


def run_session(label: str) -> tuple[str, int, int]:
    """Launch one full 005 session as a subprocess. Return
    (timestamp_prefix, wall_start_ns, wall_end_ns)."""
    pre_prefixes = {p.name for p in EXP005_RAW.glob("*-alone.csv")}
    print(f"\n========== {label}: launching 005/run.py ==========")
    print(f"  cwd: {EXP005_DIR}")
    print(f"  cmd: caffeinate -d -i -m uv run run.py")

    start_ns = time.monotonic_ns()
    proc = subprocess.run(
        ["caffeinate", "-d", "-i", "-m", "uv", "run", "run.py"],
        cwd=str(EXP005_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    end_ns = time.monotonic_ns()

    if proc.returncode != 0:
        print(f"  rc={proc.returncode}")
        print("  stdout (tail):")
        for line in proc.stdout.splitlines()[-30:]:
            print(f"    {line}")
        print("  stderr (tail):")
        for line in proc.stderr.splitlines()[-30:]:
            print(f"    {line}")
        raise RuntimeError(f"{label} 005/run.py failed (rc={proc.returncode})")

    # Find the new timestamp prefix
    post_prefixes = {p.name for p in EXP005_RAW.glob("*-alone.csv")}
    new = sorted(post_prefixes - pre_prefixes)
    if not new:
        raise RuntimeError(f"{label}: no new -alone.csv appeared in 005/raw/")
    if len(new) > 1:
        raise RuntimeError(f"{label}: multiple new -alone.csv files appeared: {new}")
    prefix = new[0].split("-alone.csv")[0]
    print(f"  done in {(end_ns - start_ns) / 1e9:.1f}s, new prefix: {prefix}")
    return prefix, start_ns, end_ns


def main() -> None:
    if not EXP005_RUN_PY.exists():
        raise FileNotFoundError(f"expected {EXP005_RUN_PY}")
    if not EXP005_RAW.exists():
        raise FileNotFoundError(f"expected {EXP005_RAW}")

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"006 cross-session orchestrator  ts={ts}")
    print(f"session gap: {SESSION_GAP_S}s ({SESSION_GAP_S / 60:.0f} min)")

    # Session A
    prefix_a, start_a_ns, end_a_ns = run_session("session A")

    # Idle gap
    print(f"\n========== sleeping {SESSION_GAP_S}s before session B ==========")
    sleep_start_ns = time.monotonic_ns()
    time.sleep(SESSION_GAP_S)
    sleep_end_ns = time.monotonic_ns()
    print(f"  slept {(sleep_end_ns - sleep_start_ns) / 1e9:.1f}s")

    # Session B
    prefix_b, start_b_ns, end_b_ns = run_session("session B")

    # Linkage file
    linkage = RAW_DIR / f"{ts}-cross-session.txt"
    lines = [
        "experiment: 006-cross-session-ratio-stability",
        f"timestamp: {ts}",
        f"session_gap_s: {SESSION_GAP_S}",
        f"session_a_prefix: {prefix_a}",
        f"session_b_prefix: {prefix_b}",
        f"session_a_wall_start_ns: {start_a_ns}",
        f"session_a_wall_end_ns:   {end_a_ns}",
        f"session_a_wall_clock_s: {(end_a_ns - start_a_ns) / 1e9:.2f}",
        f"sleep_actual_s:         {(sleep_end_ns - sleep_start_ns) / 1e9:.2f}",
        f"session_b_wall_start_ns: {start_b_ns}",
        f"session_b_wall_end_ns:   {end_b_ns}",
        f"session_b_wall_clock_s: {(end_b_ns - start_b_ns) / 1e9:.2f}",
        f"chip: {platform.processor()}  machine: {platform.machine()}",
        f"os: {platform.platform()}",
        f"python: {sys.version.splitlines()[0]}",
        # Pull architecture string from session A's meta if present
    ]
    meta_a = EXP005_RAW / f"{prefix_a}-meta.txt"
    if meta_a.exists():
        for line in meta_a.read_text().splitlines():
            if line.startswith("device:") or line.startswith("architecture:"):
                lines.append(f"a_{line}")
    meta_b = EXP005_RAW / f"{prefix_b}-meta.txt"
    if meta_b.exists():
        for line in meta_b.read_text().splitlines():
            if line.startswith("device:") or line.startswith("architecture:"):
                lines.append(f"b_{line}")
    linkage.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {linkage.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
