# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
GPU telemetry dashboard for Apple Silicon.

Wraps `powermetrics` (sudo required) and prints a live one-line-per-
sample status with the most useful fields for understanding what the
GPU is doing during an experiment. Optionally writes a CSV for
offline correlation with experiment timestamps.

Run from a separate terminal while an experiment is in progress:

    sudo uv run notes/gpu_telemetry.py
    sudo uv run notes/gpu_telemetry.py --csv /tmp/telemetry.csv
    sudo uv run notes/gpu_telemetry.py --interval-ms 500 --csv /tmp/t.csv

Stop with Ctrl-C. powermetrics gets a clean SIGTERM; partial CSV is
flushed.

What you'll see (one line per sample, default 1000 ms cadence):

    13:42:03  GPU  12.3% @  444 MHz   234 mW  Tg 42.5C  thermal=Nominal  fans 1234,0 rpm

Field meanings:
- GPU %     : GPU HW active residency (fraction of sample window the
              GPU was doing work, vs idle)
- @ X MHz   : GPU HW active frequency (the average frequency *while*
              active — flat at one P-state means stable DVFS)
- mW        : GPU package power
- Tg X C    : GPU die temperature (from SMC)
- thermal=  : macOS thermal pressure level (Nominal / Moderate /
              Heavy / Trapping / Sleeping / Critical)
- fans      : RPM of each detected fan (M4 Max MBP14 has two fans;
              MacBook Air has none -> 0,0)

CSV columns (when --csv is given) match the dashboard fields plus an
ISO timestamp and `time.monotonic_ns()` so rows can be joined with an
experiment's `wall_clock_ns` column after the fact (both processes
see the same monotonic clock on the same boot).

Why not give Claude sudo? Because powermetrics has read access to a
lot of system telemetry and an LLM-driven workflow shouldn't have
that ambient privilege. The intended pattern is: you run this in one
terminal, Claude runs experiments in another, and we both look at
the same data.

Defensive parsing: powermetrics field names have shifted across
macOS versions. Each field below has multiple regex patterns; the
first match wins. If a field is unavailable, the dashboard prints
"?" rather than failing.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Multiple patterns per field — first match wins, lets us survive
# powermetrics output format drift across macOS releases.
FIELD_PATTERNS: dict[str, list[re.Pattern]] = {
    "freq_mhz": [
        re.compile(r"^GPU\s*HW\s*active\s*frequency:\s*(\d+)\s*MHz", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^GPU\s*active\s*frequency:\s*(\d+)\s*MHz", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^GPU\s*frequency\s*as\s*fraction\s*of\s*active:.*?(\d+)\s*MHz", re.IGNORECASE | re.MULTILINE),
    ],
    "active_pct": [
        re.compile(r"^GPU\s*HW\s*active\s*residency:\s*([\d.]+)\s*%", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^GPU\s*active\s*residency:\s*([\d.]+)\s*%", re.IGNORECASE | re.MULTILINE),
    ],
    "power_mw": [
        re.compile(r"^GPU\s*Power:\s*(\d+)\s*mW", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^GPU\+EFI\s*Power:\s*(\d+)\s*mW", re.IGNORECASE | re.MULTILINE),
    ],
    "thermal": [
        re.compile(r"^Current\s*pressure\s*level:\s*(\w+)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^Thermal\s*pressure:\s*(\w+)", re.IGNORECASE | re.MULTILINE),
    ],
    "gpu_temp_c": [
        re.compile(r"^GPU\s*die\s*temperature:\s*([\d.]+)\s*C", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^GPU\s*temperature:\s*([\d.]+)\s*C", re.IGNORECASE | re.MULTILINE),
    ],
    "cpu_temp_c": [
        re.compile(r"^CPU\s*die\s*temperature:\s*([\d.]+)\s*C", re.IGNORECASE | re.MULTILINE),
    ],
    "package_power_mw": [
        re.compile(r"^Combined\s*Power\s*\(CPU\s*\+\s*GPU\s*\+\s*ANE\):\s*(\d+)\s*mW", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^Package\s*Power:\s*(\d+)\s*mW", re.IGNORECASE | re.MULTILINE),
    ],
}

FAN_PATTERN = re.compile(r"^Fan\s*(\d+)\s*speed:\s*(\d+)\s*rpm", re.IGNORECASE | re.MULTILINE)
SAMPLE_BOUNDARY = re.compile(r"^\*+\s*Sampled\s*system\s*activity", re.IGNORECASE | re.MULTILINE)


def parse_sample(buf: str) -> dict:
    """Pull every field we know about out of one accumulated sample's text block."""
    out: dict = {}
    for key, patterns in FIELD_PATTERNS.items():
        for p in patterns:
            m = p.search(buf)
            if m:
                out[key] = m.group(1)
                break
    fans: dict[str, str] = {}
    for m in FAN_PATTERN.finditer(buf):
        fans[m.group(1)] = m.group(2)
    out["fans"] = fans
    return out


def fmt_dashboard(sample: dict) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    active = sample.get("active_pct", "?")
    freq = sample.get("freq_mhz", "?")
    power = sample.get("power_mw", "?")
    temp = sample.get("gpu_temp_c", "?")
    therm = sample.get("thermal", "?")
    fans = sample.get("fans", {}) or {}
    fan_str = ",".join(fans.get(str(i), "?") for i in range(max(2, len(fans))))
    return (f"{ts}  GPU {str(active):>5}% @ {str(freq):>4} MHz  "
            f"{str(power):>5} mW  Tg {str(temp):>4}C  "
            f"thermal={therm:<8}  fans {fan_str} rpm")


def write_csv_row(writer, sample: dict) -> None:
    fans = sample.get("fans", {}) or {}
    writer.writerow([
        datetime.now().isoformat(timespec="microseconds"),
        time.monotonic_ns(),
        sample.get("active_pct", ""),
        sample.get("freq_mhz", ""),
        sample.get("power_mw", ""),
        sample.get("package_power_mw", ""),
        sample.get("gpu_temp_c", ""),
        sample.get("cpu_temp_c", ""),
        sample.get("thermal", ""),
        fans.get("0", ""),
        fans.get("1", ""),
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--interval-ms", type=int, default=1000,
                    help="powermetrics sample interval in ms (default 1000)")
    ap.add_argument("--samplers", default="gpu_power,thermal,smc",
                    help="comma-sep list of powermetrics samplers")
    ap.add_argument("--csv", type=str, default=None,
                    help="if set, write per-sample CSV to this path "
                         "(includes monotonic_ns for joining with experiment data)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress dashboard output (only useful with --csv)")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("ERROR: powermetrics requires sudo. Re-run with `sudo uv run ...`",
              file=sys.stderr)
        return 2

    cmd = [
        "powermetrics",
        "--samplers", args.samplers,
        "-i", str(args.interval_ms),
        "-f", "text",
    ]
    print(f"# launching: {' '.join(cmd)}", file=sys.stderr)
    print(f"# press Ctrl-C to stop", file=sys.stderr)
    if args.csv:
        print(f"# csv: {args.csv}", file=sys.stderr)

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "iso_ts", "monotonic_ns",
            "gpu_active_pct", "gpu_freq_mhz", "gpu_power_mw", "package_power_mw",
            "gpu_temp_c", "cpu_temp_c",
            "thermal_pressure", "fan0_rpm", "fan1_rpm",
        ])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)

    buf_lines: list[str] = []

    def flush_sample() -> None:
        if not buf_lines:
            return
        sample = parse_sample("\n".join(buf_lines))
        if not args.quiet:
            print(fmt_dashboard(sample), flush=True)
        if csv_writer:
            write_csv_row(csv_writer, sample)
            csv_file.flush()
        buf_lines.clear()

    def shutdown(_signum=None, _frame=None) -> None:
        flush_sample()
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if csv_file:
            csv_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if SAMPLE_BOUNDARY.match(line):
                flush_sample()
            buf_lines.append(line)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
