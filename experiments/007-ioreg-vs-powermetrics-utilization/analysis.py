# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Analysis for experiment 007: ioreg utilization vs powermetrics
active residency.

Loads the ioreg CSV, the user-provided powermetrics CSV, and the
phase-marker CSV, then bins both signals into 100 ms windows
(monotonic_ns rounded down) and computes per-bin disagreement in
percentage points. Reports per-phase-step summary (mean ioreg, mean
pm, mean disagreement, p95 disagreement).

Usage:
    uv run analysis.py --powermetrics-csv PATH --prefix YYYYmmddTHHMMSS

Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def percentile(xs, p):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def load_ioreg(path: Path):
    """Return list of (monotonic_ns, device_util_pct)."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = int(r["monotonic_ns"])
                u = float(r["device_util_pct"])
                out.append((t, u))
            except (ValueError, KeyError):
                continue
    return out


def load_pm(path: Path):
    """Return list of (monotonic_ns, gpu_active_pct)."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = int(r["monotonic_ns"])
                u = float(r["gpu_active_pct"])
                out.append((t, u))
            except (ValueError, KeyError):
                continue
    return out


def load_phases(path: Path):
    """Return list of (monotonic_ns, phase, target_busy_fraction)."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out.append((
                    int(r["monotonic_ns"]),
                    r["phase"],
                    float(r["target_busy_fraction"]),
                ))
            except (ValueError, KeyError):
                continue
    return out


def bin_into_windows(samples, bin_ns=100_000_000):
    """Group (t, value) tuples into bins of bin_ns. Return dict
    {bin_start_ns: mean_value}."""
    by_bin = defaultdict(list)
    for t, v in samples:
        by_bin[t - (t % bin_ns)].append(v)
    return {k: statistics.mean(v) for k, v in by_bin.items()}


def assign_phase(t_ns, phase_markers):
    """Find which phase contains this timestamp. phase_markers is a
    list of (t_ns, phase_label, target_fraction) sorted ascending."""
    last = None
    for marker in phase_markers:
        if marker[0] > t_ns:
            return last if last is not None else phase_markers[0]
        last = marker
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--powermetrics-csv", type=str, required=True,
                    help="path to the powermetrics CSV recorded during the run")
    ap.add_argument("--prefix", type=str, required=True,
                    help="timestamp prefix of the 007 run (e.g. 20260428T141500)")
    ap.add_argument("--bin-ms", type=int, default=100,
                    help="bin width in ms for joining (default 100)")
    args = ap.parse_args()

    raw_dir = Path(__file__).resolve().parent / "raw"
    ioreg_path = raw_dir / f"{args.prefix}-ioreg.csv"
    phases_path = raw_dir / f"{args.prefix}-phases.csv"
    pm_path = Path(args.powermetrics_csv)

    for p in [ioreg_path, phases_path, pm_path]:
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    ioreg = load_ioreg(ioreg_path)
    pm = load_pm(pm_path)
    phases = load_phases(phases_path)

    print(f"# ioreg samples: {len(ioreg)}")
    print(f"# powermetrics samples: {len(pm)}")
    print(f"# phase markers: {len(phases)}")

    if not ioreg or not pm or not phases:
        print("ERROR: empty data", file=sys.stderr)
        return 2

    bin_ns = args.bin_ms * 1_000_000
    ioreg_binned = bin_into_windows(ioreg, bin_ns)
    pm_binned = bin_into_windows(pm, bin_ns)

    common = sorted(set(ioreg_binned) & set(pm_binned))
    print(f"# bins with both signals: {len(common)}")

    if not common:
        print("ERROR: no overlapping bins -- did powermetrics record at the "
              "same time as ioreg? Check the CSV timestamps.", file=sys.stderr)
        return 2

    # Assign each bin to its phase
    by_phase = defaultdict(list)
    for bin_start in common:
        bin_mid = bin_start + bin_ns // 2
        phase = assign_phase(bin_mid, phases)
        if phase is None:
            continue
        ioreg_v = ioreg_binned[bin_start]
        pm_v = pm_binned[bin_start]
        diff = ioreg_v - pm_v   # signed (positive = ioreg higher)
        by_phase[(phase[1], phase[2])].append((ioreg_v, pm_v, diff))

    print()
    print("=" * 110)
    print("PRIMARY: per-phase agreement (ioreg device_util_pct vs powermetrics gpu_active_pct)")
    print("=" * 110)
    hdr = ("phase", "target%", "n_bins", "mean_ioreg%", "mean_pm%",
           "mean_diff_pp", "abs_p50_pp", "abs_p95_pp", "abs_max_pp")
    print(f"{hdr[0]:<14} {hdr[1]:>8} {hdr[2]:>7} {hdr[3]:>11} {hdr[4]:>9} "
          f"{hdr[5]:>13} {hdr[6]:>10} {hdr[7]:>10} {hdr[8]:>10}")

    overall_abs_diffs = []
    for (phase, target), rows in sorted(by_phase.items(),
                                         key=lambda kv: (kv[0][1], kv[0][0])):
        ioreg_vals = [r[0] for r in rows]
        pm_vals = [r[1] for r in rows]
        diffs = [r[2] for r in rows]
        abs_diffs = [abs(d) for d in diffs]
        overall_abs_diffs.extend(abs_diffs)
        print(f"{phase:<14} {target * 100:>7.0f}% {len(rows):>7} "
              f"{statistics.mean(ioreg_vals):>10.1f}% "
              f"{statistics.mean(pm_vals):>8.1f}% "
              f"{statistics.mean(diffs):>+12.2f}  "
              f"{percentile(abs_diffs, 50):>9.2f} "
              f"{percentile(abs_diffs, 95):>9.2f} "
              f"{max(abs_diffs):>9.2f}")

    print()
    print(f"OVERALL  N={len(overall_abs_diffs)} bins  "
          f"|diff| p50={percentile(overall_abs_diffs, 50):.2f} pp  "
          f"p95={percentile(overall_abs_diffs, 95):.2f} pp  "
          f"max={max(overall_abs_diffs):.2f} pp")

    p50 = percentile(overall_abs_diffs, 50)
    if p50 <= 5.0:
        verdict = "PASS  -- ioreg is a viable sudo-free utilization signal"
    elif p50 <= 10.0:
        verdict = "MARGINAL -- ioreg is usable with calibration"
    else:
        verdict = "FAIL  -- ioreg measures something different from powermetrics"
    print(f"VERDICT: {verdict} (per 007/README success criteria)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
