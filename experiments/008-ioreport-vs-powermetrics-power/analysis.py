# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Analysis for experiment 008: IOReport vs powermetrics GPU power.

Loads the ioreport CSV (this script's output), the user-provided
powermetrics CSV, and the phase-marker CSV. Bins both telemetry
sources into 1000 ms windows by monotonic_ns and reports per-phase-
step agreement. Verdict per 008/README success criteria.

Usage:
    uv run analysis.py --powermetrics-csv PATH --prefix YYYYmmddTHHMMSS
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


def load_pm(path: Path):
    """Return list of (monotonic_ns, gpu_power_mw)."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = int(r["monotonic_ns"])
                p = float(r["gpu_power_mw"])
                out.append((t, p))
            except (ValueError, KeyError):
                continue
    return out


def load_ioreport(path: Path):
    """Return list of (monotonic_ns, gpu_power_mw, [other buckets...])."""
    out = []
    bonus_keys = ["cpu_power_mw", "dram_power_mw", "amcc_power_mw",
                  "dcs_power_mw", "afr_power_mw", "disp_power_mw"]
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = int(r["monotonic_ns"])
                p = float(r["gpu_power_mw"])
                bonus = {k: float(r.get(k, 0) or 0) for k in bonus_keys}
                out.append((t, p, bonus))
            except (ValueError, KeyError):
                continue
    return out


def load_phases(path: Path):
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


def bin_into_windows(samples, bin_ns):
    by_bin = defaultdict(list)
    for tup in samples:
        t = tup[0]
        v = tup[1]
        by_bin[t - (t % bin_ns)].append(v)
    return {k: statistics.mean(v) for k, v in by_bin.items()}


def assign_phase(t_ns, phase_markers):
    last = None
    for marker in phase_markers:
        if marker[0] > t_ns:
            return last if last is not None else phase_markers[0]
        last = marker
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--powermetrics-csv", type=str, required=True)
    ap.add_argument("--prefix", type=str, required=True)
    ap.add_argument("--bin-ms", type=int, default=1000)
    args = ap.parse_args()

    raw_dir = Path(__file__).resolve().parent / "raw"
    ioreport_path = raw_dir / f"{args.prefix}-ioreport.csv"
    phases_path = raw_dir / f"{args.prefix}-phases.csv"
    pm_path = Path(args.powermetrics_csv)

    for p in [ioreport_path, phases_path, pm_path]:
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    pm = load_pm(pm_path)
    iorep = load_ioreport(ioreport_path)
    phases = load_phases(phases_path)
    print(f"# powermetrics samples: {len(pm)}")
    print(f"# ioreport samples:     {len(iorep)}")
    print(f"# phase markers:        {len(phases)}")

    if not pm or not iorep or not phases:
        print("ERROR: empty data", file=sys.stderr)
        return 2

    bin_ns = args.bin_ms * 1_000_000
    pm_binned = bin_into_windows(pm, bin_ns)
    ior_binned = bin_into_windows(iorep, bin_ns)

    # For bonus columns, also bin them
    bonus_keys = ["cpu_power_mw", "dram_power_mw", "amcc_power_mw",
                  "dcs_power_mw", "afr_power_mw", "disp_power_mw"]
    ior_bonus_binned = {k: defaultdict(list) for k in bonus_keys}
    for t, _, bonus in iorep:
        b = t - (t % bin_ns)
        for k in bonus_keys:
            ior_bonus_binned[k][b].append(bonus[k])
    ior_bonus_binned = {
        k: {b: statistics.mean(v) for b, v in d.items()}
        for k, d in ior_bonus_binned.items()
    }

    common = sorted(set(pm_binned) & set(ior_binned))
    print(f"# bins with both signals: {len(common)}")
    if not common:
        print("ERROR: no overlapping bins", file=sys.stderr)
        return 2

    by_phase = defaultdict(list)
    by_phase_bonus = defaultdict(lambda: defaultdict(list))
    for bin_start in common:
        bin_mid = bin_start + bin_ns // 2
        phase = assign_phase(bin_mid, phases)
        if phase is None:
            continue
        ior_v = ior_binned[bin_start]
        pm_v = pm_binned[bin_start]
        diff = ior_v - pm_v
        by_phase[(phase[1], phase[2])].append((ior_v, pm_v, diff))
        for k in bonus_keys:
            v = ior_bonus_binned[k].get(bin_start)
            if v is not None:
                by_phase_bonus[(phase[1], phase[2])][k].append(v)

    print()
    print("=" * 110)
    print("PRIMARY: per-phase GPU power agreement (ioreport vs powermetrics)")
    print("=" * 110)
    hdr = ("phase", "target%", "n_bins", "ior_mean_mW", "pm_mean_mW",
           "mean_diff_mW", "abs_p50_mW", "abs_p95_mW", "rel_diff_%")
    print(f"{hdr[0]:<14} {hdr[1]:>8} {hdr[2]:>7} {hdr[3]:>11} {hdr[4]:>10} "
          f"{hdr[5]:>13} {hdr[6]:>10} {hdr[7]:>10} {hdr[8]:>11}")

    busy_phase_rel_diffs = []
    for (phase, target), rows in sorted(by_phase.items(),
                                         key=lambda kv: (kv[0][1], kv[0][0])):
        ior_vals = [r[0] for r in rows]
        pm_vals = [r[1] for r in rows]
        diffs = [r[2] for r in rows]
        abs_diffs = [abs(d) for d in diffs]
        ior_mean = statistics.mean(ior_vals)
        pm_mean = statistics.mean(pm_vals)
        mean_pair = (ior_mean + pm_mean) / 2 if (ior_mean + pm_mean) > 0 else 1
        rel_diff_pct = abs(ior_mean - pm_mean) / mean_pair * 100
        if target >= 0.5:
            busy_phase_rel_diffs.append(rel_diff_pct)
        print(f"{phase:<14} {target * 100:>7.0f}% {len(rows):>7} "
              f"{ior_mean:>10.0f} mW "
              f"{pm_mean:>9.0f} mW "
              f"{statistics.mean(diffs):>+12.0f}  "
              f"{percentile(abs_diffs, 50):>9.0f} "
              f"{percentile(abs_diffs, 95):>9.0f} "
              f"{rel_diff_pct:>10.2f}%")

    print()
    print("=" * 110)
    print("BONUS: IOReport-side breakdown per phase (no powermetrics counterpart)")
    print("=" * 110)
    print(f"{'phase':<14} {'target%':>8} {'cpu mW':>10} {'dram mW':>10} "
          f"{'amcc mW':>10} {'dcs mW':>10} {'afr mW':>10} {'disp mW':>10}")
    for (phase, target), rows in sorted(by_phase.items(),
                                         key=lambda kv: (kv[0][1], kv[0][0])):
        bb = by_phase_bonus[(phase, target)]
        cpu = statistics.mean(bb.get("cpu_power_mw", [0]) or [0])
        dram = statistics.mean(bb.get("dram_power_mw", [0]) or [0])
        amcc = statistics.mean(bb.get("amcc_power_mw", [0]) or [0])
        dcs = statistics.mean(bb.get("dcs_power_mw", [0]) or [0])
        afr = statistics.mean(bb.get("afr_power_mw", [0]) or [0])
        disp = statistics.mean(bb.get("disp_power_mw", [0]) or [0])
        print(f"{phase:<14} {target * 100:>7.0f}% {cpu:>9.0f} mW {dram:>9.0f} mW "
              f"{amcc:>9.0f} mW {dcs:>9.0f} mW {afr:>9.0f} mW {disp:>9.0f} mW")

    print()
    if busy_phase_rel_diffs:
        med_rel = statistics.median(busy_phase_rel_diffs)
    else:
        med_rel = float("nan")
    print(f"BUSY-PHASE (target>=50%) median |rel_diff| = {med_rel:.2f}%")
    if med_rel <= 5.0:
        verdict = "PASS  -- IOReport GPU power matches powermetrics; sudo-free GPU power validated"
    elif med_rel <= 15.0:
        verdict = "MARGINAL -- usable with calibration / known offset"
    else:
        verdict = "FAIL  -- the two sources measure different things"
    print(f"VERDICT: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
