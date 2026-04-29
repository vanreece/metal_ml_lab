# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 009 analysis: per-attempt distribution + IOReport GPU power
correlation with sub-floor entry.

The pre-registration's mechanism hypothesis was: "if the state is a
DVFS upshift, IOReport GPU power should be elevated (~+50-100 mW)
during sub-floor trials relative to back-to-back floor trials in the
same attempt." This script joins the per-trial CSV with the IOReport
CSV by monotonic_ns and reports per-attempt floor-vs-sub-floor power.

Threshold for sub-floor: gpu_delta_raw < 5500 ns (matches the README
verdict threshold; well below the back-to-back floor of ~6.4 µs).
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def percentile(xs, p):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def load_attempts(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "attempt_idx": int(r["attempt_idx"]),
                "phase": r["phase"],
                "idx_within_phase": int(r["idx_within_phase"]),
                "wall_clock_ns": int(r["wall_clock_ns"]),
                "gpu_delta_raw": int(r["gpu_delta_raw"]),
            })
    return rows


def load_ioreport(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "monotonic_ns": int(r["monotonic_ns"]),
                "window_s": float(r["window_s"]),
                "gpu_power_mw": float(r["gpu_power_mw"]),
                "cpu_power_mw": float(r["cpu_power_mw"]),
                "amcc_power_mw": float(r["amcc_power_mw"]),
                "dcs_power_mw": float(r["dcs_power_mw"]),
                "afr_power_mw": float(r["afr_power_mw"]),
            })
    return rows


def gpu_power_at(monotonic_ns: int, ioreport: list[dict]) -> dict | None:
    """Find the IOReport sample whose window covers monotonic_ns."""
    for s in ioreport:
        win_start = s["monotonic_ns"] - int(s["window_s"] * 1e9)
        if win_start <= monotonic_ns <= s["monotonic_ns"]:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="timestamp prefix, e.g. 20260428T211252")
    ap.add_argument("--raw-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw")
    args = ap.parse_args()

    attempts_csv = args.raw_dir / f"{args.prefix}-attempts.csv"
    ioreport_csv = args.raw_dir / f"{args.prefix}-ioreport.csv"
    if not attempts_csv.exists():
        raise SystemExit(f"missing {attempts_csv}")
    if not ioreport_csv.exists():
        raise SystemExit(f"missing {ioreport_csv}")

    attempts = load_attempts(attempts_csv)
    ioreport = load_ioreport(ioreport_csv)
    print(f"loaded {len(attempts)} trial rows, {len(ioreport)} ioreport rows")
    print()

    # Per-attempt summary table
    by_attempt: dict[int, list[dict]] = {}
    for r in attempts:
        by_attempt.setdefault(r["attempt_idx"], []).append(r)

    print("=" * 78)
    print("Per-attempt timing distribution")
    print("=" * 78)
    print(f"{'a':>2} {'n':>3} {'min':>6} {'p05':>6} {'p50':>6} {'p95':>6} "
          f"{'max':>7} {'cv':>6} {'<5500':>5} {'<3000':>5} {'first<5500':>10}")
    for a in sorted(by_attempt):
        meas = [r for r in by_attempt[a] if r["phase"] == "measured"]
        ds = [r["gpu_delta_raw"] for r in meas]
        sd = statistics.stdev(ds) if len(ds) > 1 else float("nan")
        p50 = percentile(ds, 50)
        cv = (sd / p50) if p50 else float("nan")
        below_55 = sum(1 for d in ds if d < 5500)
        below_30 = sum(1 for d in ds if d < 3000)
        first_sub = next((r["idx_within_phase"] for r in meas
                          if r["gpu_delta_raw"] < 5500), -1)
        print(f"{a:>2} {len(ds):>3} {min(ds):>6} {percentile(ds, 5):>6.0f} "
              f"{p50:>6.0f} {percentile(ds, 95):>6.0f} {max(ds):>7} "
              f"{cv:>6.3f} {below_55:>5} {below_30:>5} {first_sub:>10}")

    # Floor-vs-sub-floor IOReport GPU power per attempt
    print()
    print("=" * 78)
    print("IOReport GPU power: floor (>=5500ns) vs sub-floor (<5500ns) per attempt")
    print("=" * 78)
    print(f"{'a':>2} {'n_flr':>5} {'gpu_mw_flr_p50':>14} "
          f"{'n_sub':>5} {'gpu_mw_sub_p50':>14} {'delta_mw':>9}")
    for a in sorted(by_attempt):
        meas = [r for r in by_attempt[a] if r["phase"] == "measured"]
        floor_mw = []
        subfloor_mw = []
        for r in meas:
            samp = gpu_power_at(r["wall_clock_ns"], ioreport)
            if samp is None:
                continue
            if r["gpu_delta_raw"] < 5500:
                subfloor_mw.append(samp["gpu_power_mw"])
            else:
                floor_mw.append(samp["gpu_power_mw"])
        flr_p50 = percentile(floor_mw, 50) if floor_mw else float("nan")
        sub_p50 = percentile(subfloor_mw, 50) if subfloor_mw else float("nan")
        delta = (sub_p50 - flr_p50) if (floor_mw and subfloor_mw) else float("nan")
        print(f"{a:>2} {len(floor_mw):>5} {flr_p50:>14.0f} "
              f"{len(subfloor_mw):>5} {sub_p50:>14.0f} {delta:>9.0f}")

    # Trial-level trajectory of attempt 0 (the canonical onset) to README-style
    # detail for the writeup.
    print()
    print("=" * 78)
    print("Per-attempt trial trajectory (gpu_delta_raw, ns) -- compact view")
    print("=" * 78)
    for a in sorted(by_attempt):
        meas = sorted([r for r in by_attempt[a] if r["phase"] == "measured"],
                      key=lambda r: r["idx_within_phase"])
        print(f"\n--- attempt {a} ---")
        line = []
        for r in meas:
            d = r["gpu_delta_raw"]
            tag = "*" if d < 5500 else " "
            line.append(f"{r['idx_within_phase']:2d}:{d:>5}{tag}")
            if len(line) == 7:
                print("  " + "  ".join(line))
                line = []
        if line:
            print("  " + "  ".join(line))

    # Cross-attempt CPU power during measurement windows (sanity)
    print()
    print("=" * 78)
    print("IOReport CPU + memory power during attempts (mean across all trials)")
    print("=" * 78)
    print(f"{'a':>2} {'cpu_mw':>8} {'amcc_mw':>8} {'dcs_mw':>8} {'afr_mw':>7}")
    for a in sorted(by_attempt):
        meas = [r for r in by_attempt[a] if r["phase"] == "measured"]
        cpus, amccs, dcss, afrs = [], [], [], []
        for r in meas:
            samp = gpu_power_at(r["wall_clock_ns"], ioreport)
            if samp is None:
                continue
            cpus.append(samp["cpu_power_mw"])
            amccs.append(samp["amcc_power_mw"])
            dcss.append(samp["dcs_power_mw"])
            afrs.append(samp["afr_power_mw"])
        def _avg(xs): return (sum(xs) / len(xs)) if xs else float("nan")
        print(f"{a:>2} {_avg(cpus):>8.0f} {_avg(amccs):>8.0f} "
              f"{_avg(dcss):>8.0f} {_avg(afrs):>7.0f}")


if __name__ == "__main__":
    main()
