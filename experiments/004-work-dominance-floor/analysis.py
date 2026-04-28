# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
One-off analysis of 004 raw output. Stdlib only.

Pools the 3 sweeps × 300 trials per (axis, level) into a single sample
of N=900 and reports robust summary statistics, plus the inter-level
p50 ratios that are the work-dominance signal we are after.
Also tabulates the 5.4 us reproduction outcome.

Not a library. Not reused. Will be retired or rewritten when the
question changes.
"""
from __future__ import annotations
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def percentile(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def robust_cv(xs):
    p50 = percentile(xs, 50)
    p25 = percentile(xs, 25)
    p75 = percentile(xs, 75)
    iqr = p75 - p25
    return (iqr / 1.349) / p50 if p50 else float("nan")


def naive_cv(xs):
    p50 = percentile(xs, 50)
    if len(xs) < 2 or p50 == 0:
        return float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return sd / p50


def load_measured(path):
    by_combo = defaultdict(lambda: defaultdict(list))  # axis -> level -> [deltas]
    by_sweep = defaultdict(lambda: defaultdict(dict))  # axis -> level -> sweep_idx -> [deltas]
    with open(path) as f:
        for row in csv.DictReader(f):
            axis = row["axis"]
            level = int(row["complexity_level"])
            sweep = int(row["sweep_idx"])
            d = int(row["gpu_delta_raw"])
            by_combo[axis][level].append(d)
            by_sweep[axis][level].setdefault(sweep, []).append(d)
    return by_combo, by_sweep


def load_repro(path):
    by_attempt = defaultdict(lambda: {"calibration": [], "measured": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            a = int(row["attempt_idx"])
            phase = row["phase"]
            d = int(row["gpu_delta_raw"])
            by_attempt[a][phase].append(d)
    return by_attempt


def fmt_axis_table(by_combo, by_sweep, axis_key, level_label, sort_levels):
    print(f"\n=== Axis: {axis_key}  (pooled N=900 per level) ===")
    print(f"{'level':>10s} {'p05':>7s} {'p50':>9s} {'p95':>9s} {'p99':>9s} "
          f"{'max':>9s} {'iqr_cv':>7s} {'naive_cv':>9s} "
          f"{'sweep_p50s':>30s}  {'ratio_p50_to_prev':>18s}")
    levels = sort_levels
    prev_p50 = None
    for level in levels:
        deltas = by_combo[axis_key][level]
        p50 = percentile(deltas, 50)
        sweep_p50s = []
        for s in sorted(by_sweep[axis_key][level].keys()):
            sweep_p50s.append(percentile(by_sweep[axis_key][level][s], 50))
        sweep_p50_str = "[" + ", ".join(f"{s:>7.0f}" for s in sweep_p50s) + "]"
        ratio_str = "-" if prev_p50 is None else f"{p50/prev_p50:>6.3f}x"
        print(f"{level:>10d} {percentile(deltas, 5):>7.0f} {p50:>9.0f} "
              f"{percentile(deltas, 95):>9.0f} {percentile(deltas, 99):>9.0f} "
              f"{max(deltas):>9d} {robust_cv(deltas):>7.3f} {naive_cv(deltas):>9.3f} "
              f"{sweep_p50_str:>30s}  {ratio_str:>18s}")
        prev_p50 = p50


def fmt_repro(by_attempt):
    print("\n=== 5.4us reproduction protocol ===")
    print(f"{'attempt':>7s} {'cal_first':>9s} {'cal_p50_rest':>13s} "
          f"{'meas_min':>8s} {'meas_p50':>9s} {'meas_p95':>9s} {'meas_max':>9s} "
          f"{'iqr_cv':>7s} {'in_low':>7s} {'in_~8us':>8s}")
    for a in sorted(by_attempt.keys()):
        cal = by_attempt[a]["calibration"]
        meas = by_attempt[a]["measured"]
        cal_first = cal[0]
        cal_p50_rest = percentile(cal[1:], 50)
        in_low = sum(1 for d in meas if 5000 <= d <= 6000)
        in_eight = sum(1 for d in meas if 8000 <= d <= 8200)
        print(f"{a:>7d} {cal_first:>9d} {cal_p50_rest:>13.0f} "
              f"{min(meas):>8d} {percentile(meas, 50):>9.0f} "
              f"{percentile(meas, 95):>9.0f} {max(meas):>9d} "
              f"{robust_cv(meas):>7.3f} {in_low:>3d}/{len(meas):<3d} "
              f"{in_eight:>4d}/{len(meas):<3d}")


def crossover_analysis(by_combo, axis_key, levels, label):
    """For a given axis, walk levels and find where p50 ratio between
    consecutive levels approaches the level-ratio (i.e. doubling threads
    doubles measured time). Print the level at which the median grows
    by at least 1.5x for at least a 1.5x level increase, sustained."""
    print(f"\n=== Crossover walk: {axis_key} ===")
    print(f"{'level':>10s} {'p50_ns':>10s} "
          f"{'level_ratio':>13s} {'p50_ratio':>13s} "
          f"{'p50_ratio/level_ratio':>25s}")
    prev_level = None
    prev_p50 = None
    for level in levels:
        p50 = percentile(by_combo[axis_key][level], 50)
        if prev_p50 is None:
            print(f"{level:>10d} {p50:>10.0f} {'-':>13s} {'-':>13s} {'-':>25s}")
        else:
            level_ratio = level / prev_level
            p50_ratio = p50 / prev_p50
            ratio_of_ratios = p50_ratio / level_ratio
            print(f"{level:>10d} {p50:>10.0f} "
                  f"{level_ratio:>11.2f}x {p50_ratio:>11.3f}x "
                  f"{ratio_of_ratios:>22.3f}")
        prev_level = level
        prev_p50 = p50
    print(f"({label}: ratio_of_ratios approaches 1.0 when work fully "
          f"dominates dispatch overhead)")


WRITE_TID_LEVELS = [
    32, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    1536, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 8388608,
]
FMA_LEVELS = [
    16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
]


def main():
    raw_dir = Path(__file__).resolve().parent / "raw"
    # Pick the most recent run by timestamp prefix so this script works
    # across re-runs (e.g. M1 Pro 20260427 + M4 Max 20260428). Caller
    # can override by passing a timestamp prefix as the first argument.
    if len(sys.argv) > 1:
        prefix = sys.argv[1]
        measured = raw_dir / f"{prefix}-measured.csv"
        repro = raw_dir / f"{prefix}-repro.csv"
    else:
        measured = sorted(raw_dir.glob("*-measured.csv"))[-1]
        repro = sorted(raw_dir.glob("*-repro.csv"))[-1]
    print(f"loading {measured.name}")
    by_combo, by_sweep = load_measured(measured)
    print(f"loading {repro.name}")
    by_attempt = load_repro(repro)

    fmt_axis_table(
        by_combo, by_sweep, "write_tid_threadcount", "threads", WRITE_TID_LEVELS
    )
    fmt_axis_table(
        by_combo, by_sweep, "fma_loop_iters", "fma_iters", FMA_LEVELS
    )
    crossover_analysis(by_combo, "write_tid_threadcount", WRITE_TID_LEVELS,
                       "thread count axis")
    crossover_analysis(by_combo, "fma_loop_iters", FMA_LEVELS, "fma iters axis")
    fmt_repro(by_attempt)


if __name__ == "__main__":
    main()
