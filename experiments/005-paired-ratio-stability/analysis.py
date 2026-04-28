# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
One-off analysis of 005 raw output. Stdlib only. Not a library.

Pools the 3 sweeps × 300 trials per condition into a single sample
of N=900 (alone) or N=900 (paired-trial / paired-ref / ratio / gap),
and answers the three primary questions of 005:

1. Does ratio robust_cv beat trial-alone robust_cv?
2. Does pairing perturb trial median by less than 5% vs alone?
3. Is the ratio stable across the 3 sweeps within 1%?

Plus subsidiaries: ref-alone vs ref-when-paired drift, gap_ns
distribution, outlier behavior comparison (naive vs robust cv).
"""
from __future__ import annotations
import csv
import math
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


def load_alone(path):
    """{(condition, sweep_idx) -> [gpu_delta]} and {condition -> [gpu_delta] pooled across sweeps}"""
    by_sweep = defaultdict(list)
    pooled = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            sweep = int(row["sweep_idx"])
            d = int(row["gpu_delta_raw"])
            by_sweep[(cond, sweep)].append(d)
            pooled[cond].append(d)
    return by_sweep, pooled


def load_paired(path):
    """Returns by_sweep[(cond, sweep)] and pooled[cond] each as a dict
    with keys ref, trial, ratio, gap."""
    by_sweep = defaultdict(lambda: {"ref": [], "trial": [], "ratio": [], "gap": []})
    pooled = defaultdict(lambda: {"ref": [], "trial": [], "ratio": [], "gap": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            sweep = int(row["sweep_idx"])
            ref_d = int(row["ref_delta_raw"])
            trial_d = int(row["trial_delta_raw"])
            ratio = float(row["ratio"])
            gap = int(row["gap_ns"])
            by_sweep[(cond, sweep)]["ref"].append(ref_d)
            by_sweep[(cond, sweep)]["trial"].append(trial_d)
            by_sweep[(cond, sweep)]["ratio"].append(ratio)
            by_sweep[(cond, sweep)]["gap"].append(gap)
            pooled[cond]["ref"].append(ref_d)
            pooled[cond]["trial"].append(trial_d)
            pooled[cond]["ratio"].append(ratio)
            pooled[cond]["gap"].append(gap)
    return by_sweep, pooled


def main():
    raw_dir = Path(__file__).resolve().parent / "raw"
    alone_path = next(raw_dir.glob("*-alone.csv"))
    paired_path = next(raw_dir.glob("*-paired.csv"))
    print(f"loading {alone_path.name}, {paired_path.name}")
    alone_by_sweep, alone_pooled = load_alone(alone_path)
    paired_by_sweep, paired_pooled = load_paired(paired_path)

    print()
    print("=" * 100)
    print("PRIMARY 1: ratio robust_cv vs trial-alone robust_cv")
    print("=" * 100)
    print(f"{'trial':>6s} {'alone_p50':>10s} {'alone_rcv':>10s} "
          f"{'pair_trial_p50':>15s} {'pair_trial_rcv':>15s} "
          f"{'ratio_p50':>10s} {'ratio_rcv':>10s} "
          f"{'verdict':>30s}")
    for label in ("T1", "T2", "T3", "T4"):
        a = alone_pooled[f"{label}_alone"]
        p = paired_pooled[f"{label}_paired"]
        a_p50 = percentile(a, 50)
        a_rcv = robust_cv(a)
        pt_p50 = percentile(p["trial"], 50)
        pt_rcv = robust_cv(p["trial"])
        r_p50 = percentile(p["ratio"], 50)
        r_rcv = robust_cv(p["ratio"])
        verdict = ("ratio < alone (good)" if r_rcv < a_rcv * 0.5
                   else "ratio ~ alone" if r_rcv < a_rcv * 1.5
                   else "ratio > alone (bad)")
        print(f"{label:>6s} {a_p50:>10.0f} {a_rcv:>10.4f} "
              f"{pt_p50:>15.0f} {pt_rcv:>15.4f} "
              f"{r_p50:>10.4f} {r_rcv:>10.4f} "
              f"{verdict:>30s}")

    print()
    print("=" * 100)
    print("PRIMARY 2: trial median paired vs alone (perturbation)")
    print("=" * 100)
    print(f"{'trial':>6s} {'alone_p50':>12s} {'paired_trial_p50':>18s} "
          f"{'shift':>8s} {'verdict':>15s}")
    for label in ("T1", "T2", "T3", "T4"):
        a = alone_pooled[f"{label}_alone"]
        p = paired_pooled[f"{label}_paired"]
        a_p50 = percentile(a, 50)
        pt_p50 = percentile(p["trial"], 50)
        shift = (pt_p50 - a_p50) / a_p50
        verdict = "OK" if abs(shift) < 0.05 else "PERTURBED"
        print(f"{label:>6s} {a_p50:>12.0f} {pt_p50:>18.0f} "
              f"{shift*100:>+7.2f}% {verdict:>15s}")

    print()
    print("=" * 100)
    print("PRIMARY 3: ratio stability across 3 sweeps (within-session)")
    print("=" * 100)
    print(f"{'trial':>6s} {'sweep0_p50':>12s} {'sweep1_p50':>12s} {'sweep2_p50':>12s} "
          f"{'spread':>10s} {'verdict':>15s}")
    for label in ("T1", "T2", "T3", "T4"):
        cond = f"{label}_paired"
        sweep_p50s = []
        for s in range(3):
            ratios = paired_by_sweep[(cond, s)]["ratio"]
            sweep_p50s.append(percentile(ratios, 50))
        spread = (max(sweep_p50s) - min(sweep_p50s)) / sweep_p50s[0]
        verdict = "OK" if spread < 0.01 else "DRIFT"
        print(f"{label:>6s} {sweep_p50s[0]:>12.4f} {sweep_p50s[1]:>12.4f} "
              f"{sweep_p50s[2]:>12.4f} "
              f"{spread*100:>+9.3f}% {verdict:>15s}")

    print()
    print("=" * 100)
    print("SUBSIDIARY: ref-alone vs ref-when-paired (does pairing perturb the ref?)")
    print("=" * 100)
    ref_alone = alone_pooled["ref_alone"]
    print(f"{'condition':>15s} {'ref_p50':>10s} {'ref_rcv':>10s} {'shift_vs_alone':>15s}")
    ra_p50 = percentile(ref_alone, 50)
    print(f"{'ref_alone':>15s} {ra_p50:>10.0f} {robust_cv(ref_alone):>10.4f} "
          f"{'-':>15s}")
    for label in ("T1", "T2", "T3", "T4"):
        cond = f"{label}_paired"
        refs = paired_pooled[cond]["ref"]
        rp = percentile(refs, 50)
        shift = (rp - ra_p50) / ra_p50
        print(f"{cond:>15s} {rp:>10.0f} {robust_cv(refs):>10.4f} "
              f"{shift*100:>+14.2f}%")

    print()
    print("=" * 100)
    print("SUBSIDIARY: inter-encoder gap_ns distribution")
    print("=" * 100)
    print(f"{'condition':>15s} {'min':>8s} {'p05':>8s} {'p50':>8s} "
          f"{'p95':>8s} {'p99':>8s} {'max':>10s}")
    for label in ("T1", "T2", "T3", "T4"):
        cond = f"{label}_paired"
        g = paired_pooled[cond]["gap"]
        print(f"{cond:>15s} {min(g):>8d} {percentile(g, 5):>8.0f} "
              f"{percentile(g, 50):>8.0f} {percentile(g, 95):>8.0f} "
              f"{percentile(g, 99):>8.0f} {max(g):>10d}")

    print()
    print("=" * 100)
    print("SUBSIDIARY: alone naive_cv vs robust_cv (outlier indicator)")
    print("=" * 100)
    print(f"{'condition':>15s} {'p50':>10s} {'p95':>10s} {'p99':>10s} "
          f"{'max':>10s} {'robust_cv':>10s} {'naive_cv':>10s}")
    for cond in ("ref_alone", "T1_alone", "T2_alone", "T3_alone", "T4_alone"):
        a = alone_pooled[cond]
        print(f"{cond:>15s} {percentile(a, 50):>10.0f} "
              f"{percentile(a, 95):>10.0f} {percentile(a, 99):>10.0f} "
              f"{max(a):>10d} {robust_cv(a):>10.4f} {naive_cv(a):>10.4f}")
    print()
    print(f"{'condition':>15s} {'p50':>10s} {'p95':>10s} {'p99':>10s} "
          f"{'max':>10s} {'robust_cv':>10s} {'naive_cv':>10s}")
    for label in ("T1", "T2", "T3", "T4"):
        cond = f"{label}_paired"
        t = paired_pooled[cond]["trial"]
        print(f"{cond+' trial':>15s} {percentile(t, 50):>10.0f} "
              f"{percentile(t, 95):>10.0f} {percentile(t, 99):>10.0f} "
              f"{max(t):>10d} {robust_cv(t):>10.4f} {naive_cv(t):>10.4f}")


if __name__ == "__main__":
    main()
