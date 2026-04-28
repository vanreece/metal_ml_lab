# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Analysis for experiment 006: cross-session ratio stability.

Reads the latest 006 linkage file (or one specified on argv) to find
the two 005 timestamp prefixes (session A and session B), then loads
both pairs of CSVs from 005/raw/ and reports the cross-session
spread for each trial kernel.

Pass / marginal / fail thresholds per 006/README.md:
  - pass:     spread <= 1 %
  - marginal: spread in (1 %, 3 %]
  - fail:     spread >  3 %

Stdlib only. One-off, not a library.
"""
from __future__ import annotations

import csv
import math
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


def robust_cv(xs):
    if not xs:
        return float("nan")
    p25 = percentile(xs, 25)
    p50 = percentile(xs, 50)
    p75 = percentile(xs, 75)
    iqr = p75 - p25
    sigma = iqr / 1.349  # IQR / 1.349 ≈ stddev for a normal
    return sigma / p50 if p50 else float("nan")


def naive_cv(xs):
    if len(xs) < 2:
        return float("nan")
    p50 = percentile(xs, 50)
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var) / p50 if p50 else float("nan")


def load_paired(path):
    """Return dict: trial_label ('T1'..'T4') -> {'ratio': [...], 'trial_delta': [...],
    'ref_delta': [...], 'gap': [...]}.

    005's paired CSV labels conditions T1_paired ... T4_paired."""
    out = defaultdict(lambda: {"ratio": [], "trial_delta": [], "ref_delta": [], "gap": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            # Strip the "_paired" suffix to get T1 / T2 / T3 / T4
            tlabel = cond.split("_")[0]
            out[tlabel]["ratio"].append(float(row["ratio"]))
            out[tlabel]["trial_delta"].append(float(row["trial_delta_raw"]))
            out[tlabel]["ref_delta"].append(float(row["ref_delta_raw"]))
            out[tlabel]["gap"].append(float(row["gap_ns"]))
    return out


def load_alone(path):
    """Return dict: trial_label -> [gpu_delta_raw, ...] (for trial-alone p50)."""
    out = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            tlabel = cond.split("_")[0]   # T1_alone -> T1, ref_alone -> ref
            out[tlabel].append(float(row["gpu_delta_raw"]))
    return out


def verdict(spread_pct):
    if spread_pct <= 1.0:
        return "PASS"
    if spread_pct <= 3.0:
        return "MARGINAL"
    return "FAIL"


def main() -> None:
    raw_dir = Path(__file__).resolve().parent / "raw"
    if len(sys.argv) > 1:
        link_path = raw_dir / sys.argv[1]
    else:
        links = sorted(raw_dir.glob("*-cross-session.txt"))
        if not links:
            print("no 006 linkage files found in raw/", file=sys.stderr)
            sys.exit(1)
        link_path = links[-1]

    print(f"# linkage: {link_path.name}")
    info = {}
    for line in link_path.read_text().splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            info[k.strip()] = v.strip()
    prefix_a = info["session_a_prefix"]
    prefix_b = info["session_b_prefix"]
    gap_s = info.get("session_gap_s", "?")
    print(f"# session A: {prefix_a}   session B: {prefix_b}   gap: {gap_s}s")
    print(f"# A wall: {info.get('session_a_wall_clock_s', '?')}s   "
          f"sleep actual: {info.get('sleep_actual_s', '?')}s   "
          f"B wall: {info.get('session_b_wall_clock_s', '?')}s")

    exp005_raw = link_path.parent.parent.parent / "005-paired-ratio-stability" / "raw"
    paired_a = load_paired(exp005_raw / f"{prefix_a}-paired.csv")
    paired_b = load_paired(exp005_raw / f"{prefix_b}-paired.csv")
    alone_a = load_alone(exp005_raw / f"{prefix_a}-alone.csv")
    alone_b = load_alone(exp005_raw / f"{prefix_b}-alone.csv")

    print()
    print("=" * 110)
    print("PRIMARY: cross-session ratio stability (per 006/README success criteria)")
    print("=" * 110)
    hdr = ("trial", "A_ratio_p50", "B_ratio_p50", "spread%", "A_ratio_rcv", "B_ratio_rcv", "verdict")
    print(f"{hdr[0]:>5}  {hdr[1]:>12}  {hdr[2]:>12}  {hdr[3]:>9}  {hdr[4]:>11}  {hdr[5]:>11}  {hdr[6]:>9}")
    for t in ["T1", "T2", "T3", "T4"]:
        a = paired_a[t]["ratio"]
        b = paired_b[t]["ratio"]
        if not a or not b:
            print(f"{t:>5}  (missing data)")
            continue
        pa = percentile(a, 50)
        pb = percentile(b, 50)
        spread = abs(pb - pa) / ((pa + pb) / 2) * 100
        ra = robust_cv(a)
        rb = robust_cv(b)
        print(f"{t:>5}  {pa:>12.5f}  {pb:>12.5f}  {spread:>8.3f}%  "
              f"{ra:>11.5f}  {rb:>11.5f}  {verdict(spread):>9}")

    print()
    print("=" * 110)
    print("SUBSIDIARY: cross-session trial-alone p50 (does the ABSOLUTE number drift, even when ratio doesn't?)")
    print("=" * 110)
    print(f"{'trial':>5}  {'A_alone_p50':>12}  {'B_alone_p50':>12}  {'shift%':>9}  {'A_alone_rcv':>11}  {'B_alone_rcv':>11}")
    for t in ["T1", "T2", "T3", "T4"]:
        a = alone_a[t]
        b = alone_b[t]
        if not a or not b:
            print(f"{t:>5}  (missing data)")
            continue
        pa = percentile(a, 50)
        pb = percentile(b, 50)
        shift = (pb - pa) / pa * 100
        print(f"{t:>5}  {pa:>12.0f}  {pb:>12.0f}  {shift:>+8.2f}%  {robust_cv(a):>11.5f}  {robust_cv(b):>11.5f}")

    print()
    print("=" * 110)
    print("SUBSIDIARY: cross-session reference kernel p50 (was the ref itself the same in both sessions?)")
    print("=" * 110)
    pa_ref = percentile(alone_a["ref"], 50) if alone_a.get("ref") else float("nan")
    pb_ref = percentile(alone_b["ref"], 50) if alone_b.get("ref") else float("nan")
    if pa_ref and not math.isnan(pa_ref):
        shift = (pb_ref - pa_ref) / pa_ref * 100
        print(f"  ref_alone   A_p50={pa_ref:.0f}  B_p50={pb_ref:.0f}  shift={shift:+.2f}%  "
              f"A_rcv={robust_cv(alone_a['ref']):.5f}  B_rcv={robust_cv(alone_b['ref']):.5f}")

    print()
    print("=" * 110)
    print("SUBSIDIARY: cross-session inter-encoder gap p50 (did the front-end behavior shift?)")
    print("=" * 110)
    print(f"{'trial':>5}  {'A_gap_p50':>10}  {'B_gap_p50':>10}  {'shift%':>9}")
    for t in ["T1", "T2", "T3", "T4"]:
        if not paired_a[t]["gap"] or not paired_b[t]["gap"]:
            continue
        pa = percentile(paired_a[t]["gap"], 50)
        pb = percentile(paired_b[t]["gap"], 50)
        shift = (pb - pa) / pa * 100 if pa else float("nan")
        print(f"{t:>5}  {pa:>10.0f}  {pb:>10.0f}  {shift:>+8.2f}%")


if __name__ == "__main__":
    main()
