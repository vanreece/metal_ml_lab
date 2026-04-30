# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 015 analysis: pointer-chase amplification.

Same structural pipeline as 014b (per-method linear fits, stable-
region detection, per-cell PWRCTRL classification, trial-level
distribution for wide-spread cells, IOReport temporal join for the
widest b2b cell). Reference numbers updated to compare against the
014b fma_loop slopes -- the headline cross-bottleneck question is
"how does memory-latency-bound slope compare to compute-bound
slope at the same per-iter constant?"
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


# 014b stable-region slopes (M4 Max, AC, 5 000-trial cells,
# fma_loop FMA_PER_ITER=64). Used for the cross-bottleneck slope
# comparison since 015 mirrors 014b exactly except for the kernel.
EXP014B_INTERNAL_SLOPE = 405.17
EXP014B_INTERNAL_INTERCEPT = 6861
EXP014B_B2B_SLOPE = 3156.67
EXP014B_B2B_INTERCEPT = 4087


def load_trials(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "cell_idx": int(r["cell_idx"]),
                "method": r["method"],
                "n_amp": int(r["n_amp"]),
                "trial_idx": int(r["trial_idx"]),
                "monotonic_ns": int(r["monotonic_ns"]),
                "gpu_delta_raw": int(r["gpu_delta_raw"]),
                "cpu_total_ns": int(r["cpu_total_ns"]),
            })
    return rows


def load_cells(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "cell_idx": int(r["cell_idx"]),
                "method": r["method"],
                "n_amp": int(r["n_amp"]),
                "start_ns": int(r["monotonic_ns_start"]),
                "end_ns": int(r["monotonic_ns_end"]),
                "trial_count": int(r["trial_count"]),
                "p10": int(r["p10"]),
                "p50": int(r["p50"]),
                "p90": int(r["p90"]),
                "p99": int(r["p99"]),
                "min": int(r["min"]),
                "max": int(r["max"]),
            })
    return rows


def load_states(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "monotonic_ns": int(r["monotonic_ns"]),
                "channel": r["channel"],
                "state_name": r["state_name"],
                "residency": int(r["residency_24Mticks"]),
            })
    return rows


def aggregate_residency(state_rows, channel, t_start, t_end):
    sums = defaultdict(int)
    for r in state_rows:
        if r["channel"] != channel:
            continue
        if not (t_start <= r["monotonic_ns"] <= t_end):
            continue
        sums[r["state_name"]] += r["residency"]
    total = sum(sums.values())
    if total == 0:
        return {}
    return {n: v / total * 100 for n, v in sums.items()}


def classify_pwrctrl(pwr_dict, deadline_threshold=25):
    if not pwr_dict:
        return "?"
    deadline = pwr_dict.get("DEADLINE", 0)
    perf = pwr_dict.get("PERF", 0)
    idle = pwr_dict.get("IDLE_OFF", 0)
    if deadline >= deadline_threshold and deadline >= perf - 10:
        return "DEADLINE"
    if perf >= 50:
        return "PERF"
    if idle >= 50:
        return "IDLE"
    return "MIXED"


def linear_fit(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    residuals = [(x, y, (intercept + slope * x), y - (intercept + slope * x))
                 for x, y in zip(xs, ys)]
    return {"slope": slope, "intercept": intercept, "r2": r2,
            "residuals": residuals, "n_points": n}


def find_stable_window(ms, stable_tol=0.10):
    if len(ms) < 3:
        return list(range(len(ms)))
    per_step = [
        (ms[i]["p50"] - ms[i - 1]["p50"]) /
        (ms[i]["n_amp"] - ms[i - 1]["n_amp"])
        for i in range(1, len(ms))
    ]
    best = (0, 0)
    for start in range(len(per_step)):
        for end in range(start + 1, len(per_step) + 1):
            window = per_step[start:end]
            med = sorted(window)[len(window) // 2]
            if med == 0:
                continue
            if all(abs(s - med) / abs(med) <= stable_tol for s in window):
                span = end - start + 1
                if span > best[1]:
                    best = (start, span)
    s, span = best
    return list(range(s, s + span))


def fmt_ns(ns):
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.2f} µs"
    return f"{ns:.0f} ns"


def percentile(sorted_xs, q):
    if not sorted_xs:
        return 0
    idx = int(round(q * (len(sorted_xs) - 1)))
    return sorted_xs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--raw-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw")
    ap.add_argument("--dispatch-overhead-ns", type=int, default=6400)
    args = ap.parse_args()

    trials_csv = args.raw_dir / f"{args.prefix}-trials.csv"
    cells_csv = args.raw_dir / f"{args.prefix}-cells.csv"
    states_csv = args.raw_dir / f"{args.prefix}-states.csv"
    for p in [trials_csv, cells_csv, states_csv]:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    trials = load_trials(trials_csv)
    cells = load_cells(cells_csv)
    states = load_states(states_csv)
    print(f"loaded: {len(trials)} trials, {len(cells)} cells, "
          f"{len(states)} state rows")

    # Per-cell PWRCTRL/GPUPH (window = the cell's actual span; no
    # widening needed because cells are long enough).
    print()
    print("=" * 110)
    print("Per-cell timing + DVFS state (full per-cell window, no widening)")
    print("=" * 110)
    print(f"{'cell':>4} {'method':<14} {'N':>5} {'trials':>6} {'p50':>10} "
          f"{'p90':>10} {'p99':>10}  {'PWRCTRL top':<26} {'P15%':>5}  "
          f"{'class':<8}")
    print("-" * 110)
    cell_state = {}
    for c in cells:
        pwr = aggregate_residency(states, "PWRCTRL", c["start_ns"], c["end_ns"])
        gpuph = aggregate_residency(states, "GPUPH", c["start_ns"], c["end_ns"])
        if pwr:
            top_pwr = sorted(pwr.items(), key=lambda kv: -kv[1])[:2]
            pwr_str = " ".join(f"{n}({pct:.0f}%)" for n, pct in top_pwr)
        else:
            pwr_str = "<no data>"
        p15 = gpuph.get("P15", 0.0)
        klass = classify_pwrctrl(pwr)
        cell_state[c["cell_idx"]] = {"pwr": pwr, "gpuph": gpuph, "p15": p15,
                                     "class": klass}
        print(f"{c['cell_idx']:>4} {c['method']:<14} {c['n_amp']:>5} "
              f"{c['trial_count']:>6} {fmt_ns(c['p50']):>10} "
              f"{fmt_ns(c['p90']):>10} {fmt_ns(c['p99']):>10}  "
              f"{pwr_str:<26} {p15:>4.0f}%  {klass:<8}")

    # Per-step slope diagnostic.
    print()
    print("=" * 100)
    print("Per-step incremental slope (forward difference, p50)")
    print("=" * 100)
    for method in ["internal-loop", "back-to-back"]:
        ms = sorted([c for c in cells if c["method"] == method],
                    key=lambda c: c["n_amp"])
        print(f"\n--- {method} ---")
        for i in range(1, len(ms)):
            dy = ms[i]["p50"] - ms[i - 1]["p50"]
            dn = ms[i]["n_amp"] - ms[i - 1]["n_amp"]
            print(f"  N {ms[i-1]['n_amp']:>4} -> {ms[i]['n_amp']:<4}  "
                  f"Δp50 = {dy:>+10} ns / ΔN = {dn:>4}  "
                  f"=> {dy / dn:>9.2f} ns/N")

    # Linear fits + stable-region detection (same heuristic as 014).
    print()
    print("=" * 100)
    print("Linear fit t(N) = a + b·N per method")
    print("=" * 100)
    fits_full = {}
    fits_stable = {}
    stable_indices = {}
    for method in ["internal-loop", "back-to-back"]:
        ms = sorted([c for c in cells if c["method"] == method],
                    key=lambda c: c["n_amp"])
        xs = [c["n_amp"] for c in ms]
        ys = [c["p50"] for c in ms]
        full_fit = linear_fit(xs, ys)
        fits_full[method] = full_fit

        stable_idx = find_stable_window(ms)
        stable_indices[method] = stable_idx
        s_xs = [ms[i]["n_amp"] for i in stable_idx]
        s_ys = [ms[i]["p50"] for i in stable_idx]
        stable_fit = linear_fit(s_xs, s_ys) if len(s_xs) >= 2 else None
        fits_stable[method] = stable_fit

        print()
        print(f"--- {method} ---")
        print(f"  full fit (all {len(xs)} N): "
              f"slope={full_fit['slope']:.2f} ns/N  "
              f"intercept={full_fit['intercept']:.0f} ns  "
              f"R²={full_fit['r2']:.4f}")
        if stable_fit is not None and len(s_xs) > 1:
            in_stable = sorted(s_xs)
            print(f"  stable-region fit (N in {in_stable}, "
                  f"{len(s_xs)} cells):")
            print(f"    slope={stable_fit['slope']:.2f} ns/N  "
                  f"intercept={stable_fit['intercept']:.0f} ns  "
                  f"R²={stable_fit['r2']:.4f}")
            for x, y, fitted, resid in stable_fit["residuals"]:
                pct = (resid / fitted * 100) if fitted else float("nan")
                print(f"      N={x:>5}: median={fmt_ns(y):>10}  "
                      f"fit={fmt_ns(fitted):>10}  "
                      f"resid={resid:>+9.0f} ns ({pct:+5.1f}%)")
        else:
            print(f"  stable-region fit: <2 cells in stable window")

    # Cross-bottleneck slope comparison: 015 (chase) vs 014b (fma).
    # Both have per-iter constant of 64, so direct ratio is the
    # memory-vs-compute per-iter latency ratio at the regime.
    print()
    print("=" * 100)
    print("Cross-bottleneck slope comparison: 015 chase vs 014b fma_loop")
    print("=" * 100)
    print(f"  Both kernels: 64 chained operations per amp-step "
          f"(CHASE_PER_ITER = FMA_PER_ITER = 64).")
    print()
    rows = [
        ("internal-loop", "slope (ns / amp-step)", EXP014B_INTERNAL_SLOPE,
         fits_stable["internal-loop"]["slope"]
         if fits_stable["internal-loop"] else None),
        ("internal-loop", "intercept (ns)", EXP014B_INTERNAL_INTERCEPT,
         fits_stable["internal-loop"]["intercept"]
         if fits_stable["internal-loop"] else None),
        ("back-to-back", "slope (ns / amp-step)", EXP014B_B2B_SLOPE,
         fits_stable["back-to-back"]["slope"]
         if fits_stable["back-to-back"] else None),
        ("back-to-back", "intercept (ns)", EXP014B_B2B_INTERCEPT,
         fits_stable["back-to-back"]["intercept"]
         if fits_stable["back-to-back"] else None),
    ]
    print(f"  {'method':<14} {'metric':<24} {'fma 014b':>14} "
          f"{'chase 015':>14} {'ratio':>8}")
    for method, metric, vfma, vchase in rows:
        if vchase is None:
            print(f"  {method:<14} {metric:<24} {vfma:>14.2f} "
                  f"{'<n/a>':>14}")
            continue
        ratio = vchase / vfma if vfma else float("nan")
        print(f"  {method:<14} {metric:<24} {vfma:>14.2f} "
              f"{vchase:>14.2f} {ratio:>7.2f}x")
    fit_int = fits_stable.get("internal-loop")
    if fit_int:
        per_load_ns = fit_int["slope"] / 64
        print()
        print(f"  Per-load latency (chase internal-loop slope / 64) = "
              f"{per_load_ns:.1f} ns")
        print(f"  Per-fma latency (fma_loop 014b slope / 64)        = "
              f"{EXP014B_INTERNAL_SLOPE / 64:.1f} ns")
        print(f"  Memory-vs-compute per-iter latency ratio          = "
              f"{per_load_ns / (EXP014B_INTERNAL_SLOPE / 64):.1f}x")

    # Trial-level distribution for cells with wide spread (p99/p50 > 1.5).
    # These are the bimodal cells where the b2b collapse lives.
    print()
    print("=" * 100)
    print("Trial-level distribution for wide-spread cells (p99/p50 > 1.5)")
    print("=" * 100)
    by_cell = defaultdict(list)
    for t in trials:
        by_cell[t["cell_idx"]].append(t)
    for c in cells:
        if c["p50"] == 0:
            continue
        spread = c["p99"] / c["p50"]
        if spread <= 1.5:
            continue
        ts = sorted(by_cell[c["cell_idx"]], key=lambda t: t["gpu_delta_raw"])
        deltas = [t["gpu_delta_raw"] for t in ts]
        print(f"\n--- cell {c['cell_idx']}: {c['method']} N={c['n_amp']}, "
              f"{c['trial_count']} trials, p99/p50 = {spread:.2f} ---")
        for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
            v = percentile(deltas, q)
            print(f"  p{int(q*100):>2}: {fmt_ns(v):>12}")
        # Histogram-ish: 10 equal-width buckets between min and max
        lo, hi = deltas[0], deltas[-1]
        nb = 12
        edges = [lo + i * (hi - lo) / nb for i in range(nb + 1)]
        counts = [0] * nb
        for d in deltas:
            b = min(nb - 1, int((d - lo) / (hi - lo) * nb)) if hi > lo else 0
            counts[b] += 1
        print(f"  histogram ({nb} buckets, lo={fmt_ns(lo)} hi={fmt_ns(hi)}):")
        max_count = max(counts) or 1
        for i, cnt in enumerate(counts):
            bar = "#" * int(60 * cnt / max_count)
            print(f"    [{fmt_ns(edges[i]):>10}, {fmt_ns(edges[i+1]):>10}) "
                  f"{cnt:>4}  {bar}")

    # Trial -> IOReport sample temporal join for the b2b N=1024 cell.
    # Question: do fast trials cluster in DEADLINE-dominant or PERF-
    # dominant 250 ms windows?
    print()
    print("=" * 100)
    print("Trial → IOReport temporal join: b2b N=1024 cell")
    print("=" * 100)
    target_cell = None
    for c in cells:
        if c["method"] == "back-to-back" and c["n_amp"] == 1024:
            target_cell = c
            break
    if target_cell is None:
        print("(no b2b N=1024 cell in this run)")
    else:
        # Per-state row stream for PWRCTRL channel within the cell
        # window. Each state row has monotonic_ns + 24M-tick residency
        # for one (state_name) within the prior 250 ms. We'll build
        # 250 ms buckets keyed by IOReport monotonic_ns and label
        # each bucket with its dominant PWRCTRL state.
        cell_states = [s for s in states
                       if s["channel"] == "PWRCTRL"
                       and target_cell["start_ns"] - 300_000_000
                       <= s["monotonic_ns"]
                       <= target_cell["end_ns"] + 300_000_000]
        # Group by sample (each unique monotonic_ns is one sample).
        per_ts = defaultdict(dict)
        for s in cell_states:
            per_ts[s["monotonic_ns"]][s["state_name"]] = s["residency"]
        sample_ts = sorted(per_ts.keys())
        sample_label = {}
        sample_window = {}
        for i, ts in enumerate(sample_ts):
            d = per_ts[ts]
            tot = sum(d.values()) or 1
            top = sorted(d.items(), key=lambda kv: -kv[1])[0]
            label = top[0]
            sample_label[ts] = label
            # Sample at ts represents activity in [prev_ts, ts].
            prev_ts = sample_ts[i - 1] if i > 0 else ts - 250_000_000
            sample_window[ts] = (prev_ts, ts)
        print(f"  {len(sample_ts)} PWRCTRL samples in cell window")
        # Bin trials by which sample's window they fall in.
        ts_to_trials = defaultdict(list)
        target_trials = by_cell[target_cell["cell_idx"]]
        for t in target_trials:
            tns = t["monotonic_ns"]
            assigned = None
            for ts in sample_ts:
                lo, hi = sample_window[ts]
                if lo <= tns < hi:
                    assigned = ts
                    break
            if assigned is None:
                continue
            ts_to_trials[assigned].append(t)
        # Per-sample summary with PWRCTRL label.
        print(f"  {'sample_idx':>10} {'pwrctrl':<10} "
              f"{'n_trials':>8} {'p50':>10} {'p90':>10} {'p99':>10}")
        rows_by_label = defaultdict(list)
        for i, ts in enumerate(sample_ts):
            tlist = ts_to_trials.get(ts, [])
            if not tlist:
                continue
            ds = sorted(t["gpu_delta_raw"] for t in tlist)
            p50 = percentile(ds, 0.5)
            p90 = percentile(ds, 0.9)
            p99 = percentile(ds, 0.99)
            label = sample_label.get(ts, "?")
            print(f"  {i:>10} {label:<10} {len(tlist):>8} "
                  f"{fmt_ns(p50):>10} {fmt_ns(p90):>10} {fmt_ns(p99):>10}")
            rows_by_label[label].extend(t["gpu_delta_raw"] for t in tlist)
        print()
        print(f"  Aggregated by PWRCTRL label:")
        for label in sorted(rows_by_label.keys()):
            ds = sorted(rows_by_label[label])
            if not ds:
                continue
            print(f"    {label:<10}  N={len(ds):>5}  "
                  f"p10={fmt_ns(percentile(ds, 0.10)):>10}  "
                  f"p50={fmt_ns(percentile(ds, 0.50)):>10}  "
                  f"p90={fmt_ns(percentile(ds, 0.90)):>10}  "
                  f"p99={fmt_ns(percentile(ds, 0.99)):>10}")

        # Mann-Whitney-ish: test whether DEADLINE p50 < PERF p50.
        if "DEADLINE" in rows_by_label and "PERF" in rows_by_label:
            d_p50 = sorted(rows_by_label["DEADLINE"])
            p_p50 = sorted(rows_by_label["PERF"])
            d_med = percentile(d_p50, 0.5)
            p_med = percentile(p_p50, 0.5)
            ratio = d_med / p_med if p_med else float("nan")
            print()
            print(f"  DEADLINE p50 / PERF p50 = {ratio:.3f}")
            if ratio < 0.6:
                print(f"  VERDICT: DVFS-EXPLAINED  "
                      f"(DEADLINE-window trials systematically faster)")
            elif ratio > 1.4:
                print(f"  VERDICT: DVFS-EXPLAINED  "
                      f"(DEADLINE-window trials systematically slower)")
            elif 0.85 <= ratio <= 1.15:
                print(f"  VERDICT: DVFS-EXCLUDED  "
                      f"(no per-window timing difference)")
            else:
                print(f"  VERDICT: DVFS-PARTIAL  "
                      f"(some difference but not the full collapse)")


if __name__ == "__main__":
    main()
