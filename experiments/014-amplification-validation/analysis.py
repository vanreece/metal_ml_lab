# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 014 analysis: linear fits per method, DVFS regime detection
via per-cell PWRCTRL/GPUPH residency, and the per-encode cost
(slope_b2b - slope_internal).

Fits use median per-cell delta as the y value (not mean) so the
N=1024 back-to-back bimodal cell doesn't drag the fit. We also
report a separate fit on the "PERF-only" subset of cells -- those
whose dominant PWRCTRL state during the cell window is PERF -- since
the pre-reg specifically expects piecewise linearity across regimes.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


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
                "p10": int(r["p10"]),
                "p50": int(r["p50"]),
                "p90": int(r["p90"]),
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


def aggregate_residency(state_rows, channel: str, t_start: int, t_end: int):
    sums: dict[str, int] = defaultdict(int)
    for r in state_rows:
        if r["channel"] != channel:
            continue
        if not (t_start <= r["monotonic_ns"] <= t_end):
            continue
        sums[r["state_name"]] += r["residency"]
    total = sum(sums.values())
    if total == 0:
        return {}
    return {name: val / total * 100 for name, val in sums.items()}


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
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "residuals": residuals,
        "n_points": n,
    }


def verdict_for_fit(fit, cells_used, dispatch_overhead_ns=6400):
    """Per-method verdict per pre-reg thresholds."""
    if fit is None:
        return "FAIL", "no fit"
    r2 = fit["r2"]
    intercept = fit["intercept"]
    max_resid_pct = max(
        abs(resid) / fitted * 100 if fitted else 0.0
        for x, y, fitted, resid in fit["residuals"]
    )
    intercept_match = (
        0.7 * dispatch_overhead_ns <= intercept <= 1.3 * dispatch_overhead_ns
    )
    notes = []
    if r2 < 0.99:
        notes.append(f"R² {r2:.4f} < 0.99")
    if max_resid_pct > 5:
        notes.append(f"max resid {max_resid_pct:.1f}% > 5%")
    if not intercept_match:
        notes.append(f"intercept {intercept:.0f} ns not within ±30% of "
                     f"{dispatch_overhead_ns} ns")
    if r2 > 0.99 and max_resid_pct < 5 and intercept_match:
        return "PASS", "clean linear"
    if r2 > 0.99 and intercept_match:
        return "MARGINAL", "; ".join(notes) or "linear but residuals high"
    return "FAIL", "; ".join(notes) or "fit not clean"


def fmt_ns(ns):
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.2f} µs"
    return f"{ns:.0f} ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--raw-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw")
    ap.add_argument("--dispatch-overhead-ns", type=int, default=6400,
                    help="prior baseline for intercept-match check (default 6400)")
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

    # Per-cell PWRCTRL/GPUPH residency (cells are short relative to the
    # 250 ms IOReport interval, so we widen the aggregation window
    # by half an interval on each side -- analysis.py's job, not run.py's).
    WIDEN_NS = 250_000_000 // 2
    print()
    print("=" * 100)
    print("Per-cell timing + DVFS state (residency window widened ± 125 ms)")
    print("=" * 100)
    print(f"{'cell':>4} {'method':<14} {'N':>5} {'p50':>10} {'p90':>10}  "
          f"{'PWRCTRL top':<22}  {'P15%':>5}")
    print("-" * 100)
    cell_state = {}
    for c in cells:
        ws = c["start_ns"] - WIDEN_NS
        we = c["end_ns"] + WIDEN_NS
        pwr = aggregate_residency(states, "PWRCTRL", ws, we)
        gpuph = aggregate_residency(states, "GPUPH", ws, we)
        if pwr:
            top_pwr = sorted(pwr.items(), key=lambda kv: -kv[1])[:2]
            pwr_str = " ".join(f"{n}({pct:.0f}%)" for n, pct in top_pwr)
        else:
            pwr_str = "<no data>"
        p15 = gpuph.get("P15", 0.0)
        cell_state[c["cell_idx"]] = {"pwr": pwr, "gpuph": gpuph, "p15": p15}
        print(f"{c['cell_idx']:>4} {c['method']:<14} {c['n_amp']:>5} "
              f"{fmt_ns(c['p50']):>10} {fmt_ns(c['p90']):>10}  "
              f"{pwr_str:<22}  {p15:>4.0f}%")

    # Per-step (forward-difference) slope diagnostic. Reveals where
    # the regime transitions live -- pre-reg predicted a kink and we
    # want to identify it from the timing data, since the IOReport
    # 250 ms window is too coarse to classify short cells.
    print()
    print("=" * 100)
    print("Per-step incremental slope (forward difference)")
    print("=" * 100)
    for method in ["internal-loop", "back-to-back"]:
        ms = sorted([c for c in cells if c["method"] == method],
                    key=lambda c: c["n_amp"])
        print(f"\n--- {method} ---")
        for i in range(1, len(ms)):
            dy = ms[i]["p50"] - ms[i - 1]["p50"]
            dn = ms[i]["n_amp"] - ms[i - 1]["n_amp"]
            print(f"  N {ms[i-1]['n_amp']:>4} -> {ms[i]['n_amp']:<4}  "
                  f"Δp50 = {dy:>+9} ns / ΔN = {dn:>4}  "
                  f"=> {dy / dn:>8.2f} ns/N")

    # Auto-detect the largest contiguous N window with stable per-step
    # slope (within ± stable_tol of the window's median per-step slope).
    # That window is the "single-regime" subset to fit on.
    def find_stable_window(ms, stable_tol=0.10):
        if len(ms) < 3:
            return list(range(len(ms)))
        per_step = [
            (ms[i]["p50"] - ms[i - 1]["p50"]) /
            (ms[i]["n_amp"] - ms[i - 1]["n_amp"])
            for i in range(1, len(ms))
        ]
        # Search for the longest contiguous run where every step slope
        # is within ±stable_tol of the run's median.
        best = (0, 0)  # (start_idx_in_ms, length)
        for start in range(len(per_step)):
            for end in range(start + 1, len(per_step) + 1):
                window = per_step[start:end]
                med = sorted(window)[len(window) // 2]
                if med == 0:
                    continue
                if all(abs(s - med) / abs(med) <= stable_tol for s in window):
                    span = end - start + 1  # cells covered = steps + 1
                    if span > best[1]:
                        best = (start, span)
        # Best run uses cells [start..start+span-1] in ms order.
        s, span = best
        return list(range(s, s + span))

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
        print(f"  per-N residuals vs full fit:")
        for x, y, fitted, resid in full_fit["residuals"]:
            pct = (resid / fitted * 100) if fitted else float("nan")
            print(f"    N={x:>5}: median={fmt_ns(y):>10}  "
                  f"fit={fmt_ns(fitted):>10}  resid={resid:>+9.0f} ns "
                  f"({pct:+5.1f}%)")
        if stable_fit is not None and len(s_xs) > 1:
            in_stable = sorted(s_xs)
            print(f"  stable-region fit (auto-detected, N in {in_stable}, "
                  f"{len(s_xs)} cells):")
            print(f"    slope={stable_fit['slope']:.2f} ns/N  "
                  f"intercept={stable_fit['intercept']:.0f} ns  "
                  f"R²={stable_fit['r2']:.4f}")
            print(f"    per-N residuals vs stable fit:")
            for x, y, fitted, resid in stable_fit["residuals"]:
                pct = (resid / fitted * 100) if fitted else float("nan")
                print(f"      N={x:>5}: median={fmt_ns(y):>10}  "
                      f"fit={fmt_ns(fitted):>10}  "
                      f"resid={resid:>+9.0f} ns ({pct:+5.1f}%)")
        else:
            print(f"  stable-region fit: <2 cells in stable window")

        v_full, n_full = verdict_for_fit(full_fit, ms,
                                         args.dispatch_overhead_ns)
        if stable_fit:
            v_stable, n_stable = verdict_for_fit(
                stable_fit, [ms[i] for i in stable_idx],
                args.dispatch_overhead_ns,
            )
        else:
            v_stable, n_stable = "n/a", "no stable subset"
        print(f"  VERDICT (full):     {v_full}  -- {n_full}")
        print(f"  VERDICT (stable):   {v_stable}  -- {n_stable}")
        if v_full != "PASS" and v_stable == "PASS" and len(s_xs) < len(xs):
            print(f"  COMBINED VERDICT:   MARGINAL "
                  f"(piecewise-linear; stable region "
                  f"N={sorted(s_xs)[0]}..{sorted(s_xs)[-1]} clean, "
                  f"{len(xs) - len(s_xs)} cells outside)")

    # Per-encode cost.
    print()
    print("=" * 100)
    print("Per-encode cost = slope_b2b - slope_internal")
    print("=" * 100)
    for label, fits in [("full fit", fits_full), ("stable-region fit", fits_stable)]:
        f_int = fits.get("internal-loop")
        f_b2b = fits.get("back-to-back")
        if f_int is None or f_b2b is None:
            print(f"  {label}: missing one method, skipping")
            continue
        diff = f_b2b["slope"] - f_int["slope"]
        print(f"  {label}:")
        print(f"    internal slope  = {f_int['slope']:.2f} ns/N")
        print(f"    back-to-back    = {f_b2b['slope']:.2f} ns/N")
        print(f"    per-encode cost = {diff:.2f} ns "
              f"(exp 005 inter-encoder gap was 833 ns)")
        if diff <= 0:
            print(f"    VERDICT: FAIL (non-positive)")
        elif diff > args.dispatch_overhead_ns:
            print(f"    VERDICT: FAIL (> dispatch overhead {args.dispatch_overhead_ns} ns)")
        elif 200 <= diff <= 3000:
            print(f"    VERDICT: PASS (a few hundred ns to a couple µs)")
        else:
            print(f"    VERDICT: MARGINAL (positive but outside expected range)")

    # Print the summary table the pre-reg specifically called out.
    print()
    print("=" * 100)
    print("Summary table (pre-reg-shaped)")
    print("=" * 100)
    print(f"{'method':<14} | {'slope (ns/N)':>14} | {'intercept (ns)':>16} | "
          f"{'R²':>8} | {'DVFS regime per N':<60}")
    print("-" * 130)
    for method in ["internal-loop", "back-to-back"]:
        fit = fits_full[method]
        ms = sorted([c for c in cells if c["method"] == method],
                    key=lambda c: c["n_amp"])
        regime_strs = []
        for c in ms:
            cs = cell_state.get(c["cell_idx"], {})
            pwr = cs.get("pwr", {})
            if not pwr:
                tag = "?"
            else:
                top = max(pwr.items(), key=lambda kv: kv[1])
                tag = top[0][:4]
            regime_strs.append(f"N={c['n_amp']}:{tag}")
        regime_blob = " ".join(regime_strs)
        print(f"{method:<14} | {fit['slope']:>14.2f} | "
              f"{fit['intercept']:>16.0f} | {fit['r2']:>8.4f} | {regime_blob:<60}")


if __name__ == "__main__":
    main()
