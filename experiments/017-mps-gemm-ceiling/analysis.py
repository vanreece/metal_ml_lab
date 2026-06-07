# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 017 analysis: MPS GEMM ceiling vs naive matmul (016).

Reads:
  - experiments/017-mps-gemm-ceiling/raw/{prefix}-cells.csv
  - experiments/016-matmul-discrimination/raw/{baseline-prefix}-cells.csv
    (default: latest run = 20260429T213959; --naive-prefix to override)

For each of 016's 34 shapes:
  - 017 MPS p50 (single-shot ns/matmul, includes dispatch overhead).
  - 016 naive: slope of t(N_AMP) → ns/matmul (compute, no overhead),
    AND p50 at N_AMP=1 → ns/matmul (with overhead). Both reported.
  - Ratio: naive / MPS, for each baseline definition.

Pre-reg verdicts applied:
  - Methodology: PASS / MARGINAL / FAIL based on per-shape p50 stability
    (CV < 30 % across trials? all 34 shapes resolvable?).
  - Cross-shape gap: WIDE (≥5×) / NARROW (<2×) / SHAPE-DEPENDENT (≥3×
    range across shapes) on the 016 compute plateau M=N=K ∈ [128, 768].
  - Absolute MPS plateau: CALIBRATED (≥14 TFLOP/s ≈ 50 % of fp32 peak
    on a large square) / UNDER-PERFORMS (< 5 TFLOP/s).
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


# Theoretical fp32 peak on M4 Max from exp 012 / state snapshot.
M4_MAX_PEAK_FP32_TFLOPS = 28.0          # at 1.578 GHz
MPS_CALIBRATED_TFLOPS = 14.0            # pre-reg threshold
MPS_UNDER_PERFORMS_TFLOPS = 5.0         # pre-reg threshold
COMPUTE_PLATEAU_M_LO = 128
COMPUTE_PLATEAU_M_HI = 768

# 016 reference numbers (from snapshot).
NAIVE_PLATEAU_TFLOPS = 2.0


def load_017_cells(path: Path):
    """017 cells: one row per shape (no n_amp)."""
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "cell_idx": int(r["cell_idx"]),
                "sweep": r["sweep"],
                "m": int(r["m"]),
                "n": int(r["n"]),
                "k": int(r["k"]),
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


def load_017_trials(path: Path):
    """017 trials: per-trial gpu_delta_raw (ns) by (m,n,k). Used for
    CV stability estimate, since the cell summary doesn't carry it."""
    deltas_by_shape = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (int(r["m"]), int(r["n"]), int(r["k"]))
            deltas_by_shape[key].append(int(r["gpu_delta_raw"]))
    return deltas_by_shape


def load_016_cells(path: Path):
    """016 cells: multiple rows per shape (one per N_AMP)."""
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "cell_idx": int(r["cell_idx"]),
                "sweep": r["sweep"],
                "m": int(r["m"]),
                "n": int(r["n"]),
                "k": int(r["k"]),
                "n_amp": int(r["n_amp"]),
                "p50": int(r["p50"]),
                "trial_count": int(r["trial_count"]),
            })
    return rows


def linear_fit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
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
    return {"slope": slope, "intercept": intercept, "r2": r2, "n_points": n}


def naive_baseline_for_shape(naive_cells_by_shape, key):
    """Returns (slope_ns, n_amp1_ns, fit_r2) for a shape's naive
    baseline. slope_ns = ns / matmul (compute, dispatch-free).
    n_amp1_ns = naive p50 at N_AMP=1 (with overhead)."""
    cells = naive_cells_by_shape.get(key, [])
    if not cells:
        return None
    xs = [c["n_amp"] for c in cells]
    ys = [c["p50"] for c in cells]
    fit = linear_fit(xs, ys)
    if fit is None:
        return None
    slope_ns = fit["slope"]
    n_amp1 = next((c["p50"] for c in cells if c["n_amp"] == 1), None)
    return {"slope_ns": slope_ns, "intercept_ns": fit["intercept"],
            "r2": fit["r2"], "n_amp1_ns": n_amp1}


def fmt_ns(ns):
    if ns is None:
        return "  -"
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.2f} µs"
    return f"{ns:.0f} ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="017 run timestamp prefix, e.g. 20260505T103000")
    ap.add_argument("--raw-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw")
    ap.add_argument("--naive-prefix", default="20260429T213959",
                    help="016 cells.csv timestamp prefix (default: latest)")
    ap.add_argument("--naive-raw-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent
                            / "016-matmul-discrimination" / "raw")
    args = ap.parse_args()

    cells_017 = load_017_cells(args.raw_dir / f"{args.prefix}-cells.csv")
    trials_017 = load_017_trials(args.raw_dir / f"{args.prefix}-trials.csv")
    cells_016 = load_016_cells(
        args.naive_raw_dir / f"{args.naive_prefix}-cells.csv"
    )
    print(f"loaded 017: {len(cells_017)} cells")
    print(f"loaded 016 (naive baseline {args.naive_prefix}): "
          f"{len(cells_016)} cells")

    # Group 016 cells by shape.
    naive_by_shape = defaultdict(list)
    for c in cells_016:
        naive_by_shape[(c["m"], c["n"], c["k"])].append(c)
    for key in naive_by_shape:
        naive_by_shape[key].sort(key=lambda c: c["n_amp"])

    # ------------------------------------------------------------------
    # 1) Per-shape MPS table.
    # ------------------------------------------------------------------
    print()
    print("=" * 130)
    print("Per-shape MPS p50 + naive baseline + ratio")
    print("=" * 130)
    print(f"{'cell':>4} {'sweep':<11} {'shape':<18} {'trials':>6} "
          f"{'MPS p50':>10} {'CV%':>6} "
          f"{'MPS ps/F':>10} {'MPS TFLOP/s':>11}  "
          f"{'naive slope':>11} {'naive N=1':>10}  "
          f"{'r_slope':>8} {'r_N=1':>7}")
    print("-" * 130)

    sweep_order = ["square", "ksweep_128", "ksweep_512", "membound"]
    cells_017.sort(
        key=lambda c: (sweep_order.index(c["sweep"]),
                       c["m"] * c["n"] * c["k"], c["k"])
    )

    rows_for_summary = []
    for c in cells_017:
        shape_str = f"{c['m']}x{c['n']}x{c['k']}"
        flops = 2 * c["m"] * c["n"] * c["k"]
        mps_p50 = c["p50"]
        # Per-shape CV% from raw deltas.
        deltas = trials_017.get((c["m"], c["n"], c["k"]), [])
        if len(deltas) >= 2 and statistics.mean(deltas) > 0:
            cv_pct = statistics.stdev(deltas) / statistics.mean(deltas) * 100
        else:
            cv_pct = float("nan")

        ps_per_flop = mps_p50 / flops * 1e3   # ns / flop * 1e3 = ps/flop
        tflops = flops / mps_p50 / 1e3

        baseline = naive_baseline_for_shape(naive_by_shape,
                                            (c["m"], c["n"], c["k"]))
        if baseline:
            slope_ns = baseline["slope_ns"]
            n_amp1_ns = baseline["n_amp1_ns"]
            r_slope = (slope_ns / mps_p50) if mps_p50 > 0 else None
            r_n1 = (n_amp1_ns / mps_p50) if (
                n_amp1_ns is not None and mps_p50 > 0
            ) else None
        else:
            slope_ns = None
            n_amp1_ns = None
            r_slope = None
            r_n1 = None

        rows_for_summary.append({
            "sweep": c["sweep"], "m": c["m"], "n": c["n"], "k": c["k"],
            "flops": flops, "mps_p50_ns": mps_p50, "cv_pct": cv_pct,
            "ps_per_flop": ps_per_flop, "tflops": tflops,
            "naive_slope_ns": slope_ns, "naive_n_amp1_ns": n_amp1_ns,
            "ratio_slope": r_slope, "ratio_n_amp1": r_n1,
            "trial_count": c["trial_count"],
        })

        cv_str = f"{cv_pct:.1f}" if not (cv_pct != cv_pct) else "  -"
        ps_str = f"{ps_per_flop:.3f}"
        tflops_str = f"{tflops:.3f}"
        slope_str = fmt_ns(slope_ns)
        n1_str = fmt_ns(n_amp1_ns)
        r_slope_str = f"{r_slope:.2f}×" if r_slope is not None else "  -"
        r_n1_str = f"{r_n1:.2f}×" if r_n1 is not None else "  -"
        print(f"{c['cell_idx']:>4} {c['sweep']:<11} {shape_str:<18} "
              f"{c['trial_count']:>6} "
              f"{fmt_ns(mps_p50):>10} {cv_str:>6} "
              f"{ps_str:>10} {tflops_str:>11}  "
              f"{slope_str:>11} {n1_str:>10}  "
              f"{r_slope_str:>8} {r_n1_str:>7}")

    # ------------------------------------------------------------------
    # 2) Sweep tables.
    # ------------------------------------------------------------------
    def sweep_table(sweep_name, title, label_fn):
        rows = [r for r in rows_for_summary if r["sweep"] == sweep_name]
        if not rows:
            return
        print()
        print("=" * 100)
        print(title)
        print("=" * 100)
        print(f"  {'shape':>14} {'MPS p50':>11} {'TFLOP/s':>9} "
              f"{'naive slope':>12} {'r_slope':>9} {'r_N=1':>9}")
        for r in rows:
            label = label_fn(r)
            slope_str = fmt_ns(r["naive_slope_ns"])
            r_slope_str = (
                f"{r['ratio_slope']:.2f}×" if r["ratio_slope"] is not None
                else "  -"
            )
            r_n1_str = (
                f"{r['ratio_n_amp1']:.2f}×" if r["ratio_n_amp1"] is not None
                else "  -"
            )
            print(f"  {label:>14} {fmt_ns(r['mps_p50_ns']):>11} "
                  f"{r['tflops']:>9.3f} {slope_str:>12} "
                  f"{r_slope_str:>9} {r_n1_str:>9}")

    sweep_table("square",
                "SQUARE DIAGONAL: M=N=K (cache-fit ceiling probe)",
                lambda r: f"{r['m']}^3")
    sweep_table("ksweep_128",
                "K-SWEEP at M=N=128",
                lambda r: f"K={r['k']}")
    sweep_table("ksweep_512",
                "K-SWEEP at M=N=512",
                lambda r: f"K={r['k']}")
    sweep_table("membound",
                "MEMORY-BOUND PROBES",
                lambda r: f"{r['m']}x{r['n']}x{r['k']}")

    # ------------------------------------------------------------------
    # 3) Verdicts.
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("Pre-registered verdicts")
    print("=" * 100)

    # --- Methodology ---
    cv_clean = [r for r in rows_for_summary
                if not (r["cv_pct"] != r["cv_pct"]) and r["cv_pct"] < 30]
    n_resolvable = sum(1 for r in rows_for_summary if r["mps_p50_ns"] > 0
                       and r["trial_count"] >= 100)
    n_total = len(rows_for_summary)
    print(f"\n  Per-shape stability: {len(cv_clean)}/{n_total} shapes "
          f"with CV < 30%; {n_resolvable}/{n_total} shapes "
          f"with ≥ 100 trials")
    if n_resolvable == n_total and len(cv_clean) >= n_total - 4:
        meth = "PASS"
    elif n_resolvable >= n_total - 4:
        meth = "MARGINAL"
    else:
        meth = "FAIL"
    print(f"  -> methodology verdict: {meth}")

    # --- Gap on 016 compute plateau (M=N=K ∈ [128, 768] square) ---
    plateau_rows = [
        r for r in rows_for_summary
        if r["sweep"] == "square"
        and COMPUTE_PLATEAU_M_LO <= r["m"] <= COMPUTE_PLATEAU_M_HI
        and r["ratio_slope"] is not None
    ]
    if plateau_rows:
        plateau_ratios = [r["ratio_slope"] for r in plateau_rows]
        plateau_med = statistics.median(plateau_ratios)
        plateau_min = min(plateau_ratios)
        plateau_max = max(plateau_ratios)
        # Range across 34 shapes
        all_ratios = [r["ratio_slope"] for r in rows_for_summary
                      if r["ratio_slope"] is not None]
        gap_range = max(all_ratios) / min(all_ratios) if all_ratios else 0
        print(f"\n  016 compute-plateau ratios (naive slope / MPS p50):")
        for r in plateau_rows:
            print(f"    {r['m']}^3:  ratio={r['ratio_slope']:.2f}×  "
                  f"(naive {r['naive_slope_ns']:.0f} ns / "
                  f"MPS {r['mps_p50_ns']:.0f} ns)")
        print(f"  median={plateau_med:.2f}×  min={plateau_min:.2f}×  "
              f"max={plateau_max:.2f}×  range across 34 shapes={gap_range:.2f}×")
        if plateau_med >= 5:
            gap_verdict = "WIDE GAP"
        elif plateau_med < 2:
            gap_verdict = "NARROW GAP"
        elif gap_range >= 3:
            gap_verdict = "SHAPE-DEPENDENT"
        else:
            gap_verdict = "MIXED"
        print(f"  -> gap verdict: {gap_verdict} "
              f"(plateau median ratio {plateau_med:.2f}×)")
    else:
        print("\n  No plateau rows for ratio analysis")
        gap_verdict = "?"

    # --- Absolute MPS plateau ---
    big_squares = [r for r in rows_for_summary
                   if r["sweep"] == "square" and r["m"] >= 1024]
    if big_squares:
        max_tflops = max(r["tflops"] for r in big_squares)
        peak_pct = max_tflops / M4_MAX_PEAK_FP32_TFLOPS * 100
        print(f"\n  MPS peak TFLOP/s on large squares (M ≥ 1024): "
              f"{max_tflops:.2f} ({peak_pct:.1f} % of {M4_MAX_PEAK_FP32_TFLOPS} "
              f"TFLOP/s fp32 peak)")
        if max_tflops >= MPS_CALIBRATED_TFLOPS:
            mps_verdict = "CALIBRATED"
        elif max_tflops < MPS_UNDER_PERFORMS_TFLOPS:
            mps_verdict = "UNDER-PERFORMS"
        else:
            mps_verdict = "MID"
        print(f"  -> MPS absolute verdict: {mps_verdict}")
    else:
        mps_verdict = "?"

    # --- Headline ---
    print()
    print("=" * 100)
    print("Headline")
    print("=" * 100)
    print(f"  methodology:    {meth}")
    print(f"  gap on 016 plateau: {gap_verdict}")
    print(f"  MPS plateau:    {mps_verdict}")

    # Quick context lines.
    print()
    print(f"  reference: naive matmul plateau (016) = ~2 TFLOP/s "
          f"on M=N=K ∈ [128, 768]")
    plateau_naive_ratios_n1 = [
        r["ratio_n_amp1"] for r in rows_for_summary
        if r["sweep"] == "square"
        and COMPUTE_PLATEAU_M_LO <= r["m"] <= COMPUTE_PLATEAU_M_HI
        and r["ratio_n_amp1"] is not None
    ]
    if plateau_naive_ratios_n1:
        med_n1 = statistics.median(plateau_naive_ratios_n1)
        print(f"  with-overhead (apples-to-apples) ratio at plateau: "
              f"median {med_n1:.2f}×")


if __name__ == "__main__":
    main()
