# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 016 analysis: matmul shape-discrimination.

Per-shape linear fits of t(N_AMP) → recover slope (ns / matmul)
and intercept. Convert slope to ns/FLOP and effective TFLOP/s for
cross-shape comparison.

Primary outputs:
  - Square-diagonal ns/FLOP curve (cache-fit transition signal).
  - Two K-sweep ns/FLOP curves at M=N=128 and M=N=512.
  - Memory-bound probe per-shape numbers.
  - Discrimination verdict using the README's PASS/MARGINAL/FAIL +
    DISCRIMINATES/AMBIGUOUS/PARTIAL/DOES-NOT criteria.

Reference numbers:
  - 014b stable-region fma_loop: 405.17 ns / amp-step at
    FMA_PER_ITER = 64 → 6.33 ns / FMA (≈ 3.17 ns / FLOP).
  - 015 stable-region chase: 21825 ns / amp-step at
    CHASE_PER_ITER = 64 → 341 ns / chained DRAM load.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


EXP014B_NS_PER_FMA = 6.33
EXP014B_NS_PER_FLOP = EXP014B_NS_PER_FMA / 2.0  # 1 FMA = 2 FLOPs
EXP015_NS_PER_CHASED_LOAD = 341.0


def load_cells(path: Path):
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
    prfboost = pwr_dict.get("PRFBOOST", 0)
    if prfboost >= 50:
        return "PRFBOOST"
    if deadline >= deadline_threshold and deadline >= perf - 10:
        return "DEADLINE"
    if perf >= 50:
        return "PERF"
    if idle >= 50:
        return "IDLE"
    return "MIXED"


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
    residuals = [
        (x, y, (intercept + slope * x), y - (intercept + slope * x))
        for x, y in zip(xs, ys)
    ]
    return {"slope": slope, "intercept": intercept, "r2": r2,
            "residuals": residuals, "n_points": n}


def fmt_ns(ns):
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.2f} µs"
    return f"{ns:.0f} ns"


def fmt_flops(flops):
    if flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    if flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    if flops >= 1e3:
        return f"{flops/1e3:.2f}K"
    return f"{flops:.0f}"


def fmt_bytes(b):
    if b >= 1 << 20:
        return f"{b/(1<<20):.1f}MiB"
    if b >= 1 << 10:
        return f"{b/(1<<10):.1f}KiB"
    return f"{b}B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--raw-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw")
    args = ap.parse_args()

    cells_csv = args.raw_dir / f"{args.prefix}-cells.csv"
    states_csv = args.raw_dir / f"{args.prefix}-states.csv"
    if not cells_csv.exists():
        raise SystemExit(f"missing {cells_csv}")
    if not states_csv.exists():
        print(f"warning: missing {states_csv}; PWRCTRL classification "
              f"will be skipped")
        states = []
    else:
        states = load_states(states_csv)

    cells = load_cells(cells_csv)
    print(f"loaded: {len(cells)} cells, {len(states)} state rows")

    # Group cells by shape.
    by_shape = defaultdict(list)
    for c in cells:
        key = (c["sweep"], c["m"], c["n"], c["k"])
        by_shape[key].append(c)
    for key in by_shape:
        by_shape[key].sort(key=lambda c: c["n_amp"])
    print(f"shapes: {len(by_shape)}")

    # ------------------------------------------------------------------
    # 1) Per-cell timing + PWRCTRL summary.
    # ------------------------------------------------------------------
    print()
    print("=" * 120)
    print("Per-cell summary (compact)")
    print("=" * 120)
    print(f"{'cell':>4} {'sweep':<11} {'shape':<18} {'N_AMP':>6} "
          f"{'trials':>6} {'p50':>10} {'p90':>10} {'p99':>10}  "
          f"{'PWRCTRL top':<24} {'P15%':>5}  {'klass':<9}")
    print("-" * 120)
    cell_state = {}
    for c in cells:
        if states:
            pwr = aggregate_residency(states, "PWRCTRL",
                                       c["start_ns"], c["end_ns"])
            gpuph = aggregate_residency(states, "GPUPH",
                                         c["start_ns"], c["end_ns"])
        else:
            pwr = {}
            gpuph = {}
        if pwr:
            top_pwr = sorted(pwr.items(), key=lambda kv: -kv[1])[:2]
            pwr_str = " ".join(f"{n}({pct:.0f}%)" for n, pct in top_pwr)
        else:
            pwr_str = "<no data>"
        p15 = gpuph.get("P15", 0.0)
        klass = classify_pwrctrl(pwr)
        cell_state[c["cell_idx"]] = {"pwr": pwr, "gpuph": gpuph, "p15": p15,
                                      "class": klass}
        shape_str = f"{c['m']}x{c['n']}x{c['k']}"
        print(f"{c['cell_idx']:>4} {c['sweep']:<11} {shape_str:<18} "
              f"{c['n_amp']:>6} {c['trial_count']:>6} "
              f"{fmt_ns(c['p50']):>10} {fmt_ns(c['p90']):>10} "
              f"{fmt_ns(c['p99']):>10}  {pwr_str:<24} "
              f"{p15:>4.0f}%  {klass:<9}")

    # ------------------------------------------------------------------
    # 2) Per-shape linear fits.
    # ------------------------------------------------------------------
    print()
    print("=" * 120)
    print("Per-shape linear fit t(N_AMP) = a + b·N_AMP")
    print("=" * 120)
    fits = {}
    rows_for_summary = []
    for key in sorted(by_shape.keys(), key=lambda k: (
        ["square", "ksweep_128", "ksweep_512", "membound"].index(k[0]),
        k[1] * k[2] * k[3]
    )):
        sweep, m, n, k = key
        ms = by_shape[key]
        xs = [c["n_amp"] for c in ms]
        ys = [c["p50"] for c in ms]
        fit = linear_fit(xs, ys)
        fits[key] = fit
        flops_per_matmul = 2 * m * n * k
        ab_bytes = 4 * (m * k + k * n)
        if fit is None:
            print(f"\n  {sweep:<11} {m}x{n}x{k}  <fit failed>")
            continue
        ns_per_flop = fit["slope"] / flops_per_matmul
        tflops = flops_per_matmul / fit["slope"] / 1e3
        residual_pct = []
        for x, y, fitted, resid in fit["residuals"]:
            if fitted > 0:
                residual_pct.append(abs(resid) / fitted * 100)
        max_resid_pct = max(residual_pct) if residual_pct else 0.0
        rows_for_summary.append({
            "key": key, "sweep": sweep, "m": m, "n": n, "k": k,
            "flops": flops_per_matmul, "ab_bytes": ab_bytes,
            "slope": fit["slope"], "intercept": fit["intercept"],
            "r2": fit["r2"], "ns_per_flop": ns_per_flop,
            "tflops": tflops, "max_resid_pct": max_resid_pct,
            "n_points": fit["n_points"],
        })
        print(f"\n  {sweep:<11} {m}x{n}x{k} (FLOPs={fmt_flops(flops_per_matmul)}, "
              f"A+B={fmt_bytes(ab_bytes)})")
        print(f"    fit: slope={fit['slope']:.2f} ns/step  "
              f"intercept={fit['intercept']:.0f} ns  "
              f"R²={fit['r2']:.4f}  ({fit['n_points']} points)")
        print(f"    derived: {ns_per_flop*1e3:.3f} ps/FLOP  "
              f"= {tflops:.2f} TFLOP/s  (max residual: {max_resid_pct:.1f}%)")

    # ------------------------------------------------------------------
    # 3) Sweep summary tables.
    # ------------------------------------------------------------------
    def sweep_table(sweep_name, title, row_label):
        rows = [r for r in rows_for_summary if r["sweep"] == sweep_name]
        if not rows:
            return
        print()
        print("=" * 110)
        print(title)
        print("=" * 110)
        print(f"  {row_label:>10} {'FLOPs':>8} {'A+B':>10} "
              f"{'slope ns/step':>14} {'intercept ns':>12} {'R²':>7} "
              f"{'ps/FLOP':>9} {'TFLOP/s':>8} {'klass':<9}")
        for r in rows:
            klass = "?"
            shape_cells = by_shape[r["key"]]
            if shape_cells and states:
                # Use the largest-N_AMP cell as representative.
                rep = shape_cells[-1]
                klass = cell_state.get(rep["cell_idx"], {}).get("class", "?")
            if sweep_name == "square":
                label = f"{r['m']}"
            elif sweep_name.startswith("ksweep"):
                label = f"K={r['k']}"
            else:
                label = f"{r['m']}x{r['n']}x{r['k']}"
            print(f"  {label:>10} {fmt_flops(r['flops']):>8} "
                  f"{fmt_bytes(r['ab_bytes']):>10} "
                  f"{r['slope']:>14.2f} {r['intercept']:>12.0f} "
                  f"{r['r2']:>7.4f} {r['ns_per_flop']*1e3:>9.3f} "
                  f"{r['tflops']:>8.3f} {klass:<9}")

    sweep_table("square",     "SQUARE DIAGONAL: M=N=K (cache-fit transition probe)",
                "M=N=K")
    sweep_table("ksweep_128", "K-SWEEP at M=N=128 (K is the AI knob)",
                "K")
    sweep_table("ksweep_512", "K-SWEEP at M=N=512 (K is the AI knob)",
                "K")
    sweep_table("membound",   "MEMORY-BOUND PROBES (narrow-output × big-reduction)",
                "shape")

    # ------------------------------------------------------------------
    # 4) Discrimination verdict.
    # ------------------------------------------------------------------
    print()
    print("=" * 110)
    print("Discrimination verdict")
    print("=" * 110)

    # Methodology PASS/MARGINAL/FAIL: how many shapes have R² > 0.99
    # and max residual < 5 %?
    n_clean = sum(1 for r in rows_for_summary
                  if r["r2"] > 0.99 and r["max_resid_pct"] < 5)
    n_fits = len(rows_for_summary)
    pct_clean = 100.0 * n_clean / n_fits if n_fits else 0.0
    print(f"\n  Methodology fits: {n_clean}/{n_fits} shapes "
          f"({pct_clean:.0f}%) with R² > 0.99 AND max residual < 5%")
    if pct_clean >= 80:
        meth_verdict = "PASS"
    elif pct_clean >= 50:
        meth_verdict = "MARGINAL"
    else:
        meth_verdict = "FAIL"
    print(f"  -> methodology verdict: {meth_verdict}")

    # Cross-shape discrimination signal.
    print()
    print(f"  ps/FLOP across shapes (lower = faster):")
    square_rows = [r for r in rows_for_summary if r["sweep"] == "square"]
    membound_rows = [r for r in rows_for_summary if r["sweep"] == "membound"]
    if square_rows:
        sq_ps = [r["ns_per_flop"] * 1e3 for r in square_rows]
        sq_min = min(sq_ps)
        sq_max = max(sq_ps)
        sq_med = statistics.median(sq_ps)
        sq_ratio = sq_max / sq_min if sq_min > 0 else float("inf")
        print(f"    square diagonal: min={sq_min:.3f}  median={sq_med:.3f}  "
              f"max={sq_max:.3f}  max/min={sq_ratio:.2f}×")
        # Largest square's ps/FLOP relative to small/mid squares
        largest_sq = max(square_rows, key=lambda r: r["m"])
        small_med = statistics.median(
            [r["ns_per_flop"] * 1e3 for r in square_rows if r["m"] <= 512]
        ) if any(r["m"] <= 512 for r in square_rows) else None
        if small_med:
            big_ratio = (largest_sq["ns_per_flop"] * 1e3) / small_med
            print(f"    largest square ({largest_sq['m']}³) ps/FLOP "
                  f"/ small-square median = {big_ratio:.2f}×")
        else:
            big_ratio = None
    else:
        sq_ratio = None
        big_ratio = None

    if membound_rows and square_rows:
        mb_ps = [r["ns_per_flop"] * 1e3 for r in membound_rows]
        sq_compute_med = statistics.median(
            [r["ns_per_flop"] * 1e3 for r in square_rows
             if 128 <= r["m"] <= 768]
        ) if any(128 <= r["m"] <= 768 for r in square_rows) else statistics.median([r["ns_per_flop"] * 1e3 for r in square_rows])
        mb_ratios = [m / sq_compute_med for m in mb_ps]
        max_mb_ratio = max(mb_ratios)
        print(f"    memory-bound probes: ps/FLOP = "
              f"{[f'{p:.2f}' for p in mb_ps]}")
        print(f"    membound max / square mid-cluster median = "
              f"{max_mb_ratio:.2f}×")
    else:
        max_mb_ratio = None

    # Apply README verdict criteria.
    print()
    sq_kink = (big_ratio is not None and big_ratio >= 3.0) or \
              (sq_ratio is not None and sq_ratio >= 3.0)
    mb_signal = max_mb_ratio is not None and max_mb_ratio >= 5.0

    if sq_kink and mb_signal:
        disc_verdict = "DISCRIMINATES"
        explanation = ("square-diagonal ns/FLOP rises ≥ 3× toward the SLC "
                       "boundary AND memory-bound probes elevate ≥ 5×")
    elif sq_kink:
        disc_verdict = "DISCRIMINATES"
        explanation = ("square-diagonal ns/FLOP shows ≥ 3× rise; memory-"
                       "bound probes too few or modest")
    elif mb_signal:
        disc_verdict = "PARTIAL DISCRIMINATION"
        explanation = ("square diagonal flat (no cache-fit signal) but "
                       "memory-bound probes elevate ≥ 5×")
    elif (max_mb_ratio is not None and max_mb_ratio >= 2.0) or \
         (big_ratio is not None and big_ratio >= 2.0):
        disc_verdict = "AMBIGUOUS"
        explanation = ("some elevation in largest squares or memory-bound "
                       "probes but < 3× / 5×")
    else:
        disc_verdict = "DOES NOT DISCRIMINATE"
        explanation = ("ns/FLOP flat across shapes; methodology produces "
                       "consistent compute rate but no cache-fit transition")

    print(f"  -> discrimination verdict: {disc_verdict}")
    print(f"     ({explanation})")

    # ------------------------------------------------------------------
    # 5) Comparison anchors.
    # ------------------------------------------------------------------
    print()
    print("=" * 110)
    print("Reference anchors")
    print("=" * 110)
    print(f"  014b fma_loop (compute-bound synthetic): "
          f"{EXP014B_NS_PER_FMA} ns/FMA = "
          f"{EXP014B_NS_PER_FLOP*1e3:.0f} ps/FLOP per thread (32-thread serial)")
    print(f"  015 chase     (memory-latency-bound):    "
          f"{EXP015_NS_PER_CHASED_LOAD} ns / chained DRAM load")
    print()
    print("  Note: matmul ps/FLOP should be much lower than 014b's "
          "per-thread reference because matmul has M·N parallelism that")
    print("  014b's 32-thread serial-chain didn't. The cross-shape "
          "*ratio* is the discrimination signal, not the absolute number.")


if __name__ == "__main__":
    main()
