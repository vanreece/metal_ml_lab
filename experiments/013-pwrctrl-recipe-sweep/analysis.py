# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 013 analysis: per-cell PWRCTRL classification + 2D grid.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_states(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "monotonic_ns": int(r["monotonic_ns"]),
                "channel": r["channel"],
                "state_idx": int(r["state_idx"]),
                "state_name": r["state_name"],
                "residency": int(r["residency_24Mticks"]),
            })
    return rows


def load_cells(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "start_ns": int(r["monotonic_ns_start"]),
                "end_ns": int(r["monotonic_ns_end"]),
                "cell_idx": int(r["cell_idx"]),
                "fma_iters": int(r["fma_iters"]),
                "sleep_us": int(r["sleep_us"]),
                "n_dispatches": int(r["n_dispatches"]),
            })
    return rows


def aggregate_in_window(state_rows, channel: str, t_start: int, t_end: int):
    sums: dict[str, int] = defaultdict(int)
    for r in state_rows:
        if r["channel"] != channel:
            continue
        if not (t_start <= r["monotonic_ns"] <= t_end):
            continue
        sums[r["state_name"]] += r["residency"]
    total = sum(sums.values())
    if total == 0:
        return []
    return sorted(
        [(name, val, val / total * 100) for name, val in sums.items()],
        key=lambda t: -t[1],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--raw-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw")
    args = ap.parse_args()

    states_csv = args.raw_dir / f"{args.prefix}-states.csv"
    cells_csv = args.raw_dir / f"{args.prefix}-cells.csv"
    if not states_csv.exists():
        raise SystemExit(f"missing {states_csv}")
    if not cells_csv.exists():
        raise SystemExit(f"missing {cells_csv}")

    states = load_states(states_csv)
    cells = load_cells(cells_csv)
    print(f"loaded {len(states)} state rows, {len(cells)} cells")

    # Per-cell residency
    print()
    print("=" * 110)
    print("Per-cell PWRCTRL + GPUPH top states")
    print("=" * 110)
    print(f"{'cell':>4} {'fma':>5} {'us':>4} {'n_disp':>6} | "
          f"{'PWRCTRL top':<24} | {'GPUPH top':<24} | {'verdict':<10}")
    cell_results = []
    for c in cells:
        pwr = aggregate_in_window(states, "PWRCTRL", c["start_ns"], c["end_ns"])
        gpuph = aggregate_in_window(states, "GPUPH", c["start_ns"], c["end_ns"])
        deadline_pct = next((pct for n, _v, pct in pwr if n == "DEADLINE"), 0.0)
        perf_pct = next((pct for n, _v, pct in pwr if n == "PERF"), 0.0)
        idle_pct = next((pct for n, _v, pct in pwr if n == "IDLE_OFF"), 0.0)
        if deadline_pct >= 50:
            verdict = "DEADLINE"
        elif perf_pct >= 50:
            verdict = "PERF"
        elif idle_pct >= 50:
            verdict = "IDLE"
        else:
            verdict = "MIXED"
        pwr_top = " ".join(f"{n}({pct:.0f}%)" for n, _v, pct in pwr[:2])
        gpuph_top = " ".join(f"{n}({pct:.0f}%)" for n, _v, pct in gpuph[:2])
        print(f"{c['cell_idx']:>4} {c['fma_iters']:>5} {c['sleep_us']:>4} "
              f"{c['n_dispatches']:>6} | {pwr_top:<24} | {gpuph_top:<24} | {verdict:<10}")
        cell_results.append({
            **c,
            "pwr_top": pwr_top,
            "deadline_pct": deadline_pct,
            "perf_pct": perf_pct,
            "idle_pct": idle_pct,
            "gpuph_top": gpuph_top,
            "verdict": verdict,
        })

    # 2D grid: PWRCTRL verdict per (FMA_ITERS, sleep_us)
    print()
    print("=" * 110)
    print("2D verdict grid: PWRCTRL mode by (FMA_ITERS, sleep_us)")
    print("=" * 110)
    fma_levels = sorted({c["fma_iters"] for c in cells})
    sleep_levels = sorted({c["sleep_us"] for c in cells})
    by_axes: dict[tuple[int, int], dict] = {}
    for c in cell_results:
        by_axes[(c["fma_iters"], c["sleep_us"])] = c

    print(f"{'FMA_ITERS':>10} | " + " | ".join(f"{s:>4}us" for s in sleep_levels))
    print("-" * (12 + len(sleep_levels) * 9))
    for f in fma_levels:
        row = [f"{f:>10}"]
        for s in sleep_levels:
            cell = by_axes.get((f, s))
            if cell:
                row.append(f"{cell['verdict']:>6}")
            else:
                row.append(f"{'?':>6}")
        print(" | ".join(row))

    # Numeric grid: DEADLINE %
    print()
    print("DEADLINE % per cell:")
    print(f"{'FMA_ITERS':>10} | " + " | ".join(f"{s:>4}us" for s in sleep_levels))
    print("-" * (12 + len(sleep_levels) * 9))
    for f in fma_levels:
        row = [f"{f:>10}"]
        for s in sleep_levels:
            cell = by_axes.get((f, s))
            if cell:
                row.append(f"{cell['deadline_pct']:>5.0f}%")
            else:
                row.append(f"{'?':>6}")
        print(" | ".join(row))

    print()
    print("PERF % per cell:")
    print(f"{'FMA_ITERS':>10} | " + " | ".join(f"{s:>4}us" for s in sleep_levels))
    print("-" * (12 + len(sleep_levels) * 9))
    for f in fma_levels:
        row = [f"{f:>10}"]
        for s in sleep_levels:
            cell = by_axes.get((f, s))
            if cell:
                row.append(f"{cell['perf_pct']:>5.0f}%")
            else:
                row.append(f"{'?':>6}")
        print(" | ".join(row))

    # GPUPH P15 residency per cell
    print()
    print("GPUPH P15 residency % per cell (correlates with peak DVFS access):")
    print(f"{'FMA_ITERS':>10} | " + " | ".join(f"{s:>4}us" for s in sleep_levels))
    print("-" * (12 + len(sleep_levels) * 9))
    for f in fma_levels:
        row = [f"{f:>10}"]
        for s in sleep_levels:
            cell = by_axes.get((f, s))
            if cell:
                # extract P15 % from gpuph by re-aggregating (above we only kept top-2)
                gpuph_full = aggregate_in_window(states, "GPUPH",
                                                 cell["start_ns"], cell["end_ns"])
                p15 = next((pct for n, _v, pct in gpuph_full if n == "P15"), 0.0)
                row.append(f"{p15:>5.0f}%")
            else:
                row.append(f"{'?':>6}")
        print(" | ".join(row))

    # Verdict on boundary shape
    print()
    print("=" * 110)
    print("Boundary shape analysis")
    print("=" * 110)
    deadline_cells = [(c["fma_iters"], c["sleep_us"]) for c in cell_results
                      if c["verdict"] == "DEADLINE"]
    perf_cells = [(c["fma_iters"], c["sleep_us"]) for c in cell_results
                  if c["verdict"] == "PERF"]
    mixed_cells = [(c["fma_iters"], c["sleep_us"]) for c in cell_results
                   if c["verdict"] == "MIXED"]
    print(f"DEADLINE cells: {len(deadline_cells)}  {deadline_cells}")
    print(f"PERF cells:     {len(perf_cells)}  {perf_cells}")
    print(f"MIXED cells:    {len(mixed_cells)}  {mixed_cells}")

    # Test for clean boundary: for each FMA_ITERS row, is the verdict
    # monotonic in sleep_us? For each sleep_us column, monotonic in FMA?
    def is_monotonic_row(fma):
        verdicts = []
        for s in sleep_levels:
            v = by_axes.get((fma, s))
            if v:
                verdicts.append(v["verdict"])
        # Check that DEADLINE -> PERF transition is monotonic (D...DPP...P)
        seen_perf = False
        for v in verdicts:
            if v == "PERF":
                seen_perf = True
            elif v == "DEADLINE" and seen_perf:
                return False, verdicts
        return True, verdicts

    rows_clean = []
    for f in fma_levels:
        clean, vs = is_monotonic_row(f)
        rows_clean.append((f, clean, vs))
        marker = "✓" if clean else "✗"
        print(f"  FMA_ITERS={f:>5} row monotonic D->P: {marker}  ({vs})")

    if all(c for _, c, _ in rows_clean) and len(deadline_cells) > 0 and len(perf_cells) > 0:
        print("\nVERDICT: PASS -- clean monotonic boundary recovered")
    elif len(deadline_cells) > 0 and len(perf_cells) > 0:
        print("\nVERDICT: MARGINAL -- boundary exists but not strictly monotonic")
    elif len(deadline_cells) == 0:
        print("\nVERDICT: FAIL -- no DEADLINE cells; sweep doesn't span boundary")
    elif len(perf_cells) == 0:
        print("\nVERDICT: FAIL -- no PERF cells; sweep doesn't span boundary")


if __name__ == "__main__":
    main()
