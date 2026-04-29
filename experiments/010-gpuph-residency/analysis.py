# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 010 analysis: per-phase GPUPH residency table + verdict.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


TICKS_PER_SECOND = 24_000_000


def load_states(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "monotonic_ns": int(r["monotonic_ns"]),
                "window_s": float(r["window_s"]),
                "group": r["group"],
                "channel": r["channel"],
                "state_idx": int(r["state_idx"]),
                "state_name": r["state_name"],
                "residency": int(r["residency_24Mticks"]),
            })
    return rows


def load_phases(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "monotonic_ns": int(r["monotonic_ns"]),
                "phase": r["phase"],
                "target": float(r["target_busy_fraction"]),
            })
    return rows


def phase_for(monotonic_ns: int, phases) -> str:
    """Return the phase whose start-bound covers monotonic_ns. Phases are
    in chronological order; the last phase whose start <= ts is the
    active one."""
    active = "<pre>"
    for p in phases:
        if p["monotonic_ns"] <= monotonic_ns:
            active = p["phase"]
        else:
            break
    return active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--raw-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw")
    ap.add_argument("--channel", default="GPUPH",
                    help="channel name to analyze (default GPUPH)")
    args = ap.parse_args()

    states_csv = args.raw_dir / f"{args.prefix}-states.csv"
    phases_csv = args.raw_dir / f"{args.prefix}-phases.csv"
    if not states_csv.exists():
        raise SystemExit(f"missing {states_csv}")
    if not phases_csv.exists():
        raise SystemExit(f"missing {phases_csv}")

    states = load_states(states_csv)
    phases = load_phases(phases_csv)
    print(f"loaded {len(states)} state rows, {len(phases)} phase markers")

    # Restrict to the channel of interest
    chan_rows = [r for r in states if r["channel"] == args.channel]
    if not chan_rows:
        raise SystemExit(f"no rows for channel {args.channel!r}")

    # Discover state index -> name mapping
    name_by_idx: dict[int, str] = {}
    for r in chan_rows:
        name_by_idx.setdefault(r["state_idx"], r["state_name"])
    print(f"\n{args.channel} states ({len(name_by_idx)}):")
    for idx in sorted(name_by_idx):
        print(f"  [{idx:>2}] {name_by_idx[idx]}")

    # Group rows by sample window, then accumulate per phase
    by_window: dict[int, list[dict]] = defaultdict(list)
    for r in chan_rows:
        by_window[r["monotonic_ns"]].append(r)

    per_phase_resid: dict[str, list[int]] = defaultdict(
        lambda: [0] * len(name_by_idx)
    )
    per_phase_window_ticks: dict[str, int] = defaultdict(int)
    per_phase_n_windows: dict[str, int] = defaultdict(int)
    for ts_ns, rows in sorted(by_window.items()):
        phase = phase_for(ts_ns, phases)
        for r in rows:
            per_phase_resid[phase][r["state_idx"]] += r["residency"]
        # Window length in ticks (any row's window_s is the same)
        per_phase_window_ticks[phase] += int(rows[0]["window_s"] * TICKS_PER_SECOND)
        per_phase_n_windows[phase] += 1

    # Per-phase summary
    print()
    print("=" * 92)
    print(f"Per-phase {args.channel} residency")
    print("=" * 92)
    header = f"{'phase':<14} {'n_win':>5} {'sum_pct':>7} {'top1_state':>11} {'top1_pct':>8} {'top2_state':>11} {'top2_pct':>8} {'mean_idx':>8}"
    print(header)
    phase_order = ["baseline", "step_000pct", "step_025pct", "step_050pct",
                   "step_075pct", "step_100pct", "tail", "subfloor"]
    rows_for_verdict = []
    for phase in phase_order:
        if phase not in per_phase_resid:
            continue
        resid = per_phase_resid[phase]
        n_win = per_phase_n_windows[phase]
        total_ticks = per_phase_window_ticks[phase]
        sum_resid = sum(resid)
        sum_pct = (sum_resid / total_ticks * 100) if total_ticks else float("nan")
        # Sort states by residency descending
        ranked = sorted(enumerate(resid), key=lambda x: -x[1])
        top1_idx, top1_val = ranked[0]
        top2_idx, top2_val = ranked[1] if len(ranked) > 1 else (None, 0)
        top1_pct = (top1_val / sum_resid * 100) if sum_resid else 0.0
        top2_pct = (top2_val / sum_resid * 100) if sum_resid else 0.0
        # Residency-weighted mean state index
        if sum_resid:
            mean_idx = sum(i * v for i, v in enumerate(resid)) / sum_resid
        else:
            mean_idx = float("nan")
        print(f"{phase:<14} {n_win:>5} {sum_pct:>6.1f}% "
              f"{name_by_idx.get(top1_idx, '?'):>11} {top1_pct:>7.1f}% "
              f"{name_by_idx.get(top2_idx, '?') if top2_idx is not None else '-':>11} "
              f"{top2_pct:>7.1f}% {mean_idx:>8.2f}")
        rows_for_verdict.append({
            "phase": phase, "mean_idx": mean_idx,
            "top1_pct": top1_pct, "top1_state": name_by_idx.get(top1_idx, '?'),
        })

    # Verdict logic
    print()
    print("=" * 92)
    print("Verdict")
    print("=" * 92)

    staircase_phases = [r for r in rows_for_verdict
                        if r["phase"].startswith("step_")]
    if len(staircase_phases) >= 2:
        # Check monotonicity of mean_idx across staircase
        means = [r["mean_idx"] for r in staircase_phases]
        n_inc = sum(1 for a, b in zip(means, means[1:]) if b >= a)
        print(f"staircase mean-idx trajectory: {[f'{m:.2f}' for m in means]}")
        print(f"  monotonic-up transitions: {n_inc}/{len(means) - 1}")

    sub = next((r for r in rows_for_verdict if r["phase"] == "subfloor"), None)
    if sub:
        print(f"subfloor top-state: {sub['top1_state']} at {sub['top1_pct']:.1f}%")

    # Apply pre-registered thresholds
    pass_strict = False
    if len(staircase_phases) == 5 and sub:
        means = [r["mean_idx"] for r in staircase_phases]
        n_inc = sum(1 for a, b in zip(means, means[1:]) if b >= a)
        # PASS criteria from README:
        # - mean idx monotonic across 5 staircase phases (>=4 of 4 transitions)
        # - >=80% top-half residency at subfloor
        # We approximate "top half" with top-state-pct >= 80
        sub_top = sub["top1_pct"]
        top_half_pct = 0.0
        # Recompute top-half over states 8-15 for subfloor
        sub_resid = per_phase_resid.get("subfloor", [])
        if sub_resid:
            n_states = len(sub_resid)
            top_half = sum(sub_resid[n_states // 2:])
            total = sum(sub_resid)
            top_half_pct = (top_half / total * 100) if total else 0.0
        print(f"subfloor top-half (idx >= {len(name_by_idx)//2}) residency: "
              f"{top_half_pct:.1f}%")
        if n_inc >= 4 and top_half_pct >= 80:
            print("VERDICT: PASS")
            pass_strict = True
        elif n_inc >= 2 or top_half_pct >= 50:
            print(f"VERDICT: MARGINAL (staircase monotonic {n_inc}/4, "
                  f"subfloor top-half {top_half_pct:.1f}%)")
        else:
            print(f"VERDICT: FAIL (staircase monotonic {n_inc}/4, "
                  f"subfloor top-half {top_half_pct:.1f}%)")

    # Free-signal: per-phase residency on other GPU Stats STATE channels
    print()
    print("=" * 92)
    print("Bonus: top-1 state per phase for other GPU Stats channels")
    print("=" * 92)
    other_channels = sorted({
        r["channel"] for r in states
        if r["group"] == "GPU Stats" and r["channel"] != args.channel
    })
    for ch in other_channels:
        ch_rows = [r for r in states if r["channel"] == ch]
        if not ch_rows:
            continue
        names = {}
        for r in ch_rows:
            names.setdefault(r["state_idx"], r["state_name"])
        per_phase_per_state: dict[str, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for r in ch_rows:
            phase = phase_for(r["monotonic_ns"], phases)
            per_phase_per_state[phase][r["state_idx"]] += r["residency"]
        line = [f"{ch:<10}"]
        for phase in phase_order:
            d = per_phase_per_state.get(phase)
            if not d:
                line.append(f"{phase[:7]:<8}=-")
                continue
            top_idx, top_val = max(d.items(), key=lambda x: x[1])
            tot = sum(d.values())
            pct = (top_val / tot * 100) if tot else 0.0
            line.append(f"{phase[:7]:<8}={names.get(top_idx, '?')}({pct:.0f}%)")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
