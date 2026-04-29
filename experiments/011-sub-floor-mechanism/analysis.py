# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 011 analysis: GPUPH + PWRCTRL residency split by attempt era.

For each attempt:
  - floor-era: wall-clock window from first trial up to the first
    sub-floor trial (gpu_delta_raw < 5500 ns)
  - sub-floor era: from first sub-floor trial through the last trial

For each inter-attempt gap: GPUPH + PWRCTRL residency during the gap.

Reports the headline table the pre-registration asks for.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


SUBFLOOR_THRESHOLD_NS = 5500


def load_attempts(path: Path):
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


def load_states(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "monotonic_ns": int(r["monotonic_ns"]),
                "window_s": float(r["window_s"]),
                "channel": r["channel"],
                "state_idx": int(r["state_idx"]),
                "state_name": r["state_name"],
                "residency": int(r["residency_24Mticks"]),
            })
    return rows


def per_window_state_breakdown(state_rows):
    """Group state rows by (monotonic_ns, channel) and return
    a dict[(ts_ns, channel)] -> dict[state_name -> residency]."""
    out: dict[tuple[int, str], dict[str, int]] = defaultdict(dict)
    for r in state_rows:
        key = (r["monotonic_ns"], r["channel"])
        out[key][r["state_name"]] = r["residency"]
    return out


def aggregate_residency(state_rows, channel: str, window_ns_set):
    """Sum residencies across the given set of (monotonic_ns)
    windows, grouped by state name. Return ordered list of
    (state_name, residency, pct)."""
    sums: dict[str, int] = defaultdict(int)
    for r in state_rows:
        if r["channel"] != channel:
            continue
        if r["monotonic_ns"] not in window_ns_set:
            continue
        sums[r["state_name"]] += r["residency"]
    total = sum(sums.values())
    if total == 0:
        return []
    return sorted(
        [(name, val, val / total * 100) for name, val in sums.items()],
        key=lambda t: -t[1],
    )


def fmt_top(breakdown, n=2):
    if not breakdown:
        return "-"
    parts = [f"{name}({pct:.0f}%)" for name, _val, pct in breakdown[:n]]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--raw-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw")
    args = ap.parse_args()

    attempts_csv = args.raw_dir / f"{args.prefix}-attempts.csv"
    states_csv = args.raw_dir / f"{args.prefix}-ioreport-states.csv"
    if not attempts_csv.exists():
        raise SystemExit(f"missing {attempts_csv}")
    if not states_csv.exists():
        raise SystemExit(f"missing {states_csv}")

    attempts = load_attempts(attempts_csv)
    states = load_states(states_csv)
    print(f"loaded {len(attempts)} trial rows, {len(states)} state rows")

    # Group attempts CSV by attempt
    by_attempt: dict[int, list[dict]] = defaultdict(list)
    for r in attempts:
        by_attempt[r["attempt_idx"]].append(r)
    n_attempts = len(by_attempt)

    # Per-attempt time windows for floor-era / sub-floor-era / full-attempt
    eras: dict[int, dict[str, tuple[int, int]]] = {}
    for a in sorted(by_attempt):
        meas = sorted(
            [r for r in by_attempt[a] if r["phase"] == "measured"],
            key=lambda r: r["idx_within_phase"],
        )
        if not meas:
            continue
        t_start = meas[0]["wall_clock_ns"]
        t_end = meas[-1]["wall_clock_ns"]
        sub = next((r for r in meas if r["gpu_delta_raw"] < SUBFLOOR_THRESHOLD_NS), None)
        t_onset = sub["wall_clock_ns"] if sub else t_end + 1
        eras[a] = {
            "full": (t_start, t_end),
            "floor": (t_start, t_onset),
            "subfloor": (t_onset, t_end),
        }

    # Map each IOReport window (keyed by monotonic_ns) to an era label
    # if its end-of-window is in [start, end]. Inter-attempt gaps are
    # whatever windows fall outside any attempt's full range.
    state_windows = sorted({r["monotonic_ns"] for r in states})
    window_label: dict[int, str] = {}
    for ts in state_windows:
        label = "<other>"
        for a, e in eras.items():
            t_start, t_end = e["full"]
            if t_start <= ts <= t_end:
                # Within attempt; assign floor or subfloor sub-era
                _t_floor_start, t_floor_end = e["floor"]
                if ts < t_floor_end:
                    label = f"a{a}_floor"
                else:
                    label = f"a{a}_subfloor"
                break
        else:
            # Not in any attempt; check gaps
            for a in range(n_attempts - 1):
                e_a = eras.get(a)
                e_b = eras.get(a + 1)
                if not e_a or not e_b:
                    continue
                if e_a["full"][1] < ts < e_b["full"][0]:
                    label = f"gap_{a}_to_{a + 1}"
                    break
        window_label[ts] = label

    # Group windows by label
    by_label: dict[str, set[int]] = defaultdict(set)
    for ts, label in window_label.items():
        by_label[label].add(ts)

    # Headline table per pre-registration
    print()
    print("=" * 110)
    print("Per-attempt PWRCTRL + GPUPH residency (top-2 states)")
    print("=" * 110)
    print(f"{'attempt':>8} {'era':>10} {'n_win':>5}  "
          f"{'PWRCTRL top':<28}  {'GPUPH top':<28}")
    for a in sorted(eras):
        for era_name in ["floor", "subfloor"]:
            label = f"a{a}_{era_name}"
            wins = by_label.get(label, set())
            pwr = aggregate_residency(states, "PWRCTRL", wins)
            gpuph = aggregate_residency(states, "GPUPH", wins)
            print(f"{a:>8} {era_name:>10} {len(wins):>5}  "
                  f"{fmt_top(pwr):<28}  {fmt_top(gpuph):<28}")

    print()
    print("=" * 110)
    print("Inter-attempt gaps")
    print("=" * 110)
    print(f"{'gap':>10} {'n_win':>5}  {'PWRCTRL top':<28}  {'GPUPH top':<28}")
    for a in range(n_attempts - 1):
        label = f"gap_{a}_to_{a + 1}"
        wins = by_label.get(label, set())
        pwr = aggregate_residency(states, "PWRCTRL", wins)
        gpuph = aggregate_residency(states, "GPUPH", wins)
        print(f"{label:>10} {len(wins):>5}  {fmt_top(pwr):<28}  {fmt_top(gpuph):<28}")

    # Verdict-relevant aggregate
    print()
    print("=" * 110)
    print("Verdict logic: PWRCTRL DEADLINE residency in sub-floor era per attempt")
    print("=" * 110)
    deadline_pcts = []
    for a in sorted(eras):
        wins = by_label.get(f"a{a}_subfloor", set())
        pwr = aggregate_residency(states, "PWRCTRL", wins)
        deadline = next((pct for name, _val, pct in pwr if name == "DEADLINE"), 0.0)
        perf = next((pct for name, _val, pct in pwr if name == "PERF"), 0.0)
        idle = next((pct for name, _val, pct in pwr if name == "IDLE_OFF"), 0.0)
        deadline_pcts.append(deadline)
        print(f"  attempt {a}: DEADLINE={deadline:>5.1f}%  PERF={perf:>5.1f}%  "
              f"IDLE_OFF={idle:>5.1f}%")

    n_with_deadline = sum(1 for p in deadline_pcts if p >= 50)
    print()
    if n_with_deadline >= 5:
        print(f"VERDICT: STRONG SUPPORT ({n_with_deadline}/5 attempts had "
              f">=50% DEADLINE in sub-floor era)")
    elif n_with_deadline >= 2:
        print(f"VERDICT: PARTIAL SUPPORT ({n_with_deadline}/5)")
    else:
        print(f"VERDICT: NO SUPPORT ({n_with_deadline}/5)")

    # Cross-attempt deepening check: GPUPH P15 residency in sub-floor era
    print()
    print("=" * 110)
    print("Cross-attempt: GPUPH P15 residency during sub-floor era per attempt")
    print("=" * 110)
    p15_pcts = []
    for a in sorted(eras):
        wins = by_label.get(f"a{a}_subfloor", set())
        gpuph = aggregate_residency(states, "GPUPH", wins)
        p15 = next((pct for name, _val, pct in gpuph if name == "P15"), 0.0)
        p15_pcts.append(p15)
        print(f"  attempt {a}: P15 residency = {p15:>5.1f}%")
    if len(p15_pcts) >= 2:
        trend = "increasing" if p15_pcts[-1] > p15_pcts[0] else "flat/decreasing"
        delta = p15_pcts[-1] - p15_pcts[0]
        print(f"  attempt 0 -> 4 trend: {trend}  (delta = {delta:+.1f} pct)")

    # All-channel cross-attempt sub-floor scan: look for any signal that
    # changes monotonically across attempts (would suggest "what carries
    # across") since PWRCTRL gaps reset to baseline.
    print()
    print("=" * 110)
    print("Cross-attempt scan: top state per attempt's sub-floor era, all GPU Stats channels")
    print("=" * 110)
    other_chans = sorted({r["channel"] for r in states})
    for ch in other_chans:
        line = [f"  {ch:<10}"]
        for a in sorted(eras):
            wins = by_label.get(f"a{a}_subfloor", set())
            br = aggregate_residency(states, ch, wins)
            line.append(f"a{a}={fmt_top(br, 1):<14}")
        print(" ".join(line))


if __name__ == "__main__":
    main()
