# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Experiment 012 analysis: parse powermetrics raw text for per-MHz and
per-P-state residency, align with IOReport GPUPH per-state residency,
report the mapping table.

Time-alignment strategy: powermetrics samples have human-readable
timestamps in their headers ("Sampled system activity (Tue Apr 28
21:30:00 2026 -0700) (250.00 ms elapsed)"). We extract the wall-clock
times, convert to a monotonic-relative offset using one anchor point,
then bin into the same phase windows as IOReport.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# powermetrics output regex patterns
SAMPLE_HEADER = re.compile(
    r"\*\*\*\s*Sampled\s*system\s*activity\s*\(([^)]+)\)\s*\(([\d.]+)\s*ms\s*elapsed\)",
    re.IGNORECASE
)
HW_ACTIVE_RES_LINE = re.compile(
    r"^GPU\s*HW\s*active\s*residency:\s*([\d.]+)\s*%\s*\(([^)]*)\)",
    re.IGNORECASE | re.MULTILINE,
)
ACTIVE_RES_LINE = re.compile(
    r"^GPU\s*active\s*residency:\s*([\d.]+)\s*%\s*\(([^)]*)\)",
    re.IGNORECASE | re.MULTILINE,
)
# Parenthesis content has tokens like "444 MHz: 0.00%" or "P0:  0.00%"
MHZ_BUCKET = re.compile(r"(\d+)\s*MHz\s*:\s*([\d.]+)\s*%")
PSTATE_BUCKET = re.compile(r"(P\d+|SW_\w+|OFF)\s*:\s*([\d.]+)\s*%")
SW_REQ_STATE_LINE = re.compile(
    r"^GPU\s*SW\s*requested\s*state:\s*\(([^)]*)\)",
    re.IGNORECASE | re.MULTILINE,
)
SW_STATE_LINE = re.compile(
    r"^GPU\s*SW\s*state:\s*\(([^)]*)\)",
    re.IGNORECASE | re.MULTILINE,
)
ACTIVE_FREQ_LINE = re.compile(
    r"^GPU\s*HW\s*active\s*frequency:\s*(\d+)\s*MHz",
    re.IGNORECASE | re.MULTILINE,
)


def parse_pmraw(path: Path) -> list[dict]:
    """Split powermetrics raw text into samples and parse each.
    Returns list of {wall_iso, elapsed_ms, mhz_residency, pstate_residency,
    sw_state_residency, active_pct, freq_mhz}."""
    text = path.read_text(errors="replace")
    parts = SAMPLE_HEADER.split(text)
    # split puts the leading text first, then alternates (header_g1, header_g2, body)
    samples = []
    # parts[0] is leading garbage. parts[1::3] is timestamps; parts[2::3] is elapsed_ms;
    # parts[3::3] is bodies.
    headers_ts = parts[1::3]
    headers_elapsed = parts[2::3]
    bodies = parts[3::3]
    for ts_str, elapsed_str, body in zip(headers_ts, headers_elapsed, bodies):
        sample = {
            "wall_str": ts_str.strip(),
            "elapsed_ms": float(elapsed_str),
        }
        # Extract per-MHz residency from active residency line. Preserve
        # positional order: powermetrics lists MHz buckets at the same
        # ordinal positions as P1..P15 in the SW state breakdown, and
        # the order is NOT monotonic in MHz value (e.g. P10=1312 > P11=1242
        # on M4 Max). Duplicates also appear (P8=P9=1182 MHz).
        m = HW_ACTIVE_RES_LINE.search(body) or ACTIVE_RES_LINE.search(body)
        if m:
            sample["active_pct"] = float(m.group(1))
            inner = m.group(2)
            mhz_ordered: list[tuple[int, float]] = []
            for mm in MHZ_BUCKET.finditer(inner):
                mhz_ordered.append((int(mm.group(1)), float(mm.group(2))))
            sample["mhz_ordered"] = mhz_ordered
            # Convenience dict (drops duplicate keys; used for "did we
            # ever see this MHz" lookups, not for positional analysis):
            mhz: dict[int, float] = {}
            for v_mhz, pct in mhz_ordered:
                mhz[v_mhz] = mhz.get(v_mhz, 0.0) + pct
            sample["mhz_residency"] = mhz
            ps_in_active = {}
            for pm in PSTATE_BUCKET.finditer(inner):
                ps_in_active[pm.group(1)] = float(pm.group(2))
            if ps_in_active:
                sample["pstate_in_active"] = ps_in_active
        # SW requested state (P-state) breakdown
        m = SW_REQ_STATE_LINE.search(body)
        if m:
            ps = {}
            for pm in PSTATE_BUCKET.finditer(m.group(1)):
                ps[pm.group(1)] = float(pm.group(2))
            sample["sw_requested"] = ps
        # SW state breakdown
        m = SW_STATE_LINE.search(body)
        if m:
            sw = {}
            for pm in PSTATE_BUCKET.finditer(m.group(1)):
                sw[pm.group(1)] = float(pm.group(2))
            sample["sw_state"] = sw
        m = ACTIVE_FREQ_LINE.search(body)
        if m:
            sample["freq_mhz"] = int(m.group(1))
        samples.append(sample)
    return samples


def parse_pm_walltime(s: str) -> float:
    """Parse a powermetrics timestamp like 'Tue Apr 28 21:30:00 2026 -0700'
    to a UNIX epoch float."""
    # Some platforms report no zone. Try a few formats.
    for fmt in (
        "%a %b %d %H:%M:%S %Y %z",
        "%a %b %d %H:%M:%S %Y",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    raise ValueError(f"could not parse pm wall time: {s!r}")


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


def load_phases(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "monotonic_ns": int(r["monotonic_ns"]),
                "phase": r["phase"],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--raw-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw")
    ap.add_argument("--pmraw", type=Path, default=None)
    args = ap.parse_args()

    raw_dir = args.raw_dir
    pmraw = args.pmraw or (raw_dir / "PMRAW.txt")
    states_csv = raw_dir / f"{args.prefix}-ioreport-states.csv"
    phases_csv = raw_dir / f"{args.prefix}-phases.csv"
    energy_csv = raw_dir / f"{args.prefix}-ioreport.csv"
    if not pmraw.exists():
        raise SystemExit(f"missing {pmraw}")
    if not states_csv.exists():
        raise SystemExit(f"missing {states_csv}")
    if not phases_csv.exists():
        raise SystemExit(f"missing {phases_csv}")

    pm_samples = parse_pmraw(pmraw)
    print(f"parsed {len(pm_samples)} powermetrics samples from {pmraw.name}")

    if not pm_samples:
        raise SystemExit("no powermetrics samples parsed -- check format")

    # Diagnostic: print first sample's keys
    s0 = pm_samples[0]
    print(f"sample 0: keys = {sorted(k for k in s0.keys() if not k.startswith('wall'))}")
    if "mhz_residency" in s0:
        print(f"  mhz buckets: {sorted(s0['mhz_residency'].keys())}")
    if "sw_requested" in s0:
        print(f"  sw_requested keys: {sorted(s0['sw_requested'].keys())}")
    if "sw_state" in s0:
        print(f"  sw_state keys: {sorted(s0['sw_state'].keys())}")
    if "pstate_in_active" in s0:
        print(f"  pstate_in_active: {sorted(s0['pstate_in_active'].keys())}")

    # Align powermetrics samples to monotonic_ns space using one anchor
    # point. The phases CSV has monotonic_ns. The ioreport CSV has both
    # iso_ts (datetime) and monotonic_ns. Use the first energy CSV row
    # as the anchor.
    anchor_mono_ns = None
    anchor_epoch = None
    if energy_csv.exists():
        with energy_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                anchor_mono_ns = int(row["monotonic_ns"])
                anchor_iso = row["iso_ts"]
                anchor_epoch = datetime.fromisoformat(anchor_iso).timestamp()
                break
    if anchor_mono_ns is None or anchor_epoch is None:
        raise SystemExit("no anchor row in ioreport energy CSV")

    # Map each pm sample's wall time to monotonic_ns
    for s in pm_samples:
        try:
            epoch = parse_pm_walltime(s["wall_str"])
            offset_ns = int((epoch - anchor_epoch) * 1e9)
            s["monotonic_ns"] = anchor_mono_ns + offset_ns
        except ValueError:
            s["monotonic_ns"] = None

    valid_pm = [s for s in pm_samples if s.get("monotonic_ns") is not None]
    print(f"aligned {len(valid_pm)} pm samples to monotonic_ns")

    # Load IOReport states + phases
    state_rows = load_states(states_csv)
    phases = load_phases(phases_csv)

    def phase_for(monotonic_ns: int) -> str:
        active = "<pre>"
        for p in phases:
            if p["monotonic_ns"] <= monotonic_ns:
                active = p["phase"]
            else:
                break
        return active

    # Per-phase aggregates
    pm_per_phase: dict[str, list[dict]] = defaultdict(list)
    for s in valid_pm:
        pm_per_phase[phase_for(s["monotonic_ns"])].append(s)

    state_per_phase: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in state_rows:
        if r["channel"] != "GPUPH":
            continue
        ph = phase_for(r["monotonic_ns"])
        state_per_phase[ph][r["state_name"]] += r["residency"]

    # Headline alignment table
    print()
    print("=" * 120)
    print("Per-phase residency: powermetrics MHz buckets vs IOReport GPUPH P-states")
    print("=" * 120)
    phase_order = ["baseline", "step_000pct", "step_025pct", "step_050pct",
                   "step_075pct", "step_100pct", "tail"]
    # Discover all MHz buckets seen
    all_mhz: set[int] = set()
    for s in valid_pm:
        for k in s.get("mhz_residency", {}):
            all_mhz.add(k)
    mhz_order = sorted(all_mhz)
    print(f"powermetrics MHz buckets observed: {mhz_order}")
    # GPUPH P-states observed
    all_pstates: set[str] = set()
    for d in state_per_phase.values():
        all_pstates.update(d.keys())
    pstate_order = sorted(all_pstates,
                          key=lambda n: (-1 if n == "OFF" else int(n.lstrip("P"))))
    print(f"GPUPH P-states observed: {pstate_order}")

    print()
    print(f"{'phase':<14} | " + " | ".join(f"{m:>6}MHz" for m in mhz_order))
    for phase in phase_order:
        if phase not in pm_per_phase:
            continue
        samples = pm_per_phase[phase]
        # Average each MHz bucket's residency across samples
        bucket_avg: dict[int, float] = {}
        for m in mhz_order:
            vals = [s["mhz_residency"].get(m, 0.0) for s in samples
                    if "mhz_residency" in s]
            bucket_avg[m] = sum(vals) / len(vals) if vals else 0.0
        cells = [f"{bucket_avg[m]:>6.2f}%" for m in mhz_order]
        print(f"{phase:<14} | " + " | ".join(cells))

    print()
    print(f"{'phase':<14} | " + " | ".join(f"{p:>8}" for p in pstate_order))
    for phase in phase_order:
        if phase not in state_per_phase:
            continue
        d = state_per_phase[phase]
        total = sum(d.values())
        cells = []
        for p in pstate_order:
            pct = (d.get(p, 0) / total * 100) if total else 0.0
            cells.append(f"{pct:>7.2f}%")
        print(f"{phase:<14} | " + " | ".join(cells))

    # Positional mapping: powermetrics output lists MHz buckets in the
    # same ordinal order as P1..P15. This gives us the canonical mapping
    # directly. Verify by checking that for each phase, the position-i
    # MHz residency matches the position-i P-state SW residency.
    print()
    print("=" * 120)
    print("Positional P-state -> MHz mapping from powermetrics (ordinal alignment)")
    print("=" * 120)
    # Pull positional mapping from the first sample with a complete list
    pos_mhz: list[int] = []
    for s in valid_pm:
        if "mhz_ordered" in s and len(s["mhz_ordered"]) >= 15:
            pos_mhz = [m for m, _ in s["mhz_ordered"]]
            break
    if pos_mhz:
        print(f"P1..P{len(pos_mhz)} -> MHz mapping (positional from powermetrics):")
        for i, mhz in enumerate(pos_mhz, start=1):
            print(f"  P{i:<2} = {mhz:>5} MHz")

        # Verification: per-phase, compare each (Pi, MHz_at_pos_i) residency
        # in powermetrics' own output. SW state (sw_state, sums to 100%)
        # should align with the per-MHz residency of the matching position.
        print()
        print("Per-phase verification (SW state pct vs MHz pct at same position):")
        print(f"{'phase':<14} | {'P-idx':>5} | {'MHz':>5} | {'sw_state %':>10} | {'mhz_res %':>10} | {'diff':>6}")
        for phase in phase_order:
            samples = pm_per_phase.get(phase, [])
            if not samples:
                continue
            # Average per-position MHz residency and per-Pi SW state across samples
            for pi in range(1, len(pos_mhz) + 1):
                mhz_at_pi_vals = [
                    s["mhz_ordered"][pi - 1][1] if "mhz_ordered" in s
                    and len(s["mhz_ordered"]) >= pi else 0.0
                    for s in samples
                ]
                sw_at_pi_vals = [
                    s.get("sw_state", {}).get(f"SW_P{pi}", 0.0) for s in samples
                ]
                mhz_avg = sum(mhz_at_pi_vals) / len(samples)
                sw_avg = sum(sw_at_pi_vals) / len(samples)
                if sw_avg > 0.5 or mhz_avg > 0.5:
                    print(f"{phase:<14} | P{pi:<4} | {pos_mhz[pi-1]:>5} | "
                          f"{sw_avg:>9.2f}% | {mhz_avg:>9.2f}% | "
                          f"{abs(sw_avg - mhz_avg):>5.2f}%")

    # Build a P-state -> MHz mapping by aligning per-phase profiles
    print()
    print("=" * 120)
    print("Mapping inference: for each non-OFF GPUPH P-state, find the MHz "
          "bucket whose per-phase residency profile correlates best")
    print("=" * 120)
    # Vectors keyed by phase
    used_phases = [p for p in phase_order if p in pm_per_phase and p in state_per_phase]
    # IOReport per-state vector (residency in each phase)
    pstate_vectors: dict[str, list[float]] = {}
    for p in pstate_order:
        if p == "OFF":
            continue
        vec = []
        for ph in used_phases:
            d = state_per_phase[ph]
            total = sum(d.values())
            vec.append((d.get(p, 0) / total * 100) if total else 0.0)
        pstate_vectors[p] = vec
    # MHz per-bucket vector
    mhz_vectors: dict[int, list[float]] = {}
    for m in mhz_order:
        vec = []
        for ph in used_phases:
            samples = pm_per_phase[ph]
            vals = [s["mhz_residency"].get(m, 0.0) for s in samples
                    if "mhz_residency" in s]
            vec.append(sum(vals) / len(vals) if vals else 0.0)
        mhz_vectors[m] = vec

    def cosine(a, b):
        import math
        num = sum(x * y for x, y in zip(a, b))
        den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
        return num / den if den else 0.0

    # For each P-state, compute cosine similarity with each MHz bucket
    print(f"{'P-state':<8} {'best MHz match':<15} {'similarity':<10} "
          f"{'2nd-best MHz':<15} {'2nd sim':<10}")
    mapping = {}
    for p, pvec in pstate_vectors.items():
        if max(pvec) < 0.5:
            continue  # state never engaged meaningfully
        sims = [(m, cosine(pvec, mhz_vectors[m])) for m in mhz_order]
        sims.sort(key=lambda t: -t[1])
        best = sims[0]
        second = sims[1] if len(sims) > 1 else (None, 0)
        mapping[p] = best[0]
        print(f"{p:<8} {best[0]!s:<15} {best[1]:<10.3f} "
              f"{second[0]!s:<15} {second[1]:<10.3f}")

    # Verdict logic refined: the cosine-similarity inference works for
    # extreme states (P1, P15) but conflates intermediate ones because
    # they share the same phase footprint. The positional mapping above
    # is the authoritative answer; the cross-check is whether sw_state %
    # at SW_Pi tracks the position-i MHz % across phases.
    print()
    print("=" * 120)
    print("Verdict")
    print("=" * 120)
    if not pos_mhz:
        print("VERDICT: FAIL -- couldn't extract positional MHz list")
        return

    # Compute mean abs diff between SW_Pi % and position-i MHz % across
    # all (phase, P_idx) cells where either is non-zero (from the table
    # above).
    diffs = []
    for phase in phase_order:
        samples = pm_per_phase.get(phase, [])
        if not samples:
            continue
        for pi in range(1, len(pos_mhz) + 1):
            mhz_avg = sum(
                s["mhz_ordered"][pi - 1][1] if "mhz_ordered" in s
                and len(s["mhz_ordered"]) >= pi else 0.0
                for s in samples
            ) / len(samples)
            sw_avg = sum(
                s.get("sw_state", {}).get(f"SW_P{pi}", 0.0) for s in samples
            ) / len(samples)
            if mhz_avg > 0.5 or sw_avg > 0.5:
                diffs.append(abs(mhz_avg - sw_avg))
    median_diff = sorted(diffs)[len(diffs) // 2] if diffs else float("nan")
    max_diff = max(diffs) if diffs else float("nan")
    print(f"Across all (phase, P-idx) cells where either residency > 0.5%:")
    print(f"  median |sw_state - mhz_at_pos| = {median_diff:.2f}%")
    print(f"  max    |sw_state - mhz_at_pos| = {max_diff:.2f}%")
    if median_diff < 1.0 and max_diff < 5.0:
        print("VERDICT: PASS -- positional mapping verified within tight tolerance")
    elif median_diff < 5.0:
        print(f"VERDICT: MARGINAL -- positional mapping holds with median "
              f"{median_diff:.2f}% / max {max_diff:.2f}% drift")
    else:
        print(f"VERDICT: FAIL -- median {median_diff:.2f}% exceeds 5% threshold")

    # Cross-validate: GPUPH P-state residency should also match position-i MHz residency
    print()
    print("Cross-validate: IOReport GPUPH SW residency vs powermetrics MHz residency at "
          "matching position")
    print(f"{'phase':<14} | {'P-idx':>5} | {'MHz':>5} | "
          f"{'gpuph_pct':>10} | {'mhz_pct':>10} | {'diff':>6}")
    cross_diffs = []
    for phase in phase_order:
        if phase not in state_per_phase or phase not in pm_per_phase:
            continue
        d = state_per_phase[phase]
        total = sum(d.values())
        samples = pm_per_phase[phase]
        for pi in range(1, len(pos_mhz) + 1):
            gpuph_pct = (d.get(f"P{pi}", 0) / total * 100) if total else 0.0
            mhz_avg = sum(
                s["mhz_ordered"][pi - 1][1] if "mhz_ordered" in s
                and len(s["mhz_ordered"]) >= pi else 0.0
                for s in samples
            ) / len(samples)
            if gpuph_pct > 0.5 or mhz_avg > 0.5:
                cross_diffs.append(abs(gpuph_pct - mhz_avg))
                print(f"{phase:<14} | P{pi:<4} | {pos_mhz[pi-1]:>5} | "
                      f"{gpuph_pct:>9.2f}% | {mhz_avg:>9.2f}% | "
                      f"{abs(gpuph_pct - mhz_avg):>5.2f}%")
    if cross_diffs:
        cross_med = sorted(cross_diffs)[len(cross_diffs) // 2]
        cross_max = max(cross_diffs)
        print(f"\nCross-source agreement: median {cross_med:.2f}% / "
              f"max {cross_max:.2f}%")


if __name__ == "__main__":
    main()
