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
        # Extract per-MHz residency from active residency line
        m = HW_ACTIVE_RES_LINE.search(body) or ACTIVE_RES_LINE.search(body)
        if m:
            sample["active_pct"] = float(m.group(1))
            inner = m.group(2)
            mhz = {}
            for mm in MHZ_BUCKET.finditer(inner):
                mhz[int(mm.group(1))] = float(mm.group(2))
            sample["mhz_residency"] = mhz
            # In case the parens contain P-states instead of MHz (some
            # powermetrics formats). Look for both.
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

    # Try to build a P-state -> MHz mapping by aligning per-phase profiles
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

    # Verdict
    print()
    print("=" * 120)
    print("Verdict")
    print("=" * 120)
    if not mapping:
        print("VERDICT: FAIL -- no mapping inferred")
        return
    # Check monotonicity: P-states sorted by index should map to MHz sorted ascending
    sorted_p = sorted(mapping.keys(), key=lambda n: int(n.lstrip("P")))
    mhz_seq = [mapping[p] for p in sorted_p]
    is_monotonic = all(b >= a for a, b in zip(mhz_seq, mhz_seq[1:]))
    print(f"Inferred mapping order: {[(p, mapping[p]) for p in sorted_p]}")
    print(f"MHz sequence monotonic non-decreasing: {is_monotonic}")
    if is_monotonic and len(set(mapping.values())) >= 4:
        print("VERDICT: PASS -- mapping is monotonic and non-trivial")
    elif is_monotonic:
        print("VERDICT: MARGINAL -- mapping monotonic but few distinct MHz")
    else:
        print("VERDICT: FAIL -- mapping not monotonic")


if __name__ == "__main__":
    main()
