# 012: Can we build a GPUPH state-index → MHz mapping by aligning IOReport per-state residency with powermetrics per-MHz residency?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-28
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~80 s (10 s baseline + 5 × 8 s staircase + 10 s
tail + 20 s recovery), with the user-runs-sudo dance from exp 008.

## The question

Exp 010 established that GPUPH's 16 states (`OFF`, `P1`-`P15`) track
workload intensity but contain no MHz information in their names.
Apple doesn't publish a GPU-state-to-frequency table. powermetrics
*does* report per-MHz residency at each sample, in a parenthesized
breakdown after the `GPU active residency` line:

    GPU active residency:  84.42% (444 MHz: 0.00% 800 MHz: 0.05% ...
                                   1180 MHz: 84.36%)

(Exact MHz buckets vary by chip and macOS version.) If we record both
sources side-by-side over the same staircase workload, we can align
them and build a state-index → MHz mapping.

**Primary question:** sampling GPUPH per-state residency (via the
exp 010 bindings) and powermetrics per-MHz active residency
*simultaneously* across a 0→100 % busy staircase, can we recover a
monotonic mapping `P_idx → MHz` such that GPUPH state X's residency
profile across phases matches powermetrics' MHz-X' residency profile?

- **PASS:** every active GPUPH P-state (P1-P15 with non-zero
  residency in some phase) has a unique powermetrics MHz bucket whose
  per-phase residency-distribution correlates ≥ 0.85 (Spearman) with
  the GPUPH state's per-phase distribution. Ordering matches:
  P_n → MHz_n monotonic in n.
- **MARGINAL:** mapping exists but is not 1:1 (e.g. powermetrics
  reports 8 MHz buckets and GPUPH has 15 active states; some MHz
  buckets span multiple P-states or some P-states are sub-states of
  one MHz bucket). Or correlations are 0.7-0.85 — directional but
  noisy.
- **FAIL:** no consistent alignment. GPUPH state ordering doesn't
  match MHz ordering (e.g. P5 maps to a higher MHz than P10), or
  residency profiles don't correlate. Either the bindings are
  reading something different from what the channel name suggests,
  or `GPUPH` indexes a different state space than powermetrics'
  P-state breakdown.

## Why this question, now

Three reasons:

1. **It's the next obvious step from 010.** GPUPH bindings work; we
   know they track workload. Without the MHz mapping, we can only
   make qualitative claims ("residency shifted to higher states");
   with it, we can make quantitative ones ("chip ran at 1140 MHz
   for X seconds").
2. **The 011 mechanism findings need a numeric layer.** We saw
   DEADLINE 33-48 % during sub-floor and P15 6-15 %. Knowing that
   P15 = N MHz (and lower P-states = M MHz) lets us compute average
   GPU clock frequency during sub-floor, which is the variable
   that drives the 1.7 µs floor calculation directly. (Cycle count
   per kernel × P15 clock → wall-clock per kernel.)
3. **Same architecture as 008.** User starts powermetrics in another
   terminal; we launch ioreport in this one; both write to CSVs
   joined by `monotonic_ns`. Pattern is established; this is a new
   data type using the same protocol.

## Hypothesis

Confidence: medium-high. Predictions:

- **Number of MHz buckets powermetrics reports:** likely 8-12, fewer
  than GPUPH's 15 active states. The "P-states" Apple ships in
  software may aggregate finer hardware substates.
- **Mapping shape:** monotonic. P1 = lowest active MHz, P15 = highest.
  The intervening states should map to MHz buckets in order, possibly
  with some MHz buckets covering multiple P-states (e.g. P3-P4 both
  map to 800 MHz).
- **At step_100, both sources should agree:** powermetrics' top MHz
  bucket residency ≈ GPUPH's P15 residency. Both ~85-90 %.
- **At step_025, both should show similar low-MHz dominance:**
  powermetrics' lowest non-zero active MHz bucket residency ≈ GPUPH's
  P1-P2 residency.
- **OFF state alignment:** powermetrics' "active residency" 16 % at
  baseline ≈ 100 % - GPUPH's `OFF` residency 84 %. (`OFF` is "GPU not
  active"; powermetrics calls this "inactive" implicitly, by
  excluding it from the active-residency breakdown.)
- **The M4 Max top GPU MHz is around 1.4 GHz.** macOS-published
  numbers for M4 Max GPU peak; we expect to see something near this
  in powermetrics' top bucket name.

## What we are NOT trying to answer

- **Whether powermetrics' MHz numbers are themselves correct.**
  Trusted as the canonical reference; cross-validating against
  external sources is out of scope.
- **Mapping for non-canonical state channels.** `BSTGPUPH`,
  `GPU_SW`, etc. are bonus inputs to analysis, not the pass/fail
  target.
- **Cross-chip generalization.** M4 Max only. Any M-series chip
  would need its own enumeration.
- **MHz mapping for GPUPH OFF.** OFF means "GPU not active" — no
  meaningful MHz. Only P1-P15 get mapped.
- **Sub-second / per-kernel MHz.** 250 ms aggregate is the
  resolution. Per-kernel-dispatch MHz claims need a separate
  experiment.
- **Whether MHz is the *only* thing distinguishing P-states.**
  Voltage, instruction-throughput, etc. could also differ. We map
  by MHz because that's what powermetrics gives us; full P-state
  characterization is bigger.

## Setup

### Prerequisites the user (not Claude) starts

In a separate terminal, before invoking `run.py`:

```
sudo uv run notes/gpu_telemetry.py \
    --csv experiments/012-gpuph-vs-powermetrics-mhz/raw/PMTELEM.csv \
    --interval-ms 250 \
    --quiet
```

This writes the per-sample dashboard fields to CSV. **gpu_telemetry.py
doesn't currently capture per-MHz residency**, so we additionally
have the user start a raw powermetrics capture:

```
sudo powermetrics --samplers gpu_power -i 250 \
    > experiments/012-gpuph-vs-powermetrics-mhz/raw/PMRAW.txt 2>&1
```

The raw output contains the per-MHz residency breakdown that we
parse in analysis. (We could fold this into gpu_telemetry.py later
if useful — for now, raw + parser keeps 012 self-contained.)

### What `run.py` does

Same staircase shape as exp 008 / 010, with phase markers:

1. Validates that `PMRAW.txt` exists and is being written.
2. Launches `notes/ioreport.py --include-states --interval-ms 250
   --csv raw/{ts}-ioreport.csv`. State data → `raw/{ts}-ioreport-states.csv`.
3. Phase 0 — baseline (10 s idle).
4. Phase 1 — staircase (5 × 8 s steps at 0/25/50/75/100 % target busy).
5. Phase 2 — tail (10 s idle).
6. SIGINT to ioreport.py for clean CSV flush.
7. Tells user to stop powermetrics.

Phase markers go to `raw/{ts}-phases.csv` keyed by `monotonic_ns` for
post-hoc joining. Same kernel as exp 008/010 (fma_loop with
FMA_ITERS=65536, 32 threads — known long enough to land in
work-dominated regime).

### What `analysis.py` does

1. Loads four sources:
   - `raw/PMRAW.txt`: parse with a custom regex for the per-MHz
     residency lines. Each sample emits a dict
     `{monotonic_ns? (parsed from sample timestamp), mhz_residency:
     {444: 0.00, 800: 0.05, ..., 1180: 84.36}, active_pct: float}`.
     We capture monotonic_ns by parsing powermetrics' "elapsed_ns"
     field if present, OR fall back to the user-process clock from
     gpu_telemetry.py's CSV (joined by sample number).
   - `raw/PMTELEM.csv`: per-sample monotonic_ns + active_pct + freq_mhz.
   - `raw/{ts}-ioreport-states.csv`: GPUPH per-state residency with
     monotonic_ns.
   - `raw/{ts}-phases.csv`: phase markers with monotonic_ns.
2. Aligns: for each phase window, aggregate per-MHz residency from
   PMRAW (averaging across samples in the window) and per-P-state
   residency from ioreport-states (summing across windows in the
   phase, normalizing).
3. Builds the alignment table:

   | phase    | pm: 444 | pm: 800 | pm: ... | pm: 1180 | iop: P1 | iop: P2 | iop: ... | iop: P15 |
   |----------|--------:|--------:|--------:|---------:|--------:|--------:|---------:|---------:|

   And reports correlations: for each GPUPH state index n with
   non-zero residency, find the powermetrics MHz bucket with the
   most similar per-phase residency profile (Spearman correlation).
4. Verdict per the success criteria above.

### What we record

`raw/PMRAW.txt`: powermetrics raw text output (~80 s of samples).
`raw/PMTELEM.csv`: gpu_telemetry CSV (joinable on monotonic_ns).
`raw/{ts}-ioreport.csv`: IOReport energy CSV.
`raw/{ts}-ioreport-states.csv`: IOReport per-state residency
(GPU Stats group).
`raw/{ts}-phases.csv`: phase markers.
`raw/{ts}-meta.txt`: env, channel inventory, alignment table,
verdict.

### What we do NOT do

- No averaging in live output.
- No retries. If powermetrics drops a sample (it sometimes does at
  high cadence), it goes in the parsed data with whatever fields
  it managed to emit.
- No discarding of "weird" residency values (negative, off-by-one).
- No interpolation across powermetrics samples — we accept the
  cadence as-is and bin into phases at the granularity recorded.

## Success criterion

The experiment **succeeds** (in the discipline's sense) if we have:

1. PMRAW.txt with per-MHz residency parsed for ≥ 80 % of expected
   samples.
2. PMTELEM.csv with ≥ 80 % expected samples.
3. ioreport-states.csv with rows spanning the workload window.
4. Phase markers covering all 7 phases.

It produces a **usable answer** if we can fill in the per-phase
alignment table and decide:

- The number of distinct MHz buckets powermetrics reports.
- Whether the GPUPH state ordering matches MHz ordering.
- Per-state correlations between GPUPH and powermetrics MHz residency.

## New questions we expect this to raise

- If powermetrics reports e.g. 8 MHz buckets and GPUPH has 15 active
  states, **multiple GPUPH P-states map to one MHz bucket**. Which
  ones? The mapping shape is its own follow-up.
- If powermetrics' top MHz at step_100 is *lower* than expected
  (~1.0-1.2 GHz when M4 Max is published as ~1.4 GHz peak), the
  staircase recipe may not push the chip hard enough to reach top
  DVFS. Worth confirming with the 009 sub-floor recipe pattern.
- If GPUPH P15 residency is much higher than powermetrics' top MHz
  residency at step_100, GPUPH may aggregate "P15 + boost" while
  powermetrics shows them separately. Cross-check with `BSTGPUPH`.
- If *all* mapping correlations are below 0.7, we may be reading
  GPUPH wrong. Re-validate ioreport bindings against macmon's
  expected output.

## After this experiment

Branches:

- **PASS.** GPUPH ↔ MHz mapping is now in the lab's reference
  inventory. Future experiments can quote "P15 at 1140 MHz" and
  similar concrete claims. exp 011's "P15 residency 6-15 % during
  sub-floor" can be re-stated as time-at-1140-MHz.
- **MARGINAL.** Map what we can map; document the parts that don't
  cleanly align (1:N or N:1 P-state ↔ MHz relationships).
- **FAIL.** Re-validate ioreport bindings vs macmon. If macmon
  shows the same numbers we do, the channel itself is misleading
  and we file the negative result. If different, our bindings are
  wrong; fix and re-run.

We do not plan past these branches.
