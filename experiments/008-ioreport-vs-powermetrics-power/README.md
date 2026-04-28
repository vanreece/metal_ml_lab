# 008: Does IOReport's GPU power agree with powermetrics' GPU power?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-28
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~60 s of GPU activity (10 s baseline + 5×8 s
staircase + 10 s tail)

## The question

`notes/ioreport.py` (committed in `cdb7972`) wraps `libIOReport.dylib`
to read GPU energy and other power-related counters **without sudo**.
The numbers are plausible at idle (~640 mW GPU, ~3.6 W CPU when
isolated to the canonical channel), but "looks plausible" isn't a
falsifiable claim. This experiment cross-checks IOReport-derived
GPU power against the canonical reference (powermetrics, sudo) over
the same staircase workload exp 007 used.

If they agree, the sudo dependency for telemetry becomes optional
and the lab's experiments can record their own per-run telemetry
without the user-runs-sudo-in-another-terminal dance.

**Primary question:** sampling IOReport's `Energy Model / GPU Energy`
delta (converted to mW) and powermetrics' `gpu_power_mw` field
*simultaneously* at 1000 ms cadence during a 0→100 % busy staircase,
what is the per-step disagreement |ioreport_mw − pm_mw| during steady-
state busy phases?

- **Pass:** median |diff| ≤ 5 % of the mean during 50 %+ busy steady-
  state phases. Sudo-free GPU power is validated.
- **Marginal:** median ≤ 15 %. Usable but with calibration, or one
  source has a systematic bias the other doesn't.
- **Fail:** median > 15 %. The two sources measure different things
  and the validation is incomplete.

## Why this question, now

Two pieces are now in place:
1. `notes/ioreport.py` — bindings + dashboard (committed today)
2. `notes/gpu_telemetry.py` — powermetrics-via-sudo collector
   (committed earlier today)

Both can record CSVs at the same cadence with `monotonic_ns`
timestamps, so they can be joined post-hoc against the same
workload's phase markers. exp 007 set up the staircase machinery
and the two-terminal coordination dance for exactly this kind of
test (it just used the wrong sudo-free signal — ioreg). exp 008
re-uses the staircase but swaps in IOReport for the comparison.

## Hypothesis

Confidence: medium-high. Predictions:

- **At idle phases (baseline, step_0, tail):** both sources will
  report similar GPU power, dominated by background compositing
  (~300-700 mW typical on this box). Agreement to within ~50 mW
  expected.
- **At full saturation (step_100):** both will report several watts
  of GPU power. Agreement to within 5 % expected because both are
  ultimately reading the same hardware energy counters — they just
  expose them through different APIs.
- **Possible systematic offset:** powermetrics may include some
  additional power overhead (kernel-side accounting, sampler
  itself) that IOReport doesn't, or vice versa. If a systematic
  offset exists, the *trend across the staircase* should still
  agree even if absolute values don't.
- **Possible window-misalignment artifact:** powermetrics samples
  at a fixed wall-clock cadence; IOReport's delta samples at the
  consumer-driven cadence (this script's). If the two windows
  drift by tens of ms, individual samples may have visibly
  different power averages. Joining at 1000 ms bin granularity
  smooths this out.

## What we are NOT trying to answer

- **Agreement on CPU / DRAM / other power.** powermetrics doesn't
  expose those at the per-channel granularity IOReport does. The
  IOReport-side CSV will record them as bonus columns but the
  pass/fail criterion is GPU-only.
- **Sub-second alignment.** Both are sampled at 1000 ms cadence;
  100 ms cadence cross-check is a follow-up if 008 passes.
- **Cross-chip agreement.** M4 Max only. M1 Pro would be a separate
  experiment if/when we want sudo-free telemetry there too.
- **Anything about utilization.** That was exp 007 (and FAILED for
  ioreg; IOReport does have an `Active Time Histogram` channel for
  utilization but we're not testing it here — power is the question).
- **What ANE / GPU SRAM / etc. are doing.** Those columns get
  recorded but aren't part of the pass/fail criterion.

## Setup

### Prerequisites the user (not Claude) starts

In a separate terminal:

```
sudo uv run notes/gpu_telemetry.py \
  --csv experiments/008-ioreport-vs-powermetrics-power/raw/PMTELEM.csv \
  --interval-ms 1000 \
  --quiet
```

Same CSV format as exp 007 (only `gpu_power_mw` column is consumed
for this comparison). Letting it run for the duration of `run.py`
plus a few seconds buffer; Ctrl-C after `run.py` reports completion.

### What `run.py` does

1. Validates the powermetrics CSV exists and has at least 2 rows.
2. **Launches `notes/ioreport.py` as a subprocess** with
   `--interval-ms 1000 --csv raw/{ts}-ioreport.csv`. This is the
   sudo-free side of the cross-check. ioreport.py runs to its own
   schedule and writes per-sample rows with `monotonic_ns`.
3. Runs the same utilization staircase as exp 007:
   - Phase 0 — baseline (10 s idle)
   - Phase 1 — staircase (5×8 s steps at 0/25/50/75/100 % target busy)
   - Phase 2 — tail (10 s idle)
4. Writes phase markers to `raw/{ts}-phases.csv`.
5. Terminates the ioreport.py subprocess (SIGINT for clean CSV flush).
6. Tells the user to stop powermetrics.

### What `analysis.py` does

1. Loads three CSVs:
   - `raw/PMTELEM.csv` (powermetrics, user-recorded)
   - `raw/{ts}-ioreport.csv` (IOReport, this experiment)
   - `raw/{ts}-phases.csv` (phase markers)
2. Bins both telemetry sources into 1000 ms windows by `monotonic_ns`.
3. For each phase / step, reports:
   - mean ioreport `gpu_power_mw`
   - mean powermetrics `gpu_power_mw`
   - mean signed diff (mW), median |diff| (mW), p95 |diff| (mW)
   - relative |diff| as % of mean (the metric for pass/fail during
     ≥ 50 % busy phases)
4. Verdict per the success criteria above.
5. Bonus: prints the IOReport-side `cpu_power_mw`, `dram_power_mw`,
   etc. per phase as informational reference (no comparison).

### What we record

`raw/{ts}-ioreport.csv` (written by `notes/ioreport.py` subprocess):
- `iso_ts`, `monotonic_ns`, `window_s`
- `gpu_power_mw`, `cpu_power_mw`, `ane_power_mw`, `dram_power_mw`,
  `amcc_power_mw`, `dcs_power_mw`, `afr_power_mw`, `disp_power_mw`,
  `isp_power_mw`, `ave_power_mw`, `gpu_sram_power_mw`
- `*_energy_nj` for each bucket (raw delta, for analysis)

`raw/{ts}-phases.csv`:
- `monotonic_ns`, `phase`, `target_busy_fraction`

## Success criterion

The experiment succeeds (in the discipline's sense) if:
1. Both telemetry CSVs are present and have rows spanning the
   workload window.
2. Phase markers are present.
3. Analysis can fill in the per-step agreement table.

It produces a usable answer if we can categorize the result PASS /
MARGINAL / FAIL per the thresholds above and decide whether
`gpu_telemetry.py`'s sudo dependency is no longer needed for GPU
power telemetry.

## New questions we expect this to raise

- If IOReport reports higher than powermetrics at idle but they
  agree under load: there's a baseline-power difference (different
  inclusion of background sleep states). Useful for calibration.
- If the staircase shape matches but absolute values differ
  systematically: the two APIs include / exclude different
  components of "GPU power" (e.g. SRAM, AFR, AMCC). Subtracting
  IOReport's other buckets from its GPU number might reconcile.
- If transitions look ragged in IOReport but smooth in powermetrics
  (or vice versa): one of them is window-averaging more
  aggressively. Actionable for understanding response time.
- If they disagree wildly in opposite directions across phases: the
  IOReport bindings have a unit / sign error somewhere. Time to
  re-validate against macmon.

## After this experiment

Branches:

- **PASS.** IOReport replaces powermetrics for GPU power. Update
  `gpu_telemetry.py` README to say "deprecated for GPU power; use
  ioreport.py." Schedule the next sudo-removal validation: IOReport
  Active Time Histogram vs powermetrics active residency
  (utilization, the question exp 007 failed on with ioreg).
- **MARGINAL.** Document the calibration / offset and use both
  side-by-side. The architecture in the agent's research synthesis
  (per-stream CSV, joined at experiment time) handles this fine.
- **FAIL.** Re-validate the IOReport bindings against macmon's
  Rust implementation (which is the reference) — line up channel
  selection, unit conversion, delta math. Re-run after fix.
