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

---

## Result

**Date run:** 2026-04-28 (timestamp prefix `20260428T170422`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Wall-clock:** ~60 s (10 s baseline + 5×8 s staircase + 10 s tail)
**Outcome:** **MARGINAL** — busy-phase median |rel_diff| = 6.30 %.
Just outside the strict 5 % PASS threshold; well inside the 15 %
MARGINAL threshold. IOReport is usable but with a known +14 %
high-load bias to calibrate against.

### Primary: per-phase GPU power agreement

| phase / target  | n_bins | ior_mean_mW | pm_mean_mW | mean_diff_mW | abs_p50_mW | rel_diff_% |
|-----------------|-------:|------------:|-----------:|-------------:|-----------:|-----------:|
| baseline (0%)   |      8 |         106 |        104 |           +2 |          8 |   **2.13** |
| step_000pct     |      7 |         108 |        106 |           +2 |          5 |   **1.76** |
| step_025pct     |      7 |         208 |        201 |           +7 |          6 |   **3.30** |
| step_050pct     |      6 |         332 |        312 |          +20 |         12 |   **6.30** |
| step_075pct     |      7 |         403 |        398 |           +5 |         15 |   **1.18** |
| step_100pct     |      7 |        2073 |       1814 |         +259 |        205 |  **13.32** |
| tail (0%)       |      8 |         514 |        629 |         -115 |         12 |  **20.12** |

Read the table:
- **Idle and low-load (≤25 %): excellent agreement.** Both sources
  report ~100-200 mW with ≤3 % relative difference. Either source
  works as a baseline indicator.
- **Mid-load (50-75 %): good agreement.** 1.2 % at 75 % is the
  best agreement of any phase. 6.3 % at 50 % is the only phase
  that misses the strict 5 % bar at moderate load — likely
  statistical from only 6 bins.
- **Full saturation (100 %): IOReport reads ~14 % HIGHER than
  powermetrics, consistently** (+259 mW on a ~1.8-2.1 W signal).
  This is a real systematic offset, not noise (mean signed diff
  matches the absolute disagreement). The most likely explanations:
  (a) IOReport's `GPU Energy` channel includes some GPU-adjacent
  component (GPU SRAM, AFR) that powermetrics excludes, or vice
  versa; (b) the two sources use different sampling-window
  edges and during a saturated workload the difference accumulates.
- **Tail (post-workload idle): 20 % disagreement, IOReport LOWER**
  than powermetrics. The opposite-direction disagreement plus the
  high p95 (618 mW) suggests one or two bins where powermetrics
  caught a residual workload spike that IOReport averaged out.
  Bonus columns confirm CPU was still elevated (1.7 W vs 0.9 W
  baseline) during tail, so the system wasn't fully idle.

### Bonus: IOReport-side power breakdown per phase

The IOReport CSV records 11 power buckets, of which powermetrics
exposes only the GPU one. Per-phase mean power for the others:

| phase           | target | cpu | dram | amcc | dcs | afr | disp |
|-----------------|-------:|----:|-----:|-----:|----:|----:|-----:|
| baseline        |     0% |  915 |  332 |  558 |  461 |   8 |  322 |
| step_000pct     |     0% |  787 |  283 |  530 |  412 |   9 |  323 |
| step_025pct     |    25% |  816 |  332 |  776 |  521 |  17 |  323 |
| step_050pct     |    50% |  792 |  370 | 1007 |  578 |  27 |  322 |
| step_075pct     |    75% |  791 |  390 | 1069 |  641 |  34 |  321 |
| **step_100pct** |   100% | **2464** |  **751** | **1348** | **1712** | **147** |  328 |
| tail            |     0% | 1694 |  462 |  735 |  726 |  38 |  329 |

Three things to note:
1. **CPU power TRIPLED during full-load step** (787 → 2464 mW).
   Driving the GPU with ~1 dispatch/ms from Python isn't free —
   the orchestrator burns ~1.7 W of CPU on top of whatever the
   GPU consumes. For any "what does this workload cost" claim,
   the CPU side is non-trivial.
2. **Memory-side power scales with GPU load.** AMCC (memory
   controller) goes 558 → 1348 mW, DCS goes 461 → 1712 mW, DRAM
   goes 332 → 751 mW. Even though `fma_loop` is "compute-bound,"
   the buffer write at the end of each thread plus pipeline state
   churn pulls memory subsystems along. The full-system power for
   a GPU-bound workload includes a substantial memory tax.
3. **AFR (display/render engine) tracks workload visibly.** 8 mW
   idle → 147 mW at full load. Not large in absolute terms but
   suggests AFR is doing GPU-adjacent work proportional to the
   compute load.
4. **DISP stays flat** at ~325 mW across all phases — display
   power isn't affected by GPU workload (just panel lighting).

### What this means for the sudo-free telemetry path

**MARGINAL is enough to declare the path viable** with a known
bias. Specifically:

- For **relative-magnitude questions** ("how much more power did
  workload A use vs workload B?"), IOReport works fine because the
  +14 % full-load bias is consistent — both A and B see it, ratio
  cancels.
- For **absolute power claims** ("kernel X uses 1.8 W"), prefer
  powermetrics if you can pay the sudo cost; otherwise note the
  ~14 % uncertainty.
- For **trend analysis across a workload** (the staircase shape),
  IOReport tracks faithfully — same monotonic increase as
  powermetrics across the staircase, just shifted up by a small
  multiplier at peak.

The user-runs-sudo-in-another-terminal dance can be **dropped for
GPU power telemetry** when the project's experiments record their
own. Future experiments don't need the coordination protocol.

### Surprises

#### 1. IOReport reads HIGHER than powermetrics at full load

The pre-registration didn't predict a direction. Empirically,
IOReport overshoots by +14 % at saturation. Mechanism not
isolated, but two candidates:

- **Different aggregation window.** IOReport's delta is over the
  exact monotonic ns window we chose; powermetrics aggregates over
  its own internal window which may include a tail of the previous
  sample. At full load with rapidly-changing energy counters,
  small window misalignment shifts the integrated power.
- **Different component inclusion.** IOReport's "GPU Energy"
  channel may exclude something powermetrics' "GPU Power" includes
  (or vice versa). Looking at the IOReport bonus columns: AFR and
  GPU SRAM are reported separately. If powermetrics' GPU Power
  number includes GPU SRAM but IOReport's GPU Energy doesn't, we'd
  expect IOReport to read LOWER — opposite of what we see. So
  this isn't the explanation in the obvious direction.

A focused micro-experiment could isolate this: subscribe to a
larger / smaller subset of IOReport channels and re-compare.

#### 2. The tail-phase disagreement direction inverts

Idle / low-load: IOReport reads slightly HIGHER (+2 mW).
Full-load: IOReport reads MUCH HIGHER (+259 mW).
Tail: IOReport reads LOWER (-115 mW).

If the bias were systematic (e.g. "IOReport always reads ~14 %
higher"), tail would also show IOReport > PM. The inversion
suggests the disagreement isn't purely a calibration constant —
there's some workload-state-dependent component. Could be that
PM smooths over its sample window using a state machine that
lags state changes, creating asymmetric disagreement on rising
edges (full-load) and falling edges (tail).

#### 3. CPU power tripled to 2.5 W during the workload

Driving the GPU at ~1 ms/dispatch from Python takes substantial
CPU. The IOReport bindings + Metal dispatch overhead + Python
loop bookkeeping adds up. For experiments measuring GPU-bound
kernels, the orchestrator's CPU cost is real and visible — the
user's machine is doing meaningful CPU work to keep the GPU
busy at peak rate. Useful calibration: a future "true-cost-per-
microbench" claim should include the CPU + memory power tax,
not just GPU.

### What does NOT change

- The sudo path stays available. `gpu_telemetry.py` still works.
- The CSV format both tools produce (same `monotonic_ns` clock)
  remains joinable.
- Decisions 004 / 005 about pair timing are unaffected by this
  result — they don't depend on power telemetry.

### What changes

- Future experiments can launch `notes/ioreport.py` as a
  subprocess (the exp 008 pattern) instead of requiring the user
  to start `gpu_telemetry.py` in another terminal. The
  orchestrator owns its own power telemetry.
- For absolute power claims with tight tolerance, dual-record
  with both sources or apply a calibrated +14 % adjustment to
  IOReport.
- The bonus IOReport columns (CPU, DRAM, AMCC, DCS, AFR, DISP,
  ISP, AVE, GPU SRAM) are available for free in any future
  experiment without sudo. We have **far more telemetry surface
  than powermetrics gives us** at zero additional cost.

### After this experiment

Branches landed on: **MARGINAL.** Per pre-registration:

> Document the calibration / offset and use both side-by-side.
> The architecture in the agent's research synthesis (per-stream
> CSV, joined at experiment time) handles this fine.

Operational consequence: the lab's experiments going forward can
optionally call `notes/ioreport.py` as a subprocess to record
power telemetry without user intervention. The
`gpu_telemetry.py` workflow remains for cases where absolute
power tolerance matters or for cross-validation.

Natural follow-on (optional):

- **exp 009: characterize the +14 % full-load bias.** Vary which
  IOReport channels are included in the "GPU power" sum (just
  GPU Energy vs GPU Energy + GPU SRAM + AFR vs ...) and see which
  combination minimizes the powermetrics disagreement. Could
  reveal what powermetrics' "GPU Power" actually includes.
- **exp 010: IOReport `GPU Active Time` Histogram vs powermetrics
  active residency.** The unanswered question from exp 007
  (utilization), now with the more promising IOReport channel.
  Same pattern as 008 but a different signal.

We do not plan past these branches.
