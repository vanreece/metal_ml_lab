# 007: Does ioreg's GPU utilization agree with powermetrics' GPU active residency?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-28
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~50 s of GPU activity (10 s baseline + 30 s
staircase workload + 10 s tail)

## The question

The macOS telemetry survey (commit `316ffea`) flagged that `ioreg`
exposes a `Device Utilization %` field under `AGXAccelerator.
PerformanceStatistics` that is reachable **without sudo** at
~10 ms per snapshot. If that field agrees closely with powermetrics'
`gpu_power active residency %` (which requires sudo), we have a
sudo-free GPU utilization signal — which means future experiments
can record their own telemetry without the user-runs-sudo dance.

This experiment is the cross-check.

**Primary question:** sampling ioreg `Device Utilization %` and
powermetrics `GPU HW active residency %` *simultaneously* at 100 ms
cadence during a known GPU workload, what is the per-sample
disagreement |ioreg − powermetrics| in percentage points?

- **Pass:** median disagreement ≤ 5 percentage points across all
  workload phases (idle, mid-utilization, full-utilization). ioreg
  is a viable sudo-free substitute for the utilization signal.
- **Marginal:** median ≤ 10 pp but with a systematic bias (one
  source consistently reads higher than the other). ioreg is
  usable but needs calibration; the bias becomes the main
  finding.
- **Fail:** median > 10 pp or unbounded disagreement during
  transitions. ioreg measures something different from powermetrics
  active residency. Sudo-free utilization remains an open problem
  and we keep the powermetrics-via-`gpu_telemetry.py` workflow.

## Why this question, now

Per the agent-research synthesis: macmon and Stats.app both lean on
sudo-free channels (IOReport + IOHID + ioreg) instead of
powermetrics. Validating the ioreg piece is the cheapest single
test of "is the sudo-free path real?" The IOReport piece (GPU energy,
P-state residencies) is more powerful but needs Python ctypes
bindings against `libIOReport.dylib`; deferring that to a follow-on
experiment (007b) once 007 establishes whether ioreg alone gets us
to "sudo-free utilization works."

## Hypothesis

Confidence: medium. Predictions:

- **At idle (no GPU client active),** both ioreg and powermetrics
  will report < 5 % utilization. They will agree to within 2 pp at
  these very-low values.
- **At full saturation** (back-to-back fma_loop dispatches, no
  inter-dispatch idle), both will report > 90 % utilization. They
  will agree to within 5 pp.
- **At intermediate utilization** (50 % busy / 50 % idle within
  each 100 ms window), they may diverge more. ioreg reads an
  instantaneous-ish "is the GPU pipeline currently active?"
  whereas powermetrics may average residency across the sample
  window — different mathematical definitions of "utilization."
  Predicted disagreement: 5-15 pp at intermediate values, with
  ioreg slightly higher because it treats brief active windows as
  fully active.
- **Steady-state agreement is better than transition-state
  agreement.** During the transition from one staircase step to
  the next, both signals are responding to a workload pattern
  change with possibly-different latencies.

If the hypothesis holds, exp 007 PASSES with notes about the
intermediate-residency divergence. If ioreg is wildly different
(e.g. always 100 % once the GPU is woken up, or always reports the
same number regardless of load), exp 007 FAILS and we need a
different sudo-free signal.

## What we are NOT trying to answer

- **GPU power agreement** (powermetrics reports mW; ioreg does
  not). That requires IOReport's `GPU Energy` channel, deferred to
  exp 007b.
- **Per-process GPU time attribution** — neither ioreg nor
  powermetrics' default sampler does this; powermetrics
  `--show-process-gpu` does, but it's a different question.
- **Frequency / DVFS agreement.** ioreg doesn't expose current GPU
  frequency in a clean field; powermetrics does
  (`GPU HW active frequency`). Future experiment.
- **Fan RPM / die temperature agreement.** ioreg has some thermal
  data via `IOHID` (separate API); powermetrics' SMC sampler is
  gone in macOS 26. Future experiment via IOHID Swift bridge or
  ctypes.
- **Cross-chip generalization.** This is M4 Max-specific; ioreg's
  `AGXAccelerator` class name differs by generation
  (`G13X` on M1 Pro, `G16X` on M4 Max).

## Setup

### Prerequisites the user (not Claude) starts

In a separate terminal, the user starts a powermetrics CSV recording
at 100 ms cadence:

```
sudo uv run notes/gpu_telemetry.py \
  --csv experiments/007-ioreg-vs-powermetrics-utilization/raw/PMTELEM.csv \
  --interval-ms 100 \
  --quiet
```

(The exact CSV path can be adjusted; `run.py` takes
`--powermetrics-csv` and verifies the file exists with at least 2
rows before starting.)

The user lets this run for the duration of the experiment, then
Ctrl-C's it after `run.py` reports completion.

### What `run.py` does

1. **Validates** the powermetrics CSV exists and has at least 2 rows.
2. **Starts ioreg sampler thread** — every 100 ms, runs
   `ioreg -rl -k "PerformanceStatistics" -d 5` (fast filter for
   nodes with the `PerformanceStatistics` property), parses out
   `Device Utilization %`, `Renderer Utilization %`,
   `Tiler Utilization %`, and writes to
   `raw/{ts}-ioreg.csv` along with `time.monotonic_ns()`.
3. **Phase 0 — baseline (10 s):** no GPU work; lets both signals
   settle to idle.
4. **Phase 1 — utilization staircase (40 s):** 5 steps × 8 s each,
   each step targeting a different time-share of `fma_loop`
   iters=65536 at 32 threads (p50 ~832 µs per dispatch, per exp 004
   M4 Max addendum). Steps:
   - 0 % busy (8 s idle)
   - 25 % busy (~24 dispatches/s)
   - 50 % busy (~48 dispatches/s)
   - 75 % busy (~72 dispatches/s)
   - 100 % busy (back-to-back, ~120 dispatches/s)
   Dispatches are paced by Python sleep between batches; the
   `target_busy_fraction` is stamped per-dispatch in a
   `phase_marker` CSV.
5. **Phase 2 — tail (10 s):** no GPU work; both signals should
   return to idle.
6. **Stops ioreg sampler.**
7. **Writes phase-marker CSV** with monotonic_ns boundaries of each
   phase / step.
8. **Tells the user** to Ctrl-C the powermetrics collector.

### What `analysis.py` does

1. Loads `{ts}-ioreg.csv`, the user-supplied
   `{PMTELEM}.csv`, and `{ts}-phases.csv`.
2. Bins both CSVs into 100 ms windows by `monotonic_ns`. For each
   bin, computes the disagreement in percentage points.
3. Reports per-phase-step:
   - Mean ioreg `Device Utilization %`
   - Mean powermetrics `GPU HW active residency %`
   - Mean disagreement, p95 disagreement, max disagreement
4. Reports overall transition-window disagreement (the 1-second
   window centered on each step transition).
5. Verdict per the success criteria above.

### What we record

Per ioreg sample (`raw/{ts}-ioreg.csv`):
- `iso_ts`, `monotonic_ns`
- `device_util_pct`, `renderer_util_pct`, `tiler_util_pct`
- `in_use_system_memory_bytes` (subsidiary)
- `recovery_count` (subsidiary; should be 0 throughout)

Per phase boundary (`raw/{ts}-phases.csv`):
- `monotonic_ns`, `phase` (e.g. "baseline", "step_0pct"), `target_busy_fraction`

Per dispatch in the staircase phase (`raw/{ts}-dispatches.csv`):
- `monotonic_ns`, `step_label`, `target_busy_fraction`,
  `dispatch_idx_within_step`

## Success criterion

The experiment succeeds (in the discipline's sense) if:
1. Both CSVs are present after `run.py` completes.
2. Phase-marker CSV is present.
3. Analysis can fill in the per-step agreement table.

It produces a usable answer if we can categorize the result as
PASS / MARGINAL / FAIL per the thresholds above, and decide whether
ioreg is a viable sudo-free substitute for powermetrics utilization.

## New questions we expect this to raise

- If ioreg has different sampling latency than powermetrics:
  visible as systematic offset during transitions. Quantifying it
  may inform a calibration step.
- If ioreg's Renderer/Tiler utilization fields tell us something
  about pipeline-side activity that powermetrics doesn't expose:
  potential bonus signal we didn't have before.
- If `In use system memory` from ioreg tracks dispatch buffer
  allocation in a useful way: maybe a signal for "is the kernel's
  output buffer reused?"
- If `recovery_count` ever increments during normal experiments:
  we've been silently triggering GPU resets and didn't know it
  (very unlikely but worth checking).

## After this experiment

Branches:

- **PASS.** ioreg is a viable sudo-free utilization signal.
  Pre-register exp 007b: IOReport's `GPU Energy` channel agreement
  with powermetrics `GPU Power × interval`. If 007b also passes,
  the entire telemetry stack can move to sudo-free, and
  `gpu_telemetry.py`'s sudo dependency becomes optional.
- **MARGINAL.** ioreg works but with a known bias. Document the
  calibration; use ioreg for relative-utilization questions and
  fall back to powermetrics for absolute claims. Still a win,
  smaller than PASS.
- **FAIL.** ioreg's utilization measures something different from
  powermetrics. Investigate what it actually measures
  (instantaneous pipeline state? Recent-window average with a
  different window?). May still be useful as a *different* signal
  but doesn't replace powermetrics. Pre-register a follow-on to
  characterize the difference.

---

## Result

**Date run:** 2026-04-28 (timestamp prefix `20260428T141616`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Wall-clock:** 60.0 s (10 s baseline + 5×8 s staircase + 10 s tail)
**Outcome:** **FAIL** — but in a specific, mechanism-revealing way.

### Headline

ioreg's `Device Utilization %` field does NOT update at a cadence
useful for 100 ms sampling. Out of **601 ioreg samples**, **599
returned `0`** (regardless of phase or actual GPU load) and only
**2 returned non-zero** (55 and 52, both within the same 700 ms
window in `step_025pct`). powermetrics' `gpu_active_pct` over the
same period correctly tracked the staircase from 0 % to 99.8 %.

| phase / target |  n_bins | mean_ioreg % | mean_pm % | mean diff (pp) |
|----------------|--------:|-------------:|----------:|---------------:|
| baseline (0 %) |      95 |       0.0    |    15.2   |    -15.18      |
| step_000pct    |      77 |       0.0    |    14.0   |    -14.04      |
| **step_025pct**|      76 |   **1.4** *  |    67.8   |    -66.40      |
| step_050pct    |      76 |       0.0    |    98.4   |    -98.35      |
| step_075pct    |      76 |       0.0    |    99.1   |    -99.13      |
| step_100pct    |      76 |       0.0    |    99.8   |    -99.79      |
| tail           |      94 |       0.0    |    25.4   |    -25.36      |

\* The 1.4 % mean for step_025pct is entirely from the two single-
sample updates; every other sample in that step was 0.

Overall median |diff| = **65.71 pp**, p95 = 100 pp, max = 100 pp.
Per the pre-registered thresholds (PASS ≤5 pp, MARGINAL ≤10 pp,
FAIL >10 pp), this is solidly FAIL.

### Mechanism

The 2 non-zero samples landed 700 ms apart in step_025pct. Their
values (55 % device, 5-6 % renderer, 2-3 % tiler) are both
plausible — *they look like a real internal driver snapshot* — but
they're isolated. Renderer/Tiler fields show the same pattern
(599 zeros, 2 non-zero, same indices).

Best explanation: **the AGX driver computes Device/Renderer/Tiler
Utilization % on its own internal cadence (probably seconds) and
writes the result into the IORegistry property dict for ioreg to
read.** The driver-side update is sparse, the field is "the latest
internal sample," and at 100 ms cadence we mostly catch stale
zeros from before the driver's next sample arrives.

Sanity check that we're reading the right node: the
`In use system memory` field in the same `PerformanceStatistics`
dict tracked the workload — 1.261 GB at baseline rising to 1.656
GB during step_050pct and back down. That's a different field with
different update semantics: it's the live MTLBuffer-allocated
total, not a periodically-published average.

### What this means

- **ioreg is NOT a sudo-free substitute for powermetrics
  utilization.** Not at 100 ms cadence; probably not at 1 s either
  given the observed update density.
- The agent-survey's recommendation that ioreg `Device
  Utilization %` is a clean "is the GPU busy right now" signal
  *was wrong*. The field exists and has the right semantics in
  principle, but the driver's update cadence makes it useless for
  microbench-scale telemetry.
- The agent's IOReport recommendation is **untested but more
  promising** because IOReport uses the
  `IOReportCreateSamples` → `IOReportCreateSamplesDelta` pattern
  — the consumer asks for state at two timestamps and computes
  the delta itself, rather than depending on the driver's internal
  publishing cadence.

### Surprises

#### 1. The renderer/tiler split when ioreg DID report

The two non-zero samples reported `device=55, renderer=6, tiler=3`
and `device=52, renderer=5, tiler=2`. The "device" number is much
higher than the renderer or tiler share — possibly "device"
includes compute work (which our fma_loop kernel is) while
renderer/tiler are graphics-pipeline-specific. Useful to remember
if we ever DO use these fields: "Device Utilization %" is the
all-up signal, "Renderer/Tiler" are graphics-only.

#### 2. powermetrics baseline of 14-25 %

Even at "0 % busy" baseline and tail phases, powermetrics reported
14-25 % GPU active residency. That's other apps' GPU activity
(WindowServer compositing, Brave/Chrome rendering, etc.) — not
noise. The "GPU at idle" assumption is wrong on a normal
interactive desktop; "GPU at idle minus user apps" requires
killing the compositor or running headless.

#### 3. The InUseMem field IS useful

`In use system memory` from the same PerformanceStatistics dict
went from 1.26 GB at baseline to 1.66 GB at peak workload — clean
correlation with the dispatch buffer allocation. Not what we
came for, but a free observation: ioreg DOES give us a usable
"GPU memory in use" signal at 100 ms cadence even when its
utilization fields don't work.

### What this changes

- **Decision: do not use ioreg utilization in any future
  experiment.** The field is too sparsely updated to be useful.
- The agent-survey's storage / architecture recommendations are
  unaffected by this result — they didn't depend on ioreg working.
- `gpu_telemetry.py`'s sudo dependency stays for now. The
  user-runs-sudo workflow is the available path until IOReport is
  validated.

### After this experiment

Branch landed on: **FAIL**. Per the pre-registration:

> FAIL. ioreg's utilization measures something different from
> powermetrics. Investigate what it actually measures
> [...] Pre-register a follow-on to characterize the difference.

The "what does ioreg actually measure" question is partially
answered above (a sparsely-updated driver-published number). A
deeper characterization isn't worth doing — the operational
finding is that ioreg utilization is unsuitable for our purposes,
which is enough.

The right next experiment is **exp 008: IOReport `GPU Energy`
delta vs powermetrics `GPU Power`** — same comparison structure,
different (and more promising) source. IOReport requires writing
ctypes bindings against `/usr/lib/libIOReport.dylib`, so it's
~2-3× the implementation cost of 007, but the agent's research
suggests it's the right path for a sudo-free telemetry stack.

Optional intermediate step: `exp 007a` could just shell out to
`macmon` (Rust binary using IOReport) if installed, sidestepping
the binding work for a first pass. Then if IOReport-via-macmon
agrees with powermetrics, the binding work for an in-process
collector becomes worth the investment.
