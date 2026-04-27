# 003: Does a warmup prefix recover sleep_0's tight distribution at noisy cadences? + GPU state observation triangulated three ways.

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-27
**Hardware target:** M1 Pro (16GB), macOS 26.3.1

## The question (primary)

Experiment 002 found that the per-dispatch GPU duration distribution
collapses to a tight floor (cv≈0.66) when dispatches run back-to-back,
explodes to cv=7.0 with bimodal warm/cool dispatches at 1ms cadence,
and re-tightens to cv=0.21 at 1s cadence. **Can a prefix of K
back-to-back warmup dispatches before each measured dispatch make the
measured cv depend on K rather than on the cadence between
measurements?**

If yes, this gives the project a recipe for honest microbenchmarking
at any cadence. If no, the chip's power-state machine has hysteresis
we cannot paper over and any later experiment must inherit cadence
noise as a fact of life.

## The question (secondary, methodology)

Three independent paths exist for observing GPU power state during a
run. We do all three at once and compare:

1. **`powermetrics` sidecar** (Apple's vendor counters, requires sudo).
   Real GPU frequency and per-P-state residency at ~200 ms sampling.
   The "ground truth" reference.
2. **Calibration kernel as thermometer**. Run a known short burst of
   the measured kernel at sleep_0 immediately before each combination.
   The first dispatch's timing reflects the GPU's incoming state; the
   tail of the burst reflects warm steady state. Pure Python, no sudo.
3. **`device.sampleTimestamps:gpuTimestamp:`** (CPU↔GPU clock
   correlation, free, already in scripts). Snapshots taken at
   well-defined points; useful for time alignment but does not
   directly expose P-state.

The methodology question: **does the calibration kernel's timing
correlate with powermetrics-observed GPU frequency well enough that we
could later drop the sudo dependency?**

## Why this experiment is the right next one

The pre-registered "obvious next" was 003 (kernel-size sweep) and
the urgent follow-up was 003a (sleep_100ms drift reproduction). This
experiment supersedes both by being higher uncertainty-reduction:

- If warmup recovery works, the entire framing of how to design later
  experiments changes (you stop worrying about cadence and start
  designing warmup prefixes).
- If powermetrics correlates with calibration kernel timing, we
  potentially get a sudo-free way to assess GPU state, which the
  project's whole "without vendor counters" thesis cares about.
- The sleep_100ms drift may be incidentally explained or made more
  mysterious by powermetrics frequency observation during this run —
  either way we learn something.
- Kernel-size sweep is conditional on knowing how to control for
  cadence noise. Doing it before 003 would inherit unexplained
  variance.

## Hypothesis

Confidence: low across the board. Specific predictions, mostly to
prevent retrofitting after the fact:

- **K=5 with the same-as-measured warmup kernel will largely recover
  sleep_0's cv at sleep_1ms.** Magnitude of recovery: median should
  return to ~8083 ns, cv should drop from 7.0 toward 1.0 at least.
  K=20 should be at least as good as K=5.
- **Heavier warmup kernels (more threads, real arithmetic) will recover
  cv faster (smaller K needed) than the lighter same-as-measured
  warmup,** because they push DVFS up harder per dispatch.
- **At sleep_0, warmup will be redundant** — the measurements already
  *are* warm. Higher K should produce no measurable change.
- **At sleep_1s, warmup may be insufficient** — the cooldown between
  measurements is so long that whatever state the warmup pushed the
  chip into has decayed by measurement time. Cv may stay near sleep_1s
  baseline regardless of K.
- **powermetrics will show GPU frequency clearly elevated during
  warmup bursts.** Frequency time-series should correlate with the
  warmup/measurement schedule visible to the eye.
- **Calibration kernel first-dispatch timing will correlate
  monotonically with powermetrics-reported GPU frequency at that
  moment.** If the correlation is strong (Spearman > 0.7), the
  calibration kernel is a viable proxy.

## What we are NOT trying to answer

- Whether the result generalizes to non-trivial kernels. Same kernel
  shape as 001/002 throughout; varying that is a separate experiment.
- Whether the result generalizes across M-series chips. M1 Pro only.
- Whether warmup persists across longer dwell times than 1s. The
  longest cadence here is 1s.
- Whether per-dispatch sample-buffer indexing scales beyond what we've
  used so far. Same shared-buffer pattern as 001/002.
- Anything about thermal effects. Run is short enough that thermals
  should not move materially.

## Setup

### Hardware / software

- M1 Pro 16GB, macOS 26.3.1, AC power, laptop awake, no other heavy
  processes.
- User-interactive QoS via `pthread_set_qos_class_self_np`.
- uv + PEP 723 inline metadata, same as 001/002.
- Same one-shared-sample-buffer-per-condition pattern as 002.

### Variables

- **Warmup kernel kind** (3 levels):
  1. `same` — identical to measured kernel: `write_tid`, 32 threads,
     1 SIMD width.
  2. `heavy_write` — `write_tid`, 1024 threads (32 threadgroups). Same
     work pattern, more of it.
  3. `fma_loop` — small kernel doing a tight loop of fused multiply-adds
     to push compute units. 32 threads, but each thread does ~1024 FMAs.
     Tests whether arithmetic-heavy warmup pushes DVFS differently than
     memory-write warmup.
- **K** (warmup count, 4 levels): `{0, 1, 5, 20}`.
- **Cadence** (5 levels): `sleep_s ∈ {0, 0.001, 0.01, 0.1, 1.0}`.
  Same set as 002 for direct comparability.

### Measurement

- For each (warmup_kind, K, cadence) combination: run a calibration
  probe (10 sleep_0 dispatches of the measured kernel, no warmup),
  then N=40 trials where each trial = sleep cadence_s, run K warmup
  dispatches (untimed), run 1 measured dispatch with timestamp counters.
- Cooldown: between combinations, sleep `max(cadence_s, 1.0)` seconds
  before the next calibration probe. Ensures the chip has time to
  decay from any high-K trail.
- Order: outer = warmup_kind (same → heavy_write → fma_loop), middle =
  cadence (ascending), inner = K (ascending). Ascending order keeps
  thermal-bias monotonic and visible.
- Total measured dispatches: 3 × 4 × 5 × 40 = **2400**.
- Total calibration dispatches: 3 × 4 × 5 × 10 = **600**.
- Estimated runtime: ~10–12 minutes (dominated by sleep_1s cadence
  sleep budget × 4 K values × 3 warmup kinds = ~480 s of pure sleep,
  plus overhead).

### powermetrics sidecar

- Started at script entry, before any GPU work, via `subprocess.Popen`.
- Command: `sudo -n powermetrics --samplers gpu_power -i 200`
  (200 ms sampling). `-n` is non-interactive: if sudo asks for a
  password we treat it as unavailable.
- Output redirected to `raw/<ts>-powermetrics.txt`.
- Stopped at script exit via SIGTERM.
- If sudo is not pre-authenticated (`sudo -n true` fails), the script
  prints clear instructions, sets `pm_proc=None`, and proceeds without
  the sidecar. The metadata file will record whether powermetrics ran.
- Pre-run instruction printed to user: `sudo -v` first, then run.

### What we record

Per measured dispatch (one big CSV per warmup_kind, three CSVs total):

- `warmup_kind`, `K`, `sleep_s`, `trial_idx_within_combo`
- `wall_clock_ns` (CPU monotonic at start of trial)
- `gpu_t_start_raw`, `gpu_t_end_raw`, `gpu_delta_raw`
- `cpu_encode_ns`, `cpu_commit_ns`, `cpu_wait_ns`, `cpu_total_ns`

Per calibration probe (one CSV total):

- `warmup_kind`, `K`, `sleep_s` (the combination this probe precedes)
- `probe_idx_within_burst` (0..9)
- `wall_clock_ns`
- `gpu_t_start_raw`, `gpu_t_end_raw`, `gpu_delta_raw`
- `cpu_*_ns` (same as above)

Per powermetrics sample (raw text from powermetrics, parsed later if
needed):

- `raw/<ts>-powermetrics.txt`. Format is powermetrics' default
  human-readable text, one block per sample, every 200 ms.

Per run (metadata):

- Device, OS, Python/PyObjC versions, QoS, power source.
- Whether powermetrics ran. PID, start/stop times.
- Per-combination summary (median, p05, p50, p95, p99, max, cv,
  in_floor count for the measured dispatches; same for calibration
  probes).
- Wall-clock duration of the whole run.
- Timestamp correlation snapshots at start and end.

### What we do NOT do

- No averaging in any live output. Means are absent on purpose.
- No retries on failure. If a buffer alloc fails, the script crashes
  loudly and the partial data is preserved.
- No warmup outside the explicit K warmup dispatches. The measured
  dispatch is whatever the chip happens to be doing; that is the point.
- No discarding of outliers. Every measured dispatch goes into the CSV.

## Success criterion

The experiment succeeds (in the sense the discipline cares about) if
we have:

1. CSVs for all three warmup kinds with all (K, cadence) combinations.
2. A calibration-probe CSV with one 10-dispatch burst per combination.
3. Either a powermetrics text file or an explicit metadata note that
   sudo was unavailable.

It produces a usable answer if we can fill in this table from the
data:

| cadence    | K=0 cv | K=1 cv | K=5 cv | K=20 cv | Recovers? |
|------------|--------|--------|--------|---------|-----------|
| sleep_0    |        |        |        |         |           |
| sleep_1ms  |        |        |        |         |           |
| sleep_10ms |        |        |        |         |           |
| sleep_100ms|        |        |        |         |           |
| sleep_1s   |        |        |        |         |           |

(Three copies, one per warmup kind.) "Recovers" = whether the cv at
some K matches sleep_0's K=0 cv.

And answer methodology questions:

- Does the calibration probe's first-dispatch timing track powermetrics
  frequency well enough to substitute? (Compute Spearman ρ between
  per-combination first-probe gpu_delta_raw and powermetrics frequency
  in the surrounding 1s window. ρ > 0.7 = "viable proxy.")
- Does warmup *kind* matter, holding K constant? (Compare cv across
  warmup kinds at the same (K, cadence).)

## New questions we expect this to raise

- If K=5 recovers sleep_1ms's cv: how persistent is the warmth across
  the cadence sleep? Implies a follow-up varying time *between* warmup
  and measurement.
- If K=20 doesn't recover sleep_1s's cv: the cooldown wins over
  warmup at long cadences. What's the half-life of warmup state? Could
  be probed with K=20 followed by varying delay-then-measure.
- If `fma_loop` is qualitatively different from `same`: arithmetic
  intensity matters for DVFS push, opening a much bigger sweep about
  what kernel shapes warm the chip "best."
- If powermetrics frequency does NOT correlate with calibration timing:
  the calibration kernel is measuring something other than (or in
  addition to) frequency — maybe pipeline state, maybe SLC state,
  maybe scheduling. Each of those is its own follow-up.
- If powermetrics shows frequency time-series with structure we did
  not predict (oscillations, hysteresis bands, multi-state stairs):
  follow-up to characterize the state machine itself.
- The sleep_100ms drift from 002 — does it show up here too? If yes,
  is it visible in powermetrics frequency drift?

## After this experiment

Branches:

- **Warmup recovers, calibration kernel is a good proxy.** Next is a
  small follow-up to characterize warmup *persistence* (how long does
  warmth last after warmup ends?), then back to 004 (kernel-size
  sweep) using a "K=N warmup before each measurement" pattern.
- **Warmup recovers but calibration kernel is not a good proxy.** Next
  is to understand what calibration is measuring instead. Probably a
  small experiment varying calibration kernel shape and re-correlating
  with powermetrics.
- **Warmup does not recover.** Hysteresis is real, the project must
  inherit cadence noise. Next is documenting *what cadence ranges are
  viable for measurement* and constraining future microbench design
  to those ranges.

We do not plan past these branches.
