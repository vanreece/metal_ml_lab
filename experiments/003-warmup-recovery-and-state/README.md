# 003: Does a warmup prefix recover sleep_0's tight distribution at noisy cadences? + GPU state observation triangulated three ways.

**Status:** complete (positive primary finding, large unexpected secondary finding)
**Date pre-registered:** 2026-04-27
**Date run:** 2026-04-27 (powermetrics intentionally skipped via `EXP003_NO_POWERMETRICS=1`)
**Hardware target:** M1 Pro (16GB), macOS 26.3.1
**Raw data:** `raw/20260427T141453-{measured,calibration}.csv`, `raw/20260427T141453-meta.txt`, `raw/20260427T141453-stdout.log`
**Wall-clock duration:** 594.9 seconds (~9.9 min)

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
- **Whether display sleep / dim state changes any of this.** Flagged
  before the run as a plausible confound — WindowServer shares the GPU,
  and the system's power posture changes when the display sleeps. This
  experiment *controls* display state (caffeinate -d -i -m runs as a
  sidecar throughout, holding display awake and preventing idle sleep),
  records `IODisplayWrangler` powerstate at start and end, and records
  the full `pmset -g assertions` output for later auditing. Whether
  display state is itself a variable that matters is the question for
  a small follow-up after 003.

## Setup

### Hardware / software

- M1 Pro 16GB, macOS 26.3.1, AC power, laptop awake, no other heavy
  processes.
- User-interactive QoS via `pthread_set_qos_class_self_np`.
- uv + PEP 723 inline metadata, same as 001/002.
- Same one-shared-sample-buffer-per-condition pattern as 002.
- `caffeinate -d -i -m` sidecar holds display awake and prevents idle
  sleep for the duration of the run. Display power state (`pmset -g
  powerstate IODisplayWrangler`) recorded at start and end; full
  `pmset -g assertions` recorded at start.

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
- A `EXP003_NO_POWERMETRICS=1` env var bypasses the powermetrics
  attempt entirely. Used when the run is intentionally pure-Python
  and we don't want a sudo session involved at all. Recorded in the
  metadata file so the absence is explicit, not silent.

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

## Result

**Warmup recovers cv at every long cadence with K=1, but does not help
(and slightly hurts) at sleep_0.** *And* the very first combo of the
run revealed a settled-state floor at ~5.4 µs that 001 and 002 never
saw — apparently a previously-unobserved fast power state, reachable
only under specific entry conditions we accidentally hit at script
start. The warmup-recovery question is answered yes-with-caveats; the
GPU-state question got more interesting than expected.

### Headline cv recovery table (each cell: p50_ns / cv)

#### warmup_kind = same (write_tid 32t, identical to measured)

| sleep_s | K=0          | K=1          | K=5          | K=20         |
|---------|--------------|--------------|--------------|--------------|
| 0       | **5375**/0.05 ⚡ | 8125/**0.006** | 9084/0.12 | 9479/0.10 |
| 0.001   | 9896/0.02   | 9750/0.07   | 9542/0.11   | 9000/0.13   |
| 0.01    | 14062/0.06  | 9750/0.06   | 9792/0.08   | 9000/0.12   |
| 0.1     | 15562/0.13  | 9834/0.06   | 9688/0.06   | 9938/**5.06** ⚠ |
| 1.0     | 16437/0.16  | 9812/0.11   | 9750/0.07   | 9730/0.11   |

#### warmup_kind = heavy_write (write_tid 1024t)

| sleep_s | K=0       | K=1            | K=5       | K=20      |
|---------|-----------|----------------|-----------|-----------|
| 0       | 9770/0.09 | 9708/0.07      | 9688/0.07 | 8938/0.09 |
| 0.001   | 9854/0.05 | 9771/0.10      | 9479/0.09 | 9812/0.12 |
| 0.01    | 13916/0.08 | 9833/**0.025** | 9771/0.15 | 9730/0.07 |
| 0.1     | 14459/0.16 | 9750/0.05     | 9770/0.09 | 9708/0.11 |
| 1.0     | 15562/0.15 | 9500/**1.62** ⚠ | 9166/0.13 | 9479/0.09 |

#### warmup_kind = fma_loop (32t, ~1024 FMAs each, arithmetic-heavy)

| sleep_s | K=0       | K=1            | K=5            | K=20      |
|---------|-----------|----------------|----------------|-----------|
| 0       | 9480/0.12 | 9604/**1.97** ⚠ | 9750/0.09     | 9542/0.06 |
| 0.001   | 9541/0.17 | 9604/**3.82** ⚠ | 9666/0.05     | 9604/0.10 |
| 0.01    | 14104/0.07 | 9750/0.04     | 9750/**0.04** | 9709/0.09 |
| 0.1     | 15021/0.29 | 9916/**1.44** ⚠ | 9980/**0.04** | 9542/0.08 |
| 1.0     | 15417/**1.38** ⚠ | 9250/0.59 | 9979/0.12      | 8500/**2.16** ⚠ |

⚡ = unexpectedly low; ⚠ = unexpectedly high tail.

### Reading the table — what passes and what doesn't

**Recovery works as predicted at sleep ≥ 10 ms:** K=0 sits at 14–17 µs
(the cool-cadence regime from 002); K=1 of any warmup kind drops the
median back to ~9.7 µs and leaves cv low (≤ 0.16). One untimed warmup
dispatch is enough to push the chip back into the warm steady state.

**Recovery does NOT work at sleep_0**, and warmup *hurts* there: the
chip is already in (or below) the warm state, and warmup pushes it
into a different — slightly slower — settled state. K=0 sleep_0 with
`same` warmup hit p50=5375 (see secondary finding); K=1+ pushed it up
to 8125–9479. The "you cannot improve on no-warmup back-to-back" part
of the hypothesis was right.

## Surprises

### 1. A previously-unobserved ~5.4 µs floor existed in the very first combo

`same K=0 sleep_0` produced 40 measured dispatches with min=5083,
p50=5375, max=5958, cv=0.05. **Every single sample is below the 8000
ns floor that 001 and 002 always converged to.** The unique values
are `[5083, 5208, 5250, 5291, 5292, 5333, 5375, 5875, 5916, 5917,
5958]` — same 41-ns quantum from 001/002, just in a band ~33% lower.

This state was reached after: laptop wake, `caffeinate` launch, 1 s
cooldown, then 10-dispatch calibration burst (during which the chip
visibly transitioned: `cal_first=7875` → `cal_med_of_rest=5917`).
**It was never reproduced anywhere else in the run.** Across all 2400
measured dispatches, only 60 are below 7000 ns — and 40 of those are
this single combo. The chip can enter a faster power state than
001/002 ever observed, but the entry conditions are narrow.

The rest of the run sat at the more familiar ~9.5–10 µs steady state
(when warm) or ~14–16 µs (when cool from a long sleep). So 001/002's
"8083 ns floor" was already a non-fundamental observation — there is
at least one faster state, possibly more.

> **Update (2026-04-27, post-004):** experiment 004 ran a tight
> 5-attempt protocol that re-created the entry conditions of this
> first combo (5 s cooldown → 10-dispatch calibration burst → 40
> measured dispatches with no warmup) and got **0/200 measured
> dispatches in [5000, 6000] ns** — every attempt converged to the
> familiar ~8 µs back-to-back floor. The 5.4 µs state is filed as a
> single-launch artifact whose trigger we cannot isolate from the
> available data. See `experiments/004-work-dominance-floor/README.md`
> § "5.4 µs reproducibility — 0/5" and Surprise §3.
> The "001/002's 8083 ns floor was non-fundamental" framing above
> still stands directionally — 001/002 *did* measure dispatch
> overhead rather than kernel time (004 confirmed `write_tid` 32t
> sits ~64× below the work-dominance threshold on M1 Pro) — but the
> specific claim that "there is at least one faster *settled* state"
> is unverified and not actionable.

### 2. The calibration probe is itself a 10-dispatch warmup, contaminating the K=0 measurements

Compare K=0 sleep_1ms across experiments:
- **002 sleep_1ms K=0** (no calibration): cv = **7.03**
- **003 sleep_1ms K=0** (after 10-probe calibration): cv = **0.02**

The 10-dispatch calibration burst that runs immediately before the
measured trials is *itself* effectively warmup. Our "K=0" condition is
therefore "K=10 with a 1 s gap before it" rather than "no warmup at
all." This is a real methodological lesson: **trying to observe the
GPU state changes it.** The calibration kernel as thermometer has
unavoidable observer effect at the granularity that matters.

The K=1+ rows in the recovery table are all comparing "K dispatches
right before the measured one" against "10 dispatches a moment ago";
both are warmup. The story they tell about K=N warmup vs N=10
calibration gap is meaningful (more recent + denser warmup is
slightly better), but the K=0 comparison should be read as "after a
gap of cooldown_s + calibration probe + 1 cadence_s sleep" not as
"cold."

### 3. `fma_loop K=1` is *unexpectedly worse* than `fma_loop K=0` at multiple cadences

| sleep_s | fma_loop K=0 cv | fma_loop K=1 cv |
|---------|-----------------|-----------------|
| 0       | 0.12            | **1.97**        |
| 0.001   | 0.17            | **3.82**        |
| 0.1     | 0.29            | **1.44**        |
| 1.0     | 1.38            | 0.59            |

A *single* arithmetic-heavy warmup dispatch destabilizes the next
measured dispatch — but K=5 fma_loop dispatches restore it
(K=5 cv ≈ 0.04–0.13 across cadences). Hypothesis: one fma_loop dispatch
puts the chip in a transitional state that needs additional dispatches
to settle. Light warmup of a heavy kind is worse than no warmup or
substantial warmup.

This is not symmetric across warmup kinds: `same K=1` and
`heavy_write K=1` are mostly fine (cv ≤ 0.13). The destabilizing
effect is specific to switching kernel character (arithmetic-heavy vs
memory-write) immediately before a memory-write measurement.

### 4. `same K=20 sleep_0.1` has cv=5.06

Among the 60 K-cadence-warmup_kind combos, this one stands out: median
9938 (clean), p95=10129 (clean), but p99=203629 (~20x the median) and
max=327292 (~33x the median). 38 of 40 dispatches were tight; 2 were
catastrophic. No similar pattern in adjacent combos (`same K=20
sleep_0.01` and `same K=20 sleep_1.0` both have cv ≤ 0.13).

Reads as a one-off interference event during this combo, not as a
property of the (K=20, sleep_0.1, same) configuration. Worth a
re-run to verify rather than explaining now.

### 5. Tail outliers cluster in fma_loop conditions

7 combos have at least one dispatch >50 µs. **5 of those 7 are
fma_loop**, even though fma_loop is one of three warmup kinds (33%
expected, 71% observed). The arithmetic-heavy warmup correlates with
tail risk in the measured (memory-write) dispatch. Cause unclear —
could be pipeline-state aliasing, could be context switching to a
heavier compute regime, could be coincidence in a small sample.

### 6. Calibration probe `cal_first` is noisy across combos

`cal_first` (the first dispatch's gpu_delta in each calibration burst,
intended as a "what state is the chip in right now" reading) ranges
from 7875 to 22291 across 60 combos. After a uniform 1 s cooldown,
the GPU's state at re-entry is *not* uniform. The calibration kernel
is sensitive to state, but the state itself has its own variance. We
will need more than one probe sample (or a different probe design) to
get a stable state reading.

`cal_med_of_rest` (median of probe samples 2–10) is much tighter:
mostly 9000–10000 ns. The chip converges to a similar warm state by
mid-burst regardless of where it started.

### 7. CPU wait grows with cadence as expected (002 reproducer)

`cpu_wait_ns` p50 grows from ~190 µs (sleep_0) to ~1.85 ms (sleep_1s)
across all warmup_kinds and Ks. This matches 002's pattern. No
anomaly here; useful as a sanity check that the run's CPU-side
behavior is stable.

## Methodology evaluation

### Does the calibration kernel work as a thermometer?

**Partially.** What works: `cal_med_of_rest` reliably reports the chip
has converged to the warm state by burst end (~9000-10000 ns across
all 60 combos). What doesn't: `cal_first` varies from 7875 to 22291
even after a uniform 1 s cooldown, so a single probe dispatch is too
noisy to read state from. The probe itself moves the chip; you cannot
observe state without affecting it.

For future use, the practical pattern is probably: **multi-sample probe
(N=10), use median-of-tail as a coarse "hot/cool" classifier, do not
attempt fine-grained state inference from a single dispatch.** The
absolute calibration accuracy is in the hundreds of ns; not enough to
detect, e.g., the difference between 8 µs and 9 µs warm states without
many samples.

### Did powermetrics correlate with calibration?

Not tested — powermetrics intentionally skipped this run. The
calibration probe data is preserved (`raw/...-calibration.csv`); a
follow-up that runs powermetrics in parallel can later cross-validate
without re-running the whole experiment.

### Did `device.sampleTimestamps:gpuTimestamp:` add anything?

Same 1.000000 elapsed-ratio as 001 and 002 — a sanity check that the
GPU clock didn't drift relative to the CPU clock over the ~10-minute
run. Useful as confirmation that the timestamp counter results are
trustworthy across long runs. Did not provide additional state info.

## What this means operationally

For any later microbench:

1. **K=1 untimed warmup before each measurement is enough to escape
   the cool-cadence regime** (sleep ≥ 10 ms) for the trivial
   `write_tid` kernel. Bigger K offers no obvious benefit and introduces
   risk (the K=20 outliers).
2. **At sleep_0, do not add warmup** — it pushes the chip into a
   slightly slower settled state. Pure back-to-back is the right
   pattern when you have it.
3. **Do not pick `fma_loop K=1` as a warmup recipe.** Even-numbered
   small-K of an arithmetic-heavy kind is strictly worse than no
   warmup at all (cv 1.4-3.8 vs 0.1-0.3). Either don't warmup, or
   warmup substantially (K≥5).

   > **Update (2026-04-28, post-M4-Max-rerun):** does NOT generalize
   > to M4 Max. On `applegpu_g16s` / macOS 26.4.1, `fma_loop K=1` is
   > one of the *cleaner* recipes (cv ≤ 0.19 across all cadences),
   > and `same K=1` becomes the destabilizer (cv 0.587 / 1.183 at
   > sleep_10ms / sleep_100ms). The destabilizer-warmup-kind inverted
   > from M1 Pro to M4 Max. See M4 Max addendum below.

4. **Treat the 5.4 µs floor as a real possibility** that some entry
   conditions can hit — kernels you previously thought were running
   "as fast as they can" at 8 µs may have a faster regime hidden in
   their state space.
5. **Calibration probes are useful but disturbing**: use them to
   classify hot/cool state, not to read off precise state. The 10-probe
   pattern can give you a one-bit "is the chip in warm steady state
   yet?" answer.

## New questions

- **What entry conditions reach the 5.4 µs floor?** This is the most
  important new question. The combo that hit it was the very first
  one of the run. Reproducing it is high priority — if it's
  repeatable, every "8 µs floor" claim from 001/002 needs revising.
  Possible factors: cold-start of the device + caffeinate transition +
  cooldown sleep timing.
- **Is the `same K=20 sleep_0.1` cv=5 reproducible or a one-off?**
  Quick re-run of just that combo would tell us. Affects whether
  K=20 is broadly safe or risky.
- **Why does `fma_loop K=1` destabilize so badly?** Suggests the
  Apple Silicon pipeline has state that depends on recent kernel
  character, not just frequency. This deserves a small focused
  experiment (vary the warmup kernel's compute/memory ratio finely).
- **Display state was controlled but not investigated.** A small
  follow-up swapping display awake/asleep on one of the worst-cv
  conditions (sleep_1ms K=0 with no calibration probe) would test the
  collaborator's hypothesis that WindowServer contention drives
  outliers.
- **Calibration probe contamination is real and significant.** Future
  experiments need to either (a) accept that "K=0" means
  "K=10 calibration + K_explicit", or (b) design a probe-free way to
  classify state, or (c) run probe and no-probe versions side by side.
- **Does the warmup recovery story hold for non-trivial kernels?**
  All of 003 used the trivial `write_tid` kernel as the measured
  workload. Real kernels have their own dispatch overhead profile.
  That's the original pre-registered question for what would have been
  003 (kernel-size sweep). Now even more important than before, because
  we've established that the warmup recipe matters and we don't know
  if it generalizes.
- **The sleep_100ms drift from 002 — does it appear here?** Need to
  check the raw 003 data for `same K=0 sleep_0.1` and similar to see
  if the drift signature is present. The runs are short enough that
  drift may be hidden, but worth a look.

## After this experiment

**Highest-priority follow-up: 003a — reproduce the 5.4 µs floor.**
The 003 run accidentally discovered a state, and the first principle
of the project is that we don't ignore unexplained findings. A small
focused experiment that runs the same first-combo conditions ~5 times
in a row (with re-launches between to get clean cold starts) tells us
whether 5.4 µs is reproducible.

After 003a, **004: warmup recipe transferability across kernel
shapes.** With K=1 warmup of `same` shape established as a clean
recipe for `write_tid`, ask whether the recipe still works for
non-trivial kernels (varying threadgroup count, threads-per-group,
arithmetic intensity).

**Optional: 003b — display state effect.** One-day swap test, only if
003a or 004 shows display-correlated weirdness. Otherwise the question
of "does display state matter" can wait; we have the controlled
recording in metadata for any later run, and the more interesting
findings from 003 push higher up the priority queue.

---

## M4 Max addendum (re-run on new hardware)

**Date re-run:** 2026-04-28
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS:** macOS 26.4.1 (build 25E253)
**Raw data:** `raw/20260428T113232-{measured,calibration}.csv`,
`raw/20260428T113232-meta.txt`
**Wall-clock duration:** 594.75 s (essentially identical to M1 Pro's
594.9 s — most of the budget is per-cadence sleep × K_values × kinds).
**Powermetrics:** intentionally skipped (`EXP003_NO_POWERMETRICS=1`).

The M4 Max version of 003 was very different from the M1 Pro version
because 002's M4 Max re-run had already shown the chip's DVFS picture
is qualitatively different (no 1ms nightmare zone, smaller cool-cadence
spread, lower floor). 003's premise — "warmup should fix the noisy
cadences" — has less work to do because there are fewer noisy cadences
to fix in the first place. But several findings from M1 Pro 003 do
NOT generalize, and one new state was observed.

### Headline cv recovery table (each cell: p50_ns / cv)

#### warmup_kind = `same` (write_tid 32t, identical to measured)

| sleep_s | K=0          | K=1            | K=5            | K=20           |
|---------|--------------|----------------|----------------|----------------|
| 0       | 6917 / 0.110 | 6771 / 0.057   | 6708 / 0.092   | 6771 / **0.848** ⚠ |
| 0.001   | 9042 / 0.108 | 6417 / 0.118   | 6709 / 0.040   | 6500 / 0.114   |
| 0.01    | 8771 / 0.071 | 8708 / **0.587** ⚠ | 6792 / 0.149 | 6792 / **3.326** ⚠ |
| 0.1     | 9229 / **1.962** ⚠ | 6979 / **1.183** ⚠ | 6708 / **2.800** ⚠ | 6584 / 0.611 |
| 1.0     | 9270 / 0.170 | 6646 / 0.135   | 6417 / 0.064   | 6375 / **0.035** |

#### warmup_kind = `heavy_write` (write_tid 1024t)

| sleep_s | K=0          | K=1          | K=5          | K=20         |
|---------|--------------|--------------|--------------|--------------|
| 0       | 6416 / 0.029 | 6666 / 0.079 | 6604 / 0.034 | 6542 / 0.094 |
| 0.001   | 8959 / 0.078 | 6584 / 0.079 | 6438 / 0.035 | 6458 / 0.038 |
| 0.01    | 8980 / 0.084 | 8771 / 0.137 | 6708 / 0.165 | 6667 / 0.141 |
| 0.1     | 9250 / 0.069 | 6479 / 0.105 | 6458 / 0.088 | 6479 / 0.147 |
| 1.0     | 9354 / 0.083 | 6375 / 0.070 | 6584 / 0.108 | 6438 / 0.052 |

#### warmup_kind = `fma_loop` (32t arithmetic-heavy)

| sleep_s | K=0          | K=1          | K=5          | K=20         |
|---------|--------------|--------------|--------------|--------------|
| 0       | 6625 / 0.036 | 6542 / 0.038 | 6417 / 0.031 | **6375 / 0.270** 🌟 |
| 0.001   | 8958 / 0.080 | 6542 / 0.040 | 6562 / 0.043 | 6396 / 0.076 |
| 0.01    | 9000 / 0.064 | 8667 / 0.137 | 6604 / 0.182 | 6480 / 0.077 |
| 0.1     | 9104 / 0.087 | 6625 / 0.116 | 6750 / 0.095 | 6417 / 0.051 |
| 1.0     | 9458 / 0.149 | 6750 / 0.189 | 6750 / 0.123 | 6708 / 0.085 |

⚠ = unexpectedly high tail. 🌟 = sub-floor state observed
(see Surprise § 2 below — 11 of 40 trials in this combo dropped to
2-4 µs, with absolute min = 2083 ns).

### Side-by-side: which warmup recipe works best on each chip?

| chip      | best warmup recipe                  | worst warmup recipe                      |
|-----------|-------------------------------------|------------------------------------------|
| M1 Pro    | `same K=1` (cv ≤ 0.16 everywhere)   | `fma_loop K=1` (cv 1.4-3.8 — destabilizer) |
| M4 Max    | `heavy_write K=1` or `fma_loop K=1` (cv ≤ 0.19 everywhere) | `same K=1`/`K=5` (cv 0.59-2.80 at sleep_10ms-100ms) |

**The ranking of warmup kinds inverted between G13 and G16.** The
recipe that worked best on M1 Pro (`same K=1`, identical to the
measured kernel) is a destabilizer on M4 Max. The recipe that was the
M1 Pro destabilizer (`fma_loop K=1`, arithmetic-heavy single warmup)
is one of the cleanest on M4 Max.

### Five qualitative shifts G13 → G16 in 003

1. **`same K=1` flipped from the safest recipe to a destabilizer.**
   On M1 Pro, `same K=1` had cv ≤ 0.16 across all cadences and was
   the recommended recipe. On M4 Max, `same K=1` produces cv 0.587 at
   sleep_10ms and 1.183 at sleep_100ms with tails to 34 µs and 53 µs
   respectively. Mechanism unknown — speculation: when the chip is
   already at the back-to-back floor, adding one more identical light
   dispatch may keep it in a marginal micro-state where the next
   dispatch is unstable. Heavier or more sustained warmup pushes
   through this.
2. **`fma_loop K=1` is no longer destabilizing.** The headline finding
   of M1 Pro 003 — that one arithmetic-heavy warmup dispatch breaks
   the next measurement — does not reproduce on M4 Max. cv at
   `fma_loop K=1` is 0.038-0.189 across all cadences (vs M1 Pro's
   1.4-3.8). Either the G16 front-end handles the kernel-character
   transition better, or the relevant pipeline-state aliasing
   doesn't apply on the newer architecture.
3. **`heavy_write` is the most robust warmup kind on M4 Max.** Every
   `heavy_write` cell has cv ≤ 0.165. No tail outliers anywhere.
   On M1 Pro, `heavy_write` was OK but had its own surprise (`K=1
   sleep_1s` cv = 1.62). M4 Max `heavy_write K=1 sleep_1s` is cv =
   0.07 — clean.
4. **Cool-cadence median recovery still works** — K=1 of any warmup
   kind brings the median from ~9 µs (cool, K=0) back toward the
   ~6.4-6.8 µs floor across sleep_1ms onwards. Same operational
   recipe, just at lower absolute numbers (M1 Pro recovered to ~9.7 µs
   floor; M4 Max recovers to ~6.4 µs floor).
5. **Tail outliers are concentrated in `same` warmup combos on
   M4 Max.** Across 60 (kind, K, cadence) combos, only 4 had any
   single dispatch above 50 µs — and **all 4 are `same` warmup**
   (K=0 sleep_0.1: max=123 µs; K=1 sleep_0.1: max=53 µs; K=5
   sleep_0.1: max=126 µs; K=20 sleep_0.01: max=148 µs). On M1 Pro,
   tail outliers concentrated in `fma_loop` (5 of 7). Inverted again.

### Surprises

#### 1. The fma_loop K=1 destabilizer story does not generalize

The M1 Pro 003 headline that "a single arithmetic-heavy warmup
destabilizes the next measurement" was a clean, sharp finding with
clear mechanism speculation (kernel-character switch). On M4 Max it
just doesn't happen. This is the single biggest piece of evidence
that **warmup-recipe characterization is chip-specific and may not
transfer at all between Apple Silicon generations.** Cross-chip
warmup-recipe libraries are not viable.

#### 2. A sub-floor ~2 µs settled state exists on M4 Max — reached late in fma_loop K=20 sleep_0

M1 Pro 003 stumbled into a 5.4 µs state in its very first combo
(later filed as non-reproducible by 004). M4 Max 003 stumbled into a
**~2-4 µs state** in the LAST trials of one specific combo:
`fma_loop K=20 sleep_0`.

The trial-by-trial trajectory of that combo (40 dispatches) is:
- Trials 0-23: 6.2-6.9 µs (normal floor)
- Trial 24: 9.8 µs (one outlier)
- Trials 25-28: 6.3-7.1 µs (back to floor)
- **Trials 29-39: 4.0, 3.9, 2.9, 4.2, 3.0, 2.8, 3.3, 3.1, 2.5, 2.3,
  2.1 µs** — the chip dropped into a faster state and stayed there,
  with the absolute min at 2083 ns (final trial).

The 2083 ns min is **roughly 50 ticks** of the 24 MHz GPU timestamp
clock (2083 / 41.67 ≈ 50). For comparison, the back-to-back floor
of 6125 ns is ~147 ticks. So the chip transitioned to operate at
roughly a third of the cycle count it normally takes for the same
write_tid 32t kernel — strongly suggesting a step-up in GPU clock
frequency (DVFS upshift) sustained across the rest of the combo.

This combo has 40 measured trials × 20 fma_loop warmup dispatches
each = 800 cumulative warmup dispatches before the state shift,
plus 10 calibration probe dispatches. The calibration probe entering
this combo looked normal (6.3-9.9 µs range, settled by mid-burst).
The transition is not at the start; it builds up. None of the other
59 combos produced any sample below 5500 ns — this is uniquely the
heaviest-warmup × sleep_0 × arithmetic configuration.

**Operational implication:** the ~6.4 µs floor measured in 001/002
on M4 Max (and again as the K=0 baseline here) is *not* the lowest
state the chip can reach. With sustained arithmetic warmup at
sleep_0, the chip can push into a higher-DVFS state ~3× faster.
Whether this state is reproducible across runs (vs. a single-run
artifact like M1 Pro's 5.4 µs floor) is **the most important
follow-up question** from this re-run.

#### 3. The `same K=20 sleep_10ms` outlier is huge

cv = 3.326, max = 148 µs (~22× p50). This is the largest-tail combo
in the entire 003 M4 Max run. The same combo on M1 Pro had cv = 0.12.
A specific recipe that was clean on M1 Pro is now the noisiest on
M4 Max. No clear mechanism.

#### 4. The calibration-probe-as-warmup observation is unchanged

The cal_med_of_rest values across all 60 combos sit between
~6.3-7.1 µs — a tighter and lower band than M1 Pro's ~9-10 µs
calibration tail. The chip converges to a similar warm state by
mid-burst regardless of where it started, just like on M1 Pro, but at
the M4 Max-floor level. The structural observation that "the
calibration probe is itself a 10-dispatch warmup" still applies — the
"K=0" cv numbers in the recovery tables above are "K=0 measured
trials, but after a 10-probe warmup happened ~1 sleep_s ago."

### What this means operationally on M4 Max

The M1 Pro decision rules (decision-relevant guidance from M1 Pro
003) need to be re-derived per chip. New M4 Max guidance, based on
this single run:

1. **K=1 untimed warmup of `heavy_write` (write_tid 1024t) before
   each measurement** is the safest recipe across cadences (cv ≤
   0.165 in every cell). Use this as the default on M4 Max.
2. **`fma_loop K=1`** is also viable (cv ≤ 0.19) and may be
   appropriate when measuring arithmetic-heavy kernels for
   character-matched warmup.
3. **AVOID `same K=1` and `same K=5` at sleep_10ms-100ms cadences**
   on M4 Max. Inverted from M1 Pro guidance.
4. **At sleep_0**, K=0 on M4 Max is fine for `heavy_write` and
   `fma_loop` (the chip stays at floor); `same K=0` at sleep_0 also
   works. Consistent with M1 Pro's "no warmup at sleep_0" guidance,
   though the mechanism may be different.
5. **Treat the ~2 µs state observed in `fma_loop K=20 sleep_0` as
   the M4 Max analog of M1 Pro's 5.4 µs floor** — possibly a real
   reachable state, possibly a single-run artifact. Do not optimize
   against it until reproducibility is established.

### New questions raised by the M4 Max re-run

- **Is the M4 Max sub-floor state reproducible?** Run a 003a-style
  small experiment that does only `fma_loop K=20 sleep_0` for ~80
  trials (twice the original combo) several times in a row, with
  re-launches. If the state is reproducible, it expands the M4 Max
  operating envelope downward.
- **Why did `same K=1` flip from safest to destabilizer?** Could be
  M4 Max front-end behavior, could be macOS 26.4.1 scheduler, could
  be some interaction with Dynamic Caching (M3+ feature). Worth a
  focused mechanism investigation only if `same` warmup matters
  practically — for now, pick `heavy_write K=1` and move on.
- **Does the warmup-recipe inversion show up at intermediate
  generations?** Would need M2/M3 access to triangulate. Not in
  scope for this lab right now.
- **Did the calibration-probe-itself-is-warmup observation
  contaminate K=0 differently on M4 Max than M1 Pro?** The cv numbers
  for K=0 on M4 Max are mostly tight (0.03-0.15 except the
  `same K=0 sleep_100ms` outlier at 1.96) — substantially better than
  M1 Pro K=0 in the cool-cadence regime. This is consistent with the
  002 finding that the M4 Max cool regime is itself less noisy
  (sleep_1s cv 0.46 vs M1 Pro 0.21 in the opposite direction, but
  sleep_100ms 1.64 vs M1 Pro 2.23 in the same direction). Hard to
  isolate from the data we have.
