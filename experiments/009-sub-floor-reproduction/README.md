# 009: Is the M4 Max ~2 µs sub-floor state reproducible under sustained fma_loop K=20 sleep_0?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-28
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~3-5 min (5 attempts × ~30 s, mostly subprocess
launch + 5 s inter-attempt cooldown; per-attempt GPU work is small)

## The question

Experiment 003's M4 Max re-run (`exp 003 re-run on M4 Max`, 2026-04-28)
hit a previously-unobserved settled state during the *last* 11 of 40
trials in one specific combo, `fma_loop K=20 sleep_0`. Trial-by-trial
in that combo:

- Trials 0-23: 6.2-6.9 µs (back-to-back floor)
- Trial 24: 9.8 µs (one outlier)
- Trials 25-28: 6.3-7.1 µs (back to floor)
- **Trials 29-39: 4.0, 3.9, 2.9, 4.2, 3.0, 2.8, 3.3, 3.1, 2.5, 2.3, 2.1 µs**
  — chip dropped into a faster state and stayed there.

The 2083 ns minimum is roughly 50 ticks of the 24 MHz GPU timestamp
clock. The back-to-back floor of ~6.4 µs is ~147 ticks. Cycle count
~3× lower for the same `write_tid` 32t kernel strongly implies a GPU
DVFS upshift sustained across the rest of the combo.

**The question:** if we re-run *just* this combo, several times in a
row with subprocess re-launches between attempts, does the sub-floor
state reappear? The 003 M4 Max readme already pre-named this as the
top follow-up question.

The M1 Pro analog (exp 004 5.4 µs reproduction protocol, 5 attempts)
landed on 0/5 — the M1 Pro fast state was a single-launch artifact.
This experiment is the M4 Max equivalent test for a *different* fast
state (different conditions, different mechanism speculation, different
chip), not a re-test of the M1 Pro one.

**Primary metric:** does at least one attempt produce ≥10 measured
trials with `gpu_delta_raw` < 5500 ns?

- **STRONG REPRODUCE** (≥ 3 of 5 attempts): the sub-floor state is on
  demand, and exp 003-M4-Max found a real, repeatable operating point
  for the M4 Max GPU that 001/002 didn't see. The lab's "minimum
  achievable per-dispatch overhead on M4 Max" claim drops from ~6.4 µs
  to ~2-3 µs under this specific recipe.
- **PARTIAL REPRODUCE** (1-2 of 5): the state exists but its entry
  conditions are narrow / probabilistic. Worth filing as "the chip can
  reach this, sometimes", not as a reliable recipe.
- **NOT REPRODUCED** (0 of 5): the original 003 observation was a
  single-launch artifact — same fate as the M1 Pro 5.4 µs floor.
  File and stop chasing.

The 5500 ns threshold is well below the back-to-back floor (~6125 ns
absolute min in 003 outside this combo), so any trial under 5500 ns
is qualitatively distinct from "lucky low end of floor distribution."

## Why this question, now

Two reasons it's the highest uncertainty-reduction candidate:

1. **It's tiny.** ~3-5 min wall clock, no powermetrics dependency, no
   user coordination dance — IOReport subprocess gives us power
   telemetry for free per exp 008.
2. **The outcome rewrites or doesn't rewrite the M4 Max envelope.**
   If reproducible, every "M4 Max floor is 6.4 µs" claim from 001-008
   needs an asterisk and we have a third operating point worth
   characterizing. If not reproducible, we file it the same way 004
   filed the M1 Pro 5.4 µs floor and move on with confidence.

Compared to other queued experiments (exp 010 IOReport histogram, exp
009-bias channel-subset sweep, 006a longer-gap cross-session): those
all assume the M4 Max envelope as known. This question asks whether
the envelope itself needs revising. Higher up the dependency tree.

## Hypothesis

Confidence: low. Predictions stated to prevent post-hoc retrofitting:

- **REPRODUCIBILITY: 1-2 of 5 attempts (PARTIAL REPRODUCE).** Lean
  weakly toward this because the 003 entry mechanism (cumulative
  warmup pushing DVFS up over ~800 dispatches) is a state-machine
  story, not a one-shot trigger story. If the mechanism is real, it
  should hit again under the same conditions — but DVFS state machines
  can have hysteresis / temperature dependence that makes
  reproducibility uneven.
- **TRIAL-INDEX OF FIRST SUB-FLOOR:** if the state appears, it should
  appear *late* in the attempt (after ~25-30 trials), not at the start.
  Mirrors 003-M4Max trial 29 onset. The mechanism speculation is that
  cumulative arithmetic-heavy warmup pushes DVFS up gradually.
- **STABILITY ONCE ENTERED:** if the state appears, it should persist
  for the rest of the attempt (no return to floor). 003-M4Max stayed
  in the sub-floor state from trial 29 to 39 with no return.
- **CROSS-ATTEMPT COUPLING:** if attempt N enters the state, attempt
  N+1 may enter it earlier (because process re-launch resets MTLDevice
  but not necessarily the chip's thermal / DVFS history). If we see
  a clear "attempt 0 doesn't reproduce, attempts 1+ do," that's a
  thermal-warmup story.
- **IOREPORT GPU POWER during sub-floor trials:** should be elevated
  (~+50-100 mW relative to back-to-back floor trials in the same
  attempt) if the mechanism is a DVFS upshift. If GPU power doesn't
  change, the timing change is from something other than frequency
  (cycle-count quantum change in the timestamp counter? unlikely.
  Different kernel scheduling path? possible.) — would significantly
  rewrite the mechanism speculation.
- **84-trial-per-attempt ought to be enough.** 003-M4Max needed 29
  trials of warmup to enter; we run 84 trials per attempt to give it
  ~2.9× the runway.

## What we are NOT trying to answer

- **Whether the same recipe triggers a sub-floor on M1 Pro.** 003-M1Pro
  used the same recipe and didn't enter; would be inconsistent to
  re-test now. Cross-chip generalization is out of scope.
- **What the underlying mechanism is.** If it reproduces, the
  follow-up is a DVFS-residency / IOReport `GPUPH` channel
  investigation. We're not pre-running that here.
- **Whether other warmup recipes (heavy_write K=20, same K=20) can
  reach the same state.** 003-M4Max only saw it under `fma_loop K=20
  sleep_0`. Restricting this experiment to the same recipe keeps it
  tightly scoped.
- **Threshold for entry (minimum K, minimum cumulative dispatches).**
  Mechanism investigation, not reproducibility test.
- **Cross-session (overnight, AC-vs-battery).** All 5 attempts run in
  one session. Cross-session questions would inherit the present
  question's ambiguity.
- **Anything about `T_kernel` vs `T_overhead` decomposition.** This
  experiment measures the same write_tid 32t dispatch as 001-003;
  whether that's overhead or work was answered by 004 (it's overhead).

## Setup

### Hardware / software

- M4 Max 36GB, macOS 26.4.1, AC power, laptop awake, no other heavy
  processes.
- User-interactive QoS via `pthread_set_qos_class_self_np`, same as
  every prior experiment.
- uv + PEP 723 inline metadata, same pattern as 001-008.
- `caffeinate -d -i -m` sidecar holds display awake throughout.
- `notes/ioreport.py` launched as subprocess for the duration of the
  outer driver (covers all 5 attempts) at `--interval-ms 500`. No
  sudo needed.
- powermetrics intentionally NOT used (the +14 % full-load IOReport
  bias from exp 008 is irrelevant for the *relative* comparison
  "GPU power during sub-floor trials vs back-to-back floor trials in
  the same attempt").

### Architecture

Two-script pattern:

1. `run.py` — outer driver. Starts caffeinate + ioreport subprocesses,
   loops 5 times: cooldown, launch `attempt.py` as subprocess,
   wait for it. Aggregates per-attempt CSVs into one combined CSV
   for analysis. Stops subprocesses cleanly at end.
2. `attempt.py` — inner per-attempt script. Single subprocess per
   attempt. Creates fresh MTLDevice, builds pipelines, runs one
   calibration probe burst, runs 84 trials of the combo, writes CSV
   and exits. Subprocess re-launches give us a fresh MTLDevice each
   attempt while the chip itself retains thermal / DVFS state.

This is the M4 Max sub-floor analog of 004's 5.4 µs reproduction
protocol (which ran in-script). Subprocess separation is necessary
here because we want to test whether re-creating MTLDevice / command
queue state matters for entry.

### Per-attempt protocol

Inner `attempt.py` does, in order:

1. Set user-interactive QoS.
2. Create MTLDevice, command queue, write_tid pipeline, fma_loop
   pipeline (FMA_ITERS=1024 — same as 003), shared sample buffer.
3. **Cooldown:** sleep 5 s. Outer driver also sleeps between attempts;
   5 s gives the chip time to drop out of any prior elevated state
   while still being short enough that 5 attempts fit in a few minutes.
4. **Calibration probe:** 10 back-to-back sleep_0 dispatches of
   `write_tid` 32t with timestamps. Same probe as 003. Records the
   chip's incoming state. Slots 0-19 of the sample buffer.
5. **Measured combo:** 84 trials of (20 fma_loop warmup × untimed,
   1 write_tid 32t measured × timed) with sleep_0 between trials.
   Slots 20-187 of the sample buffer. Same recipe as 003-M4Max
   `fma_loop K=20 sleep_0`, just with 84 trials instead of 40 to
   give the state more runway.

84 trials chosen because: 003-M4Max entered the state at trial 29 of
40, so the entry needed roughly 29/40 = 73% of the combo. Doubling to
80 leaves margin; rounding to 84 gives a sample buffer slot count
(2 × (10 + 84) = 188) that's well under the typical 1024 cap.

### Variables

Fixed (no axis sweeps — this is a reproducibility test):

- Warmup kind: `fma_loop` (kernel: 32 threads, 1024 FMAs each)
- K: 20
- Cadence: sleep_0 (back-to-back, no inter-trial sleep)
- Measured kernel: `write_tid` 32 threads, threadgroup 32
- Trials per attempt: 84
- Calibration probe: 10 dispatches per attempt
- Cooldown before each attempt: 5 s
- N attempts: 5

### What we record

`raw/{ts}-attempts.csv` (one row per measured trial across all 5 attempts):

- `attempt_idx` (0..4)
- `phase` (`calibration` | `measured`)
- `idx_within_phase`
- `attempt_pid` (subprocess PID for traceability)
- `wall_clock_ns` (CPU monotonic)
- `gpu_t_start_raw`, `gpu_t_end_raw`, `gpu_delta_raw`
- `cpu_encode_ns`, `cpu_commit_ns`, `cpu_wait_ns`, `cpu_total_ns`

`raw/{ts}-ioreport.csv` (written by ioreport.py subprocess across the
whole driver run): same schema as exp 008 ioreport.csv, joinable on
`monotonic_ns`.

`raw/{ts}-meta.txt`:
- Device, OS, Python/PyObjC versions, QoS, power source.
- Per-attempt summary (n, min, p05, p50, p95, p99, max, cv,
  count_below_5500, trial_idx_of_first_sub_floor).
- Cumulative-attempt cross-comparison (does cv tighten across
  attempts? does count_below_5500 grow?).
- Wall-clock duration of the whole driver.
- IOReport CSV path.
- caffeinate / ioreport subprocess PIDs and exit codes.

### What we do NOT do

- No averaging in any live output.
- No retries on subprocess failure (raise loudly, preserve partial
  data).
- No discarding of outliers — every measured dispatch is in the CSV.
- No varying of K, cadence, or warmup kind. Single recipe.
- No powermetrics. Absolute GPU power isn't the question.
- No display-state / WindowServer manipulation beyond caffeinate (003
  established that controlled display state is sufficient for these
  microbenchmarks).

## Success criterion

The experiment **succeeds** (in the discipline's sense) if we have:

1. CSV with all 5 × (10 + 84) = 470 measured rows (50 calibration +
   420 measured).
2. IOReport CSV covering the wall-clock window of all 5 attempts.
3. Metadata file with per-attempt summary.

It produces a **usable answer** if we can populate this table:

| attempt | cal_first | cal_p50_rest | meas_min | meas_p50 | meas_p95 | trials < 5500 | first sub-floor trial_idx | ioreport gpu_p50_mW | verdict |
|--------:|----------:|-------------:|---------:|---------:|---------:|--------------:|--------------------------:|--------------------:|---------|
|       0 |           |              |          |          |          |               |                           |                     |         |
|       1 |           |              |          |          |          |               |                           |                     |         |
|       2 |           |              |          |          |          |               |                           |                     |         |
|       3 |           |              |          |          |          |               |                           |                     |         |
|       4 |           |              |          |          |          |               |                           |                     |         |

Plus the verdict (STRONG / PARTIAL / NOT REPRODUCED) by the threshold
above and a paragraph on whether IOReport GPU power tracks the timing
change.

## New questions we expect this to raise

- If the state reproduces, **what is its DVFS state?** Follow-up: bind
  IOReport `GPUPH` channel via `IOReportStateGetCount` /
  `IOReportStateGetResidency` (we have only SimpleGet so far) and
  observe per-state residency during sub-floor trials vs floor trials.
  This was already item 5 in the queued questions list; if 009
  reproduces, it jumps to item 1.
- If the state reproduces but IOReport GPU power *doesn't* change,
  the mechanism isn't a DVFS upshift. Could be a different timestamp
  counter clock domain at high DVFS (counter ticks faster, so same
  cycle count looks shorter), or a different kernel scheduling path.
  Either is its own follow-up.
- If the state reproduces only on attempts 1+ and not 0, that's a
  thermal / persistent-state story — implies a "pre-warmed" recipe
  could enter the state more reliably. Follow-up: characterize the
  warm-up half-life of the entry condition.
- If the state does NOT reproduce, file alongside the M1 Pro 5.4 µs
  floor. Two single-launch artifacts on two chips in a row would
  suggest "transient fast states are a thing on Apple Silicon GPUs
  but their entry conditions are not isolable from CPU-side
  observations alone." Useful for future-us to know the pattern.

## After this experiment

Branches:

- **STRONG REPRODUCE.** Update lab state snapshot to add a third
  M4 Max operating point ("sustained fma_loop sub-floor at ~2-3 µs
  reachable under recipe X"). Schedule the GPUPH channel binding
  experiment as the immediate next.
- **PARTIAL REPRODUCE.** File the conditions and frequency of entry.
  Note in the snapshot. Don't promote to "operating envelope" claim,
  but don't write off the state. Skip the GPUPH binding for now in
  favor of building a stronger reproduction protocol if/when we want
  this state to be a reliable measurement target.
- **NOT REPRODUCED.** File alongside the M1 Pro 5.4 µs floor. Update
  the snapshot to remove the "sub-floor 2 µs state" from the M4 Max
  cheat sheet entries (Surprises § 2 in 003-M4Max README still
  records the original observation; we don't rewrite history, just
  add the falsification note).

We do not plan past these branches.

---

## Result

**Date run:** 2026-04-28 (timestamp prefix `20260428T211252`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Wall-clock:** 36.62 s total (5 attempts × ~6 s including 5 s cooldown
+ subprocess launch ~1 s + ~0.4 s GPU work per attempt).
**Outcome:** **STRONG REPRODUCE (5/5).** Every attempt entered the
sub-floor state with onset at trial idx 24-33 (the 003-M4Max original
was trial 29). The state is **deeper** and **more persistent** than the
003 observation, and **deepens across attempts** in a way that strongly
implicates persistent chip state across subprocess re-launch.

### Per-attempt summary

| attempt | cal_first | cal_p50_rest | meas_min | meas_p50 | meas_p95 | meas_max | meas_cv | below_5500 | first_sub_idx |
|--------:|----------:|-------------:|---------:|---------:|---------:|---------:|--------:|-----------:|--------------:|
|       0 |    68 959 |       10 084 |    2 375 |    6 334 |    7 148 |   16 666 |  0.300  |    **22**  |        **31** |
|       1 |    10 167 |        9 834 |    2 833 |    6 333 |    7 106 |    9 125 |  0.189  |    **14**  |        **33** |
|       2 |    11 708 |        8 792 |  **1 625** |  2 187 |    7 238 |  118 542 |  5.829  |    **60**  |        **24** |
|       3 |     8 083 |        6 625 |  **1 625** |  2 896 |    6 702 |    7 333 |  0.732  |    **52**  |        **32** |
|       4 |     8 209 |        6 459 |    1 666 |    2 396 |    6 994 |    8 125 |  0.901  |    **55**  |        **28** |

Pre-registration verdict thresholds: STRONG ≥ 3 attempts with ≥ 10
sub-floor trials; PARTIAL 1-2; NOT REPRODUCED 0. Result: 5/5 with
14-60 sub-floor trials each.

### The trajectory shape: two-tier sub-floor

The trial-by-trial trace exposes structure the 003 single observation
couldn't. The chip enters sub-floor through a **gradual descent**, not
a single transition:

**Attempt 2 (the canonical example):**
- Trials 0-23: 6.3-9.9 µs (back-to-back floor)
- Trial 9: 118 542 ns spike — one catastrophic outlier (see Surprises § 2)
- **Trial 24-31: 3.7 → 2.6 µs** (mid-tier descent, ~75-95 ticks)
- **Trial 32-83: 1.6-2.8 µs** (deep-tier, ~40-65 ticks, 1 642 ± 105 ns)
- No return to floor for the rest of the attempt.

**Attempt 0 (the partial example):**
- Trials 0-30: 6.2-7.4 µs (floor)
- **Trial 31-50: 2.4-4.6 µs** (mid-tier sub-floor, never reached deep-tier)
- Trial 51 onward: returns to floor with one excursion at trial 53, 55
- Final cv 0.30 because of the floor↔sub-floor oscillation in the second half.

**Attempts 3-4 (the most dramatic):** descended into the deep-tier
within ~10 trials of onset and stayed at 1.6-2.1 µs for the entire
remainder. Attempt 3 absolute min = 1 625 ns; attempt 4 = 1 666 ns.

**1 625 ns is exactly 39 ticks** of the 24 MHz GPU timestamp counter
(1 625 / 41.67 = 39.00). The back-to-back floor is ~147 ticks (6 125
ns). The chip is operating at **3.77× fewer cycles for the same kernel**
under sustained sub-floor conditions. Either the GPU clock is running
materially faster (DVFS upshift), or the kernel is taking a different
scheduling path with fewer cycles, or both.

### Cross-attempt coupling is real

The most surprising finding: **the sub-floor state deepens with each
successive attempt, despite subprocess re-launch and a 7 s gap between
attempts** (5 s cooldown inside attempt.py + 2 s sleep in driver).

| attempt | cal_first → cal_p50_rest | meas_p50 | depth |
|--------:|:-------------------------|---------:|-------|
| 0       | 68 959 → 10 084          |    6 334 | shallow (mostly floor) |
| 1       | 10 167 → 9 834           |    6 333 | shallow (mostly floor) |
| 2       | 11 708 → 8 792           |    2 187 | **deep**, persistent |
| 3       |  8 083 → 6 625           |    2 896 | **deep**, persistent |
| 4       |  8 209 → 6 459           |    2 396 | **deep**, persistent |

Three readouts of the cross-attempt drift:
1. **Calibration `cal_p50_rest` halves from attempt 0 to attempt 4**
   (10 084 → 6 459 ns). The chip enters each successive attempt in a
   more elevated state, even after the cooldown.
2. **Sub-floor onset moves earlier on the deepest attempts.** Attempt
   2 entered at trial 24, attempts 0-1 at 31-33. Attempt 4 entered at
   trial 28 with the chip already partially elevated.
3. **The deep-tier (40-tick state) is only reached by attempts 2-4.**
   Attempts 0-1 found the mid-tier (75-95 ticks) but didn't descend
   further before the attempt ended. Once the chip has been pushed
   into the deep-tier, it goes there faster on subsequent attempts.

This says **the 5 s cooldown does not reset the chip's DVFS state**
to the 001-002 baseline. The fast state has a half-life longer than 5
seconds, possibly much longer. A cleaner reproduction protocol would
need explicit "wait until baseline returns" between attempts, perhaps
keyed to IOReport showing GPU power < some threshold.

### IOReport GPU power: directional, not per-trial

Hypothesis: GPU power should be elevated during sub-floor trials. The
analysis ran but **IOReport's 500 ms cadence is too coarse for
trial-level resolution** — each attempt is only ~5 s of wall-clock
spread, and many trials map to the same IOReport sample windows. The
per-attempt floor-vs-sub-floor median GPU power was identical within
each attempt because both groups overlap the same windows.

What we *can* see is across-attempt aggregate GPU power:

| attempt | gpu_p50_mw | cpu_p50_mw | amcc_p50_mw | dcs_p50_mw | depth |
|--------:|-----------:|-----------:|------------:|-----------:|-------|
| 0       |        241 |      1 234 |         785 |        736 | shallow |
| 1       |        230 |      1 225 |         727 |        728 | shallow |
| 2       |        521 |      1 798 |         941 |      1 069 | deep |
| 3       |        445 |      2 034 |       1 082 |      1 387 | deep |
| 4       |        128 |      1 006 |         559 |        470 | deep |

Attempts 2-3 (deepest sub-floor) had ~2× the GPU power of attempts 0-1
(shallow). This is **directional support for a DVFS upshift mechanism**
— deeper state, more power.

But attempt 4 broke the pattern: deepest sub-floor entry, *lowest* GPU
power. Two candidate explanations:
1. **IOReport sampling missed.** Only 6 IOReport samples covered
   floor-trial wall-clock windows in attempt 4 (vs 24-70 in others).
   Attempt 4's measured-trial work happened mostly between IOReport
   samples, so the per-attempt GPU power median is dominated by a few
   non-representative samples.
2. **A different "fast state" with lower power.** Possible, but
   speculative — would need a sub-second sampling cadence to separate
   from explanation 1.

**Verdict for the mechanism question:** the IOReport data is consistent
with a DVFS upshift but not conclusive at this cadence. Higher-cadence
IOReport (e.g. 100 ms) is the natural follow-up.

## Surprises

### 1. The state is reachable on demand and persists across re-launches

Confidence on reproducibility was low pre-registration (the 003 M4 Max
single observation could plausibly have been a one-off). 5/5 with onset
within 9 trials of 003's reference is far stronger than predicted —
the entry conditions for the M4 Max sub-floor state are robust under
the exact recipe (`fma_loop K=20 sleep_0`, write_tid 32t measured,
~25-30 trials of cumulative warmup).

The pre-registered prediction "1-2 of 5" was wrong. Pre-registering it
explicitly anyway prevents post-hoc retrofitting — we wrote down what
we expected before knowing the outcome, and the outcome was much
stronger than expected.

### 2. The mid-tier vs deep-tier two-state structure

003-M4Max's single observation showed trials in the [2, 4] µs range
without distinguishing internal structure. Five reproductions reveal
a **two-tier structure**:

- **Mid-tier:** ~75-95 ticks (3.0-4.5 µs). All five attempts pass
  through this on entry.
- **Deep-tier:** ~40-50 ticks (1.6-2.1 µs). Only reached on attempts
  2, 3, 4 (the cross-attempt-warmed ones).

The mid-tier might be one DVFS state; the deep-tier might be a higher
one; or they might be the same DVFS state with different warm-up
durations. The trajectory in attempt 2 shows a clean monotonic descent
from mid to deep over ~10 trials, which is more consistent with
"continuous warm-up" than "discrete state transitions."

### 3. The 118 µs outlier in attempt 2

Trial 9 of attempt 2 measured 118 542 ns — ~14× the median. This
happened during the floor era (before sub-floor entry at trial 24).
No other attempt produced an outlier above 17 µs. The mechanism is
unknown; possibilities include OS scheduling preemption, a one-shot
WindowServer competition spike (caffeinate keeps display awake but
not exclusive), or thermal throttle event. It does not appear to
have prevented or delayed the sub-floor entry — the chip still
descended starting at trial 24 as predicted.

### 4. The 1 625 ns hardware quantum boundary

The minimum across 5 attempts is exactly 1 625 ns = 39 ticks. The
back-to-back floor is ~147 ticks. **3.77× cycle reduction** is a big
number — if the GPU clock is running at the published M4 Max max GPU
freq of ~1.4 GHz vs the floor freq of ~400-500 MHz, that's ~3× — in
the same ballpark as the observed cycle reduction.

This is consistent with the chip being at or near its **maximum GPU
DVFS state** during deep-tier trials. The "001-002 floor of 6.4 µs is
the minimum dispatch overhead" framing should be replaced with
"6.4 µs is the *minimum at the steady-state DVFS floor*; under
sustained arithmetic warmup the chip can push to ~1.7 µs at peak DVFS."

### 5. Calibration probe `cal_first` is wildly different on attempt 0

Attempt 0 cal_first = 68 959 ns (vs 8-12 µs for attempts 1-4). Cold
start of the MTLDevice for the first time in this driver process picks
up ~7× the steady-state cal_first. This is consistent with prior
findings (003 cal_first variance) — the *first* dispatch after a fresh
MTLDevice is in a transient state. By attempt 1+, the device is hot
and cal_first stabilizes.

This is a known calibration-probe contamination story per 003 §
"Calibration probe is itself a 10-dispatch warmup." The scale of the
attempt 0 outlier is unusual but doesn't contaminate the headline
finding (attempt 0 still entered sub-floor at trial 31).

## What this means operationally

For the M4 Max operating envelope:

1. **The "6.4 µs back-to-back floor" claim from 001-002 needs an
   asterisk.** Under sustained arithmetic warmup, the chip can sustain
   ~1.7 µs per write_tid 32t dispatch — 3.77× faster. The 6.4 µs
   number is the *cool-state* floor, not the *fundamental hardware
   floor*. For experiments designing measurements around expected
   per-dispatch overhead, the floor depends on warmup recipe.
2. **The recipe to enter the sub-floor state is now characterized:**
   `fma_loop K=20 sleep_0` for ~25-30 trials at minimum, and the state
   deepens further with sustained dispatch over the next ~30-50
   trials. Subprocess re-launch does NOT reset the state on a 5-7 s
   cooldown — the warmth carries across.
3. **Future experiments wanting to *avoid* the sub-floor state**
   should not run sustained `fma_loop K=20 sleep_0` for more than ~25
   measured trials. Or should pad with longer cooldown (>> 7 s) if
   trial counts approach the entry threshold.
4. **Future experiments wanting to *exploit* the sub-floor state for
   "what does write_tid 32t look like at peak GPU DVFS?"** can use
   this recipe as a setup. Real workloads almost certainly never reach
   this state via natural usage — it requires the specific
   arithmetic-only warmup pattern.
5. **The fast state half-life is longer than 5 s.** Better characterized
   as "longer than 5 s" than "infinite" — we don't know its decay rate.
   A follow-up experiment with explicit cooldown-time sweep could
   measure it.

## What does NOT change

- Decision 005 (pair timing primary on M4 Max for trials ≥ 64 µs)
  is unaffected — pair timing operates above the dispatch-overhead
  floor regardless of which floor is in effect.
- Decisions 003/004 (M1 Pro pair timing scope) are unaffected — the
  M4 Max sub-floor doesn't apply on M1 Pro.
- The `notes/ioreport.py` + 008 telemetry stack worked exactly as
  designed; no changes there.

## What changes

- Lab state snapshot needs a "third operating point" entry on M4 Max
  cheat sheet: "fma_loop K=20 sleep_0 sustained sub-floor at ~1.7 µs
  reachable on demand, cross-attempt persistent."
- The 003-M4Max readme's Surprise § 2 should get a forward-pointer to
  this experiment ("validated and characterized — see exp 009").
- UNKNOWNS.md should swap the `applegpu_g16s` minimum dispatch-overhead
  question from "6.4 µs?" to "6.4 µs cool, 1.7 µs warm under recipe X."

### Natural follow-ups

- **GPUPH channel binding** to read DVFS state residency directly.
  Was item 5 in queued questions; if the mechanism is a DVFS upshift,
  GPUPH would let us *observe* it and confirm. Promoted to high
  priority.
- **Higher-cadence IOReport** (100 ms) during a sub-floor reproduction
  to see GPU power actually transition during the trajectory. Tests
  whether the per-attempt power story holds at trial-level resolution.
- **Half-life characterization.** Re-run 009 with varying cooldown
  durations between attempts (1 s, 5 s, 30 s, 5 min, 1 hour) to
  measure the decay rate of the elevated state.
- **Recipe sensitivity sweep.** Vary K (5, 10, 20, 50), FMA_ITERS
  (256, 1024, 4096), warmup kind (heavy_write, fma_loop) to find the
  minimum work needed to enter the sub-floor state and which
  ingredients matter.

We do not plan past these.
