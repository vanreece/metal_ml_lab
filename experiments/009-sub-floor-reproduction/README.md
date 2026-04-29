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
