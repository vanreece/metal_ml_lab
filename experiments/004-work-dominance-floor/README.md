# 004: Work-dominance floor (where does timing measure work, not overhead?) + 5.4 µs floor reproducibility

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-27
**Hardware target:** M1 Pro (16GB), macOS 26.3.1
**Estimated runtime:** ~12-18 min unattended

## The question

Three related sub-questions, all about a noise floor distinct from
the two we have already characterized:

**Primary 1 — `write_tid` thread-count axis.** Sweep `write_tid`
thread count from 32 to ~8M (`THREADS_PER_GRID` powers and
intermediate halves). At what thread count does measured `gpu_delta`
start scaling linearly with thread count (work dominates), versus
plateau at the per-dispatch overhead floor (overhead dominates)?

**Primary 2 — `fma_loop` per-thread-work axis.** With thread count
fixed at 32, sweep FMA iterations per thread from 16 to ~64K. At what
FMA-iter count does `gpu_delta` start scaling linearly with iteration
count?

**Bonus / secondary — 5.4 µs floor reproducibility.** Experiment 003
accidentally hit a settled-state floor at p50=5375 ns with cv=0.05
on its very first combo (`write_tid` 32-thread, K=0 warmup, sleep_0)
— 33% faster than the 8083 ns floor that 001 and 002 always
converged to. It was never reproduced anywhere else in 003. Before
running the main 004 sweep, run a tight protocol to test whether
that floor is reachable on demand: 5 sequential attempts, each
attempt a fresh post-cooldown calibration burst followed by 40
back-to-back `write_tid` 32-thread measurements with no warmup.
Save these probes separately.

The third noise floor we are characterizing here ("work-dominance
floor") is qualitatively distinct from the two prior ones:

- **001's noise floor:** timing quantum (~42 ns / tick on 24 MHz GPU
  timestamp counter)
- **002's noise floor:** cv depends on inter-dispatch idle, recoverable
  with K=1 warmup at long cadences (per 003)
- **004's noise floor (this experiment):** below some kernel
  complexity, measured `gpu_delta` reflects dispatch/launch overhead,
  not work. Above some complexity, work dominates. The crossover
  position determines the minimum kernel complexity for which timing
  carries usable signal about the kernel itself.

This matters because **001-003 measured the smallest possible kernel
the whole time**, and the ~8 µs floor we kept hitting is presumably
mostly `T_overhead`. So our cv numbers have been measuring overhead
variance, not work variance — those are different things, and any
later experiment about a real kernel needs to know which side of the
crossover that kernel sits on.

## Hypothesis

Confidence: medium-low. Stating predictions to prevent post-hoc
retrofitting:

- **Thread-count crossover at ~1024-4096 threads.** Below 32-thread
  steady state at ~8 µs (mostly overhead). Linear scaling at very high
  thread counts. Crossover where doubling threads doubles measured
  time, predicted somewhere in 1K-4K thread range.
- **FMA-iter crossover at ~512-2048 iters.** Below: ~9-10 µs steady
  state. Linear scaling at very high iters. Crossover at the iter
  count where compute time matches dispatch overhead, predicted
  somewhere in 500-2000 range.
- **cv shape across complexity:** at the small-kernel end, cv is
  dominated by the ~5-15 µs envelope of overhead variance (the
  numbers from 002/003). As work dominates, cv should drop because
  the "denominator" (median) grows while the absolute envelope of
  overhead variance is approximately fixed. Prediction: cv ≈ 0.1-0.2
  at small end, dropping to ≈ 0.01-0.05 at large end.
- **5.4 µs floor reproducibility:** uncertain. It could be a
  single-launch artifact (would fail to reproduce in any of the 5
  attempts) or it could be reachable by the cold-start + cooldown +
  calibration pattern (would reproduce in most attempts). Either
  outcome is informative.
- **Same hardware quantum (~42 ns) should be visible across all
  complexity levels.** No expectation that work-dominated regimes
  hide it.

## What we are NOT trying to answer

- **Whether the warmup recipe still works for non-trivial kernels.**
  We use the K=1 warmup recipe established by 003 (one warmup of the
  same kernel kind right before each measurement). But we do not
  re-test K=0 vs K=1 vs K=5 here — that was 003. Future experiments
  may revisit per-kernel-shape warmup if 004 shows weirdness.
- **Multi-kernel state profiling.** That's planned for 005, after we
  know what complexity range to operate at. Here we use one kernel
  kind per axis.
- **Threadgroup size effects.** Threadgroup size is fixed at 32
  (matches Apple Silicon SIMD width). Varying it is a future axis.
- **Memory access pattern effects.** `write_tid` is purely sequential
  contiguous writes. Other access patterns are future.
- **Anything about M4 Max.** Hardware not yet available.
- **Display state / WindowServer interaction.** Same display-state
  controls as 003 (caffeinate -d -i -m sidecar). Whether display
  state itself matters is a separate small experiment, not folded in.

## Setup

### Hardware / software

- M1 Pro 16GB, macOS 26.3.1, AC power, laptop awake.
- User-interactive QoS via `pthread_set_qos_class_self_np`.
- uv + PEP 723 inline metadata, same pattern as 001-003.
- `caffeinate -d -i -m` sidecar holds display awake throughout.
- `EXP004_NO_POWERMETRICS=1` env var skips the powermetrics sidecar
  by default. If the operator wants powermetrics in the run, run
  `sudo -v` first and unset the env var.

### Variables

- **Axis A: `write_tid` thread count.** 22 levels:
  `[32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
   4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
   1048576, 8388608]`. Densest at the predicted-inflection zone
  (32-1024). Threadgroup size fixed at 32. Largest is 8M threads ≈
  256K threadgroups.
- **Axis B: `fma_loop` per-thread iters.** 17 levels:
  `[16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096,
   8192, 16384, 32768, 65536]`. Thread count fixed at 32. Densest
  at the predicted-inflection zone (128-1024). Largest is 64K
  iterations per thread, roughly 2M FP-ops per thread.

`write_tid` and `fma_loop` are the same kernels used in 003, with the
difference that here `fma_loop` is parameterized by `FMA_ITERS` at
compile time (we generate one `MTLLibrary` per FMA-iters level). See
"Implementation notes" below.

### Per-combination protocol

For each (axis, complexity_level) — 39 unique combinations total:

1. Cooldown: sleep 2s before the combination.
2. Calibration probe: 10 sleep_0 dispatches of `write_tid` 32-thread
   with timestamp sampling. Captured separately. (Caveat from 003:
   the calibration probe is itself a 10-dispatch warmup; treat it as
   part of the protocol, not as a "cold" reference.)
3. Measured trials: N=300 trials of (1 untimed warmup of the
   combination's measured kernel/complexity, then 1 measured dispatch
   with timestamp sampling). No inter-trial sleep within a
   combination. Slot indices in the shared sample buffer rotate
   through the available slots.

### Sweep repetitions

The full 39-combination sweep runs **3 times in sequence** within a
single script invocation, with 30s cooldown between sweeps. Each
sweep produces its own row-tagged data. This characterizes
within-script between-sweep variance — i.e., does running the same
sweep twice produce the same answer?

### 5.4 µs reproducibility protocol

Run BEFORE any other GPU work in the script. 5 sequential attempts.
Each attempt:

1. Cooldown: sleep 5s (longer than per-combination cooldown).
2. Calibration probe: 10 sleep_0 dispatches of `write_tid` 32-thread
   with timestamps. Captured.
3. Measured: 40 sleep_0 dispatches of `write_tid` 32-thread with
   timestamps and **NO** warmup. (Replicates the 003 first-combo
   conditions exactly.)

Saved separately as `repro/<ts>-attempt_N-{calibration,measured}.csv`
(or as one combined CSV with attempt_idx column — implementation
choice).

### What we record

Per measured dispatch (one big CSV per axis, two CSVs total):

- `axis` (`write_tid_threadcount` | `fma_loop_iters`)
- `complexity_level` (the value being swept)
- `sweep_idx` (0..2 — which of the 3 sweep repetitions)
- `trial_idx_within_combo` (0..299)
- `wall_clock_ns` (CPU monotonic at start of trial)
- `gpu_t_start_raw`, `gpu_t_end_raw`, `gpu_delta_raw`
- `cpu_encode_ns`, `cpu_commit_ns`, `cpu_wait_ns`, `cpu_total_ns`

Per calibration probe (one CSV total):

- `axis`, `complexity_level`, `sweep_idx`, `probe_idx_within_burst`
- same dispatch fields

Per 5.4 µs reproduction attempt (one CSV total):

- `attempt_idx` (0..4), `phase` (`calibration` | `measured`),
  `idx_within_phase`
- same dispatch fields

Per run (metadata file):

- Device name, registry ID, architecture string
- OS, Python, PyObjC versions
- QoS, power source, display powerstate, pmset assertions
- Whether powermetrics ran (and how it was started/stopped)
- Whether caffeinate ran
- Wall-clock duration
- Timestamp correlation snapshots at start and end
- Per-combination per-sweep summary (median, p05, p50, p95, p99,
  max, cv, in_floor) for both measured trials and calibration probes
- Per 5.4 µs reproduction attempt summary

### What we do NOT do

- No averaging in any live output.
- No exception swallowing or retries.
- No discarding of outliers.
- No varying of warmup count K (fixed at K=1, the recipe established
  by 003).
- No varying of threadgroup size (fixed at 32).
- No varying of cadence between measurements (fixed at sleep_0 within
  a combination, since cadence question was answered in 002+003).

## Success criterion

The experiment succeeds (in the discipline's sense) if we have:

1. Measured CSV for axis A (`write_tid` thread count) covering all
   22 levels × 3 sweeps × 300 trials = 19800 rows.
2. Measured CSV for axis B (`fma_loop` iters) covering all 17 levels
   × 3 sweeps × 300 trials = 15300 rows.
3. Calibration CSV with one 10-probe burst per (axis, complexity,
   sweep) = 39 × 3 × 10 = 1170 rows.
4. 5.4 µs reproduction CSV with 5 attempts × (10 calibration + 40
   measured) = 250 rows.
5. Metadata file with per-combo summaries.
6. Stdout log preserved alongside the CSVs.

It produces a usable answer if we can fill in:

**Question 1 — work-dominance floor for write_tid thread count:**
At what thread count `T*` does `median(gpu_delta) - 8083` (the
overhead floor) start being well-approximated by a linear function of
thread count? Equivalently: at what thread count does doubling
threads produce a (very nearly) doubling of `gpu_delta`?

**Question 2 — work-dominance floor for fma_loop iters:**
Same question for `fma_loop` per-thread iters.

**Question 3 — 5.4 µs reproducibility:**
Of the 5 attempts, how many produce a 40-trial median in the
[5000, 6000] ns range (the 003 signature)? 0-1 = "non-reproducible
artifact"; 4-5 = "reproducible state, important to characterize";
2-3 = "intermittently reachable, mechanism unclear, follow-up
needed".

## New questions we expect this to raise

- If `T*` for write_tid is at, say, 4096 threads — does that recipe
  generalize to other memory-write kernels (different access
  patterns)? This becomes a future axis.
- If `T*` is much larger or smaller than predicted — what's special
  about the inflection point? Is it related to threadgroup count
  filling some hardware queue?
- If FMA-iter crossover is sharply different from thread-count
  crossover, that says something about how the GPU pipelines compute
  vs. memory work. Would inform the 005 multi-kernel state profile
  design.
- If between-sweep variance is large (sweep_2 doesn't match sweep_0),
  the chip is in different states at different sweep starts, and
  even our "2s cooldown" is not enough to control state.
- If between-sweep variance is small but the 5.4 µs reproductions
  vary wildly, the reproducibility question has a complicated answer
  ("the state can be entered but only sometimes").
- The cv at the high end of each axis should be very low (work
  dominates, overhead variance amortized). If it's not, something
  about the long-duration regime introduces its own variance — could
  be context switching / preemption, GPU memory bandwidth contention
  with the compositor, or something stranger.
- Does the GPU timestamp counter behave the same way for very long
  dispatches? Specifically: does it stay monotonic, does the quantum
  remain ~42 ns, are there any wraparound concerns at ~ms-scale
  durations?

## Implementation notes (for the operator)

### How to run

```bash
cd experiments/004-work-dominance-floor
EXP004_NO_POWERMETRICS=1 uv run --quiet run.py 2>&1 | tee /tmp/exp004-run.log
```

Total runtime estimated 12-18 minutes. Run it unattended; the script
prints sparse progress. Do not interact with the laptop while it
runs (no other GPU-heavy apps, no display sleep — caffeinate
sidecar handles the second).

### How to enable powermetrics

If you want the powermetrics sidecar to run alongside (for later
correlation against the calibration / multi-kernel timings):

```bash
sudo -v   # in another terminal, then promptly run:
unset EXP004_NO_POWERMETRICS
uv run --quiet run.py 2>&1 | tee /tmp/exp004-run.log
```

powermetrics samples at 200 ms by default in this script; output
goes to `raw/<ts>-powermetrics.txt`.

### Compilation note for fma_loop

The `fma_loop` kernel is parameterized by `FMA_ITERS` via Metal
shader source string substitution. The script generates one
`MTLLibrary` per FMA-iters level (17 libraries total). Each library
contains one `fma_loop` function compiled with the level baked in as
a constant. This avoids the cost of a kernel uniform read and keeps
the generated code at each level honestly representative of "this
many FMAs per thread."

### Sample buffer sizing

Same shared-sample-buffer pattern as 002/003. One buffer per script
invocation, sized to hold the largest single combination's
calibration + measured trials = 2 × (10 + 300) = 620 slots. We
re-use slots across combinations (writing over previous data; no
need for retention).

### Order of operations

1. QoS, caffeinate, environment probes.
2. (Optional) powermetrics sidecar start.
3. Device and pipelines.
4. **5.4 µs reproduction protocol** (5 attempts).
5. Sweep 0 of axis A (`write_tid`), then sweep 0 of axis B
   (`fma_loop`), then 30s cooldown.
6. Sweep 1 of A, sweep 1 of B, 30s cooldown.
7. Sweep 2 of A, sweep 2 of B.
8. Write CSVs and metadata.
9. Stop sidecars.

The 5.4 µs protocol runs first because we want it to land on the
freshest possible chip state — no GPU work has happened yet in this
script invocation.

## After this experiment

Branches:

- **Both work-dominance floors found cleanly + 5.4 µs is reproducible.**
  Move to 005 (multi-kernel state profiling) using the work-dominated
  region as the operating range.
- **Floors found cleanly + 5.4 µs is NOT reproducible.** Same as
  above for 005. The 5.4 µs floor is filed as "occasionally
  reachable, no known recipe" until a future experiment finds the
  trigger.
- **Floors are noisy or non-monotonic.** Something about how `T_work`
  scales with kernel parameters is more complex than additive.
  Investigate before 005 — maybe a small experiment on threadgroup
  size, maybe on memory-access pattern.
- **Between-sweep variance is large.** State control is worse than we
  thought. May need to rethink cooldown durations or measurement
  protocol before 005.

We do not plan past these branches.

## Notes for the post-run writeup

When this experiment is run and written up, the README should be
updated with sections matching the convention from 001-003:

- `## Result` (with summary tables for each axis)
- `## Surprises` (numbered, specific, with values)
- `## What this means operationally`
- `## New questions`
- `## After this experiment` updated to reflect actual findings

`UNKNOWNS.md` should be updated with the new "work-dominance floor"
answer. `notes/answered-questions.md` should gain entries for the
two work-dominance floor questions and (depending on outcome) for
the 5.4 µs reproducibility question.

If the 5.4 µs floor IS reproducible, that is a substantive finding
that retroactively re-characterizes the 8 µs floor from 001/002 and
the calibration-probe-end floor from 003. Update those experiment
READMEs with cross-references.
