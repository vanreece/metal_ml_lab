# 004: Work-dominance floor (where does timing measure work, not overhead?) + 5.4 µs floor reproducibility

**Status:** complete
**Date pre-registered:** 2026-04-27
**Date run:** 2026-04-27 (`raw/20260427T152008-*`)
**Hardware:** Apple M1 Pro 16GB, macOS 26.3.1, AC power, `applegpu_g13s`
**Actual runtime:** 359 s (much shorter than the 12-18 min estimate
because the per-combo cooldown was the dominant cost and per-trial
work was small; only the very longest fma_loop dispatches added
non-trivial GPU time).

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

## Result

**Both work-dominance floors exist and are clean enough to operate on,
but they're in *different places* than predicted.** The 5.4 µs floor
from 003 is **not reproducible** by the cold-start + cooldown +
calibration recipe — 0/200 measured trials across 5 attempts landed
in [5000, 6000] ns; every attempt converged to the familiar ~8 µs
back-to-back floor. The 003 first-combo observation is filed as a
single-launch artifact.

Pooled across 3 sweeps × 300 trials = N=900 per level. Robust cv =
IQR / (1.349 × p50). Naive cv (sd/p50) is reported alongside because
its divergence from robust cv is itself signal — it tracks rare
outliers (probably preemption or page faults) that don't move the
median.

### Axis A — `write_tid` thread count (22 levels, threadgroup=32)

|     threads |    p50 (ns) | level_ratio | p50_ratio | p50_ratio / level_ratio | iqr_cv | naive_cv |
|-----------:|------------:|------------:|----------:|------------------------:|-------:|---------:|
|         32 |       9 084 |       —     |     —     |          —              |  0.078 |    2.083 |
|         64 |       9 292 |    2.00×    |   1.023×  |        0.51             |  0.080 |    4.227 |
|        128 |       8 708 |    2.00×    |   0.937×  |        0.47             |  0.111 |    0.095 |
|        256 |       9 084 |    2.00×    |   1.043×  |        0.52             |  0.082 |    9.317 |
|        512 |       9 666 |    2.00×    |   1.064×  |        0.53             |  0.112 |    0.248 |
|       1024 |       9 042 |    2.00×    |   0.935×  |        0.47             |  0.126 |    2.839 |
|       2048 |       9 750 |    2.00×    |   1.078×  |        0.54             |  0.099 |    2.556 |
|       4096 |       9 917 |    2.00×    |   1.017×  |        0.51             |  0.100 |    0.108 |
|       8192 |      10 583 |    2.00×    |   1.067×  |        0.53             |  0.073 |    1.857 |
|      16384 |      11 250 |    2.00×    |   1.063×  |        0.53             |  0.082 |    8.095 |
|      32768 |      13 584 |    2.00×    |   1.207×  |        0.60             |  0.036 |    0.052 |
|      65536 |      17 042 |    2.00×    |   1.255×  |        0.63             |  0.034 |    1.552 |
|     131072 |      22 750 |    2.00×    |   1.335×  |        0.67             |  0.053 |    0.768 |
|     262144 |      40 417 |    2.00×    |   1.777×  |    **0.89**             |  0.021 |    0.186 |
|     524288 |      62 291 |    2.00×    |   1.541×  |        0.77             |  0.030 |    0.246 |
|    1048576 |     102 750 |    2.00×    |   1.650×  |        0.83             |  0.038 |    0.061 |
|    8388608 |     393 584 |    8.00×    |   3.831×  |    **0.48**             |  0.006 |    0.136 |

`p50_ratio / level_ratio` is the work-dominance test: 1.0 means doubling
threads doubled measured time (work fully dominates dispatch overhead);
near 0.0 means the median didn't move (overhead-dominated). The signal
**stays around 0.5 from 32 threads up through 16K threads**, then
climbs gradually through 32K-131K, and reaches ~0.9 at 262K threads
before sliding back toward ~0.5 at 8M.

**Crossover for `write_tid`: T\* ≈ 131 072 - 262 144 threads.** Below
~32K threads, the dispatch is overhead-dominated and timing tells you
about overhead variance, not the kernel. The overhead floor itself
sits at ~9-11 µs (slightly higher than 001's 8 µs because of the K=1
warmup recipe). Above 1M threads, scaling goes sublinear again
(8M takes only 3.83× the time of 1M, not 8×) — bandwidth saturation
(8M × 4 B = 32 MiB writes) is now the bottleneck.

### Axis B — `fma_loop` per-thread iters (17 levels, 32 threads)

|  fma_iters |    p50 (ns) | level_ratio | p50_ratio | p50_ratio / level_ratio | iqr_cv | naive_cv |
|-----------:|------------:|------------:|----------:|------------------------:|-------:|---------:|
|         16 |       9 083 |     —       |    —      |        —                |  0.082 |    2.503 |
|         32 |       9 583 |    2.00×    |   1.055×  |        0.53             |  0.077 |    0.088 |
|         64 |       9 542 |    2.00×    |   0.996×  |        0.50             |  0.094 |    0.092 |
|         96 |      10 834 |    1.50×    |   1.135×  |        0.76             |  0.074 |    2.597 |
|        128 |      11 500 |    1.33×    |   1.061×  |        0.80             |  0.081 |    0.154 |
|        192 |      12 625 |    1.50×    |   1.098×  |        0.73             |  0.073 |    1.514 |
|    **256** |  **34 167** |    1.33×    |  **2.706×**| **2.03 ⚠**             |  0.021 |    0.027 |
|        384 |      46 250 |    1.50×    |   1.354×  |        0.90             |  0.013 |    0.450 |
|        512 |      58 500 |    1.33×    |   1.265×  |        0.95             |  0.014 |    0.169 |
|        768 |      82 750 |    1.50×    |   1.415×  |        0.94             |  0.012 |    0.356 |
|       1024 |     107 375 |    1.33×    |   1.298×  |        0.97             |  0.003 |    0.199 |
|       2048 |     204 541 |    2.00×    |   1.905×  |        0.95             |  0.002 |    0.159 |
|       4096 |     398 916 |    2.00×    |   1.950×  |        0.98             |  0.147 |    0.252 |
|       8192 |     788 083 |    2.00×    |   1.976×  |        0.99             |  0.516 |    0.312 |
|      16384 |     472 083 |    2.00×    |   0.599×  |    **0.30 ⚠**           |  1.722 |    1.050 |
|      32768 |     939 667 |    2.00×    |   1.990×  |        0.99             |  0.002 |    0.407 |
|      65536 |   1 875 209 |    2.00×    |   1.996×  |        1.00             |  0.379 |    0.446 |

**Crossover for `fma_loop`: sharp step at iters = 192 → 256.** Below
192 iters: p50 climbs slowly from 9 → 12.6 µs (overhead-dominated,
work contributing in the noise). Between 192 and 256 iters: a discrete
**21 µs jump** — `p50_ratio = 2.71×` for a `level_ratio` of just
1.33×. From 256 onward: clean linear scaling (`p50_ratio /
level_ratio` ∈ [0.90, 1.00] up through 65K iters). The inflection
is not the smooth crossover predicted; it's a step.

### Bimodality / between-sweep variance at fma_iters 4096-16384

For most levels, sweep p50s match within ~3%. Strong divergence in
one range:

| fma_iters | sweep 0 p50 | sweep 1 p50 | sweep 2 p50 | spread |
|----------:|------------:|------------:|------------:|-------:|
|      4096 |    398 875 |    399 208 |    398 833 |  ≈0%   |
|  **8192** |  **316 666** | **789 125** | **396 166** | **2.5×** |
| **16384** |  **471 938** | **628 250** | **472 000** | **1.33×** |
|     32768 |    939 709 |    939 584 |    939 667 |  ≈0%   |

At fma_iters = 8192-16384, the chip is in *different operating modes*
on different sweep starts despite a 30-second cooldown between sweeps.
The pattern is roughly bimodal — measurements cluster near either
~470K or ~789K ns (close to 1× and 2× the iters=4096 baseline). Looks
like a quantized state effect rather than continuous variance. Outside
this band, between-sweep p50s match.

### 5.4 µs reproducibility — 0/5

| attempt | cal_first |  cal_p50_rest | meas_min | meas_p50 | meas_p95 |  iqr_cv | in_low [5000,6000] | in_8µs [8000,8200] |
|--------:|----------:|--------------:|---------:|---------:|---------:|--------:|-------------------:|-------------------:|
|       0 |    12 250 |         8 250 |    8 041 |    8 146 |    8 546 |  0.019  |      **0/40**      |          27/40     |
|       1 |    12 500 |         8 333 |    8 208 |    8 333 |   22 133 |  0.011  |      **0/40**      |           0/40     |
|       2 |    15 709 |         8 209 |    7 875 |    8 084 |    8 169 |  0.008  |      **0/40**      |          37/40     |
|       3 |    11 834 |         8 041 |    7 500 |    8 166 |   10 583 |  0.016  |      **0/40**      |          21/40     |
|       4 |    15 333 |         9 417 |    8 041 |    8 334 |    9 428 |  0.024  |      **0/40**      |           1/40     |

**0 of 200 measurements landed in the 003 first-combo signature
window.** Every attempt converged to the 8 µs back-to-back floor that
001/002/003 always observed elsewhere. Per the success-criterion
threshold ("0-1 = non-reproducible artifact"), the 5.4 µs floor is a
single-launch artifact — possibly tied to first-ever GPU activity in a
session, or to the ~17-second-old `caffeinate` activation in the 003
log, or to some other one-shot trigger we cannot isolate without
re-creating the exact macOS process tree at 003 launch. We are not
going to chase it further; it's now filed.

## Surprises

### 1. write_tid work-dominance crossover is ~64× later than predicted

Hypothesis predicted T\* in 1024-4096 threads. Actual: T\* ≈ 131K-262K
threads. From 32 to ~16K threads, p50 stays in the 9-11 µs band
(overhead-dominated). That means **all of 001, 002, and 003 measured
write_tid in the overhead-dominated regime** — even at the most
generous reading (16K threads = 512 threadgroups), we never put enough
work on the GPU to make the work fall out of the noise. Cv numbers
from those experiments characterize per-dispatch overhead variance,
not write_tid kernel variance. This retroactively re-frames every
"noise floor" claim from earlier experiments as a claim about
*dispatch overhead noise floor*, not *kernel timing noise floor*.

### 2. fma_loop work-dominance is *earlier* than predicted, but it's a step, not a slope

Hypothesis predicted a smooth crossover at 512-2048 iters. Actual:
flat 9-13 µs from 16-192 iters, then a **+21 µs step** between 192 and
256 iters, then clean linear scaling from 256 onward.

The step is far too big to be 64 extra `fma` operations doing real
work. 64 FMAs × 32 threads / 24 MHz tick ≈ 0.085 µs per thread of
useful work — three orders of magnitude smaller than the observed 21
µs jump. This is much more likely a compiler / hardware threshold:
register spilling kicking in, an unrolling heuristic flipping, an
instruction-cache crossover, or a switch from one issue strategy to
another. We have not isolated which.

Practical consequence: **between 192 and 256 fma iters there is a
non-linear region** where doubling iters does not give 2× time, and
neither does 1.33×. Operating points must be either solidly below
the step or solidly above.

### 3. 5.4 µs is not a reachable steady state — it was a one-time launch artifact

The pre-registered prediction said 0-1 reproductions = non-reproducible
artifact. We got 0/5. Confidence on this is high because the 200
samples are tightly distributed in [7500, 8550] for all attempts (with
a few tail excursions to 22-75 µs in attempts 1 and 4, but no
sub-7000 ns samples). The chip is not staying near 5.4 µs in any
attempt — it's nowhere near it.

This rules out hypothesis "cold-start + caffeinate + cooldown +
calibration burst is the recipe" and pushes us toward "the 003 first
combo hit a transient state that depends on something we did not
log" (e.g., very-recent Python/MLX/system-level GPU activity from
prior shell sessions). Without a reproducer, it is not actionable.
The headline operational floor for `write_tid` 32t is back to ~8 µs.

### 4. write_tid scaling goes sublinear above 1M threads (bandwidth ceiling)

8M threads takes 3.83× the time of 1M, not 8×. The 8M dispatch writes
32 MiB; at 393 µs that's ~85 GB/s. M1 Pro's published peak DRAM
bandwidth is ~200 GB/s but realistic STREAM-style write-only is in
the 80-120 GB/s range, so this looks like a real bandwidth saturation
rather than an instrumentation artifact. We are not currently
verifying it (a STREAM-style bench is in `UNKNOWNS.md` but not
scheduled), just noting.

### 5. Bimodal operating modes at fma_iters = 8192-16384

Outside this range, between-sweep variance is small. Inside this
range, sweep p50s land near either "the expected linear value" or
"~½ the expected value" — looks like the chip sometimes runs
this dispatch in two passes and sometimes in one, or sometimes
hits a faster cache state. The 30-second between-sweep cooldown
is not enough to make this state choice deterministic. The same
range corresponds to dispatch durations in 0.4-1.6 ms, i.e.,
inside the "1-10 ms transition zone of the GPU power state machine"
that 002 originally flagged as the worst-cv region. Plausibly
related, but we have not measured it directly.

### 6. The 8 µs floor is the *overhead* floor, not a kernel floor

This was implicit in the experiment design but worth saying out loud.
Across both axes, the small-kernel p50 sits in 9-13 µs regardless of
what the kernel does (write_tid 32-1024 threads ≈ 9-9.7 µs;
fma_loop 16-192 iters ≈ 9-13 µs). The overhead has at least two
floors visible: the back-to-back floor at ~8 µs (with K=0 / no
warmup) and the K=1-warmup floor at ~9 µs. Neither one tells us
anything about the kernel — they are dominated by encoder setup,
command buffer commit, GPU front-end dispatch, and waitUntilCompleted
latency.

## What this means operationally

For all subsequent experiments measuring kernel performance on M1 Pro
via `MTLCounterSampleBuffer`:

1. **`write_tid` 32-thread is not a kernel measurement — it's a
   dispatch-overhead measurement.** Using it to characterize "GPU
   state" is fine (the overhead floor itself reflects GPU power
   state), but calling its time "the cost of writing 32 uints" is
   wrong. The cost of writing 32 uints is buried under ~9 µs of
   overhead.
2. **Minimum operating point for a kernel-dominated `write_tid`-class
   benchmark on M1 Pro: ≥ 262K threads** (and prefer 524K-1M, where
   `p50_ratio / level_ratio` is most consistent and bandwidth has not
   yet saturated).
3. **Minimum operating point for an `fma_loop`-class compute benchmark:
   ≥ 256 iters per thread.** The 192→256 step artifact means *avoid*
   the 96-256 iter range — measurements there are non-monotonic with
   work.
4. **Avoid the fma_iters 4096-16384 zone for between-run-comparable
   measurements.** Variance across sweeps inside a single script
   invocation is up to 2.5× — comparing across script invocations
   would be even worse.
5. **The 8 µs floor is a dispatch-overhead constant.** Any future
   "kernel time" report should subtract or footnote this overhead
   rather than reporting raw `gpu_delta`. Equivalently: report
   `gpu_delta - 8083` as "approximate kernel time" only above the
   work-dominance threshold; below it, the subtraction is meaningless
   because the residual is dominated by overhead variance.
6. **The 5.4 µs floor from 003 is filed.** Until it reproduces, the
   working-floor for downstream calibration probes is 8 µs, and no
   experiment gets to claim sub-8-µs as a steady state without
   reproducing it first.

## New questions

- **What is the +21 µs step at fma_iters = 192→256?** Compiler
  threshold, hardware register-pressure threshold, instruction cache
  effect? Could be answered cheaply by inspecting Metal AIR / GPU
  assembly at iters=192 vs iters=256 (`xcrun metal -S` on the
  source). This is a small experiment worth doing before 005 because
  if the step is compiler-driven, it tells us our "compute kernel
  complexity" axis isn't smooth and we can't rely on continuous
  scaling.
- **Why is fma_loop scaling bimodal at 4096-16384 iters?** Specific
  thermal / DVFS state quantization at this duration band? Worth
  characterizing as a sub-experiment before relying on this duration
  range for any downstream measurement.
- **Does the `write_tid` thread-count crossover at ~131K-262K
  generalize across access patterns?** Strided writes, scatter/gather,
  read-modify-write — all open. This becomes a future axis if 005's
  multi-kernel state profile uses them.
- **Where exactly does the bandwidth ceiling start for write_tid?**
  Sublinear scaling at 8M threads suggests we're hitting it; a
  finer sweep between 1M and 8M would pin down the knee. STREAM-style
  bandwidth bench is in `UNKNOWNS.md` and would answer this directly.
- **Do `caffeinate` activation timing or display power state actually
  influence the 5.4 µs floor reproducibility?** The current attempt
  did not attempt to recreate the 003 first-combo's exact process
  tree (which had a brand-new caffeinate process and a powermetrics
  sidecar starting). A future attempt could recreate that more
  exactly. Low priority — likely chasing noise.
- **Is the 8 µs overhead floor a single value or a stack?** With K=0
  back-to-back at sleep_0 we saw 8083 ns in 001/002. With K=1 warmup
  we see 9-9.7 µs. With longer cadence we see 14-17 µs (cool path).
  Are these three discrete floors or a continuum? Affects whether
  "overhead-subtraction" before reporting kernel time is even
  well-defined.

## After this experiment

Branch we landed on: **floors found cleanly + 5.4 µs is NOT
reproducible + step discontinuity at fma 192→256 is a new wrinkle.**

Two viable next moves, in priority order:

1. **Tiny experiment 005a:** characterize the 192→256 fma step.
   Compile the Metal source at both iters values, dump AIR / GPU
   assembly, look for register-spill instructions or unrolling
   heuristic differences. This is a one-off investigation, ~30 lines
   of code + reading some assembly, and answers a methodology
   question we cannot afford to ignore before designing 005's
   multi-kernel probe vector.
2. **Experiment 005:** multi-kernel state profiling. Operating range
   informed by 004:
   - memory-bound probe: `write_tid` at 524K-1M threads (work-dominated,
     not yet bandwidth-saturated)
   - compute-bound probe: `fma_loop` at iters ∈ [256, 2048] (above
     the step, below the bimodality zone)
   - latency-bound probe: TBD — this is the kernel kind we have not
     yet built
   - dispatch-overhead probe: `write_tid` 32t (deliberately
     overhead-dominated, kept as a "what's the dispatch overhead
     right now" reference)

005's pre-registration is **not yet written** — we should write the
decision note for "operating points come from 004's findings" before
the experiment README, and resolve the 192→256 step question first
unless we decide it's not on the critical path.

## Notes on the post-run writeup

`UNKNOWNS.md` updated to mark the 5.4 µs reproducibility unknown as
closed (single-launch artifact) and to add the work-dominance floor
findings.

`notes/answered-questions.md` gains entries for:
- Work-dominance floor for `write_tid` thread count.
- Work-dominance floor for `fma_loop` iters (with the step caveat).
- 5.4 µs floor reproducibility.

`experiments/003-warmup-recovery-and-state/README.md` does *not* need
the "5.4 µs floor is the headline finding" framing reversed (it
correctly flagged reproducibility as the open question), but its
"Surprises" §1 should reference 004's null result so readers don't
chase the artifact further than the data warrants. Cross-reference
added.

A small `analysis.py` next to `run.py` does the pooled-across-sweeps
analysis for this writeup. It is a one-off, not a library.
