# Answered questions

Questions that started in `UNKNOWNS.md` and got definitively (or
substantively) answered by an experiment. Kept here so that the answer
travels with the project even after the question leaves the active list.

Format: question, answer, hardware/software it applies to, and the
experiment that closed it.

## Does `MTLDevice.supportsCounterSampling(at:)` return True for any sampling point besides `atStageBoundary` on M-series?

**Answer:** No, on M1 Pro / macOS 26.3.1 *and* on M4 Max / macOS 26.4.1.

`supportsCounterSampling_` returns:
- `atStageBoundary`: True
- `atDrawBoundary`: False
- `atBlitBoundary`: False
- `atDispatchBoundary`: False
- `atTileDispatchBoundary`: False

Identical on both chips and across the 26.3.1 → 26.4.1 macOS update.
Practical consequence: any per-dispatch timing must be done at the
compute-pass-encoder level (one pass = one timing window) rather than
sampling individual `dispatchThreads` calls. Multi-dispatch encoders
cannot be timed per-dispatch on either chip generation.

**Hardware/software (verified):**
- Apple M1 Pro 16GB / `applegpu_g13s` / macOS 26.3.1, PyObjC via
  `pyobjc-framework-Metal`.
- Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1, PyObjC via
  `pyobjc-framework-Metal`.

**Closed by:** experiment 001 (M1 Pro, 2026-04-27),
extended to M4 Max by `notes/probe-counter-sets.py` re-run + experiment
001 re-run (2026-04-28).
**Still open for:** future macOS versions, all other Apple Silicon
variants (base M-series, Ultra-class, future generations).

## Can we read GPU timestamps from Python via PyObjC at all?

**Answer:** Yes. `MTLCounterSampleBuffer` with the `timestamp` counter
set, attached to a `MTLComputePassDescriptor`'s
`sampleBufferAttachments[0]` with start/end indices, then resolved with
`resolveCounterRange_((start, count))`, returns nanosecond-scaled GPU
timestamps as raw uint64 little-endian bytes in NSData. No Swift bridge
needed.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, PyObjC.
**Closed by:** experiment 001.

## What is the noise floor (σ/μ) of MTLCounterSampleBuffer timing on M1 Pro?

**Answer:** It is *not* a single number — it depends sharply on how
much idle time sits between consecutive dispatches. For the trivial
`write_tid` kernel (32 threads, 1 SIMD width) on M1 Pro with macOS
26.3.1 at user-interactive QoS on AC power:

| inter-dispatch sleep | median gpu_delta_raw (ns) | cv (σ/p50) |
|----------------------|---------------------------|------------|
| 0                    | 8083                      | 0.66       |
| 1 ms                 | 8291                      | **7.03**   |
| 10 ms                | 11292                     | 2.71       |
| 100 ms               | 13791                     | 2.23       |
| 1 s                  | 15229                     | **0.21**   |

The non-monotonic cv is the headline. The 1 ms condition is by far the
worst — bimodal with most dispatches near the back-to-back floor and
~5/200 dispatches taking 30–80x longer because the GPU is in a
transition zone of its power-state machine. The 1 s condition is by
far the *best* (tightest distribution), even though the absolute median
is highest. "More idle = more noise" is wrong; "more idle = different
power state, possibly tighter" is right.

**Practical consequence:** any microbench design must explicitly choose
its cadence and inherit that cadence's noise structure. There is no
"pin to one frequency" workaround on Apple Silicon. The 1–10 ms range
is poison and should be avoided unless characterizing it is the point.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power,
laptop awake, no other heavy processes.
**Closed by:** experiment 002.
**Still open for:** how this scales with kernel duration / size / load,
and whether the same picture holds on M4 Max.

## What counter sets does Metal actually expose on M1 Pro and M4 Max?

**Answer:** Exactly one on both chips — `timestamp`, with one counter
inside it (`GPUTimestamp`). No `StageUtilization`, no `Statistic`, no
per-stage cycle counts, no `ComputeKernelInvocations`, no occupancy.
The Metal headers define all of these as constants and PyObjC happily
exposes them, but the Apple GPU driver does not populate them on
`applegpu_g13s` (M1 Pro) *or* on `applegpu_g16s` (M4 Max).

This means **timing is the only "free" GPU-side signal Metal gives
us, on both M1 Pro and M4 Max.** Anything else (occupancy, stage
activity, cycle counts) requires either powermetrics (sudo, sliding
window, no per-kernel detail), Instruments / `xctrace` -> `.gputrace`
(sudo, opaque format, reverse-engineering out of scope), or private
Apple SPIs we cannot reach.

The project's "without vendor-internal counters" thesis is therefore
not a self-imposed constraint but a real architectural gap that Apple
has chosen to keep closed in the public API across at least two
chip generations and at least two macOS releases (26.3.1 and 26.4.1).

**Hardware/software (verified):**
- Apple M1 Pro 16GB / `applegpu_g13s` / macOS 26.3.1, PyObjC.
- Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1, PyObjC.

**Closed by:** `notes/counter-sets-on-m1-pro.md` (probe) on 2026-04-27;
extended to M4 Max by `notes/counter-sets-on-m4-max.md` (re-probe) on
2026-04-28.
**Still open for:** future macOS versions on either chip (Apple could
in principle add support in a driver update — worth re-running the
probe periodically); other Apple Silicon variants (base M-series,
Ultra-class).

## Does a warmup prefix recover the "cool cadence" noise from 002?

**Answer:** Yes, with caveats, for cadences ≥ 10 ms. **No, and don't
try, at sleep_0.**

For inter-dispatch sleep ≥ 10 ms, K=0 puts p50 at 14–17 µs (the
"cool" regime from 002). One untimed warmup dispatch (K=1) of the
same kernel kind drops p50 back to ~9.7 µs and leaves cv ≤ 0.16.
Bigger K offers no obvious benefit; K=20 occasionally introduces tail
risk (one combo at cv=5).

At sleep_0, the chip is already in (or below) the warm steady state.
Adding warmup pushes it to a slightly slower settled state (p50 climbs
from 5.4–8 µs to ~9 µs).

Warmup *kind* matters in unexpected ways: a single arithmetic-heavy
warmup dispatch (`fma_loop K=1`) is **strictly worse** than no warmup
at all (cv 1.4–3.8 vs 0.1–0.3 at the same cadences). K≥5 of the same
kind recovers; one is destabilizing. The destabilization is specific
to switching kernel character before the measured (memory-write)
dispatch.

**Practical recipe for `write_tid` 32-thread microbenches:** at
cadence ≥ 10 ms, `K=1` of the same kernel kind right before each
measurement is the right warmup. At sleep_0, no warmup. Avoid one
dispatch of an arithmetic-heavy kernel as a "light" warmup.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power,
display held awake via `caffeinate -d -i -m`.
**Closed by:** experiment 003.
**Still open:** whether the recipe transfers to non-trivial kernels
(planned for 004); how to escape the 1-10 ms transition zone (003
contaminated the K=0 baseline at this cadence via the calibration
probe and could not test cleanly).

## At what `write_tid` thread count does measured time start scaling with work rather than reflecting dispatch overhead?

**Answer:** T\* ≈ 131 072 - 262 144 threads on M1 Pro (threadgroup size 32).
Below ~32K threads the median sits in 9-11 µs regardless of thread count
— this is dispatch-overhead-dominated and timing tells you about overhead
variance, not the kernel. From 32K to 131K threads the work begins to
contribute (`p50_ratio / level_ratio` rises from 0.6 to 0.67). At 131K →
262K threads, doubling threads multiplies median time by 1.78× —
work-dominated. Linear scaling continues through 1M threads, then goes
sublinear at 8M (8× threads → 3.83× time) because the 32 MiB write hits
DRAM bandwidth saturation (~85 GB/s observed, in line with M1 Pro's
realistic write-only bandwidth).

**Practical consequence:** any `write_tid`-class memory-bound benchmark
on M1 Pro must operate at ≥ 262K threads to be measuring the kernel,
not the dispatch overhead. The 524K-1M range is the cleanest operating
point. **Retroactively this means experiments 001, 002, 003 all
measured `write_tid` 32t — i.e., were measuring dispatch overhead, not
write_tid kernel time.** Their cv numbers describe *overhead* variance,
not write-kernel variance.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power, K=1
warmup recipe (per 003).
**Closed by:** experiment 004.
**Still open for:** M4 Max; other access patterns (strided, scatter);
threadgroup sizes other than 32; whether the bandwidth-saturation knee
sits between 1M and 8M threads (no fine-grained sweep yet).

## At what `fma_loop` per-thread iter count does measured time start scaling with work?

**Answer:** Sharp step at iters = 192 → 256 on M1 Pro (32 threads,
threadgroup 32). Below 192 iters: median climbs slowly from 9 to 12.6
µs (overhead-dominated). At 192 → 256 iters: a discrete **+21 µs jump**
in median (12.6 → 34.2 µs) — far too large to be explained by the
extra 64 FMA instructions doing useful work. From 256 onward: clean
linear scaling (`p50_ratio / level_ratio` ∈ [0.90, 1.00] up to 65K
iters), with one bimodal-variance band at iters = 8192-16384 where
between-sweep p50s vary 2.5× even with 30 s between-sweep cooldowns.

**Practical consequence:** any `fma_loop`-class compute benchmark must
use ≥ 256 iters per thread to be measuring the kernel. The 96-256
range is non-monotonic and unsuitable for any "increase work,
measure proportional increase in time" methodology. The 4096-16384
range has bimodal between-run state effects and should also be
avoided for between-run-comparable measurements.

**Open mechanism:** the +21 µs step at 192→256 has not been isolated.
Most likely explanation is a compiler / hardware threshold (register
pressure, loop-unrolling heuristic flip, instruction-cache crossover)
rather than work actually appearing — kept as an open question in
`UNKNOWNS.md`.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power, K=1
warmup recipe (per 003).
**Closed by:** experiment 004.
**Still open for:** M4 Max; the mechanism behind the +21 µs step;
whether the bimodal band at 8192-16384 iters is thermal, DVFS, or a
queue-depth effect.

## Is the ~5.4 µs settled state observed in 003's first combo reproducible by the cold-start + cooldown + calibration recipe?

**Answer:** No. Across 5 sequential attempts with the protocol that
recreated the 003 first-combo conditions (5 s cooldown → 10-dispatch
calibration burst → 40 measured dispatches, no warmup, sleep_0
between trials), 0 of 200 measured dispatches landed in [5000, 6000]
ns. Every attempt converged to the ~8 µs back-to-back floor that
001/002/003 always observed elsewhere (per-attempt p50: 8146, 8333,
8084, 8166, 8334 ns). Per the 004 success criterion's threshold (0-1
reproductions = "non-reproducible artifact"), this state is filed.

**Practical consequence:** the operational floor for `write_tid`
32-thread back-to-back dispatches on M1 Pro is **8 µs (K=0) / 9 µs
(K=1)** for downstream calibration-probe work. The 5.4 µs observation
in 003 is a single-launch artifact whose trigger we cannot reconstruct
from logged state. Future experiments do not get to claim sub-8-µs
steady-state without independently reproducing it.

**What we ruled out:** that the recipe of "cold-start + caffeinate +
short cooldown + calibration burst" reaches the state. What we did
not test: that *the very first GPU activity in a fresh process tree
that has had no recent prior MLX/Metal activity at all* reaches it.
But this is unfalsifiable in practice (we cannot confidently produce
a "no recent prior GPU activity" environment without a full reboot
and a measurement protocol that does not contaminate it).

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power.
**Closed by:** experiment 004.

## Does paired co-encoded ratio timing reduce within-session variance vs single-kernel timing?

**Answer:** No, on M1 Pro for kernels in the 50-400 µs duration
range. Across 4 trial kernels (T1 fma_loop iters=512, T2 fma_loop
iters=4096, T3 write_tid 524K threads, T4 write_tid 1M threads)
each measured both alone and paired with the reference (fma_loop
iters=1024 at 32t), the paired ratio's robust cv was equal to or
**worse** than the trial-alone robust cv in every case:

| trial | alone robust_cv | ratio robust_cv | verdict        |
|-------|----------------:|----------------:|----------------|
| T1    |          0.0095 |          0.0112 | ratio ≈ alone  |
| T2    |          0.0012 |          0.0057 | **4.75× worse** |
| T3    |          0.0346 |          0.0374 | ratio ≈ alone  |
| T4    |          0.0392 |          0.0410 | ratio ≈ alone  |

Mechanism: variance composition for ratios of independent measurements
is `cv²(A/B) ≈ cv²(A) + cv²(B)`. The ratio inherits noise from both
the trial and the reference rather than canceling shared noise,
because the dominant within-session noise sources for kernels at
warm steady state (timestamp quantization, OS scheduling /
preemption, per-encoder setup variance) are not state that's
shared between two encoders 42 µs apart in the same command buffer.
The "ratios cancel DVFS variance" intuition only applies when DVFS
variance is the dominant noise source, which it isn't inside a
2-minute hot run.

**Practical consequence:** for within-session cv-bound questions,
single-kernel timing with N samples and robust statistics is the
right tool. Pair timing is *not* a variance reducer and should not
be invoked for that purpose. See decisions/004-narrowed-pair-timing-scope.md
for the narrowed operating envelope.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power.
**Closed by:** experiment 005.
**Still open for:** kernels at much longer durations (>1 ms) where
DVFS state changes might happen mid-run (untested); cross-session
ratio stability (which is a different question).

## Does paired co-encoded timing perturb the trial's underlying behavior?

**Answer:** No. Across all 4 trial kernels in 005, the trial's
median when measured paired vs alone shifted by less than ±1 % (T1
-0.93%, T2 -0.13%, T3 -0.68%, T4 +0.81%). The reference dispatch
that immediately precedes the trial inside the same command buffer
does not meaningfully change the trial's behavior.

**Practical consequence:** any per-trial number from a paired
measurement can be compared directly to a single-kernel measurement
of the same kernel. Pair timing does not introduce a systematic bias
on the trial.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power.
**Closed by:** experiment 005.

## Is the paired ratio stable across within-session sweep repetitions?

**Answer:** Yes, within ~1 %. Across 3 sweep repetitions of the
same paired conditions with 30 s between-sweep cooldowns, the
ratio's per-sweep median agreed within:

| trial | spread |
|-------|-------:|
| T1    |  0.23% |
| T2    |  0.20% |
| T3    |  0.98% |
| T4    |  0.60% |

T3 sits right at the boundary, mostly reflecting T3 itself being
the noisiest trial (alone robust_cv = 0.035) rather than ratio-
specific drift.

**Practical consequence:** within a single script invocation, the
paired ratio is a stable relative-magnitude metric. "Kernel A is
~0.54× kernel B's duration on this chip in this session" is a
sound claim if A and B are paired with the same reference.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power.
**Closed by:** experiment 005.
**Still open for:** cross-session ratio stability (untested; a
follow-up experiment would run the same conditions across separate
process invocations / time-of-day / thermal states).

## What is the inter-encoder gap inside a single MTLCommandBuffer on M1 Pro?

**Answer:** ~42 µs at p50, with low variance (p95 in [46, 48] µs,
p99 in [49, 74] µs across 4 trial kinds × 900 paired measurements
each). Two consecutive `MTLComputeCommandEncoder` passes inside a
single `MTLCommandBuffer` are *not* atomically tight; the GPU
front end inserts substantial idle time between them.

This is roughly 4 × the dispatch-overhead floor (~9 µs) and is
consistent across kernel kind (fma_loop and write_tid), suggesting
it's a property of the encoder model itself rather than of the
trial kernel. Mechanism (per-encoder setup cost, GPU front-end
stall, command-list reordering) not isolated.

**Practical consequence:** any analysis that depends on "ref and
trial in the same chip state" inside a paired command buffer has
~42 µs of resolution limit. For DVFS / thermal state changes that
happen on millisecond-or-longer timescales, this is fine; for
cycle-accurate or sub-microsecond comparisons, the same-buffer
pattern is not tight enough.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power,
PyObjC + `MTLCounterSamplingPointAtStageBoundary`.
**Closed by:** experiment 005.
**Still open for:** whether different encoder patterns (separate
command buffers, explicit barriers, queue properties) reduce the
gap; whether the gap differs on M4 Max.

## What is the GPU timestamp counter's hardware tick resolution on M1 Pro and M4 Max?

**Answer:** ~24 MHz (one tick ≈ 41.67 ns) on **both** M1 Pro and M4 Max.
Apparent in 001's back-to-back distribution as quantization at
8000, 8041, 8083, 8125, 8167, ... raw units on M1 Pro, and 6125, 6167,
6208, 6250, 6291, 6333, 6375, 6417, ... on M4 Max — different
absolute floors, identical step (41 or 42 ns; min/41.6667 ≈ 147 on
M4 Max, exact integer multiples of the tick). Metal's
`resolveCounterRange_` returns these raw counter values already scaled
to nanoseconds — i.e. each integer "nanosecond" we read is actually
one of every ~42 contiguous nanoseconds. The
`device.sampleTimestamps:gpuTimestamp:` API returns matching values
(elapsed-ratio = 1.000000 across ~99-100 s in 001 on both chips),
confirming the two paths expose the same clock at the same granularity.

**Practical consequence:** smallest distinguishable GPU duration delta is
~42 ns on both M1 Pro and M4 Max. Anything reported as a sub-42-ns
delta would be impossible. Tick-rate-dependent reasoning (e.g. ratio
of duration to tick to estimate cycle count) carries one-to-one between
the two chips at this layer.

**Hardware/software (verified):**
- Apple M1 Pro 16GB / `applegpu_g13s` / macOS 26.3.1.
- Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1.

**Closed by:** experiment 001 (M1 Pro, 2026-04-27); extended to M4 Max
by experiment 001 re-run (2026-04-28).
**Still open for:** future Apple Silicon generations (G17+ etc.) — the
24 MHz figure could change at any chip revision; worth re-validating
on each new family.

## What is the dispatch-overhead floor on M4 Max for write_tid 32t?

**Answer:** ~6.4 µs at p50 / ~6.1 µs at min, vs ~8.0 µs / ~8.0 µs on
M1 Pro. A clean ~20 % reduction in the per-dispatch envelope from
G13 to G16 / macOS 26.3.1 to 26.4.1.

| metric (write_tid 32t back-to-back) | M1 Pro | M4 Max | delta |
|-------------------------------------|-------:|-------:|------:|
| min                                 |  8 000 |  6 125 |  -23% |
| p50                                 |  8 083 |  6 417 |  -21% |
| p95                                 |  8 667 |  8 131 |   -6% |
| max                                 | 925 583 | 24 667 |  -97% |

**Practical consequence:** every M1-Pro-baseline number that 001-005
anchored against the 8 µs floor (work-dominance threshold thread
counts, +21 µs step duration, ~42 µs inter-encoder gap, paired-ratio
variance composition) needs to be re-evaluated against M4 Max's
6.4 µs floor. The qualitative methodology framework (decision 004's
narrowed pair-timing scope, percentile reporting, "raw before robust")
carries over; the absolute numbers do not.

**Tail behavior is dramatically tighter on M4 Max** in this single
N=100 run: max in back-to-back was 25 µs vs 925 µs on M1 Pro (a 37×
reduction in worst-case). One run is not enough to lock this in; it
could be statistical, macOS 26.4 scheduler improvements over 26.3,
or G16 front-end behavior. Flagged for revisit in 002-style re-runs.

**Hardware/software:** Apple M4 Max 36GB, `applegpu_g16s`,
macOS 26.4.1, AC power, user-interactive QoS, K=0 (no warmup).
**Closed by:** experiment 001 re-run on 2026-04-28.
**Still open for:** how this floor scales with cadence (002 re-run
needed); whether the dramatic tail-tightening is reproducible across
runs and macOS releases.
