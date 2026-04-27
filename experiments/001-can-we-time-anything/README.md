# 001: Can we time anything at all from Python?

**Status:** complete (positive)
**Date pre-registered:** 2026-04-27
**Date run:** 2026-04-27
**Hardware target:** M1 Pro (16GB), macOS 26.3.1
**Raw data:** `raw/20260427T132650-{backtoback,spaced1s}.csv`, `raw/20260427T132650-meta.txt`, `raw/20260427T132650-stdout.log`

## The question

Can we get GPU-side timing of a Metal compute dispatch from a Python process,
using `MTLCounterSampleBuffer` with the `MTLCommonCounterSetTimestamp` set,
via PyObjC?

This is the most basic possible question about whether the rest of the
project is feasible. If the answer is no — if PyObjC can't reach the
relevant APIs, or if the timestamps don't make sense, or if the variance is
so bad that no signal is recoverable — then the entire approach needs
rethinking before any other experiment is worth running.

## Hypothesis

Yes, this works. PyObjC bridges Metal sufficiently for compute workloads;
several open-source projects have done dispatch-style work this way. The
specific concern is not "does it work" but "what's the noise floor."

## What we are NOT trying to answer in this experiment

- What the noise floor is. (That's experiment 002.)
- Whether we can sample at dispatch boundaries. (That's a later experiment;
  the public docs strongly suggest stage-boundary is the only supported
  point on Apple Silicon and we'll start there.)
- Anything about M4 Max. (Hardware not yet available.)
- Anything about correctness, roofline, or bottleneck inference. Those are
  much later.

## Setup

- Python 3.11+ via uv or pyenv (decision pending in 002).
- PyObjC, specifically the `Metal` and `Foundation` frameworks.
- Single trivial compute kernel: a kernel that writes `tid` into a buffer.
  Chosen because it has near-zero meaningful work, so any non-trivial timing
  is dominated by dispatch overhead. We want to see the overhead clearly,
  not have it drowned out by compute.
- Run on AC power, with the laptop awake, at user-interactive QoS, with no
  other heavy processes. Capture `powermetrics` output for the same window
  in a separate process so we can correlate later if needed (raw save only;
  not analyzed in this experiment).

## What we'll record

For a single dispatch:
- The two timestamps from the counter sample buffer
- Their delta (the GPU duration)
- CPU-side wall clock around the command buffer commit/wait

For 100 dispatches in a loop, same kernel, same input size:
- All 100 deltas, raw, in a CSV
- Min, median, p95, max
- The histogram

For 100 dispatches with a 1-second sleep between each:
- Same metrics, separate CSV

These two conditions (back-to-back vs spaced) probably look very different
because of DVFS. We don't need to model that yet, just observe it.

## Success criterion

We can produce a CSV of 100 GPU-side dispatch durations and look at them.
That's it. If the numbers are plausible (microseconds, not nanoseconds or
seconds; not all identical; not all NaN), the experiment succeeds.

If the numbers are *not* plausible — if `supportsCounterSampling(at:)`
returns false for `atStageBoundary`, or if PyObjC can't construct the
buffer, or if the timestamps are zero — the experiment also succeeds in the
sense that we've learned something important. We just learned the path
needs to change.

## New questions we expect this to raise

- What's the ratio of CPU wall-clock to GPU duration? (Probably huge for a
  trivial kernel. How huge?)
- Do the back-to-back and spaced distributions look different? How much?
- Is there a warmup effect within the first few dispatches?
- Does the timestamp resolution match what `device.timestampPeriod`-style
  APIs claim?

## After this experiment

If timing works at all, the next experiment is characterizing its noise
properties under controlled conditions. If timing doesn't work, the next
experiment is trying a Swift bridge or `xctrace`-based path. We do not
plan past that branch point.

## Result

**Yes — PyObjC can time Metal compute dispatches end-to-end.** We produced
two CSVs of 100 GPU-side dispatch durations each, all values plausible.
The branch point is taken: timing works, so the next experiment is
characterizing noise (the 002 path), not bridging through Swift.

Headline numbers (per-dispatch GPU duration in raw counter units; see
"Surprises" below for the unit interpretation):

| condition  | min   | median | p95    | max     |
|------------|-------|--------|--------|---------|
| backtoback | 8000  | 8083   | 8667   | 925583  |
| spaced1s   | 8209  | 14730  | 17089  | 288709  |

CPU wall clock around `commit`/`waitUntilCompleted` (nanoseconds):

| condition  | min     | median  | p95     | max       |
|------------|---------|---------|---------|-----------|
| backtoback | 149084  | 186750  | 321675  | 4351542   |
| spaced1s   | 243500  | 728646  | 913029  | 11867000  |

## Surprises

1. **Per-dispatch sample buffer allocation runs out after ~30 dispatches.**
   The first attempt allocated a fresh `MTLCounterSampleBuffer` (sampleCount=2)
   per dispatch, called `resolveCounterRange_` after each `waitUntilCompleted`,
   and let the Python reference go out of scope. After ~31 successful
   dispatches the next `newCounterSampleBufferWithDescriptor:error:` returned
   `MTLCounterErrorDomain Code=0 "Cannot allocate sample buffer"`. Either the
   underlying hardware has a small finite pool and Python's GC is too lazy to
   release in time, or the buffers are simply not released on dealloc the way
   we'd hope. **Mitigation used:** one shared sample buffer per condition
   with `2 * NUM_DISPATCHES` slots, dispatch `i` writes to indices
   `(2i, 2i+1)`. Documented pattern, no behavioral cost. The failed-run log
   is preserved at `raw/20260427T132603-FAILED-perdispatch-buffer.log` —
   negative result, retained.

2. **GPU timestamp counter is ~24 MHz, not 1 GHz.** Back-to-back deltas
   quantize to ~41-42 ns increments (8000, 8041, 8083, 8125, 8167, 8208,
   8250, ...). One tick ≈ 41.67 ns ⇒ 24 MHz hardware clock. Metal scales
   the raw counter into nanoseconds for us, but the underlying resolution
   is one tick. **The smallest GPU duration we can distinguish from "the
   next tick" is ~42 ns.** Anything smaller than ~8 µs is dominated by the
   kernel-launch floor (more on that below). The
   `device.sampleTimestamps:gpuTimestamp:` API returns the GPU value in the
   same nanosecond units, and the elapsed-ratio across the whole experiment
   came out to exactly 1.000000 — i.e. `resolveCounterRange` and
   `sampleTimestamps:gpuTimestamp:` are the same clock, just exposed at
   different granularities.

3. **Apparent kernel-launch floor at ~8 µs.** 84/100 back-to-back samples
   fall in the [8000, 8200] ns window. Min observed value is exactly 8000.
   The "near-zero work" kernel (32 threads, one SIMD-width, write `tid` to
   buffer) hits this floor consistently. This is presumably the
   stage-boundary sampling envelope plus the unavoidable per-dispatch GPU
   overhead, not the kernel's compute time.

   > **Update (2026-04-27, post-004):** experiment 004 swept `write_tid`
   > thread count from 32 up to 8M and confirmed this directly — the
   > 32-thread dispatch is **~64× below the work-dominance threshold**
   > on M1 Pro (T\* ≈ 131K-262K threads). The 8 µs floor measured here
   > is essentially pure dispatch overhead and tells you nothing about
   > write_tid's per-thread cost. Cv numbers in this experiment
   > characterize *dispatch-overhead variance*, not kernel variance.

4. **Back-to-back vs spaced-1s distributions are dramatically different.**
   Spaced-1s has **zero** samples in the [8000, 8200] floor window the
   back-to-back run lives in — its minimum is 8209, median is 14730 (1.8x).
   The DVFS-rampdown between dispatches puts the GPU in a lower-frequency
   state at the start of each spaced dispatch. This is a *very* clean
   signature of why "Apple Silicon DVFS cannot be locked" matters for any
   methodology built on top of this. Both conditions also show occasional
   high outliers (1ms in back-to-back, 12ms CPU-wait once in spaced-1s) —
   probably OS scheduling / GPU contention with the compositor. Untouched
   in the raw data.

5. **CPU wall clock dominates GPU duration by 20–100x.** GPU side does ~8 µs
   of "work" but the round-trip (encode + commit + wait) is 150–700 µs
   median. This is the headline number for "what is the cost of measurement
   itself" and matters for any tight-loop microbench design later.

6. **Capability matrix on M1 Pro / macOS 26.3.1 is starkly clean:**
   `atStageBoundary=True`; `atDrawBoundary`, `atBlitBoundary`,
   `atDispatchBoundary`, `atTileDispatchBoundary` all `False`. This
   definitively answers one of the foundational unknowns and was not in
   doubt for `atDispatchBoundary` per the public docs, but is now
   verified on actual hardware.

## New questions

- What does the *intra-burst* distribution look like at finer than 100
  samples — is the mode at 8083 ns truly stable, or do we see drift over
  thousands of dispatches as the chip warms? (Feeds 002.)
- Does pre-warming with a "wakeup" kernel before timing change the spaced-1s
  distribution? (Feeds 002 — what does "warm" even mean operationally.)
- Why did the per-dispatch sample buffer allocation fail at ~30, not 100,
  not 5? Is there a hardware-published count we can query
  (`MTLDevice.maxBufferLength` or analogous)? Worth a 5-minute side check
  before doing anything else with sample buffers.
- The p95 outliers in back-to-back (~3-4x median) — are they correlated
  with anything observable from the Python side (Python GC pauses,
  command-queue depth)? Or are they pure GPU-side OS noise?
- Does the `sampleTimestamps:gpuTimestamp:` correlation drift across
  long runs? Across thermal events? (We took two snapshots and they
  matched 1:1 over ~100 seconds. Not yet stressed.)
- The high-outlier in spaced-1s' cpu_wait_ns (11.8 ms — one sample) is
  suspicious. Was that one dispatch's GPU duration also an outlier, or
  is it pure CPU-side scheduling? Worth a one-line look at the CSV.

## What this means for experiment 002

002 was framed as "characterize the noise floor under controlled
conditions". 001 has already shown the noise floor is condition-dependent
in a *huge* way (1.8x median shift between back-to-back and spaced-1s).
002 needs to grapple with DVFS state as a first-class variable, not
something to control out. Probable shape: vary inter-dispatch sleep across
{0, 1ms, 10ms, 100ms, 1s, 10s} and characterize the distribution at each.
That's a separate pre-registration; this experiment is closed.
