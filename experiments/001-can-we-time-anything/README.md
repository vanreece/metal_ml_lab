# 001: Can we time anything at all from Python?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-27
**Hardware target:** M1 Pro (16GB)

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
