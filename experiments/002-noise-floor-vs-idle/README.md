# 002: How does the GPU dispatch-timing distribution depend on inter-dispatch idle?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-27
**Hardware target:** M1 Pro (16GB), macOS 26.3.1

## The question

For the trivial `write_tid` kernel from experiment 001, what is the
shape of the per-dispatch GPU duration distribution as a function of the
sleep interval inserted between consecutive dispatches?

Specifically:
- At what sleep value does the median of the distribution leave the
  back-to-back floor (~8 µs from 001) and start to climb?
- Where, between back-to-back and 1-second-spaced (the two endpoints
  measured in 001), does the bulk of the shift happen?
- Is the transition sharp (a step function around some idle threshold) or
  gradual (a continuous DVFS rampdown)?
- How does σ/μ behave across the sweep — does the noise floor scale
  with the median, or does it have its own structure?

## Why this question, now

Experiment 001 showed that the back-to-back distribution (median 8083 ns,
84% of samples in [8000, 8200] ns) and the 1-second-spaced distribution
(median 14730 ns, *zero* samples in the back-to-back floor window) are
dramatically different. That gap of 1 second hides the entire interesting
part of the curve. Until we know what that curve looks like, every later
microbench has to either:

- pin to back-to-back operation (and thereby measure something only
  reachable in artificial conditions), or
- accept whatever DVFS state it happens to land in (and inherit the noise
  we don't yet understand).

The 002 question is the prerequisite for any honest noise budget on this
hardware.

## Hypothesis

The distribution shifts continuously rather than as a sharp step, and the
crossover to "consistently above the back-to-back floor" happens
somewhere in the 1–10 ms idle range. σ/μ stays roughly constant or
shrinks at higher idle (because the floor's hard quantization becomes a
smaller fraction of the absolute value). The 1 ms condition is the most
uncertain — could plausibly look like back-to-back, like spaced-1s, or
something in between.

Confidence: low. The primary value of the experiment is data, not
hypothesis confirmation. We are explicitly NOT going to reframe results
to fit this hypothesis after the fact.

## What we are NOT trying to answer in this experiment

- **Whether kernel size matters.** Kernel is identical to 001 (32-thread
  `write_tid`). Varying kernel size is a separate experiment.
- **What thermal pressure does to the picture.** Run is short enough
  (~7-8 minutes total) and GPU work is light enough that thermals
  should not be a factor. They will be a separate experiment.
- **Whether `powermetrics` GPU frequency correlates with the variance.**
  That correlation is the question for 003 (or wherever it lands).
  002 captures only what's reachable from inside the Python process.
- **Whether warmup ("wakeup kernel before timing") changes the spaced
  distributions.** Also separate. We will not introduce warmup.
- **Anything about M4 Max.** Hardware not yet available.

## Setup

- M1 Pro 16GB, macOS 26.3.1, AC power, laptop awake, no other heavy
  processes running.
- User-interactive QoS (same as 001, via
  `pthread_set_qos_class_self_np`).
- uv + PEP 723 inline metadata for dependencies. Reproducible with
  `uv run run.py` from a clean machine.
- Same kernel as 001 (`write_tid`, 32 threads, 1 SIMD width threadgroup).
- Same shared-sample-buffer pattern as 001 (one buffer with `2 * N`
  slots per condition).
- Conditions: sleep_s in **{0, 0.001, 0.01, 0.1, 1.0}**. N=200 dispatches
  per condition. Each condition runs to completion before the next
  starts. Conditions run in fixed order (sleep ascending) so any
  thermal-state drift is monotonically biased and visible in the data
  rather than hidden by random ordering. Estimated total runtime:
  0.1 + 0.4 + 4 + 40 + 400 ≈ 7.5 minutes plus per-condition setup.
- One CSV per condition under `raw/`, plus a single metadata file with
  device info, OS, Python/PyObjC versions, QoS confirmation, power
  source, and start/end timestamp correlation snapshots.

## What we'll record

Per dispatch:
- `sample_idx` (0..199 within condition)
- `sleep_s` (the condition label)
- `wall_clock_ns` (CPU monotonic at the start of the dispatch — for
  later cross-correlation if we want it)
- `gpu_t_start_raw`, `gpu_t_end_raw`, `gpu_delta_raw`
- `cpu_encode_ns`, `cpu_commit_ns`, `cpu_wait_ns`, `cpu_total_ns`

Per condition (printed live and stored in metadata):
- N, min, p05, median, p95, p99, max of `gpu_delta_raw`
- σ/μ of `gpu_delta_raw`
- Same five quantiles for `cpu_wait_ns`
- Count of samples in the [8000, 8200] back-to-back floor window
  (this is the answer to "did this condition reach the warm floor at all")

Per run (once, in the metadata file):
- Timestamp correlation (CPU/GPU) snapshot at very start and very end
- Wall-clock duration of the whole experiment

We will NOT compute means in any of the live output. Means hide the
shape of distributions that we already know are not Gaussian.

## Success criterion

We have CSVs for all five conditions and can answer, by inspection of
the percentiles:

1. At which sleep value does the median visibly leave the
   back-to-back floor?
2. Is the transition sharp or gradual?
3. Does the sleep=0 condition reproduce 001's back-to-back distribution
   within reasonable tolerance? (If not, something has changed about the
   environment between runs and we need to understand what before
   trusting any of 002.)

If sleep=0 does not reproduce 001 well — different median, different
floor-window count, different shape — that is a successful experiment
in the same sense 001's per-buffer alloc failure was successful: it
tells us the thing we were measuring is more environment-dependent than
we assumed. We follow up before any later experiment.

## New questions we expect this to raise

- What does the curve look like *between* the two sleep values where
  the median jumps? May want a follow-up sweep with finer resolution
  in whatever decade contains the transition.
- What does the curve look like *above* 1s? (Capped here for runtime
  reasons. The 10-second case is a 30+ minute run on its own and may
  not be informative beyond 1s.)
- Are the high-tail outliers (1-3x median) clustered in time within a
  condition, or uniformly distributed? (Time-clustering would suggest
  external interference; uniform suggests intrinsic GPU-side variance.)
- Does the sleep=0 distribution drift across 200 samples vs. 001's 100?
  More samples = more chance to see slow effects.
- How does cpu_wait_ns track gpu_delta_raw within a condition? If the
  CPU overhead is a constant offset, that's good news for measurement
  budget; if it scales with GPU duration, that's a confound.

## After this experiment

If the curve is well-behaved (smooth, monotonic) and reproduces 001's
endpoints, the next experiment is most likely 003: characterize how the
above changes with kernel size — varying threadgroup count from 1 to many,
and threads-per-group from 32 to 1024. That informs whether the
8-µs-floor effect is constant or scales.

If the curve is ill-behaved (multimodal, non-monotonic, or the sleep=0
condition fails to reproduce 001), the next experiment is investigating
what makes the timing environment-dependent before doing anything else.
