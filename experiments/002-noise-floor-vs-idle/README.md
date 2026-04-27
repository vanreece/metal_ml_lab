# 002: How does the GPU dispatch-timing distribution depend on inter-dispatch idle?

**Status:** complete (positive, with one new oddity flagged)
**Date pre-registered:** 2026-04-27
**Date run:** 2026-04-27
**Hardware target:** M1 Pro (16GB), macOS 26.3.1
**Raw data:** `raw/20260427T134031-{sleep_0,sleep_1ms,sleep_10ms,sleep_100ms,sleep_1s}.csv`, `raw/20260427T134031-meta.txt`, `raw/20260427T134031-stdout.log`
**Wall-clock duration:** 224.1 seconds

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

## Result

**The median `gpu_delta_raw` ramps up gradually and monotonically across
two decades of inter-dispatch idle, but the per-condition coefficient of
variation is wildly non-monotonic — sleep_1ms is a *nightmare* condition
and sleep_1s is unexpectedly clean.** Sleep_0 reproduces 001's
back-to-back distribution within tolerance. The success criterion is met.

Headline numbers (per-dispatch GPU duration in raw counter units, ns):

| condition    | min   | p05   | p50   | p95   | p99    | max    | cv (σ/p50) | in [8000,8200] |
|--------------|-------|-------|-------|-------|--------|--------|------------|----------------|
| sleep_0      | 7875  | 8000  | 8083  | 8250  | 11586  | 83750  | 0.66       | 177/200 (88.5%)|
| sleep_1ms    | 7958  | 8083  | 8291  | 9381  | 258385 | 663208 | **7.03**   | 66/200 (33.0%) |
| sleep_10ms   | 7500  | 10834 | 11292 | 15646 | 77819  | 319708 | 2.71       | 0/200 (0%)     |
| sleep_100ms  | 7208  | 10873 | 13791 | 19711 | 133210 | 409834 | 2.23       | 0/200 (0%)     |
| sleep_1s     | 10333 | 13125 | 15229 | 20252 | 21083  | 48250  | **0.21**   | 0/200 (0%)     |

CPU wall clock around `commit`/`waitUntilCompleted` (ns):

| condition    | p50      | p95      | p99      | max      |
|--------------|----------|----------|----------|----------|
| sleep_0      | 189188   | 250315   | 299373   | 425666   |
| sleep_1ms    | 215166   | 284492   | 490529   | 913291   |
| sleep_10ms   | 581604   | 885156   | 2053220  | 3412709  |
| sleep_100ms  | 803084   | 2090102  | 2874797  | 4682333  |
| sleep_1s     | 1882666  | 2514224  | 2886739  | 5710750  |

### Sanity check vs experiment 001

| metric        | 001 backtoback (N=100) | 002 sleep_0 (N=200) | match? |
|---------------|------------------------|---------------------|--------|
| min           | 8000                   | 7875                | ✅     |
| median        | 8083                   | 8083                | ✅ exact |
| p95           | 8667                   | 8250                | ✅     |
| in_floor frac | 84%                    | 88.5%               | ✅     |

| metric        | 001 spaced1s (N=100)   | 002 sleep_1s (N=200)| match? |
|---------------|------------------------|---------------------|--------|
| median        | 14730                  | 15229               | ✅ ~3% |
| p95           | 17089                  | 20252               | ~     |
| max           | 288709                 | 48250               | 002 quieter on max (no big outlier this run) |

Both endpoints reproduce — the timing setup is stable across runs and
the data trustworthy.

> **Update (2026-04-27, post-004):** the cv numbers in this table
> describe *dispatch-overhead* variance, not `write_tid` kernel
> variance. Experiment 004 swept `write_tid` thread count and showed
> the 32-thread dispatch sits ~64× below the work-dominance threshold
> on M1 Pro — at this size the median is dominated by encoder/commit/
> wait overhead, and any cv is the variance of that overhead under
> different cadences. Reading these numbers as "the noise floor of
> Metal timing" is fine; reading them as "the noise floor of running
> a memory-write kernel" is not. See
> `experiments/004-work-dominance-floor/README.md` § "What this means
> operationally".

## Surprises

1. **The 1ms condition has cv=7.03 — the *highest* of any condition by
   3x.** The hypothesis predicted noise to scale roughly with the
   median, or perhaps shrink at higher idle. Reality: at sleep=1ms the
   GPU is in a transitional power state where most dispatches still
   complete fast (66/200 land in the back-to-back floor window) but a
   handful of dispatches take **30–80x longer** (p99 = 258 µs vs p95 =
   9.4 µs — a 27x gap between the 95th and 99th percentile). 5 samples
   exceeded 50 µs, with one at 663 µs. This means sleep=1ms is, by an
   enormous margin, the *worst* cadence to microbench at — both
   warm-fast and cold-slow regimes happen, the boundary is inside this
   condition, and you can't tell from the median that the tail is
   elsewhere. Critical lesson for any later microbench design: **never
   pick a sleep value in this transition zone unless you specifically
   want to characterize it.**

2. **sleep_1s is unexpectedly tight — cv=0.21, the *lowest* of any
   condition.** Once the GPU has fully settled into its idle/cool state,
   it bounces back to a remarkably consistent execution time. p99
   exceeds p95 by only 4%. The "fully cold" regime is a much better
   place to live than the "transitioning" regime. This inverts the
   intuition that "more idle = more variability".

3. **The outliers in sleep_1ms are not clustered in time** — they appear
   at indices 0, 37, 43, 45, and 174 across the 200-dispatch run. Not a
   warmup phenomenon (would expect early-only); not a thermal phenomenon
   (would expect late-only); not a periodic phenomenon (no obvious
   period). Looks like genuine intrinsic variance from the GPU's
   power-state machine making different decisions on adjacent dispatches.

4. **sleep_100ms shows real drift across the 20-second run.** Median of
   the first 20 samples is 11667 ns; median of the last 20 samples is
   14542 ns — a +2875 ns shift (+25%) over the run. All other
   conditions drift by ≤84 ns first-20-vs-last-20 (i.e. nothing).
   Specifically sleep_1s, which ran 200 seconds (10x longer), drifts
   only +83 ns. So the drift is not "the chip is warming up over wall
   time" — it's something specific to the 100ms cadence. This is a new
   surprise that 002 didn't pre-register and we're flagging for follow-up
   rather than explaining now.

5. **The 24 MHz quantum has a sub-tick rounding signature.** The unique
   values in sleep_0 are spaced as `[..., 7958, 7959, 8000, 8041, 8042,
   8083, 8084, 8125, 8166, 8167, 8208, 8209, ...]` — diffs alternate
   `41, 1, 41, 1, 41, 41, 1, ...`. The underlying tick is 41.67 ns and
   Metal rounds it to integer ns, so the same physical tick sometimes
   reads as N and sometimes as N+1 nanoseconds. Confirms the 001
   interpretation; doesn't change anything operationally; nice to have
   the artifact visible in the raw data.

6. **Some conditions show `min` *below* the back-to-back floor.**
   sleep_10ms min=7500, sleep_100ms min=7208. The "floor" is therefore
   not a hard physical floor of the kernel but the typical settled value
   of one specific power state. At least one observed dispatch in the
   higher-idle conditions completed faster than any sleep_0 dispatch.
   This was unexpected but not surprising in retrospect — different
   power states presumably have different timing characteristics, and
   the "fast outlier" presumably represents the GPU briefly being in a
   higher-frequency state than back-to-back ever reaches.

7. **Total runtime came in 9% over budget** (224s vs 222s estimated).
   Sleep budget alone was 221.2 s; the 2.9 s overhead across 1000
   dispatches matches the per-dispatch CPU wait totals. No surprise,
   noted only because pre-registration made the prediction.

## What this means operationally

If you want to microbench a kernel on this hardware, your noise budget
is roughly:

| cadence target              | recommended sleep | expected cv | notes |
|-----------------------------|-------------------|-------------|-------|
| "as fast as possible"       | 0 (back-to-back)  | ~0.66 *     | hard quantization at 8083 ns; outlier risk |
| "interactive responsive"    | **avoid 1–10 ms** | 2.7–7.0     | unpredictable bimodal regime |
| "patient measurement"       | ≥ 1 s             | ~0.21       | very tight; pay 1 s/sample |

\* sleep_0's cv is dominated by one 83-µs outlier in 200 samples; the
inlier cv is much smaller. The hard quantization makes nominal cv
misleading at this scale.

## New questions

- What does the curve between sleep=0 and sleep=10ms look like at
  finer resolution? sleep_1ms is the noisy one; is sleep_2ms still
  noisy, or has the bimodal regime passed? (Worth a follow-up sweep
  before 003.)
- **Why does sleep_100ms drift while no other condition does?** Is this
  reproducible (run twice and compare) or a one-off artifact of this
  specific run? The drift's direction (median grows) is the opposite of
  what compositor-warmup would predict. This is the highest-priority
  follow-up.
- The 5 outliers in sleep_1ms have values 99500, 179333, 256875, 407833,
  663208 ns. These look like multiples of something — possibly the
  duration of "one full DVFS rampup" being inserted into specific
  dispatches? Worth checking if the values are integer multiples of
  some base wakeup latency.
- The first dispatch of sleep_1ms (index 0) is also an outlier (256875).
  But sleep_0 index 0 is *not* an outlier (only 11208 — well within
  normal range). Whatever's special about "first dispatch after a fresh
  command queue" depends on the cadence. New question: does the very
  first dispatch in any session have its own distribution?
- How much of the cpu_wait_ns growth at higher idle is the actual
  GPU work taking longer vs. the OS scheduler taking longer to wake the
  Python thread back up? Both contribute and we can't separate them
  from inside the process. May want to instrument from outside (dtrace
  or similar) for a follow-up if this becomes a measurement-budget
  bottleneck.
- Do the sleep_10ms / sleep_100ms maxes (319708, 409834) reflect rare
  full DVFS-warmup transitions, or external interference (compositor,
  system services)? p99 is much smaller than max in both, suggesting
  these are 1-in-200 events, not the typical worst case.

## After this experiment

The natural next experiment per the pre-registration is 003:
characterize how the noise structure changes with kernel size. But the
sleep_100ms drift is more anomalous than 003 will be informative, and
"highest uncertainty reduction" probably means investigating the drift
first. Pre-registration of 003 (kernel-size sweep) and 003a (drift
reproduction at sleep_100ms) will be done as separate decisions; this
experiment is closed.
