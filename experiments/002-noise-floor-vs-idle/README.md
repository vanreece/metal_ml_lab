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

---

## M4 Max addendum (re-run on new hardware)

**Date re-run:** 2026-04-28
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS:** macOS 26.4.1 (build 25E253)
**Raw data:** `raw/20260428T112556-sleep_{0,1ms,10ms,100ms,1s}.csv`,
`raw/20260428T112556-meta.txt`, `raw/20260428T112556-stdout.log`
**Wall-clock duration:** 223.7 s (essentially identical to M1 Pro's
224.1 s — wall time is dominated by `sleep_1s` x 200).

Re-ran 002 unchanged (one small addition: meta now records
`device.architecture().name()`). Same 5 cadences, same N=200.

### Headline: the M1 Pro "1ms nightmare zone" is GONE

**This is the largest qualitative finding of the M4 Max re-run series
so far.** On M1 Pro, sleep_1ms had cv=7.03 — *the worst by 3×*, with a
bimodal distribution where 5/200 dispatches took 30-80× longer than
the median. On M4 Max, sleep_1ms has cv=**0.07** — *the best by an
order of magnitude over every other cadence*, with a tight unimodal
distribution and max=12.5 µs (vs 663 µs on M1 Pro at the same cadence).

The M1 Pro recipe of "**never** pick a sleep value in the 1-10 ms
transition zone" does **not** transfer to M4 Max. Either the GPU's
power-state machine has fewer / different states near 1 ms, or the
states transition fast enough that 1 ms idle no longer falls inside
a transition window, or both. The structural lesson — "DVFS state
matters and noise is cadence-dependent" — survives, but the specific
operating constraints are chip-specific.

### Side-by-side: M1 Pro vs M4 Max

GPU duration in raw counter units (ns), per condition:

| condition    | metric  | M1 Pro (g13s) | M4 Max (g16s) | delta |
|--------------|---------|--------------:|--------------:|-------|
| sleep_0      | min     | 7 875         | **6 000**     | -24%  |
|              | p50     | 8 083         | **6 458**     | -20%  |
|              | p95     | 8 250         | 8 379         | ~     |
|              | p99     | 11 586        | 58 350        | **+5×** |
|              | max     | 83 750        | 163 084       | +95%  |
|              | cv      | 0.66          | **1.92**      | M4 worse |
| **sleep_1ms**| p50     | 8 291         | 6 791         | -18%  |
|              | p95     | 9 381         | 7 208         | -23%  |
|              | p99     | 258 385       | **7 470**     | **-97%** |
|              | max     | 663 208       | **12 500**    | **-98%** |
|              | **cv**  | **7.03**      | **0.07**      | **100× cleaner** |
| sleep_10ms   | p50     | 11 292        | 8 917         | -21%  |
|              | p95     | 15 646        | 21 502        | +37%  |
|              | p99     | 77 819        | 55 084        | -29%  |
|              | max     | 319 708       | 137 458       | -57%  |
|              | cv      | 2.71          | 1.38          | -49%  |
| sleep_100ms  | p50     | 13 791        | 9 000         | -35%  |
|              | p95     | 19 711        | 20 015        | ~     |
|              | p99     | 133 210       | 27 101        | -80%  |
|              | max     | 409 834       | 158 833       | -61%  |
|              | cv      | 2.23          | 1.64          | -26%  |
| sleep_1s     | min     | 10 333        | 7 667         | -26%  |
|              | p50     | 15 229        | 9 042         | -41%  |
|              | p95     | 20 252        | 22 519        | +11%  |
|              | p99     | 21 083        | 25 168        | +19%  |
|              | max     | 48 250        | 29 208        | -39%  |
|              | cv      | **0.21**      | **0.46**      | **2.2× worse** |

### Five distinct shifts G13 → G16

1. **Floor dropped uniformly ~20-25 %.** Min and p50 across every
   cadence are 18-41 % lower on M4 Max. Consistent with the 6.4 µs
   floor measured in the 001 re-run.
2. **The 1 ms transition zone vanished.** sleep_1ms on M4 Max behaves
   like an idealized version of sleep_0: tight, unimodal, no big tail.
   This is the most striking qualitative shift between G13 and G16
   we have observed so far.
3. **sleep_0 got NOISIER on M4 Max.** cv 0.66 → 1.92, driven by 1-2
   very large outliers (max 163 µs vs 84 µs). The bulk distribution
   is tighter (most samples in the new ~6.5 µs floor window, very few
   in [8000, 8200]) but the worst-case grew. The M4 Max appears to
   have a thinner, more cliffside back-to-back floor: when it lands,
   it lands clean; when it misses, it misses bigger.
4. **The "fully cold gives tight cv" intuition broke.** M1 Pro
   sleep_1s was the cleanest condition (cv=0.21); M4 Max sleep_1s is
   middle-of-pack (cv=0.46). The cold settled state on M4 Max is
   noisier than on M1 Pro, both in absolute (max 29 µs vs 48 µs is
   *better* but cv is worse because the median moved less) and in
   relative terms.
5. **The ordering of cv across cadences inverted.** M1 Pro:
   `sleep_1ms (7.03) > sleep_10ms (2.71) > sleep_100ms (2.23) >
   sleep_0 (0.66) > sleep_1s (0.21)`. M4 Max:
   `sleep_0 (1.92) > sleep_100ms (1.64) > sleep_10ms (1.38) >
   sleep_1s (0.46) > sleep_1ms (0.07)`. Almost the reverse for the
   1ms case; jumbled elsewhere.

### Floor-window count: the metric needs to move with the floor

The "in_floor [8000, 8200]" count from M1 Pro is essentially zero on
M4 Max in every condition — the floor *moved*. Going forward, this
metric needs to be calibrated to the chip-specific floor (somewhere
near [6000, 6500] for M4 Max sleep_0). For now, treat the M4 Max
in_floor counts as evidence the floor shifted, not as evidence the
chip never reaches its floor.

### Sanity check vs M4 Max experiment 001

| metric         | 001 backtoback (N=100) | 002 sleep_0 (N=200) | match? |
|----------------|------------------------|---------------------|--------|
| min            | 6 125                  | 6 000               | ✅     |
| p50            | 6 417                  | 6 458               | ✅ ~1% |
| p95            | 8 131                  | 8 379               | ✅     |
| max            | 24 667                 | 163 084             | 002 has worse outlier |

| metric         | 001 spaced1s (N=100)   | 002 sleep_1s (N=200)| match? |
|----------------|------------------------|---------------------|--------|
| min            | 6 208                  | 7 667               | ~      |
| p50            | 9 188                  | 9 042               | ✅ ~1% |
| p95            | 23 315                 | 22 519              | ✅ ~3% |
| max            | 27 083                 | 29 208              | ✅     |

Bulk distributions reproduce cleanly between 001 and 002 on M4 Max.
The single 163 µs sleep_0 outlier in 002 is the kind of sample 001
N=100 didn't catch — N=200 is enough to start seeing the M4 Max
sleep_0 tail.

### What does NOT change G13 → G16

- The qualitative claim "DVFS state matters and noise is
  cadence-dependent" still holds — just with completely different
  per-cadence numbers.
- Median climbs monotonically with sleep duration on both chips
  (8.0 → 8.3 → 11.3 → 13.8 → 15.2 on G13; 6.5 → 6.8 → 8.9 → 9.0 →
  9.0 on G16) — though the M4 Max curve flattens above 10 ms
  whereas M1 Pro keeps climbing.
- Tail outliers exist at every cadence on both chips, just at
  different magnitudes and concentrations.
- 24 MHz tick quantization apparent in raw values on both chips.
- `sampleTimestamps:gpuTimestamp:` ratio = 1.000000 on both runs.

### What this changes operationally

The M1 Pro "noise budget" table is *wrong* on M4 Max. New M4 Max
table:

| cadence target              | recommended sleep | observed cv | notes |
|-----------------------------|-------------------|-------------|-------|
| "as fast as possible"       | 0 (back-to-back)  | ~1.9        | bulk tight ~6.5 µs but rare large outliers |
| **"interactive responsive"**| **1 ms**          | **~0.07**   | **opposite of M1 Pro — sleep_1ms is the cleanest** |
| "patient measurement"       | ≥ 1 s             | ~0.46       | not as tight as on M1 Pro |
| AVOID                       | none observed yet | -           | no clear "nightmare zone" in this sweep |

Decision 004's narrowed pair-timing scope is unchanged by this
finding — it depends on math (variance composition) and on
within-session ratio stability, neither of which 002 measured. But
the *baseline* against which paired-ratio cv is compared has to be
re-measured per chip.

### New questions raised by the M4 Max re-run

- **Does the M4 Max "sleep_1ms is cleanest" finding reproduce across
  runs?** One run, one set of conditions. If reproducible, sleep_1ms
  becomes the recommended cadence for any single-kernel microbench
  on M4 Max — the inverse of the M1 Pro recommendation. Worth a
  second run with the same conditions.
- **Where does the M4 Max "transition zone" actually live, if at
  all?** A finer sweep around the 1ms-10ms decade (sleep_2ms,
  sleep_5ms) would tell us whether the transition just shifted to a
  different cadence band or genuinely flattened out.
- **What causes the rare large outliers at sleep_0 on M4 Max?**
  163 µs max with a 6.5 µs median — that's a 25× spike. On M1 Pro
  sleep_0, max was 84 µs / 8 µs median = 10×. The M4 Max
  worst-case is *worse* in absolute terms even though the typical
  case is better. Mechanism unclear.
- **Why is M4 Max sleep_1s noisier than M1 Pro sleep_1s?** The
  long-idle "cold settled" regime that M1 Pro had at cv=0.21 doesn't
  exist on M4 Max in this run. Could be macOS scheduler differences,
  could be different DVFS rest state, could be more frequent
  background activity contention.
- **Did the sleep_100ms drift from M1 Pro 002 reproduce on M4 Max?**
  **No.** M1 Pro showed +2875 ns drift in median between first-20 and
  last-20 samples (+25 %). M4 Max shows:

  | condition   | first-20 p50 | last-20 p50 | shift |
  |-------------|-------------:|------------:|------:|
  | sleep_0     | 6 688        | 6 458       |  -229 |
  | sleep_1ms   | 6 542        | 6 750       |  +208 |
  | sleep_10ms  | 8 917        | 9 042       |  +125 |
  | sleep_100ms | 8 896        | 9 146       |  +250 |
  | sleep_1s    | 8 875        | 9 125       |  +250 |

  All M4 Max conditions show shifts in [-250, +250] ns — well inside
  noise. The M1 Pro sleep_100ms +2875 ns drift was either G13-specific,
  macOS-26.3.1-specific, or a one-off run artifact. Doesn't reproduce
  on G16 / 26.4.1.
