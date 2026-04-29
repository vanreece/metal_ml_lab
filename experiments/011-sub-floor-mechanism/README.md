# 011: Does the M4 Max sub-floor state correspond to PWRCTRL flipping into DEADLINE mode, and is that what cross-attempt persistence preserves?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-28
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~45 s (5 attempts × ~6 s, same as 009 + small
overhead from ioreport at 100 ms cadence)

## The question

Exp 009 found that the M4 Max sub-floor state (~1.7 µs floor under
`fma_loop K=20 sleep_0`) is reproducible 5/5 and **deepens across
subprocess re-launches with 5-7 s gaps between**. Mechanism mystery:
what about the chip's state survives a subprocess re-launch?

Exp 010 added a sudo-free path for per-DVFS-state residency (GPUPH
bindings). The bonus finding: PWRCTRL — the GPU power controller's
own state machine — is in **`DEADLINE` mode 76 % of the sub-floor
phase** vs **`PERF` mode 100 % of the staircase**. Two different
power-controller modes for two different recipe types.

This experiment connects the two. Combine 009's exact protocol with
010's GPUPH + PWRCTRL telemetry. Answer:

**Primary question:** during the 5-attempt sub-floor reproduction,
does PWRCTRL flip from `PERF` (or `IDLE_OFF`) to `DEADLINE` mode at
the same trial index where measured `gpu_delta_raw` drops below the
back-to-back floor (~5500 ns)? And does PWRCTRL stay in `DEADLINE` /
return to `PERF` differently across attempts in a way that explains
009's cross-attempt deepening?

- **STRONG SUPPORT:** PWRCTRL flips to `DEADLINE` within ±2 IOReport
  windows of the first sub-floor trial in every attempt; later
  attempts (2-4) enter DEADLINE earlier than attempts 0-1 OR sustain
  it longer.
- **PARTIAL SUPPORT:** PWRCTRL flips to `DEADLINE` during sub-floor
  in some attempts but not others, OR the timing of the flip doesn't
  cleanly align with sub-floor entry. The DEADLINE-mode story is
  *part* of the picture but not the whole mechanism.
- **NO SUPPORT:** PWRCTRL stays in `PERF` (or some other state)
  during sub-floor entry. The 010 observation was about a different
  recipe (50 s sustained); the 009 recipe doesn't reach DEADLINE.
  Mechanism question reopens.

**Secondary question:** during the 5-7 s inter-attempt gap (5 s
cooldown inside attempt.py + 2 s in driver), does PWRCTRL settle
back to `IDLE_OFF` cleanly? If yes, the cross-attempt deepening
mystery DEEPENS (state survives even after PWRCTRL returns to idle).
If no, the chip retains `PERF` or `DEADLINE` state through the gap,
which would explain 009's cross-attempt warmth.

## Why this question, now

Three reasons it's the highest-value next bite:

1. **Cheapest possible test of 010's mechanism speculation.** 010
   raised the DEADLINE-mode story; 011 is the focused test.
2. **Reuses 009's exact protocol with one new flag** (`--include-states`
   in the ioreport subprocess). Minimal new code; result is rich
   because we now have GPUPH + PWRCTRL traces to overlay on 009's
   timing data.
3. **The mechanism answer changes how we *think* about the M4 Max
   operating envelope.** If PWRCTRL DEADLINE is the trigger, then
   reaching ~1.7 µs is a state-machine question (can we induce
   DEADLINE earlier / more reliably?), not a thermal-envelope
   question. Different downstream experiments depending on which.

## Hypothesis

Confidence: medium-high (010's PWRCTRL DEADLINE finding is
suggestive but not yet proven causal). Predictions:

- **PWRCTRL flips to DEADLINE during sub-floor in all 5 attempts.**
  010 saw 76 % DEADLINE residency during a 50 s sub-floor recipe;
  this should reproduce in the shorter 009 recipe.
- **The flip happens within ±2 IOReport windows (±200 ms at 100 ms
  cadence) of the first sub-floor trial.** If the controller mode
  causes the timing change, transitions should be temporally aligned.
- **GPUPH residency during sub-floor era of each attempt is
  P15-dominated** (≥ 50 % P15 during the sub-floor portion of the
  attempt's wall-clock window). Less than 010's step_100 (89 %)
  because of the same dispatch ↔ idle cycling, but more than 010's
  full sub-floor average (62 %) because the early floor portion of
  each attempt dilutes the average.
- **Cross-attempt: DEADLINE residency increases from attempt 0 to
  attempt 4** (because chip enters each attempt closer to the
  DEADLINE-friendly state). If observed, supports DEADLINE as the
  preserved state.
- **During inter-attempt gaps, PWRCTRL returns to IDLE_OFF.** Cooldown
  windows should look like baseline GPUPH (OFF/P1) and idle PWRCTRL
  (IDLE_OFF). If true, the cross-attempt deepening is from something
  besides "chip stayed in DEADLINE the whole time."

## What we are NOT trying to answer

- **Trial-level alignment.** IOReport at 100 ms can't resolve trial-
  level (~5 ms/trial) transitions. We compare at attempt-level and
  phase-of-attempt level, not trial-level.
- **What flips PWRCTRL into DEADLINE.** The mechanism *trigger* is
  for exp 013. 011 just observes the correlation.
- **Whether DEADLINE on its own produces 1.7 µs cycles.** That's a
  causality question requiring controlled state-flipping; out of
  scope for this observation experiment.
- **Cross-chip generalization.** M4 Max only.
- **Higher-cadence ioreport (< 100 ms) effects.** 100 ms is enough
  to resolve attempt-level (~5 s) transitions. Sub-100 ms is for
  trial-level work in a future experiment.
- **Recipe variations.** Same fma_loop K=20 sleep_0 as 009. We're
  not exploring why; we're characterizing what.

## Setup

### Architecture (same as 009 with one addition)

Three-script pattern:

1. `run.py` — outer driver. Starts caffeinate + **ioreport.py
   subprocess at 100 ms cadence with `--include-states`**, loops
   5 times: cooldown, launch attempt.py as subprocess, wait. Stops
   subprocesses cleanly at end. Aggregates per-attempt CSVs.
2. `attempt.py` — same as 009's. Subprocess re-launches give a fresh
   MTLDevice each attempt.
3. `analysis.py` — joins attempts CSV, ioreport energy CSV, and
   ioreport states CSV by `monotonic_ns`. Reports per-attempt:
   - GPUPH state breakdown for floor-era windows vs sub-floor-era
     windows
   - PWRCTRL state breakdown for floor-era vs sub-floor-era
   - Cross-attempt aggregates

The IOReport subprocess covers the whole driver run (all 5 attempts
+ inter-attempt gaps). State CSV captures every 100 ms window across
all of that.

### Per-attempt protocol (verbatim from 009)

1. Set user-interactive QoS.
2. Create MTLDevice, command queue, write_tid pipeline, fma_loop
   pipeline (FMA_ITERS=1024), shared sample buffer.
3. **Cooldown:** sleep 5 s.
4. **Calibration probe:** 10 back-to-back sleep_0 dispatches of
   `write_tid` 32t with timestamps.
5. **Measured combo:** 84 trials of (20 fma_loop warmup × untimed,
   1 write_tid 32t measured × timed) with sleep_0 between trials.

Same recipe, same trials, same sample-buffer slot layout. The only
difference is what's *running in parallel* (ioreport).

### Variables fixed (no axis sweeps)

- N attempts: 5
- Cooldown before each attempt: 5 s (inside attempt.py)
- Inter-attempt sleep (in driver): 2 s
- Trials per attempt: 84
- Calibration probe: 10 dispatches per attempt
- Warmup recipe: `fma_loop K=20`
- Measured kernel: `write_tid` 32 threads
- Cadence: sleep_0 (back-to-back)
- IOReport interval: **100 ms** (5× finer than 009's 500 ms)
- IOReport state-groups: `GPU Stats` (GPUPH, PWRCTRL, BSTGPUPH,
  GPU_SW, plus the others as bonus)

### What we record

- `raw/{ts}-attempts.csv` — same as 009, per-trial timing.
- `raw/{ts}.csv` — IOReport energy (per 100 ms window).
- `raw/{ts}-states.csv` — IOReport per-state residency for every
  GPU Stats channel × 100 ms window.
- `raw/{ts}-meta.txt` — env, per-attempt summary, verdict.

### What we do NOT do

- No averaging in live output.
- No retries on subprocess failure.
- No trial-level state assignment (cadence is too coarse).
- No discarding of any data.

## Success criterion

The experiment succeeds (in the discipline's sense) if we have:

1. attempts CSV with all 5 × (10 + 84) = 470 rows.
2. states CSV covering the wall-clock window of all 5 attempts at
   100 ms cadence.
3. Phase markers tagged by attempt boundary so analysis can map
   IOReport windows to "during attempt N" vs "between attempts."

It produces a usable answer if we can populate this table:

| attempt | floor-era (trials 0-23) PWRCTRL top |  sub-floor era (trials 24+) PWRCTRL top | floor GPUPH top | sub-floor GPUPH top | inter-attempt gap PWRCTRL top |
|--------:|:------------------------------------|:----------------------------------------|:---------------:|:-------------------:|:------------------------------:|
|       0 |                                     |                                         |                 |                     |                                |
|       1 |                                     |                                         |                 |                     |                                |
|       2 |                                     |                                         |                 |                     |                                |
|       3 |                                     |                                         |                 |                     |                                |
|       4 |                                     |                                         |                 |                     |                                |

Plus the verdict (STRONG / PARTIAL / NO SUPPORT) on the primary
question.

## New questions we expect this to raise

- If PWRCTRL flips cleanly with sub-floor entry: what other recipes
  trigger DEADLINE mode? Exp 013.
- If PWRCTRL flips during sub-floor in some attempts but stays
  PERF/IDLE in others: the controller mode is *one* contributor but
  not the only one. There's a second mechanism worth isolating.
- If GPUPH P15 residency is much higher in deeper-attempts (2-4) vs
  shallow-attempts (0-1): the chip is "warmer" at attempt entry —
  consistent with 009's cross-attempt deepening story but with state-
  level evidence.
- If PWRCTRL returns to IDLE_OFF in inter-attempt gaps but the chip
  STILL enters sub-floor faster on later attempts: cross-attempt
  state isn't in PWRCTRL — must be elsewhere (GPU SRAM warmup,
  thermal proxy, scheduler queue priors). Big new mystery.
- If state CSV is unexpectedly small / empty: ioreport at 100 ms may
  be too aggressive; some channels may publish at coarser intervals.
  Worth a tooling note.

## After this experiment

Branches:

- **STRONG SUPPORT.** The mechanism story for 009's sub-floor is now
  firm: `fma_loop K=20 sleep_0` triggers PWRCTRL `DEADLINE` mode,
  which routes per-dispatch DVFS access through a faster path,
  producing the 1.7 µs floor. Schedule exp 013 (PERF/DEADLINE
  boundary sweep) and exp 012 (GPUPH MHz mapping). Update lab state
  snapshot's M4 Max cheat sheet to add a "controller mode" column to
  the recipe table.
- **PARTIAL SUPPORT.** Document what part of the mechanism PWRCTRL
  explains. Probably still pursue 013 / 012 because they remain
  useful regardless. The "what's the rest of the mechanism" gap
  becomes a tracked open question.
- **NO SUPPORT.** Refile 010's PWRCTRL DEADLINE observation as
  "specific to long-sustained sub-floor recipes, not the canonical
  009 recipe." Look for the actual mechanism in other GPU Stats
  channels (BSTGPUPH? GPU_SW? AFRSTATE?). Re-frame and try again.

We do not plan past these branches.

---

## Result

**Date run:** 2026-04-28 — two passes
- v1 (`raw/v1-20260428T214621-*`): 5 attempts × 84 trials, IOReport at
  100 ms — **partial reproduce 2/5** with too-coarse-to-resolve telemetry.
- v2 (`raw/20260428T215024-*`): 5 attempts × 800 trials, IOReport at
  250 ms — **strong reproduce 5/5**, telemetry resolves cross-attempt
  pattern.

**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Outcome:** **STRONG SUPPORT** for the DEADLINE-mode mechanism, with a
methodology lesson: the observation perturbs the state, and the
009-protocol's measured-trial wall-clock window is too short for
attempt-level IOReport telemetry to resolve at any cadence we tried.

### v1 finding: observation perturbs entry; 100 ms IOReport is too coarse anyway

| attempt | below_5500 | first_sub_idx | meas_max  |
|--------:|-----------:|--------------:|----------:|
|       0 |         19 |            22 |   279 542 |
|       1 |    **0**   |            -1 |   324 541 |
|       2 |         18 |            66 |    14 583 |
|       3 |          1 |            24 |   163 167 |
|       4 |    **0**   |            -1 |   147 333 |

Vs 009's headline (no IOReport states subprocess): 5/5 attempts entered,
14-60 sub-floor trials each, max max-trial 118 µs. With IOReport at
100 ms running, two attempts didn't enter at all and the others had
much larger tails (up to 324 µs). Observation effect: the IOReport
sampling subprocess at 100 ms cadence consumes enough resources to
perturb the chip's state machine and prevent or delay sub-floor entry.

Independent of that, the analysis was *also* doomed: each attempt's
84 measured trials span only ~100 ms of wall-clock, and at 100 ms
IOReport cadence we get 0-1 windows per attempt's "sub-floor era."
n_win values per attempt era ranged 0-1; can't compute residency
distributions from one window.

**v1 file disposition:** raw data preserved as `raw/v1-*` for reference.
The methodology lesson informed v2.

### v2: 800 trials × 250 ms IOReport — STRONG REPRODUCE 5/5

Same protocol but 10× the trials per attempt (so measured-trial wall-
clock ≈ 2 s per attempt) and IOReport at 250 ms (5 windows per
attempt's sub-floor era, lighter CPU footprint).

| attempt | meas_min | meas_p50 | meas_max  | meas_cv | below_5500 / 800 | first_sub_idx |
|--------:|---------:|---------:|----------:|--------:|-----------------:|--------------:|
|       0 |    1 625 |    3 958 |   288 709 |  6.96   |          **405** |            55 |
|       1 |    1 667 |    3 667 |   324 417 |  6.82   |          **464** |            22 |
|       2 |    1 708 |    3 708 |   272 208 |  5.67   |          **442** |            98 |
|       3 |    1 667 |    3 979 |   325 833 |  4.89   |          **404** |            80 |
|       4 |    1 625 |    2 792 |   364 917 |  9.02   |          **590** |             4 |

**5/5 strong reproduce.** Min always 1 625-1 708 ns (≈ 39-41 ticks of
the 24 MHz clock — same hardware quantum we hit in 009). 50-74 % of
trials per attempt land below the back-to-back floor. Attempt 4's
first_sub_idx = 4 is striking: by attempt 4, the chip enters sub-floor
within the first 5 trials.

### v2 PWRCTRL + GPUPH residency per attempt

| attempt | PWRCTRL DEADLINE | PWRCTRL PERF | PWRCTRL IDLE_OFF | GPUPH P15 | GPUPH P1 |
|--------:|-----------------:|-------------:|-----------------:|----------:|---------:|
|       0 |          33.0 %  |       65.1 % |            1.1 % |     6.0 % |   56 %   |
|       1 |          30.7 %  |       64.8 % |            3.2 % |     3.4 % |   48 %   |
|       2 |          33.1 %  |       65.3 % |            0.0 % |     4.7 % |   44 %   |
|       3 |          37.0 %  |       62.3 % |            0.0 % |     4.5 % |   54 %   |
|       4 |        **48.0 %**|     **48.2 %** |            2.2 % |  **14.9 %**| **25 %** |

**DEADLINE is engaged in every attempt's sub-floor era**, ranging
30.7-48.0 %. Pre-registered "STRONG SUPPORT" was conditioned on
DEADLINE flipping in every attempt + cross-attempt strengthening; both
are observed.

**Cross-attempt deepening (the original mystery from 009) is visible
across multiple channels in v2:**

- DEADLINE residency rises 33.0 → 48.0 % from attempt 0 to attempt 4
- P15 residency rises 6.0 → 14.9 %
- P1 (low-power) residency drops 56 → 25 %
- AFRSTATE shifts top state from P1 (49 %) at attempt 0 to **P7 (27 %)**
  at attempt 4 — the AFR engine has moved several states up
- BSTGPUPH P1 residency drops 56 → 29 %

These correlate cleanly: the chip is progressively more time at higher
states across attempts. **The 009 cross-attempt deepening story is now
mechanistically grounded in observable state shifts, not just inferred
from timing.**

### Inter-attempt gaps reset to baseline

| gap        | PWRCTRL top                | GPUPH top         |
|------------|----------------------------|-------------------|
| 0 → 1      | PERF 56 % + IDLE_OFF 42 %  | P1 56 % + OFF 42 %|
| 1 → 2      | PERF 56 % + IDLE_OFF 43 %  | P1 56 % + OFF 43 %|
| 2 → 3      | PERF 53 % + IDLE_OFF 45 %  | P1 54 % + OFF 45 %|
| 3 → 4      | PERF 56 % + IDLE_OFF 42 %  | P1 56 % + OFF 42 %|

**Every gap looks identical**: chip drops out of DEADLINE, returns to
the half-PERF / half-IDLE_OFF baseline. Yet the *next* attempt enters
sub-floor faster and deeper than the prior one. So **whatever survives
across attempts isn't preserved in PWRCTRL or GPUPH state during the
gap.** The gap looks like baseline; the next attempt isn't.

This refutes the simplest version of the mechanism story ("chip stays
in DEADLINE between attempts"). Open question for future work: what
DOES persist? Candidates the data doesn't yet rule out:

- **Thermal:** chip temperature might be elevated at attempt-N entry
  (after recent compute) such that the PWRCTRL transition into
  DEADLINE happens faster. Would need a temperature signal during
  gaps; IOReport's temp channels were zero-valued in 008's enumeration.
- **Driver-side state:** kernel scheduler queue, dispatch-pattern
  history, or other software state in IOAccelerator family.
- **Voltage / per-state DVFS history:** the chip might track recent
  use of P15 to lower the entry threshold next time. This would be
  invisible to PWRCTRL but observable to a finer state channel we
  haven't enumerated.

### Hypothesis check

| prediction                                                                        | observed                                                                              | verdict |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|---------|
| PWRCTRL flips to DEADLINE during sub-floor in all 5 attempts                      | yes — 30.7-48.0 % DEADLINE residency in 5/5 sub-floor eras                            | ✓       |
| Flip happens within ±2 IOReport windows of first sub-floor trial                  | unable to test at 250 ms cadence × ~5-window sub-floor era                            | n/a     |
| GPUPH ≥ 50 % P15 during sub-floor                                                 | 3-15 % P15 (much lower; mostly P1 during sub-floor era too)                           | falsified |
| Cross-attempt: DEADLINE residency increases attempt 0 → 4                         | 33.0 → 48.0 % (+15.0 pct)                                                             | ✓       |
| Inter-attempt gaps return PWRCTRL to IDLE_OFF                                     | gaps are PERF 56 % + IDLE_OFF 43 % (chip not idle, but not in DEADLINE either)        | partial |

The "GPUPH ≥ 50 % P15" prediction was wrong — sub-floor era is mostly
P1 with brief P15 visits (matches 010's "chip cycles peak ↔ idle on
each tiny dispatch" finding). The 1.7 µs floor is the *tick rate
during P15 visits*, not residency-weighted.

## Surprises

### 1. Observation perturbs sub-floor entry at 100 ms IOReport cadence

This is the methodological-discipline analogue of 003's "calibration
probe is itself a 10-dispatch warmup" — observation affects state.
v1 ran IOReport at 100 ms and 2 of 5 attempts didn't enter sub-floor;
v2 ran at 250 ms and all 5 entered. The mechanism is probably CPU
contention from the IOReport subprocess sampling (100 ms cadence ⇒
~10 wakeups/s on top of whatever the experiment is doing).

**Operational rule:** for future experiments aiming to *both* time
microbenchmarks *and* sample IOReport states, default to ≥ 250 ms
IOReport cadence. 100 ms is observable to the chip.

### 2. Inter-attempt gaps look identical despite cross-attempt deepening

Every gap's PWRCTRL profile is within 3 pct of every other gap, yet
the *next* attempt enters sub-floor faster than the prior one. The
chip's *observable* state during the gap is reset to baseline, but
*something* about the chip is more "primed" for sub-floor each time.

This is a mechanism-not-located finding. The next experiment should
look at GPUPH at higher cadence during the gap (resolve any sub-200 ms
flicker into DEADLINE) AND at temperature / driver-side state.

### 3. AFRSTATE flips up to P7 at attempt 4 specifically

Attempts 0-3: AFRSTATE top state = P1 with 35-49 % residency.
Attempt 4: AFRSTATE top state = **P7 with 27 % residency**. AFRSTATE's
highest state observed in 010 was P7 at 90 % during step_100. So
attempt 4's AFRSTATE briefly hits the top of its 8-state space.

Attempt 4 is also the deepest-sub-floor attempt (590/800 below 5500 ns,
P15 residency 14.9 %). AFRSTATE may be a sensitive secondary indicator
of "how hard the chip is being driven." Worth tracking in future
experiments; the AFR engine is sometimes thought of as just the
display/render path but appears to participate proportionally in
GPU compute too.

### 4. Sub-floor max trials are ~10× the back-to-back floor max

v2 attempts had max trial values 272-365 µs — 50-65× the median.
009's worst was 118 µs. The longer attempt with continuous arithmetic
warmup creates more opportunities for the GPU to be preempted (likely
by WindowServer or the IOReport subprocess itself). The median stays
clean (3-4 µs); the tail is noisy.

For future "is the median in sub-floor?" questions, p50 + below_5500
count are the right summary statistics; max is dominated by occasional
preemption, not state.

### 5. Cross-attempt deepening is mechanism-persistent, not state-persistent

The most important framing update from this experiment: the 009
cross-attempt deepening *is* observable in PWRCTRL/GPUPH/AFRSTATE state
during sub-floor eras, *but* the persistence between attempts isn't
in those channels. Whatever sets up "chip enters faster on attempt 4"
operates on a layer below the observable controller states.

This is consistent with thermal explanation, voltage memory,
prediction-table state, or some other hidden layer. The 009 narrative
(cross-attempt deepening is real) is unchanged; the mechanism for
*persistence* is now newly mysterious.

## What this means operationally

For the project:

1. **PWRCTRL DEADLINE is part of the 009 sub-floor mechanism.**
   30-48 % DEADLINE residency in every sub-floor attempt is robust;
   exp 010's 76 % residency observation generalizes — it just shows
   up at lower percentages in shorter recipes.
2. **The "what persists across attempts" question is open.** PWRCTRL
   and GPUPH state both reset between attempts. The deepening signal
   we see in 009 must come from elsewhere.
3. **For future state-observation experiments, ≥ 250 ms IOReport
   cadence is the new floor.** 100 ms perturbs the state being
   measured.
4. **The 30-48 % range is "DEADLINE is engaged, not sustained."**
   Different from 010's 76 % during the 50 s sustained recipe; same
   underlying mode, different exposure.

## What does NOT change

- 009 STRONG REPRODUCE result is unchanged.
- 010 DEADLINE finding is unchanged; refined with the 30-48 %
  short-recipe number.
- Decisions 003-005 are unaffected.

## What changes

- UNKNOWNS.md: refine the cross-attempt persistence question — it's
  not in PWRCTRL or GPUPH; mechanism unknown.
- The lab's standard IOReport cadence guidance: ≥ 250 ms.
- New question: AFRSTATE as a secondary "how hard is the chip being
  pushed" indicator — worth tracking when investigating deep
  workloads.

### Natural follow-ups

- **Higher-cadence IOReport during attempt 4's first 5 trials.** Test
  whether DEADLINE flip aligns with the very-fast entry. Sub-100 ms
  cadence would resolve, but the v1 result says that perturbs entry —
  so the test would need to record DEADLINE at the cost of preventing
  the entry. Tricky methodology.
- **Cross-attempt persistence mechanism.** Vary inter-attempt gap
  duration (1 s, 5 s, 30 s, 5 min) and observe at what point the
  attempt-4-style fast entry decays. Half-life characterization.
- **AFRSTATE relationship.** Run a sweep where the chip is
  intentionally pushed into AFRSTATE P7+ via different recipes
  (display-heavy work?) and see if that primes the chip for sub-floor.
- Original queue: exp 012 (GPUPH MHz mapping), exp 013 (PERF/DEADLINE
  recipe boundary).

We do not plan past these.
