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
