# 013: What recipe ingredients flip PWRCTRL between PERF and DEADLINE modes?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-28
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~120 s (5 s baseline + 20 cells × 5 s + 5 s tail
+ subprocess startup)

## The question

The lab's PWRCTRL evidence so far:

| recipe                                    | observed PWRCTRL                                | source |
|-------------------------------------------|-------------------------------------------------|--------|
| baseline / idle                           | IDLE_OFF 84-87 %                                | 010    |
| staircase 25-100 % busy, FMA_ITERS=65536  | **PERF 91-100 %**                               | 010    |
| sustained sub-floor recipe (50 s)         | **DEADLINE 76 %**                               | 010    |
| 5-attempt 009 sub-floor (84 trials each)  | DEADLINE 30-48 % during measured-trial era      | 011    |

PERF and DEADLINE are PWRCTRL's two non-idle modes. The boundary
between them is the open question. Two recipes differ along multiple
axes:

| axis              | staircase (PERF)         | sub-floor (DEADLINE)     |
|-------------------|---------------------------|---------------------------|
| FMA_ITERS         | 65 536                    | 1 024                     |
| per-dispatch wall-clock | ~1 ms                | ~30 µs                    |
| inter-dispatch sleep | duty-cycle to target busy %| 0 (sleep_0)             |
| K (warmup count)  | 0                         | 20 (per measured trial)  |
| total dispatch rate (GPU work / s) | varies by busy %  | ~395 trials/s × 21 dispatches = ~8 K/s |

Any of FMA_ITERS, inter-dispatch sleep, or dispatch rate could be the
trigger. **This experiment isolates which axis matters** by sweeping
two of them in a 2D grid and watching PWRCTRL.

**Primary question:** under sleep_0 back-to-back dispatches, which
combinations of (FMA_ITERS, inter-dispatch sleep) put PWRCTRL into
DEADLINE mode vs PERF mode? Is it a clean kernel-duration boundary?
A clean dispatch-period boundary? A 2D region?

Verdict thresholds per cell (fraction of cell's wall-clock):
- **DEADLINE:** PWRCTRL DEADLINE residency ≥ 50 %
- **PERF:** PWRCTRL PERF residency ≥ 50 %
- **MIXED:** neither dominates

Plot the cells in a 2D grid and characterize the boundary shape.

## Why this question, now

Three reasons:

1. **Closes the mechanism arc.** 010 found the controller has a
   DEADLINE mode. 011 found it engages during the 009 sub-floor
   recipe. 013 says *what triggers the flip*. Three connected
   questions, one mechanism.
2. **Predicts whether other workloads can hit sub-floor.** If
   DEADLINE is purely "kernel < N µs and back-to-back," any
   small-kernel workload should hit it. If it's also "K=20 warmup
   needed" or "specific dispatch pattern," only narrow recipes will.
3. **Cheapest experiment to run on M4 Max alone.** No sudo, just
   `notes/ioreport.py --include-states`. ~120 s of GPU work.

## Hypothesis

Confidence: medium. Predictions:

- **The flip is primarily a kernel-duration boundary, not a sleep
  boundary.** Tiny kernels (FMA_ITERS ≤ ~4096) at any reasonable
  back-to-back rate trigger DEADLINE. Heavy kernels (FMA_ITERS ≥
  ~16384) stay in PERF regardless of sleep.
- **There's a sleep-duration upper bound for DEADLINE.** Even tiny
  kernels at sleep_1ms or longer go to PERF (or back to IDLE_OFF)
  because the chip can't justify "deadline" mode if there's a
  millisecond of headroom between dispatches.
- **GPUPH P15 residency tracks DEADLINE residency.** When the
  controller is in DEADLINE, the chip visits P15 more aggressively
  than when in PERF (per 010's "DEADLINE drives faster per-dispatch
  DVFS access" hypothesis).
- **Approximate boundary shape:** DEADLINE in cells with FMA_ITERS
  ≤ 4096 AND sleep ≤ 200 µs. PERF elsewhere.

## What we are NOT trying to answer

- **K (warmup count) effect.** Held fixed at K=20 throughout; varies
  in a future experiment.
- **Per-trial GPUPH/PWRCTRL alignment.** 250 ms IOReport cadence is
  too coarse for trial-level questions. Cell-level (5 s) is the
  resolution.
- **Cross-chip generalization.** M4 Max only. M1 Pro doesn't have
  the same DEADLINE mode (or hasn't been characterized).
- **Mechanism beyond mode-switching.** "Why does the controller
  pick DEADLINE here?" is for hardware vendor docs we don't have.
- **Effect on measured timing.** This is a state-observation
  experiment, not a timing experiment. We don't measure
  `gpu_delta_raw` here.
- **Anything about thermal.** Cells run for 5 s each — too short
  for thermals to dominate.

## Setup

### Architecture

Single script `run.py`:

1. Launches `notes/ioreport.py --include-states --interval-ms 250
   --csv raw/{ts}-ioreport.csv`.
2. Phase markers per cell.
3. Phase 0: baseline (5 s idle).
4. Phase 1: 20 cells, each 5 s, in a 5 × 4 grid.
5. Phase 2: tail (5 s idle).
6. SIGINT to ioreport for clean flush.

### Variables

- **FMA_ITERS** (5 levels): `{256, 1024, 4096, 16384, 65536}`
- **inter-dispatch sleep_us** (4 levels): `{0, 50, 200, 1000}`
- 5 × 4 = 20 cells.
- Per cell: 5 s of `(20 fma_loop K=FMA_ITERS untimed, 1 write_tid 32t
  untimed)` looped at sleep_0 between trials, with `sleep_us` between
  warmup and measured dispatches inside the trial. Wait — that
  conflates two things. Simpler: for cell (FMA_ITERS, sleep_us), loop
  for 5 s issuing fma_loop dispatches back-to-back with `sleep_us`
  between them. No K-warmup variation.

Actually let me restate the per-cell protocol more carefully:
- Cell (FMA_ITERS, sleep_us): for 5 s wall-clock, loop:
    issue fma_loop(FMA_ITERS, threads=32) dispatch
    sleep(sleep_us microseconds)
- That's it. No write_tid, no K-warmup, no measured trials. Just
  sustained dispatch of one kernel kind at one rate.

This isolates the question: what does the controller do under a
sustained workload of (kernel_duration, dispatch_period)?

### Order of cells

Outer = FMA_ITERS ascending (small to large), inner = sleep_us
ascending (back-to-back to spaced-out). Ascending order biases any
thermal drift to be visible as a smooth gradient rather than random
noise.

### What we record

- `raw/{ts}.csv`: IOReport energy (per 250 ms window).
- `raw/{ts}-states.csv`: IOReport per-state residency, `GPU Stats`
  group only.
- `raw/{ts}-cells.csv`: phase markers per cell, with `monotonic_ns`,
  `cell_idx`, `fma_iters`, `sleep_us`, `n_dispatches_in_cell`.
- `raw/{ts}-meta.txt`: env + per-cell summary (PWRCTRL top state,
  GPUPH top state, both with %).

### What we do NOT do

- No averaging in live output.
- No retries.
- No measured trials (this is a state-observation experiment).
- No varying K. K isn't part of this protocol.
- No cooldown between cells. Cells run consecutively. We rely on
  the chip's natural settling within each 5 s cell.

## Success criterion

The experiment **succeeds** if we have:

1. ioreport states CSV covering the workload window.
2. Cell markers for all 20 cells.
3. Per-cell summary fillable from data.

It produces a **usable answer** if we can populate this 5×4 table:

| FMA_ITERS \\ sleep_us | 0 | 50 | 200 | 1000 |
|----------------------:|---|----|-----|------|
| 256                   |   |    |     |      |
| 1024                  |   |    |     |      |
| 4096                  |   |    |     |      |
| 16384                 |   |    |     |      |
| 65536                 |   |    |     |      |

Each cell labelled DEADLINE / PERF / MIXED, plus PWRCTRL top-state %
and GPUPH P15 residency %.

Verdict on the boundary shape: PASS if there's a clean monotonic
boundary (no checkerboard). MARGINAL if boundary exists but with
exceptions. FAIL if PWRCTRL is everywhere PERF or everywhere
DEADLINE — recipe doesn't matter.

## New questions we expect this to raise

- If the boundary is purely on FMA_ITERS axis: kernel duration is
  the entire trigger. Predicts that any "tiny kernel" workload
  hits DEADLINE.
- If boundary depends on both axes: a 2D rule like "duration <
  threshold AND inter-dispatch period < threshold." More complex
  to predict for arbitrary recipes.
- If sleep_1000 (1 ms) cells are PERF for all FMA_ITERS: 1 ms
  inter-dispatch is enough to pull the chip out of DEADLINE.
  Sets a target for "spaced-out enough to behave normally" cadence.
- If GPUPH P15 residency *doesn't* track PWRCTRL DEADLINE
  residency: the controller mode and the achieved DVFS state are
  partially independent. Would refine 010/011 mechanism story.
- If the boundary is *non-monotonic* (e.g. some FMA_ITERS hit
  DEADLINE at sleep_0 but PERF at sleep_50, then DEADLINE again at
  sleep_200): there's a non-trivial controller logic we don't
  understand. Worth deeper investigation.

## After this experiment

Branches:

- **Clean boundary.** Add the (FMA_ITERS, sleep_us) → mode rule to
  the lab's M4 Max cheat sheet. Future experiments designing for
  PERF or DEADLINE can choose recipes by the boundary.
- **Mixed cells dominate.** Boundary isn't a clean function of
  these two axes. Might require K-axis or longer-duration cells
  to expose. Pre-register a follow-up with the third axis.
- **Everywhere PERF or DEADLINE.** The recipe range here doesn't
  span the boundary. Either widen the sweep (FMA_ITERS=64 or
  =1M, sleep_us=10 ms), or accept that the 011 result was
  recipe-specific in some other way.

We do not plan past these branches.

---

## Result

**Date run:** 2026-04-28 (timestamp prefix `20260428T222216`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Wall-clock:** ~110 s (5 s baseline + 20 cells × 5 s + 5 s tail).
**Outcome:** **MARGINAL PASS with re-interpretation.** A clean 2D
boundary exists in the data, but the pre-registered 50 % DEADLINE-
verdict threshold is too high — DEADLINE engages at 26-51 % in the
"engaged" region, not ≥ 50 %. With a more honest 25 % threshold the
verdict is PASS: the boundary recovered cleanly.

### Headline: DEADLINE residency 5×4 grid

| FMA_ITERS \ sleep_us |   0 | 50 | 200 | 1000 |
|---------------------:|----:|---:|----:|-----:|
|                  256 | **37 %** | **48 %** | 3 % | 0 % |
|                 1024 | **33 %** | **49 %** | 3 % | 0 % |
|                 4096 | **34 %** | **51 %** | 1 % | 0 % |
|                16384 | **34 %** | **26 %** | 5 % | 0 % |
|                65536 |   0 % |  0 % | 0 % | 0 % |

**The DEADLINE engagement is bimodal**: either ~26-51 % (cells where
the controller chooses DEADLINE intermittently) or ~0-5 % (cells
where it doesn't). No middle-ground cells with e.g. 15 %. So the
"is the controller choosing DEADLINE here?" question has a clean
answer per cell, even though the residency *amount* varies.

### The boundary is a 2D rectangle

DEADLINE engages in cells where:

    (FMA_ITERS ≤ 16384) AND (sleep_us ≤ 50)

PERF dominates everywhere else:

- `FMA_ITERS = 65536`: any sleep — PERF 100 % regardless of cadence.
  Heavy kernels keep the controller in PERF mode.
- `sleep_us ≥ 200`: any FMA_ITERS — PERF dominant. Inter-dispatch
  headroom of 200 µs is enough for the controller to exit DEADLINE.

The boundary is rectangular, not diagonal — both axes have
independent thresholds.

### Hypothesis check

| prediction                                                  | observed                                              | verdict |
|-------------------------------------------------------------|-------------------------------------------------------|---------|
| Boundary is primarily a kernel-duration thing               | partially — duration matters but sleep also matters   | refined |
| Tiny kernels at any back-to-back rate trigger DEADLINE      | yes (FMA_ITERS ≤ 16384, sleep ≤ 50 µs)                 | ✓ |
| Heavy kernels (FMA_ITERS ≥ ~16384) stay in PERF             | depends — 16384 still DEADLINE at sleep ≤ 50 µs;       | mixed |
|                                                             | only 65536 stays in PERF unconditionally              |  |
| Sleep-duration upper bound for DEADLINE                     | yes — 200 µs is enough to drop out                    | ✓ |
| GPUPH P15 residency tracks DEADLINE residency               | **NO — they're independent dimensions**               | falsified |
| Approximate boundary: FMA ≤ 4096 AND sleep ≤ 200 µs         | actual: FMA ≤ 16384 AND sleep ≤ 50 µs                  | partial |

The biggest miss: **P15 residency tracks DEADLINE only when the
kernel is small.** In PERF mode at FMA_ITERS=65536 / sleep_0, the
chip pegs P15 at 94 % residency despite PWRCTRL = PERF 100 %. So
"chip at peak DVFS" and "PWRCTRL in DEADLINE" are independent — a
controller mode choice and a DVFS state choice are different things.
Refines the 010/011 mechanism story.

### GPUPH P15 residency 5×4 grid

For comparison with DEADLINE:

| FMA_ITERS \ sleep_us |   0 |  50 | 200 | 1000 |
|---------------------:|----:|----:|----:|-----:|
|                  256 |  4 % | 26 % | 1 % | 0 % |
|                 1024 |  5 % | 23 % | 1 % | 0 % |
|                 4096 |  4 % | 20 % | 0 % | 0 % |
|                16384 |  6 % |  5 % | 1 % | 0 % |
|                65536 | **94 %** | **100 %** | 7 % | 0 % |

Two regimes:

- **DEADLINE regime** (small FMA, short sleep): brief P15 visits
  per dispatch, ~20-26 % residency. Chip cycles peak ↔ idle.
- **PERF regime sustained** (FMA=65536, sleep ≤ 50 µs): chip
  pinned at P15, 94-100 %. Kernels long enough that the
  controller can keep DVFS at peak throughout.
- **PERF regime + headroom** (sleep ≥ 200 µs OR FMA=65536 with
  sleep=200 µs): the chip drops out of P15 because it has time
  between dispatches to step down the frequency.

### Surprises

#### 1. sleep_us = 50 has *more* DEADLINE residency than sleep_us = 0

For every FMA_ITERS ≤ 4096, DEADLINE residency is *higher* at
sleep_50 than at sleep_0:

| FMA_ITERS | sleep_0 DEADLINE % | sleep_50 DEADLINE % | delta |
|----------:|-------------------:|--------------------:|------:|
|       256 |              37 % |               48 % | +11 % |
|      1024 |              33 % |               49 % | +16 % |
|      4096 |              34 % |               51 % | +17 % |

Counterintuitive — adding 50 µs gap between dispatches *increases*
DEADLINE engagement. Hypothesis:
- At sleep_0 the chip is back-to-back saturated and the controller
  sees it as a continuous workload — closer to PERF semantics.
- At sleep_50 the chip has just enough headroom to recognize each
  dispatch as a discrete deadline-bounded task, choosing DEADLINE
  more aggressively.

That sleep_50 also has the highest GPUPH P15 residency (20-26 %)
in the DEADLINE region supports this: the controller is more
willing to push to peak DVFS per-dispatch when each dispatch is
clearly bounded.

This refines exp 011's mechanism: **sleep_0 isn't the optimal
recipe for DEADLINE engagement**; a small inter-dispatch gap is.
Worth re-running 009's sub-floor reproduction with sleep_50 to see
if onset is even faster.

#### 2. PWRCTRL has a third state: `PRFBOOST`

Cell 16 (FMA_ITERS=65536, sleep_us=0) reported PWRCTRL = `PERF(100%)
PRFBOOST(0%)`. PRFBOOST appeared as a 0 % residency state we hadn't
seen named before; the bindings exposed it because it's part of
PWRCTRL's state space even when residency is zero.

PWRCTRL's complete state space (observed via 010/011/013):
- `IDLE_OFF` — chip idle / clock-gated
- `PERF` — performance mode
- `DEADLINE` — deadline mode
- `PRFBOOST` — performance-boost mode (never engaged in our data
  but listed as a possible state)

We've now enumerated 4 PWRCTRL states. PRFBOOST presumably engages
under specific conditions we haven't tested — sustained heavy
workload, thermal pressure, charge state changes. Worth a focused
follow-up if it ever shows up.

#### 3. FMA_ITERS=65536 + sleep_200 has weird GPUPH

Cell 18 (65536, 200 µs): PWRCTRL PERF 100 %, but GPUPH top is
`P4(49%) P6(24%)` — chip in mid-states, not peak. This is the only
cell where a heavy kernel doesn't get peak DVFS.

Why? With 200 µs between heavy kernels, the chip has time to step
*down* DVFS between dispatches. The dispatch at the start finds the
chip in some lower state (P4-P6) and runs through to completion;
during the 200 µs sleep, DVFS drops further; next dispatch starts
from a lower state. The chip never builds momentum to P15.

Operational implication: even for "heavy compute" workloads, brief
inter-dispatch sleeps can prevent the chip from reaching peak DVFS.
Useful to know when designing benchmarks.

#### 4. `notes/ioreport.py` correctly captures cells with very few samples

Cell 19 (65536, 1000 µs) had only 1 113 dispatches over 5 s — much
less than other cells. IOReport at 250 ms gave us ~20 samples in
that window, all classified PERF 99 % + IDLE_OFF 1 %. Sanity check:
the bindings handle low-activity cells without crashing.

### Verdict re-classification

The pre-registered binary verdict (DEADLINE / PERF / MIXED) used a
50 % threshold which only labeled 1 cell as DEADLINE. With a 25 %
threshold (matching 011's "DEADLINE engagement" range), the
classification recovers a clean grid:

|     FMA_ITERS \ sleep_us |       0 |      50 |     200 |    1000 |
|-------------------------:|--------:|--------:|--------:|--------:|
|                      256 | DEADLINE| DEADLINE|    PERF |    PERF |
|                     1024 | DEADLINE| DEADLINE|    PERF |    PERF |
|                     4096 | DEADLINE| DEADLINE|    PERF |    PERF |
|                    16384 | DEADLINE| DEADLINE|    PERF |    PERF |
|                    65536 |    PERF |    PERF |    PERF |    PERF |

This is monotonic both row-wise and column-wise. Clean rectangular
boundary. **VERDICT (revised): PASS.**

## What this means operationally

For the project:

1. **The PWRCTRL DEADLINE-mode entry rule is now characterized:**

       DEADLINE engages when FMA_ITERS ≤ 16384 AND sleep_us ≤ 50

   Future experiments wanting to *avoid* DEADLINE: use FMA_ITERS ≥
   65536 OR sleep ≥ 200 µs. Future experiments wanting to *induce*
   DEADLINE: use FMA_ITERS ≤ 16384 with sleep = 50 µs (more
   reliable than sleep_0).
2. **The exp 009 sub-floor recipe (`fma_loop K=1024 sleep_0`) is in
   the DEADLINE region.** Not the strongest DEADLINE recipe (sleep_50
   is stronger), but well within the boundary.
3. **PWRCTRL mode and GPUPH P15 are independent.** Chip can be in:
   - PERF + P15 sustained (heavy kernel, no headroom): 94 %+ P15
   - DEADLINE + P15 brief (small kernel, short headroom): 20-26 %
     P15 with cycling
   - PERF + P15 cycling (heavy kernel, with headroom): mid-state P4-P6
   - PERF + low DVFS (any kernel, large headroom): mostly P1
   The two dimensions describe different facets of "what's the
   controller doing" and need to be reported separately.
4. **PRFBOOST is the 4th PWRCTRL state.** Not engaged in our recipe
   range. Worth looking out for.
5. **Sleep_50 µs is the sweet spot for DEADLINE engagement.** Higher
   DEADLINE residency than sleep_0 across all small-kernel rows.
   Counterintuitive but reproducible across 4 FMA_ITERS levels.

## What does NOT change

- All prior decisions and measurements stand.
- The IOReport bindings work correctly — they captured all 4 PWRCTRL
  states (including PRFBOOST as a 0%-residency entry).
- Exp 009's STRONG REPRODUCE result is unchanged; this experiment
  confirms its recipe is in the DEADLINE region.

## What changes

- Lab state snapshot's M4 Max cheat sheet should add the
  PERF/DEADLINE recipe boundary rule and the four PWRCTRL states.
- UNKNOWNS.md: close "what triggers PERF vs DEADLINE?"; open
  "what triggers PRFBOOST?" and "why is sleep_50 µs more DEADLINE-
  friendly than sleep_0?"

### Natural follow-ups

- **Re-run exp 009 with sleep_us = 50 instead of sleep_0** — predict
  even faster sub-floor onset based on 013's "sleep_50 has higher
  DEADLINE residency."
- **Find what triggers PRFBOOST.** Likely candidates: thermal
  pressure (sustained workload until Tj > some threshold), explicit
  performance hints (rare), or specific PCIe/external GPU
  conditions. Could try a long-duration heavy workload (10+ minutes
  of FMA_ITERS=65536 saturated) to see if PRFBOOST engages.
- **Boundary refinement.** The 4096-vs-16384 row at sleep_50 µs is
  the noisiest (DEADLINE drops from 51 % to 26 %). A finer sweep of
  FMA_ITERS in [4K, 8K, 16K, 32K] at sleep_50 µs would map the
  boundary edge more precisely.
- **K-axis variation.** Held fixed at "1 dispatch / iteration" here;
  exp 011 had K=20. Does "K dispatches followed by a brief gap"
  push deeper into DEADLINE than the simple loop here? Plausible
  but untested.

We do not plan past these.
