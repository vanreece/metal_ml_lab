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
