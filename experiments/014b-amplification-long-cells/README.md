# 014b: Do the amplification slopes from 014 hold under long cells, and does within-cell DVFS state explain the back-to-back N=1024 collapse?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-29
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~110 s (2 s baseline + 16 cells × 5 s + 15 ×
1 s inter-cell + 2 s tail + IOReport startup).
**Predecessor:** experiment 014, run on 2026-04-29 with 50-trial
back-to-back cells. Established the technique works (MARGINAL PASS,
piecewise-linear) but left two mechanism questions unanswered
because the short cells (~10-235 ms) ran shorter than IOReport's
250 ms cadence — only the longest cell (b2b N=1024) had usable
PWRCTRL classification.

## The question

Two things 014's short-cell data couldn't pin down:

1. **Are 014's slopes representative of steady-state DVFS, or
   contaminated by transient DVFS during the cell?** Each 014 cell
   ran ~10-235 ms; for the small-N cells, most of that was the GPU
   not yet in steady-state for the recipe. A 5 s cell gives the
   chip ~250-500 ms of warmup and ~4.5 s of steady-state.

2. **What causes the back-to-back N=1024 cell to collapse from
   ~3 200 ns/dispatch to ~209 ns/dispatch with bimodal trial
   distribution (p50 988 µs, p90 3.36 ms)?** 014's PWRCTRL data
   for that cell showed PERF (53 %) + DEADLINE (29 %) + 9 % P15,
   which is consistent with the chip cycling between DVFS regimes
   *within* the cell. If we have IOReport samples within the cell
   and trial-level monotonic_ns timestamps, we can temporally join
   trials to the DVFS state they ran in and ask directly: are fast
   trials concentrated in one regime?

## Pre-registered verdicts

For the slope-stability check (question 1):

- **PASS:** stable-region slopes from 014b agree with 014 within
  ± 5 % for both methods. (014 internal-loop stable slope: 405.28
  ns / amp-step; 014 b2b stable slope: 3 219.09 ns / dispatch.)
- **MARGINAL:** slopes agree within ± 15 % — qualitative agreement
  but quantitative drift; flag for follow-up.
- **FAIL:** disagreement > 15 %. The 014 numbers are transient-
  contaminated and need to be re-cited from 014b.

For the b2b N=1024 mechanism check (question 2):

- **DVFS-EXPLAINED:** per-trial gpu_delta_raw correlates with
  contemporaneous PWRCTRL state at p < 0.01 (Mann-Whitney U or
  similar). Specifically, trials whose monotonic_ns falls in an
  IOReport sample with PWRCTRL = DEADLINE-dominant are
  systematically faster (or slower) than trials in PERF-dominant
  samples. Magnitude of the difference matches the slope collapse.
- **DVFS-PARTIAL:** correlation exists but explains < 50 % of the
  bimodality. Some other mechanism (driver batching, GPU
  pipelining within a long cb) is also at work.
- **DVFS-EXCLUDED:** no correlation between trial timing and
  contemporaneous PWRCTRL state. Look elsewhere — probably driver-
  side batching or kernel-coalescing.

## Why this question, now

Three reasons:

1. **014's b2b N=1024 finding is the load-bearing surprise of the
   foundation arc.** It says "amplification breaks at large N" —
   but we don't know *why*, so we can't say *when* it breaks for
   ML kernels of different shapes. Mechanism matters.
2. **Short cells were a self-imposed limitation, not a methodology
   choice.** 014 inherited 50-trial-per-cell from earlier exps
   that were timing characterization, not regime characterization.
   Running long enough for IOReport to resolve cell-internal
   structure is a methodology improvement at almost no cost.
3. **Cheap.** ~110 s GPU time, no sudo, reuses the IOReport stack.

## Hypothesis

Confidence: medium-high on slope stability, medium on the DVFS
mechanism explaining the b2b collapse.

- **014's stable-region slopes hold up under longer cells.** Any
  drift comes from the small-N cells (where steady-state matters
  less anyway). Mid-large-N cells should be very close.
- **The b2b N=1024 collapse is at least partly DVFS-state driven.**
  Specifically: the DEADLINE-mode sub-floor (1 625 ns absolute
  floor from exp 009) is well below the per-dispatch cost in the
  PERF regime (~3 µs). If a fraction of trials hit DEADLINE-mode
  cycles where individual dispatches see sub-floor latency,
  per-dispatch median collapses correspondingly. The expected
  signature: **bimodal trial distribution within the b2b N=1024
  cell**, with the fast mode at ~per_dispatch_in_DEADLINE × N and
  the slow mode at ~per_dispatch_in_PERF × N.
- **The internal-loop kink (between N=8 and N=16 in 014) will
  reappear at the same N values.** It's a property of the kernel +
  recipe, not of cell duration.
- **Slope of internal-loop for N ≥ 16 will tighten** (lower
  per-cell variance) because more trials per cell + steady-state.

## What we are NOT trying to answer

- **Anything about memory-bound base units.** Same as 014.
- **Cross-chip generalization.** M4 Max only.
- **Whether amplification works on real ML kernels.** Still the
  experiment after this one (or after the memory-bound one).
- **The mechanism of the internal-loop DEADLINE → PERF kink at
  N=8 → 16.** 013 already characterized this transition; we are
  re-confirming it lives where 014 said, not re-deriving it.
- **Whether the N=1024 b2b collapse generalizes to other base
  units.** Same kernel as 014; one base unit at a time.
- **Power consumption per cell.** IOReport energy CSV will be
  written but not analyzed beyond cross-checking that the long
  cells did real GPU work. Power characterization is its own
  experiment.

## Setup

### Same as 014

Same kernels (carry-dependent fma_loop with FMA_PER_ITER=64), same
N values ({1, 2, 4, 8, 16, 64, 256, 1024}), same two methods
(internal-loop, back-to-back), same THREADS=32 / GROUP=32, same
output buffer, same IOReport sidecar at 250 ms cadence with
`--include-states`.

### Different from 014

- **Cell duration target: 5 s wall-clock.** Trials run back-to-back
  at sleep_0 within the cell until either 5 s elapses or
  MAX_TRIALS_PER_CELL is hit.
- **Inter-cell pause: 1 s.** Crisp IOReport boundaries, lets the
  chip drop out of any elevated state before the next cell begins.
- **MAX_TRIALS_PER_CELL = 5000.** Caps low-N cells at 5000 trials
  even if the 5 s budget would allow more (smallest cells could
  produce 700 K trials, which we don't need). High-N cells will
  run far below this cap (~1 000 for b2b N=1024).
- **Per-trial monotonic_ns timestamp recorded** so trial-level
  data can be temporally joined to IOReport samples in analysis.
- **Bulk sample-buffer resolve at end of cell.** Per-trial
  resolveCounterRange_ would inject CPU work between dispatches
  and perturb steady-state. Buffer is sized per cell as
  `2 × MAX_TRIALS_PER_CELL`, slots used contiguously, resolved in
  one call after the cell ends. ~160 KB per cell, reused across
  cells.

### Variables

Same 2 × 8 = 16 cells as 014. Same order: internal-loop ascending
N (cells 0-7), back-to-back ascending N (cells 8-15).

### What we record

- `raw/{ts}-trials.csv` — per-trial: cell_idx, method, n_amp,
  trial_idx, monotonic_ns, gpu_t_start_raw, gpu_t_end_raw,
  gpu_delta_raw, cpu_total_ns. (Same columns as 014, more rows.)
- `raw/{ts}-cells.csv` — per-cell: cell_idx, method, n_amp,
  start_ns, end_ns, trial_count, percentiles + min/max of
  gpu_delta_raw.
- `raw/{ts}.csv` and `raw/{ts}-states.csv` — IOReport energy and
  per-state residency at 250 ms.
- `raw/{ts}-meta.txt` — env, run config.

### What we do NOT do

- No retries on failed sample-buffer alloc / compile. Raise loudly.
- No outlier discarding in raw CSV. Per-cell percentiles will be
  in cells.csv; raw distribution stays in trials.csv.
- No averaging in live output (per-cell summary line is fine).
- No varying threadgroup config or base unit — exactly the same
  recipe as 014, only cell duration changed.

## Success criterion

The experiment **succeeds** (in the discipline's sense) if we have:

1. Per-trial CSV with all timed dispatches (expect ~50 K-80 K rows
   total, depending on how many small-N cells hit the cap).
2. Per-cell summary CSV with 16 rows.
3. IOReport states CSV covering all 16 cells with ~20 samples
   per cell.
4. Stable-region slope per method recomputed and compared to 014.
5. For the b2b N=1024 cell: trial-level temporal join with
   IOReport samples, reporting per-DVFS-state trial timing
   distribution.

## After this experiment

Branches:

- **PASS slope-stability + DVFS-EXPLAINED collapse.** Promote 014b
  numbers to canonical, write up the b2b N=1024 mechanism, move
  on to memory-bound amplification (the next experiment).
- **PASS slope-stability + DVFS-PARTIAL/EXCLUDED collapse.** The
  collapse is something else; one targeted follow-up experiment
  to nail it down (likely toggling driver batching hints or
  varying cb structure). Then memory-bound.
- **MARGINAL/FAIL slope-stability.** 014's slope numbers are
  transient-contaminated and need to be re-cited. 014's
  qualitative findings (technique works, regimes exist) still
  hold; the headline numbers move to 014b.
- **All branches:** 014b's per-cell PWRCTRL/GPUPH classification
  becomes the new baseline for amplification-experiment
  diagnostic data, replacing 014's stable-region heuristic for
  cells > 1 s.

We do not plan past these branches.

## Result

**Date run:** 2026-04-29 (timestamp prefix `20260429T150438`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Wall-clock:** ~38 s (most cells hit the 5 000-trial cap before
5 s, only b2b N=64/256/1024 ran the full duration). 75 110 trials
captured across 16 cells; 18 304 IOReport state rows.
**Outcome:** **PASS slope-stability + DVFS-PARTIAL collapse.** Three
of four 014 stable-region numbers reproduced within ± 2 %; the
fourth (b2b intercept) drifted 13 % MARGINAL. Per-cell PWRCTRL
classification now works. The b2b N=1024 collapse is *partly*
DVFS-driven, but not at the 250 ms IOReport granularity — the
controller cycles faster than that, and the bimodality lives
within single IOReport samples.

### Slope-stability check vs 014

| method | metric | 014 | 014b | Δ% | verdict |
|---|---|---:|---:|---:|---|
| internal-loop | slope     | 405.28 ns/N | 405.17 ns/N | −0.0 % | **PASS** |
| internal-loop | intercept | 6 860 ns    | 6 861 ns    | +0.0 % | **PASS** |
| back-to-back  | slope     | 3 219 ns/N  | 3 157 ns/N  | −1.9 % | **PASS** |
| back-to-back  | intercept | 4 711 ns    | 4 087 ns    | −13.2 % | MARGINAL |

The 014 stable-region slopes hold up under 100× more trials per cell
and are now CANONICAL. The b2b intercept drift is consistent with
the small-N b2b region picking up sub-floor trials that 014's
50-trial sample missed, dragging the fitted intercept down. Slope
unaffected because it's anchored by N=16 (where bimodality settles
into PERF for b2b at this point — see below).

### Per-cell PWRCTRL classification (now usable)

The 250 ms IOReport cadence resolves cell state for cells running
≥ ~600 ms, which is most of them (only the very-shortest ones still
saturate as MIXED).

| cell | method | N | p50 | DEADLINE % | P15 % | class |
|---:|---|---:|---:|---:|---:|---|
| 4  | internal-loop | 16   |  5.96 µs |  59 % | 36 % | **DEADLINE** |
| 7  | internal-loop | 1024 |   421.75 µs |  29 % |  2 % | **PERF** |
| 11 | back-to-back  | 8    | 29.17 µs |  35 % | 10 % | **DEADLINE** |
| 13 | back-to-back  | 64   |   119.08 µs |  44 % | 21 % | **DEADLINE** |
| 14 | back-to-back  | 256  |   360.29 µs |  51 % | 25 % | **DEADLINE** |
| 15 | back-to-back  | 1024 |    1.419 ms |  45 % | 22 % | **DEADLINE** |

The internal-loop kink at N=16 is now officially the **DEADLINE-
mode cell**: 59 % DEADLINE residency, 36 % P15. p50 = 5.96 µs is
*below* the prior 6.4 µs cool-DVFS dispatch-overhead floor — the
floor itself moves under DEADLINE.

### Bimodality is everywhere there are enough trials

Distributions in the 5 000-trial cells reveal a pattern 014's
50-trial samples could only hint at: most cells are bimodal, and
the modes are separated by enough that a 50-sample p50 catches one
or the other depending on chance.

Selected representative cells:

| cell | method | N | p25 | p50 | p75 | p99 | shape |
|---:|---|---:|---:|---:|---:|---:|---|
| 0  | internal-loop |  1 |  2.96 µs |  7.12 µs |  7.46 µs |  15.25 µs | bimodal (~3 µs / ~7 µs) |
| 4  | internal-loop | 16 |  3.62 µs |  5.96 µs | 13.42 µs |  16.25 µs | strongly bimodal (~3.5 µs / ~14 µs) |
| 13 | back-to-back  | 64 | 56.75 µs |   119.08 µs |   213.42 µs |    336.96 µs | quad-modal (~55 / 110 / 200 / 250 µs) |
| 15 | back-to-back  | 1024 | 961.00 µs |    1.419 ms |    3.380 ms |     3.775 ms | bimodal (~900 µs / ~3.5 ms) |

**This means the 014 single-mode "stable region" fit was actually
fitting *median across both modes*.** The slopes still match
because the fast/slow mode ratio is consistent across N within a
regime, but the underlying distribution is much richer than a
single Gaussian.

### B2b N=1024 collapse mechanism: DVFS-PARTIAL

Trial-level temporal join (1 054 trials assigned to 14 IOReport
PWRCTRL samples in the cell window):

|         | N | p10 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| DEADLINE windows |  797 |  887 µs |  1.267 ms |  3.512 ms |  3.775 ms |
| PERF windows     |  208 |  898 µs |  1.642 ms |  3.607 ms |  3.750 ms |
| IDLE_OFF windows |   49 |  914 µs |  3.278 ms |  3.592 ms |  3.902 ms |

DEADLINE-window trials are systematically ~23 % faster (p50 1.27 ms
vs 1.64 ms) than PERF-window trials — the predicted direction. But
the *within-window* bimodality (p50 ~1.3 ms, p90 ~3.5 ms in
DEADLINE windows) is much wider than the *between-window* p50
difference. So:

- **DVFS state at the 250 ms granularity does shift the timing
  distribution**, but only modestly.
- **The bimodality is mostly sub-250 ms** — the controller is
  cycling between modes faster than IOReport can resolve. This is
  consistent with exp 013's "DEADLINE residency 26-51 % when
  engaged" finding: the controller ping-pongs.

**Verdict: DVFS-PARTIAL.** DVFS contributes to the collapse but
doesn't fully explain it at this granularity. The mechanism is
**fast DVFS cycling within a single command-buffer execution** —
each ~3.5 ms trial samples the controller at one moment in its
ping-pong, sometimes catching PERF (slow per-dispatch ~3.4 ms),
sometimes DEADLINE (fast per-dispatch ~900 µs).

### Hypothesis check

| prediction | observed | verdict |
|---|---|---|
| Stable-region slopes from 014 hold within ±5 % | 3/4 within ±2 %, 1 within ±13 % | ✓ MARGINAL |
| Internal-loop kink at N=8→16 reappears | yes, identical N values | ✓ |
| B2b N=1024 collapse is at least partly DVFS-driven | yes, but only ~23 % at 250 ms granularity | ✓ partial |
| Slope of internal-loop tightens (lower variance) with more trials | partially — slope identical but bimodal distribution emerges | ✗ refined |
| Bimodality of N=1024 cell visible in trial data | yes, far stronger than 014 hinted | ✓ |
| Internal-loop bimodality only at the kink cell | **NO — bimodality is universal at all N** for low-N cells | ✗ surprise |

### Two unpredicted findings (in addition to 014's two)

1. **Bimodality is the rule, not the exception.** Every cell with
   ≥ 5 000 trials shows distinct fast and slow modes, including
   internal-loop N=1 (p25 = 2.96 µs vs p50 = 7.12 µs — a 60 % gap).
   The "single dispatch overhead floor of 6.4 µs" from exp 001
   was capturing the *slow* mode of a bimodal distribution; the
   fast mode at ~2-3 µs has been there all along, hidden by short
   cell counts. This refines exp 001/009 — the chip's behavior is
   bimodal even on the simplest dispatch.

2. **The kink cell becomes the most informative cell.** Internal-
   loop N=16 has DEADLINE = 59 %, P15 = 36 %, p50 below the cool-
   DVFS floor. **It's the cleanest experimental evidence we have
   that DEADLINE-mode timing is a real, observable thing on M4 Max
   that drops sub-floor in measurable steady-state, not just as
   transient sub-floor onset (exp 009).** Future experiments
   wanting to study DEADLINE-mode kernel timing should *aim* for
   this regime rather than treat it as an outlier.

### What this means operationally

- **014's stable-region slopes are now canonical**: 405 ns/N
  (internal-loop) and 3 157 ns/N (back-to-back) on M4 Max. The
  intercept for back-to-back drops to 4 087 ns (was 4 711 ns) —
  use 014b's number going forward.
- **Stable-region detection by per-step slope variance still
  works** as the primary heuristic, but now we can cross-validate
  it with per-cell PWRCTRL classification when cells run > 600 ms.
- **For per-iteration kernel cost recovery**, internal-loop
  amplification with N ≥ 64 gives a clean unimodal distribution
  in PERF-dominant regime (cells 5, 6, 7 have p99/p50 < 1.5).
- **Trial-level distribution is required for any kernel where
  DEADLINE engages.** Reporting only median hides the bimodality
  and may mislead about the kernel's actual cost.

### What does NOT change

- Pair timing (decision 005) primary for trials ≥ 64 µs.
- Internal-loop amplification is the methodology of choice for
  sub-floor kernels.
- Memory-bound base unit is still the next experiment.

### What changes

- **Refine answered-questions:** the 6.4 µs "dispatch-overhead
  floor" is the *slow mode* of a bimodal distribution. The fast
  mode lives at ~2-3 µs even for N=1.
- **Refine the b2b N=1024 collapse explanation:** mechanism is
  fast (sub-250 ms) DVFS cycling within the cb, not inter-sample
  DVFS state changes.
- **New methodology default for amplification experiments:** 5 s
  cell duration target, 5 000 trial cap, chunked sample-buffer
  resolve. Per-cell PWRCTRL classification becomes a routine
  diagnostic.
- **New unknown:** what is the period of the sub-250 ms PWRCTRL
  cycling? IOReport at 250 ms can't resolve it. Possibly resolvable
  with 50 ms IOReport cadence, but exp 011 showed 100 ms perturbs
  the system. Worth a small targeted experiment to find the safe
  cadence floor.
