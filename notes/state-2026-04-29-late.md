# Lab state snapshot — 2026-04-29 (late)

A second dated snapshot for 2026-04-29, taken at end of day after
experiments 014 → 014b → 015 ran on M4 Max. The morning snapshot
(`state-2026-04-29.md`) covered up through "exp 014 pre-registered,
ready to run." This one covers the **completion of the foundational
measurement arc**: amplification methodology validated on both halves
of the roofline, PRFBOOST trigger discovered, ready to apply the
methodology to real ML kernels.

Reading order if you're new to the repo: `README.md` → previous
morning snapshot (`state-2026-04-29.md`) for context on 001-013 →
this file → recent experiment READMEs (014, 014b, 015) → `notes/
answered-questions.md`.

## Hardware in scope (unchanged)

| chip | RAM | macOS | architecture | role |
|------|----:|-------|--------------|------|
| Apple M1 Pro 16 GB | 16 | 26.3.1 (Tahoe) | `applegpu_g13s` | original baseline; experiments 001-005 first run here |
| Apple M4 Max 36 GB | 36 | 26.4.1 (Tahoe) | `applegpu_g16s` | current primary; experiments 001-015 run here |

The M1 Pro hasn't been re-run with the post-008 telemetry stack or
post-014 amplification methodology. **Cross-chip generalization of
009-015 findings is untested.**

## Completed experiments (M4 Max state of play)

| #   | name                                | status | headline finding |
|-----|-------------------------------------|:------:|------------------|
| 001 | can-we-time-anything                | ✅     | timing infra works; ~6.4 µs floor on M4 Max |
| 002 | noise-floor-vs-idle                 | ✅     | "1 ms nightmare zone" gone on M4 Max; cv ordering inverted |
| 003 | warmup-recovery-and-state           | ✅     | warmup-kind ranking inverted G13→G16; new `heavy_write K=1` default |
| 004 | work-dominance-floor                | ✅     | write_tid needs ≥ 524 K threads on M4 Max; +13 µs step at fma_loop 192→256 |
| 005 | paired-ratio-stability              | ✅     | 42 µs inter-encoder gap collapsed to 833 ns on M4 Max |
| 006 | cross-session-ratio-stability       | ✅     | M4 Max cross-session ratios stable to ≤ 0.6 % for trials ≥ 10 × floor |
| 007 | ioreg-vs-powermetrics-utilization   | ❌     | ioreg utilization FALSIFIED |
| 008 | ioreport-vs-powermetrics-power      | ⚠️    | IOReport GPU power MARGINAL pass — agrees within 5 %, +14 % full-load bias |
| 009 | sub-floor-reproduction              | ✅     | M4 Max sub-floor STRONG REPRODUCE 5/5; 1 625 ns absolute floor |
| 010 | gpuph-residency                     | ✅     | GPUPH bindings PASS; PWRCTRL `DEADLINE` mode discovered |
| 011 | sub-floor-mechanism                 | ✅     | DEADLINE residency 30-48 %; cross-attempt persistence isn't in PWRCTRL/GPUPH state |
| 012 | gpuph-vs-powermetrics-mhz           | ⚠️    | M4 Max GPUPH→MHz table extracted; peak 1 578 MHz; non-monotonic in P-index |
| 013 | pwrctrl-recipe-sweep                | ✅     | DEADLINE entry rule: `(FMA_ITERS ≤ 16384) AND (sleep_us ≤ 50)`; 4th PWRCTRL state `PRFBOOST` enumerated |
| **014** | **amplification-validation**    | ✅     | MARGINAL PASS piecewise-linear; both methods clean within stable region |
| **014b** | **amplification-long-cells**   | ✅     | Slopes reproduce within ±2 %; bimodality universal; b2b N=1024 collapse is DVFS-PARTIAL |
| **015** | **memory-bound-amplification**  | ✅     | Pointer-chase amplification PASS internal-loop; PRFBOOST trigger discovered; b2b invalid for memory-bound |

## What changed since the morning snapshot

### New methodology: validated amplification

- **Internal-loop amplification + many-point linear fit** is now
  the canonical method for recovering per-iteration kernel cost
  on kernels below the dispatch-overhead floor. Validated on:
  - Compute-bound (fma_loop, FMA_PER_ITER=64): slope =
    **405.17 ns / amp-step** = **6.33 ns / FMA** at running DVFS.
  - Memory-latency-bound (chase, CHASE_PER_ITER=64): slope =
    **21 825 ns / amp-step** = **341 ns / chained DRAM load** at
    PRFBOOST DVFS.
- **Both halves of the roofline now have reference per-iter
  costs.** Memory:compute slope ratio = 53.9× on M4 Max.
- **Stable-region detection by per-step slope variance** replaces
  PWRCTRL classification when cells run shorter than the 250 ms
  IOReport interval. With 5 000-trial cells, both signals work.
- **Back-to-back amplification has known limits:**
  - Compute-bound: works for N ≤ 256, collapses at N=1024
    (DVFS-PARTIAL — fast PWRCTRL cycling sub-250 ms).
  - Memory-bound: **invalid** at any N. Each new dispatch restarts
    `addr = tid` so cache reuse is artificial.

### New PWRCTRL knowledge

- **PRFBOOST trigger closed** (was open from exp 013): sustained
  memory-bound dispatches with single-dispatch wall-clock ≥ ~5 ms.
  Compute-bound dispatches at the same wall-clock do NOT engage it.
- **PWRCTRL state behavior on M4 Max** is more granular than 013
  established. Per-cell observations from 015:
  - Small dispatches (< 50 µs): MIXED / IDLE_OFF dominated (cells
    too short for PWRCTRL to pick a stable mode).
  - Small dispatches with extended sustained workload: DEADLINE.
  - Mid dispatches (~100 µs - 5 ms) sustained: PERF.
  - Memory-bound dispatches ≥ 5 ms sustained: **PRFBOOST**.
  - Both DEADLINE and PRFBOOST ramp P15 residency dramatically
    (30-97 %).

### New refinements

- **Bimodality is the rule, not the exception** (014b). With ≥
  5 000 trials per cell, almost every cell shows a fast mode and
  a slow mode. The 6.4 µs "dispatch-overhead floor" from exp 001
  was always the *slow* mode of a bimodal distribution; the fast
  mode at ~2-3 µs has been there all along.
- **The 014b internal-loop N=16 cell is the cleanest steady-state
  evidence we have that DEADLINE-mode timing is a real, measurable
  regime** — not just sub-floor onset transient (exp 009). 59 %
  DEADLINE residency, 36 % P15, p50 = 5.96 µs (below the cool-DVFS
  floor).

## Methodology decisions in force

- [Decision 001](../decisions/001-lab-in-public.md): in-public lab,
  agent-first tooling. Active.
- [Decision 002](../decisions/002-uv-with-pep-723.md): uv + PEP 723
  inline metadata. Active.
- [Decision 003](../decisions/003-paired-kernel-ratio-timing.md):
  pair timing as primary methodology. Superseded by 004 on M1 Pro,
  partly restored by 005 on M4 Max.
- [Decision 004](../decisions/004-narrowed-pair-timing-scope.md):
  narrow pair timing on M1 Pro to relative-magnitude / tail-
  suppression / sweep-stability. **Active on M1 Pro. Superseded on
  M4 Max by decision 005.**
- [Decision 005](../decisions/005-restore-pair-timing-on-m4-max.md):
  pair timing primary on M4 Max for trials ≥ ~64 µs. **Active on
  M4 Max only.**

**No new decisions** from 014/014b/015. The amplification methodology
is informally adopted but does not need a decision document — it
applies to a specific kernel-size regime (sub-floor) that pair timing
doesn't cover.

## Operational rules per chip

### M1 Pro / `applegpu_g13s` (unchanged from morning snapshot)

- Dispatch-overhead floor: ~8 µs back-to-back.
- Warmup recipe: `same K=1` (avoid `fma_loop K=1`). Decision 004.
- Work-dominance: write_tid ≥ 131 K-262 K threads, fma_loop ≥ 256.
- **No state-channel telemetry, amplification, or PRFBOOST work
  validated here yet.** Everything 009-015 is M4 Max only.

### M4 Max / `applegpu_g16s`

#### Timing
- Dispatch-overhead floor: ~6.4 µs cool / ~1.7 µs DEADLINE-mode
  sub-floor. **Both are slow / fast modes of a bimodal
  distribution** with the fast mode appearing for almost any kernel
  with enough trials (014b refinement).
- Warmup recipe: `heavy_write K=1`. Decision 004 inverted on M4 Max.
- Work-dominance: write_tid ≥ 524 K threads, fma_loop ≥ 256 iters.
- Pair timing primary for trials ≥ ~64 µs (decision 005).
- **Sub-floor amplification:** internal-loop with stable-region fit
  for kernels below ~50 µs. Down to ~64-op base units.

#### DVFS state
- 16 GPUPH states: `OFF + P1..P15`. Read sudo-free via
  `notes/ioreport.py --include-states`.
- Per-state MHz mapping (positional, M4 Max specific, exp 012):

      P1 = 338    P6 = 1056   P11 = 1242
      P2 = 618    P7 = 1062   P12 = 1380
      P3 = 796    P8 = 1182   P13 = 1326
      P4 = 924    P9 = 1182   P14 = 1470
      P5 = 952    P10 = 1312  P15 = 1578

  **Peak GPU: 1 578 MHz.** Mapping NOT monotonic in P-index.

#### Power-controller modes (4 known states)
- **`IDLE_OFF`** — between dispatches.
- **`PERF`** — normal sustained execution.
- **`DEADLINE`** — engaged when `(FMA_ITERS ≤ 16384) AND (sleep_us
  ≤ 50)` (exp 013); cycles peak↔idle on a sub-250 ms timescale.
  DEADLINE residency 26-51 % when engaged; chip ping-pongs.
- **`PRFBOOST`** — engaged for sustained memory-bound dispatches
  with wall-clock ≥ ~5 ms (exp 015). Pegs P15 residency at 93-97 %
  and produces tight unimodal trial distributions (no bimodality
  cycling).
- **PWRCTRL mode and GPUPH P15 residency are independent
  dimensions.** Same chip can be PERF + sustained P15 (heavy
  kernel back-to-back), or DEADLINE + cycling P15 (small kernel,
  sleep_50 µs), or PRFBOOST + sustained P15 (memory-bound large
  dispatch).

#### Reference per-iter costs (running DVFS regime)
- Compute-bound (fma_loop carry chain): **6.33 ns / FMA**.
- Memory-latency-bound (chase, SLC-defeating table): **341 ns /
  chained DRAM load**.
- Memory:compute ratio: **53.9×**.

## Telemetry stack (unchanged)

| tool | location | sudo? | what it gives | when to use |
|------|----------|------:|---------------|-------------|
| `gpu_telemetry.py` | `notes/` | yes | powermetrics-derived: GPU active residency, freq, power, thermal | absolute power claims with tight tolerance |
| `ioreport.py` | `notes/` | **no** | libIOReport-derived: 11 power buckets + per-state residency for any STATE-format channel | default for any new experiment |

Standard cadence: **250 ms.** 100 ms perturbs the system being
measured. Subprocess pattern: launch ioreport.py from inside run.py,
SIGINT at end for clean flush.

## Standard experiment template (014b/015 pattern, current default)

For amplification or DVFS-characterization experiments:
- Cell duration target: 5 s wall-clock.
- Trial cap: 5 000 per cell (Metal counter sample buffer is 32 768 B
  = 4 096 slots = 2 048 trials, so resolve in chunks of 2 000 to
  reuse the buffer within a cell).
- Inter-cell idle: 1 s.
- Trials at sleep_0 within cell.
- IOReport sidecar at 250 ms with `--include-states`.
- Per-trial monotonic_ns recorded for IOReport temporal join.
- Per-cell PWRCTRL/GPUPH classification works for cells ≥ ~600 ms.

## Strategic frame: where the project actually is

**The foundational measurement arc (001-015) is complete on M4 Max.**

We can now:
- Time any kernel ≥ 50 µs single-shot (001-006).
- Time arbitrary sub-floor kernels via internal-loop amplification
  (014/014b/015).
- Read DVFS / PWRCTRL state per-cell at 250 ms granularity (007-013).
- Place a kernel on the roofline with reference per-iter costs for
  both halves (015).

**What the foundation has NOT been used for yet:** classifying a
real (non-synthetic) ML kernel by its bottleneck class. The
discrimination thesis from CLAUDE.md / UNKNOWNS.md was the original
target; everything 001-015 is foundation, not the discrimination
test itself.

The bridge is built but has only been driven across by synthetic
kernels we already knew the answer for. Whether the methodology
holds for kernels we *don't* know the answer for — small matmul,
LayerNorm slice, attention scores, etc. — is the open question that
the next experiment should answer.

## Most useful unanswered questions

Re-ordered by mission relevance after the foundational arc closure:

1. **Does the amplification methodology classify a real ML kernel's
   bottleneck class on M4 Max?** The actual discrimination test.
   Pre-reg target: small matmul or LayerNorm slice; vary internal
   parameters to perturb compute or memory intensity; compare
   recovered slope to 015's reference per-iter costs.
2. **Cross-chip: does the post-008 telemetry stack and post-014
   amplification methodology work on M1 Pro?** Required for any
   cross-chip claims about bottleneck classification. One focused
   experiment to re-run 014/014b/015 on M1 Pro.
3. **External sanity check: peak FLOPS and memory bandwidth
   matching Philip Turner's published M1 Pro numbers.** Cheap
   on M1 Pro; would catch silent timing-methodology bugs.
4. **Memory bandwidth (vs latency) characterization.** 015 measured
   latency via pointer chase. Bandwidth via STREAM-style parallel
   loads is a separate experiment.

Lower priority — interesting but not load-bearing:

5. **Period of sub-250 ms PWRCTRL cycling** that drives the b2b
   N=1024 fma collapse.
6. **What carries cross-attempt deepening across PWRCTRL/GPUPH
   gaps?** (009/011) — likely thermal or driver-side.
7. **Half-life of the M4 Max sub-floor state.** (009)
8. **Why is GPUPH index ordering non-monotonic in MHz?** (012)
9. **Why is sleep_50 µs more DEADLINE-friendly than sleep_0?** (013)
10. **+13/+21 µs step at fma_loop iters 192→256.** (004)
11. **Cache-reuse asymmetry quantification** (015) — sweep table
    size to find where b2b/internal-loop slopes converge.

## What we learned about the agent-first lab discipline (carry-over + 014/14b/15 additions)

Carry-over:
- Per-chip decisions are now the norm.
- Pre-registration before running every experiment held up across
  16 experiments (no temptation to retro-fit).
- Negative results pile up cheaply.
- "Raw before robust" exposes structure (every cell with enough
  trials shows bimodality the small-N pre-regs didn't predict).
- Observation perturbs state — IOReport at 250 ms is the safe
  cadence floor.

New from 014/014b/015:
- **Pre-reg + immediate follow-up (014 → 014b) is a successful
  pattern when the first run reveals a methodology blind spot.**
  Don't try to over-engineer the first run; ship it, see what
  breaks, run the follow-up.
- **Long-cell methodology (5 s, 5 000 trials)** is now the default
  for amplification/DVFS work. The 50-trial-per-cell pattern
  inherited from 001-005 was a methodology mismatch for regime-
  characterization work.
- **Stable-region detection from timing data** is a more robust
  primary signal than PWRCTRL classification when cell durations
  are uncertain. Use both when available.
- **Cross-bottleneck comparison** is a methodology check: when
  you can mirror an experiment across two kernels with the same
  per-iter constant, divergent slopes are informative
  (53.9× ratio established compute vs memory).

## Pointer to next experiment

The next experiment is the **discrimination test** — applying the
amplification methodology to a real (non-synthetic) ML kernel and
asking whether the recovered slope identifies its bottleneck class.

Suggested setup (not pre-registered):
- Base unit: small matmul (e.g., 32×32 × 32×K for varying K, or
  similar) or LayerNorm slice — small enough to need amplification,
  representative of ML decoder-side work.
- Apply internal-loop amplification across the same N grid as
  014/014b/015.
- Compare recovered slope to 014b's compute-bound reference
  (6.3 ns / op) and 015's memory-bound reference (341 ns / chained
  load). For an unambiguously compute-bound matmul, the slope per
  op should land near the compute reference. For a memory-bound
  shape (e.g., very tall-skinny), it should land near the memory
  reference.
- The discrimination test is whether the slope ratio actually
  discriminates between known compute-vs-memory-bound shapes.

Do not pre-register past this branch. The result of the
discrimination test will determine whether the methodology is
ready for application or needs another foundation-layer
refinement.
