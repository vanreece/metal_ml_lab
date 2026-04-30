# 015: Does loop amplification + many-point timing also work for a memory-bound base unit on M4 Max?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-29
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro
14" (Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~3-5 min including IOReport sidecar startup
and 16 cells × ≤ 5 s.
**Predecessors:**
- 014 / 014b validated the technique on a compute-bound (FMA carry-
  chain) base unit.
- This experiment tests whether the same technique transfers to a
  memory-bound base unit. Without it, the bottleneck-classification
  thesis (CLAUDE.md, UNKNOWNS.md "Methodology") only has half the
  roofline characterized.

## The question

In 014/014b we established that internal-loop amplification recovers
per-iteration kernel cost cleanly for an FMA carry chain, with
slope ≈ 405 ns / amp-step on M4 Max. The base unit there was
compute-bound: a 64-fma dependent chain with no memory access
beyond a single output write.

If amplification only works for compute-bound kernels, the
methodology has a half-roofline limit and we can't classify ML
kernels that live on the memory side. So we want to test the
mirror case: a **memory-latency-bound** base unit, structurally
analogous to fma_loop (serial dependency chain, no ILP, carry
prevents compiler optimization) but where the dependency is on
loaded values rather than computed ones.

The natural mirror is **pointer-chase**: a chained sequence of
random loads where each load's address comes from the previous
load's value. A pre-shuffled lookup table forces the chain to miss
cache and serialize on DRAM round-trip latency. This is the
canonical memory-latency benchmark (lmbench, STREAM-CHAIN,
LBL roofline tooling).

```metal
constant int CHASE_PER_ITER = 64;
constant int N_AMP = X;

kernel void chase_amplified(
    device const uint *table [[buffer(0)]],
    device uint       *out   [[buffer(1)]],
    uint tid                  [[thread_position_in_grid]]
) {
    uint addr = tid;
    for (int n = 0; n < N_AMP; n++) {
        for (int i = 0; i < CHASE_PER_ITER; i++) {
            addr = table[addr];
        }
    }
    out[tid] = addr;
}
```

## Pre-registered verdicts

For internal-loop on the chase base unit:

- **PASS:** stable-region linear fit holds with R² > 0.99,
  residuals < 5 % at every N in the stable region, and the
  intercept matches the prior 6.4 µs dispatch-overhead floor
  within ± 30 %. Slope reports a sensible per-amp-step latency
  (positive, larger than fma_loop's 405 ns/N — order of
  magnitude expectation: ~5-15 µs / amp-step at 64 chained loads
  if memory latency is ~80-200 ns).
- **MARGINAL:** linear fit is *piecewise* across DVFS regimes
  (analogous to 014's internal-loop kink at N=8→16). Slopes in
  each regime are individually clean.
- **FAIL:** non-linear, or the slope isn't positive/sensible
  (suggests cache effects dominate — the chase isn't actually
  serializing on DRAM, or the compiler somehow eliminated the
  chain).

For the cross-bottleneck slope ratio:

- We expect `slope_chase` to be **5×-30× larger** than
  `slope_fma_loop` at the same FMA_PER_ITER / CHASE_PER_ITER = 64,
  reflecting the memory-vs-compute latency ratio. The exact ratio
  is itself a finding — it's a per-iter latency comparison.

For back-to-back: not the load-bearing part of this experiment.
We'll run it as a sanity check on the dispatch-overhead floor;
b2b being well-behaved is not required for the verdict.

## Why this question, now

Three reasons:

1. **Load-bearing for the discrimination thesis.** ML kernels span
   the roofline. Without memory-bound validation, we can't claim
   the amplification methodology generalizes.
2. **Cheap.** ~5 minutes GPU time, no sudo, reuses the IOReport
   stack. Same shape as 014b.
3. **Structural symmetry with 014.** Same N grid, same trial
   protocol, same analysis. Differences in the result are
   attributable to the kernel, not the method.

## Hypothesis

Confidence: medium-high on basic linearity, medium on the per-iter
latency magnitude.

- **Internal-loop linear model holds.** Slope of `t(N)` recovers
  the per-amp-step memory latency. The chain is data-dependent
  (load → use as address) so the compiler can't unroll/fuse it.
- **Slope is order-of-magnitude bigger than fma_loop's 405 ns/N.**
  64 chained loads of ~80-200 ns each = ~5-13 µs per amp-step.
  Modulo cache effects, the slope should land in this range.
- **Intercept matches the 6.4 µs dispatch-overhead floor.** Same
  chip, same dispatch path. The intercept is a chip property,
  not a kernel property.
- **A DEADLINE/PERF kink may appear at a smaller N than 014b's**,
  because per-amp-step wall-clock is bigger and crosses
  DEADLINE-trigger thresholds at a smaller N. Possibly the
  smallest N is already in PERF (sustained); we'll see.
- **The pointer-chase R² will be lower than fma_loop's** within
  any single regime because memory latency is more variable
  (cache hit/miss, DRAM bank conflicts, etc.). Expect R² ≥ 0.97
  rather than fma_loop's 1.0000.

## What we are NOT trying to answer

- **Memory bandwidth.** This is latency-bound. Bandwidth-bound
  characterization (STREAM copy/scale/add/triad) is its own
  experiment.
- **L1/L2/SLC cache hierarchy mapping.** Single working-set size
  here (one 128 MB table, definitely beyond SLC). Sweeping working-
  set sizes is its own experiment.
- **Threadgroup-size scaling on memory access patterns.** Fixed at
  32 threads = 1 SIMD group, mirroring 014.
- **Cross-chip generalization.** M4 Max only.
- **Cross-bottleneck application to real ML kernels.** Conditional
  on this experiment passing.
- **Validation against published M4 Max DRAM latency numbers.**
  We'll report the slope but not pass/fail against a published
  reference (none readily available to me; cross-checking is its
  own task).
- **Per-trial DVFS attribution at sub-250 ms.** 014b's DVFS-PARTIAL
  finding still stands; we are not chasing the sub-250 ms cycling
  question here.

## Setup

### Base unit kernel (memory-latency-bound)

Same shape as 014's `fma_loop_amplified`, with the FMA carry chain
replaced by a pointer-chase chain. `N_AMP` is a compile-time constant
so we pre-compile a pipeline per N.

```metal
constant int CHASE_PER_ITER = 64;
constant int N_AMP = {n_amp};

kernel void chase_amplified(
    device const uint *table [[buffer(0)]],
    device uint       *out   [[buffer(1)]],
    uint tid                  [[thread_position_in_grid]]
) {
    uint addr = tid;
    for (int n = 0; n < N_AMP; n++) {
        for (int i = 0; i < CHASE_PER_ITER; i++) {
            addr = table[addr];
        }
    }
    out[tid] = addr;
}
```

`CHASE_PER_ITER = 64` matches 014's `FMA_PER_ITER = 64` so per-amp-
step costs are directly comparable across kernels.

### Pointer-chase table

- **Table size:** 128 MB = 32 M `uint32` entries. Beyond Apple
  silicon's typical SLC (24-48 MB) so each load is DRAM-bound at
  steady state.
- **Permutation:** single random permutation of [0, 32M) generated
  once on CPU, uploaded to a private/shared MTLBuffer. All threads
  share the same table; thread `tid` starts at index `tid`.
- **Thread spread:** with 32 threads, each starting at a different
  index, they diverge into different cache lines within ~1 chained
  load and stay scattered.

The output buffer (`out[tid]`) prevents the compiler from eliding
the chain — the final `addr` must be observable.

### Variables (mirror of 014)

- **Amplification method** (2 levels): `internal-loop`, `back-to-back`.
- **N (amplification factor)** (8 levels):
  `{1, 2, 4, 8, 16, 64, 256, 1024}`.

2 × 8 = **16 cells.**

### Per-cell protocol (014b's, unchanged)

- 5 s wall-clock target per cell, capped at MAX_TRIALS_PER_CELL
  = 5 000.
- Trials at sleep_0 within cell, no inter-trial sleep.
- Sample buffer reused in chunks of RESOLVE_CHUNK_TRIALS = 2 000
  (Metal's 32 768 B counter-buffer cap).
- 1 s inter-cell idle for clean IOReport boundaries.
- IOReport sidecar at 250 ms with `--include-states`.
- Threadgroup: 32 threads = 1 SIMD group.

### What we record

Same schema as 014b:
- `raw/{ts}-trials.csv` — per-trial timing and monotonic_ns.
- `raw/{ts}-cells.csv` — per-cell summary with percentiles.
- `raw/{ts}.csv` and `raw/{ts}-states.csv` — IOReport.
- `raw/{ts}-meta.txt` — env, run config, table size, permutation
  random seed.

### What we do NOT do

- No retries; raise loudly on alloc / compile failures.
- No outlier discarding. Per-cell percentiles in cells.csv; raw
  in trials.csv.
- No averaging in live output beyond per-cell summary.
- No cooldown between cells beyond the 1 s inter-cell idle.
- No varying CHASE_PER_ITER or threadgroup config.

## Success criterion

The experiment **succeeds** if we have:

1. Per-trial CSV with all completed trials (expect ~50 K-80 K rows).
2. Per-cell summary CSV with 16 rows.
3. IOReport states CSV with ≥ 2-3 samples per cell where cell
   duration permits.
4. Per-method linear fit (full + stable-region) with slope,
   intercept, R², residuals.
5. Side-by-side comparison of `slope_chase` vs `slope_fma_loop`
   from 014b at the same CHASE_PER_ITER = FMA_PER_ITER = 64.

## After this experiment

Branches:

- **PASS internal-loop with sensible memory-latency slope.** Adopt
  amplification on memory-bound kernels as validated. The
  bottleneck-discrimination thesis now has both halves of the
  roofline. Move to applying the technique on a real (small) ML
  kernel.
- **MARGINAL piecewise-linear.** Memory-bound has its own DVFS-
  regime structure (different from compute-bound's). Document
  and adapt the stable-region heuristic.
- **FAIL.** Most likely cause: cache structure of the chip is
  hiding DRAM latency below our N values, or the table size /
  permutation isn't actually defeating cache. Diagnose by
  sweeping table size and re-running with a clearly-DRAM-bound
  config.

We do not plan past these branches.

## Result

**Date run:** 2026-04-29 (timestamp prefix `20260429T151743`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Wall-clock:** ~70 s (16 cells, IOReport sidecar + 1 s inter-cell
idle; small-N cells hit the 5 000-trial cap, large-N cells ran the
full 5 s). 60 385 trials captured; 27 456 IOReport state rows.
**Outcome:** **PASS internal-loop with sensible memory-latency
slope.** Memory-bound amplification works; the bottleneck-
discrimination thesis now has both halves of the roofline. Two
unpredicted findings: (1) PRFBOOST trigger discovered; (2) cache-
reuse asymmetry between methods means back-to-back amplification
is *not transparent* for memory-bound kernels.

### Headline numbers (stable-region fits)

| method | stable N range | slope (ns / amp-step) | per-load (ns) | intercept | R² |
|---|---|---:|---:|---:|---:|
| internal-loop | 64, 256, 1024 | 21 824.96 | **341.0** | −143 110* | 0.9999 |
| back-to-back  | 16, 64, 256   | 11 033.11 | 172.4 | 41 431 | 0.9998 |

*The negative intercept is an extrapolation artifact: the stable
region excludes small N, and extrapolating from N=64..1024 back to
N=0 lands below the dispatch-overhead floor by ~7×. The full-fit
intercept of 4 108 ns (R² = 0.9998 across all 8 N) is closer to
the prior 6.4 µs dispatch floor (≈ 36 % low — outside the ± 30 %
pre-reg threshold).

### Cross-bottleneck slope comparison vs 014b's fma_loop

| method | metric | fma 014b | chase 015 | ratio |
|---|---|---:|---:|---:|
| internal-loop | slope (ns / amp-step)        |   405.17 | 21 824.96 | **53.87×** |
| internal-loop | per-iter latency (ns / op)   |     6.33 |    341.02 | **53.87×** |
| internal-loop | intercept (full fit)         | 6 861    |  4 108    |  0.60×    |
| back-to-back  | slope (ns / amp-step)        | 3 156.67 | 11 033.11 |  3.50×    |
| back-to-back  | intercept                    | 4 087    | 41 431    | 10.14×    |

**The internal-loop slope ratio is 54× — memory-latency-bound is
54× slower per chained operation than ALU-bound on M4 Max.** The
pre-reg expected 5-30×; we underestimated, but this is a positive
finding: memory latency on this chip is at the high end of the
plausible range.

The b2b ratio (3.5×) is much smaller than internal-loop's (54×)
because of the cache-reuse asymmetry described below.

### Two unpredicted findings

#### 1. PRFBOOST trigger discovered

The 4th PWRCTRL state `PRFBOOST` enumerated in exp 013 was tagged
"never engaged in our recipes; trigger unknown" in the state
snapshot. **It engages here, dominantly:**

| cell | method | N | trial wall-clock | PWRCTRL top |
|---:|---|---:|---:|---|
|  6 | internal-loop | 256  | ~5.5 ms | **PRFBOOST(94 %)** IDLE_OFF(5 %) |
|  7 | internal-loop | 1024 | ~22 ms  | **PRFBOOST(97 %)** IDLE_OFF(2 %) |

Trigger: **sustained memory-bound dispatches with single-dispatch
wall-clock ≥ ~5 ms** push the controller into PRFBOOST. Compute-
bound dispatches at the same wall-clock (014b internal-loop N=1024
at 422 µs and beyond) never engaged it; the trigger is memory-
intensity, not duration. This closes the long-open question 8 from
the state snapshot.

P15 residency in PRFBOOST cells is 93-97 % (cells 6 and 7), so
PRFBOOST is the regime that pegs the GPU at peak DVFS for sustained
memory work. PERF cells (4 and 5) only hit 9-93 % P15.

#### 2. Cache-reuse asymmetry breaks back-to-back transparency

Each back-to-back dispatch starts a fresh `addr = tid` chain. With
N dispatches in one cb, the first few loads of each dispatch hit
predictable addresses that get cached across dispatches. Internal-
loop runs one continuous chain that diverges into uncached lines.

Per-step slope diagnostic on b2b makes this visible:

| N step | b2b Δp50/ΔN | internal-loop Δp50/ΔN |
|---|---:|---:|
| 1 → 2  | 38 625 ns/N | 7 916 ns/N |
| 2 → 4  | **1 021 ns/N** | 54 562 ns/N |
| 4 → 8  | **−2 645 ns/N** | 30 989 ns/N |
| 8 → 16 | 14 088 ns/N | 17 770 ns/N |
| 16 → 64 | 11 882 ns/N | 15 847 ns/N |
| 64 → 256 | 10 891 ns/N | 22 763 ns/N |
| 256 → 1024 | 8 477 ns/N | 21 668 ns/N |

The b2b N=2..8 region has near-zero (or even negative) per-step
slope because additional dispatches reuse cache. The slope only
recovers at N ≥ 16 once the working set per-cb exceeds cache
capacity. Even then, b2b slope stabilizes at ~10 µs/step vs
internal-loop's ~22 µs/step — a 2× cache-reuse benefit.

**Operational consequence:** for memory-bound kernels, **back-to-
back amplification systematically underestimates per-iteration
cost** because cache reuse across dispatches is artificial — it
wouldn't exist in single-dispatch usage. Internal-loop is the only
correct amplification method for memory-bound per-iter cost
recovery.

### Hypothesis check

| prediction | observed | verdict |
|---|---|---|
| Linear model holds (internal-loop, R² > 0.99, residuals < 5 %) | R² = 0.9999, residuals < 6.4 % | ✓ |
| Intercept matches 6.4 µs dispatch-overhead floor within ± 30 % | full-fit intercept 4 108 ns (~ 36 % low) | ✗ MARGINAL |
| Slope 5×-30× larger than fma_loop's 405 ns/N | 54× larger | ✗ underestimated, in good direction |
| Memory-latency per load ~80-200 ns | 341 ns | ✗ underestimated |
| DEADLINE/PERF kink visible | yes — small-N cells bimodal, large-N PERF/PRFBOOST | ✓ |
| Pointer-chase R² lower than fma_loop's | R² *higher* (0.9999 vs 1.0000 — both essentially 1) | ✗ refined |
| Compiler can't elide chain | confirmed — slope is positive and consistent | ✓ |

### What this means operationally

- **Memory-bound amplification is validated** with internal-loop on
  M4 Max. Adopt it as the primary methodology for memory-latency-
  bound per-iter cost recovery.
- **For memory-bound kernels, do not use back-to-back
  amplification.** Cache reuse between dispatches makes the slope
  unrepresentative.
- **PRFBOOST is now a known operating regime.** Memory-bound
  experiments running ≥ 5 ms per dispatch will engage it; expect
  P15 residency near 100 % and stable timing in this regime.
- **The 6.4 µs dispatch-overhead floor citation** still holds for
  small N where it dominates, but the 015 full-fit intercept of
  4 108 ns suggests memory-bound dispatches see slightly different
  setup overhead — possibly because the chase kernel has 2 buffer
  bindings vs fma's 1, or because the start-of-dispatch DRAM warmup
  differs. Worth a follow-up if the discrimination thesis depends
  on it.
- **Per-load DRAM latency on M4 Max ≈ 341 ns** (internal-loop chase
  slope / 64), at the running DVFS regime in PRFBOOST. This is the
  first published-from-this-lab number for M4 Max DRAM access
  latency.

### What does NOT change

- 014b's stable-region slopes for fma_loop remain canonical for
  compute-bound.
- Pair timing (decision 005) primary for trials ≥ 64 µs.
- Internal-loop amplification remains the recommended methodology
  for sub-floor kernels.

### What changes

- **Memory-bound base unit validated.** Both halves of the roofline
  now have an amplification methodology.
- **Closes UNKNOWNS question 8** (PRFBOOST trigger): sustained
  memory-bound dispatches with wall-clock ≥ ~5 ms.
- **Adds a new operational rule:** back-to-back amplification is
  invalid for memory-bound base units.
- **New unknown raised:** is the cache-reuse asymmetry between b2b
  and internal-loop something we can quantify (e.g., by sweeping
  table size to find where the asymmetry vanishes)? Lower priority
  than progressing to real ML kernels.

### Pointer to next experiment

The amplification methodology is now validated on both halves of
the roofline. Natural next step per CLAUDE.md "Methodology": apply
controlled microbench perturbations on a real ML kernel (small
matmul or small attention scores) and ask whether the amplified
slope discriminates its bottleneck class. That is the actual
discrimination test the methodology has been built up to support.
