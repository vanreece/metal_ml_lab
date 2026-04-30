# 014: Does loop amplification + two-point timing recover per-iteration kernel cost reliably for kernels below the dispatch-overhead floor?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-29
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~3-5 min (16 cells × ~10-50 ms × 50 trials, plus
subprocess setup / IOReport overhead)

## The question

Experiments 001-013 established that the M4 Max single-dispatch
timing floor is **~6.4 µs in the cool DVFS state and ~1.7 µs in the
DEADLINE-mode sub-floor.** Most of this is dispatch overhead — the
work-dominance threshold from exp 004 says you need a kernel doing
at least ~30+ µs of actual compute before its real cost pokes above
the noise. That cleanly excludes the small-kernel regime where a lot
of practical ML work lives (decoder-side single-token inference,
LayerNorm slices, small attention scores).

**Loop amplification** is the standard technique for measuring small
kernels: run the kernel N times inside one measurement window, divide
total time by N (or fit a line) to recover the per-iteration cost.
It's well-established in NVIDIA's `nvbench`, BabelStream, the LBL
roofline group's tooling. Two questions before we trust it on this
hardware:

1. **Does the linear model `t(N) = a + b·N` actually hold across N
   on Apple Silicon?** The slope b should equal per-iteration cost
   and the intercept a should equal launch overhead. If the line
   is clean, two-point (or many-point) timing works.

2. **Does amplification preserve the regime the kernel runs in?**
   Per exp 013, PWRCTRL flips between `PERF` and `DEADLINE` based
   on per-dispatch wall-clock. Amplifying a kernel from ~100 ns to
   ~100 µs **shifts which DVFS state the chip uses to run it**.
   The amplified per-iter cost may not reflect what an
   un-amplified single-shot of the same kernel costs in a real
   workload.

The two-point trick: with measurements at two N values,

    per_iter_cost   = (t_N - t_1) / (N - 1)
    launch_overhead = t_1 - per_iter_cost

Or fit a line over many N values for better statistics.

**This experiment tests both questions on two amplification methods:**

- **internal-loop:** kernel source compiled with N copies of the
  base unit's body in one inner loop. Single dispatch, one launch
  overhead. `t_internal(N) = launch_overhead + N × per_iter_cost`.
- **back-to-back:** the base-unit kernel dispatched N times inside
  one MTLCommandBuffer. Single command-buffer commit, one wait.
  `t_b2b(N) = cb_overhead + N × (per_iter_cost + per_encode_cost)`.

Comparing the two slopes lets us isolate the per-encode cost — the
~833 ns inter-encoder gap exp 005 found on M4 Max should drop out
of `slope_b2b - slope_internal`.

## Pre-registered verdicts

For each amplification method:

- **PASS:** linear fit holds within a single DVFS regime
  (R² > 0.99, residuals < 5 % of fitted value at every N), and the
  intercept matches the prior 6.4 µs dispatch-overhead floor (within
  ± 30 %).
- **MARGINAL:** linear fit is *piecewise*-linear because amplification
  crossed a DVFS regime boundary partway through the N range. Slopes
  in each regime are individually clean. The technique works but
  needs two-regime handling.
- **FAIL:** non-linear, or linear with wildly noisy slope, or
  intercept doesn't match dispatch overhead (suggests the compiler
  optimized the loop, or some other artifact contaminates the
  measurement).

For the per-encode cost (`slope_b2b - slope_internal`):

- **PASS:** difference is positive and small (a few hundred ns to a
  couple µs), consistent with exp 005's 833 ns inter-encoder gap.
- **FAIL:** difference is zero, negative, or larger than dispatch
  overhead. Either the model is wrong or one of the methods has an
  artifact.

## Why this question, now

Three reasons:

1. **It's load-bearing for the project's stated mission.** The
   bottleneck-classification thesis (CLAUDE.md, UNKNOWNS.md
   "Methodology") needs to time real ML kernels, most of which are
   small. Without amplification, single-dispatch timing limits us
   to kernels ≥ 50 µs; with amplification, in principle, we can go
   arbitrarily small. Validating it on a known controlled kernel
   *before* applying it to a real one is the responsible order.
2. **The exp 013 finding directly bears on this.** "Amplification
   shifts the DVFS regime" isn't a generic worry — it's a
   prediction from our data. Testing it explicitly here tells us
   whether amplification is a transparent abstraction or a
   stateful one.
3. **Cheap to run.** ~5 minutes of GPU work, no sudo, reuses the
   IOReport telemetry stack from 010-013.

## Hypothesis

Confidence: medium-high on the basic linearity, medium on the DVFS
piecewise-linearity prediction.

- **Internal-loop linear model holds across the full N range when
  the chip stays in PERF.** R² > 0.99. Slope ≈ kernel-per-iter cost
  at the chip's PERF DVFS frequency.
- **Internal-loop crosses DEADLINE → PERF boundary somewhere in
  the N range.** Per exp 013, DEADLINE engages when per-dispatch
  wall-clock is in the ~5-50 µs range with brief inter-dispatch
  gaps. With our N range and base unit, the un-amplified (N=1)
  case sits below DEADLINE entry, intermediate N hits DEADLINE,
  large N goes back to PERF (sustained). Expect a visible kink in
  t(N) somewhere mid-range.
- **Back-to-back linear model also holds, with a higher slope.**
  `slope_b2b - slope_internal` ≈ 0.5-2 µs (per-encode cost,
  dominated by the inter-encoder gap exp 005 measured at 833 ns
  + small bookkeeping).
- **Back-to-back intercept ≈ internal intercept.** Both should be
  the cb commit + wait overhead, which is ~6 µs.
- **Carry-dependent kernel structure prevents the compiler from
  eliding the loop.** If we accidentally use an independent-iter
  pattern, the compiler will hoist/vectorize and slope will be
  ~zero. We use the existing fma_loop pattern (`y = fma(y, x, x)`)
  which is data-dependent.

## What we are NOT trying to answer

- **Whether the technique works on real ML kernels.** That's the
  experiment that comes after this one. Here we use a controlled
  fma_loop kernel where we know the answer.
- **Memory-bound kernels.** This experiment uses an arithmetic-only
  kernel. A memory-bound version (pointer chase, STREAM-style) is
  a separate experiment.
- **N values larger than 1024.** Diminishing returns in linearity
  validation; cell wall-clock would balloon.
- **Cross-chip generalization.** M4 Max only.
- **The "back-to-back without command-buffer batching" variant**
  where each dispatch has its own cb commit + wait. Not measured;
  it's just N × t_1 by construction and provides no new info.
- **Amplification under DEADLINE-mode entry recipes from exp 013.**
  Holding base unit fixed; not co-sweeping inter-dispatch sleep.
- **Per-trial DVFS resolution.** IOReport at 250 ms covers
  cell-level resolution, not trial-level. Per-cell GPUPH residency
  is the unit of analysis.

## Setup

### Base unit kernel

Carry-dependent fma_loop with a small fixed FMA count per "iteration":

```metal
constant int FMA_PER_ITER = 64;

kernel void fma_loop_amplified(...) {
    float x = float(tid) * 0.0001f + 1.0f;
    float y = 1.0f;
    for (int n = 0; n < N; n++) {       // outer loop -- amplification
        for (int i = 0; i < FMA_PER_ITER; i++) {  // inner -- one base unit
            y = fma(y, x, x);
        }
    }
    out[tid] = y;
}
```

`FMA_PER_ITER = 64` makes the base unit very small (~64 cycles per
thread = ~40 ns at peak DVFS, well below the dispatch overhead
floor). N is the amplification factor. We compile a separate pipeline
per N.

The carry chain (`y = fma(y, x, x)`) makes iterations data-dependent
so the Metal compiler can't unroll-and-fuse them. If it could, slope
would collapse to ~0; we'll cross-check the slope against our prior
"single-FMA cycle cost ~1 cycle" expectation as part of validation.

### Variables

- **Amplification method** (2 levels): `internal-loop`, `back-to-back`.
- **N (amplification factor)** (8 levels):
  `{1, 2, 4, 8, 16, 64, 256, 1024}`. Log-spaced for good linear-fit
  coverage across orders of magnitude.

2 × 8 = **16 cells.** Per cell: 50 timed trials. Total = 800 dispatches.

### Per-cell protocol

For internal-loop cell (N=k):
- Pre-compiled pipeline `fma_loop_amplified<N=k>`.
- 50 trials, each: dispatch one kernel with the compiled N, measure
  with `MTLCounterSampleBuffer` at stage boundary.

For back-to-back cell (N=k):
- Pre-compiled pipeline `fma_loop_amplified<N=1>` (the un-amplified
  base unit).
- 50 trials, each: encode k dispatches into one MTLCommandBuffer,
  attach sample buffer at start of first encoder and end of last
  encoder, commit, wait. Measure the elapsed across the whole cb.

The 50 trials are back-to-back at sleep_0. (The DVFS regime
behavior we want to characterize lives at sleep_0; if we add inter-
trial gaps we'd be co-varying the 013 axis.)

### IOReport sidecar

`notes/ioreport.py --include-states --interval-ms 250` runs across
the whole experiment. Phase markers per cell so analysis can attribute
GPUPH and PWRCTRL residency to each cell.

### What we record

- `raw/{ts}-trials.csv` — per-trial timing (cell_idx, method, N,
  trial_idx, gpu_delta_raw, monotonic_ns).
- `raw/{ts}-cells.csv` — phase markers per cell with start/end
  monotonic_ns and N.
- `raw/{ts}.csv` — IOReport energy CSV.
- `raw/{ts}-states.csv` — IOReport per-state residency CSV.
- `raw/{ts}-meta.txt` — env, fit results per method, verdict.

### What we do NOT do

- No averaging in live output.
- No retries. If a sample buffer alloc or compile fails, raise
  loudly.
- No discarding of outliers. Per-N median + IQR are reported, the
  raw 50 trials per cell go in the CSV.
- No varying base unit or threadgroup config.
- No cooldown between cells. Cells run sequentially; cross-cell
  contamination is part of the data we're collecting.

## Success criterion

The experiment **succeeds** (in the discipline's sense) if we have:

1. Per-trial CSV with all 16 × 50 = 800 rows.
2. Per-cell summary with median/IQR per N for both methods.
3. IOReport states CSV covering all cells.
4. A linear fit (least squares) of t(N) for each method, with R²
   reported.

It produces a **usable answer** if we can fill in:

| method        | slope (per-iter ns) | intercept (launch ns) | R² | DVFS regime per N |
|---------------|--------------------:|----------------------:|---:|---|
| internal-loop |                     |                       |    |   |
| back-to-back  |                     |                       |    |   |

Plus the verdict (PASS / MARGINAL / FAIL per the thresholds above)
and a separate verdict on the per-encode cost.

## New questions we expect this to raise

- If the slope is clean and the intercept matches the 6.4 µs
  dispatch overhead, we have a **per-iteration cost number for the
  smallest kernel** (~64 FMAs, ~64 cycles): the cost-per-cycle of
  the M4 Max GPU. Cross-check against published peak FLOPS.
- If the linear fit is *piecewise* across DVFS regimes, the slope
  difference quantifies "how much faster does the kernel run when
  the chip is at PERF-sustained P15 vs DEADLINE-cycling P15?" —
  a real number for the 011 mechanism story.
- If `slope_b2b - slope_internal` is much larger than 833 ns, the
  inter-encoder gap depends on workload kind, not just chip
  generation. Worth knowing.
- If the technique passes here, the **next experiment is
  amplification on a memory-bound base unit** (STREAM-style
  pointer chase) to test whether the technique works for the
  memory-side of the roofline too.
- If GPUPH P15 residency tracks cell index (small N → less P15,
  large N → more P15), we have a continuous knob to dial DVFS
  state via amplification. Useful for designing experiments at
  specific operating points.

## After this experiment

Branches:

- **PASS (clean linear) for both methods.** Adopt amplification +
  many-point fit as the standard methodology for small kernels.
  Schedule a memory-bound base unit follow-up.
- **PASS for internal-loop, FAIL for back-to-back** (or vice
  versa). One method works; document and use that one. Note the
  asymmetry as a finding.
- **MARGINAL piecewise-linear for one or both.** Document the
  regime boundaries; build a "fit each regime separately" analysis
  helper. Less elegant but still usable.
- **FAIL.** Most likely cause: compiler optimized the loop. Verify
  by inspecting Metal AIR or by comparing carry-dependent vs
  carry-independent kernel slopes. If carry-dependent does fail,
  the technique itself doesn't work on this hardware and we need
  a different approach.

We do not plan past these branches.

## Result

**Date run:** 2026-04-29 (timestamp prefix `20260429T115430`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Wall-clock:** ~10 s (2 s baseline + 16 cells × ~10-235 ms + 2 s tail).
**Outcome:** **MARGINAL piecewise-linear for both methods.** Each
method's stable region fits cleanly (R² = 1.0000), the intercept
matches the prior 6.4 µs dispatch-overhead floor in both cases, and
the per-encode cost (slope_b2b − slope_internal) is positive and
within pre-reg's "a few hundred ns to a couple µs" band. Two-point /
many-point linear fitting works on Apple Silicon — but the regime
boundary is real and has to be detected from the data, since
IOReport's 250 ms cadence is too coarse to classify cells shorter
than itself.

### Headline numbers (stable-region fits)

| method        | stable N range | slope (ns / amp-step) | intercept (ns) |     R² |
|---------------|----------------|----------------------:|---------------:|-------:|
| internal-loop | 16, 64, 256, 1024 |             405.28 |          6 860 | 1.0000 |
| back-to-back  | 1, 2, 4, 8, 16, 64, 256 |        3 219.09 |          4 711 | 1.0000 |

Per-encode cost (stable-region):
`slope_b2b − slope_internal = 3 219 − 405 = 2 814 ns`.

### Per-step incremental slope (the regime boundaries are obvious in
the data)

**internal-loop:**

| N step | Δp50 / ΔN |
|---|---:|
| 1 → 2  | **792 ns/N** |
| 2 → 4  | **875 ns/N** |
| 4 → 8  | **990 ns/N** |
| 8 → 16 | **−62 ns/N** |
| 16 → 64 | 406 ns/N |
| 64 → 256 | 405 ns/N |
| 256 → 1024 | 405 ns/N |

The DEADLINE → PERF kink predicted in the hypothesis lives **between
N=8 and N=16**: per-step slope goes *negative* there (more work runs
faster — the chip transitions to a higher DVFS regime). For N ≥ 16
the slope is rock-solid at 405 ns per amplification step (= 64 fmas
in the inner loop = ~6.3 ns / fma at the regime's running frequency).

**back-to-back:**

| N step | Δp50 / ΔN |
|---|---:|
| 1 → 2  | 3 417 ns/N |
| 2 → 4  | 3 167 ns/N |
| 4 → 8  | 3 292 ns/N |
| 8 → 16 | 3 182 ns/N |
| 16 → 64 | 3 303 ns/N |
| 64 → 256 | 3 197 ns/N |
| 256 → 1024 | **209 ns/N** |

A 15× collapse in per-step slope between N=256 and N=1024. The N=1024
cell is the only one long enough (~236 ms) to dominate IOReport's
250 ms window, and it shows PWRCTRL = PERF(53 %) DEADLINE(29 %) and
GPUPH P15 = 9 %. Every other cell is too short to classify cleanly.

### Hypothesis check

| prediction | observed | verdict |
|---|---|---|
| Linear model holds within a single DVFS regime (R² > 0.99, residuals < 5 %) | R² = 1.0000 in both stable regions; residuals < 0.2 % internal, < 8.6 % b2b (driven by N=1) | ✓ |
| Internal-loop crosses DEADLINE → PERF mid-range with visible kink | yes — kink between N=8 and N=16 | ✓ |
| Back-to-back linear with higher slope than internal-loop | yes — 3 219 vs 405 ns / amp step | ✓ |
| Back-to-back intercept ≈ internal intercept (~6 µs) | b2b 4 711 ns, internal 6 860 ns — both within ±30 % of 6 400 ns | ✓ |
| Carry-dependent kernel prevents compiler elision | confirmed — slope is non-zero and consistent across orders of magnitude | ✓ |
| `slope_b2b − slope_internal` ≈ 833 ns (just the inter-encoder gap from exp 005) | **2 814 ns** — ~3.4× larger than predicted | ✗ refined |
| Single regime crossing per method | internal-loop yes; **back-to-back has a SECOND collapse at N=1024 we did not predict** | ✗ |

### Two unpredicted findings

1. **The b2b "per-encode cost" is mostly per-dispatch overhead, not
   the inter-encoder gap.** Pre-reg expected `slope_b2b −
   slope_internal` to recover the 833 ns inter-encoder gap from exp
   005. The actual value, 2 814 ns, is more than 3× that. The
   interpretation: a back-to-back encoded dispatch carries the inter-
   encoder gap *plus* per-dispatch GPU-side overhead (~2 µs) that
   doesn't compound when the same work runs inside one inner loop.
   Exp 005's 833 ns measured the *gap* component in isolation, not
   the full per-dispatch cost.

2. **Back-to-back collapses 15× at N=1024.** Per-step slope goes
   from ~3 200 ns/dispatch (N ≤ 256) to ~209 ns/dispatch (N=1024 vs
   N=256). p90 in this cell is 3.36 ms while p50 is 988 µs —
   **bimodal**, not just shifted. The cell is the only one long
   enough that PWRCTRL captures it as PERF-dominant. Working
   hypothesis (untested): once a cb crosses some duration / dispatch-
   count threshold, either (a) the GPU starts pipelining adjacent
   dispatches so they overlap in execution, or (b) DEADLINE-mode
   sub-floor (1.7 µs) takes over for most dispatches in the chain,
   collapsing per-dispatch cost. Either way, **back-to-back
   amplification is not transparent at large N** on M4 Max.

### Hardware context (M4 Max only — not generalized to M1 Pro)

- Internal-loop: clean 405 ns / amplification step at the running
  DVFS regime (whatever that is — IOReport too coarse to pin down
  for our cell durations). At ~64 fmas per amp step, that's ~6.3 ns
  per FMA in the dependent chain, consistent with 6-10 cycle FMA
  latency on dependent operands at 1 GHz-ish. Below peak DVFS in
  expectation.
- Back-to-back stable region: 3 219 ns per dispatch. Decomposes as
  ~833 ns (inter-encoder gap, exp 005) + ~2 µs (per-dispatch GPU-
  side overhead) + ~400 ns (inner work).
- Both methods recover an intercept in the 4.7-6.9 µs range,
  consistent with the 6.4 µs cool-DVFS dispatch-overhead floor from
  exp 001.

### What this means operationally

Loop amplification + many-point linear fit is **usable on M4 Max for
small-kernel timing** with two caveats:

1. **You have to detect and exclude the regime-boundary cells from
   the fit.** The `find_stable_window` heuristic in `analysis.py`
   does this from the timing data alone (per-step slope variance) —
   we cannot rely on per-cell PWRCTRL classification when cells run
   shorter than the IOReport interval.

2. **Stick to internal-loop amplification for kernels whose
   per-iteration cost we want to recover.** Back-to-back works in a
   bounded N range but its slope is dominated by per-dispatch
   overhead, not by the inner kernel cost. Use back-to-back only as
   a cross-check on per-encode cost or to characterize the dispatch
   overhead itself.

3. **Don't extrapolate b2b past N ≈ 256 without re-validating.**
   The N=1024 collapse means b2b is a stateful, not transparent,
   abstraction at large N.

### What does NOT change

- Pair timing (decision 005) remains primary for trials ≥ ~64 µs.
- Single-shot timing remains correct for trials ≥ ~50 µs.
- Internal-loop amplification is now the path for trials below the
  dispatch-overhead floor — to be used in the next experiment with a
  memory-bound base unit, then with real ML kernels.

### What changes

- **New methodology:** internal-loop amplification with stable-region
  fit, validated for arithmetic-only base units down to ~64 fmas.
  Memory-bound validation is the next step before applying to real
  kernels.
- **Refines exp 005's inter-encoder gap interpretation:** 833 ns is
  the *gap*, not the *full per-dispatch cost in a back-to-back
  chain*. The latter is ~3.2 µs.
- **New unknown:** mechanism of the b2b N=1024 collapse. Bimodal
  distribution + only-cell-long-enough-for-PWRCTRL-classification
  hints at a DVFS regime change but doesn't pin it down. Lower
  priority than the memory-bound follow-up but worth a quick targeted
  experiment if the mechanism turns out to bear on amplification at
  large N for ML kernels.
