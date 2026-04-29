# What we know we don't know

Maintained list of open questions, ordered roughly by how much each one
matters for whether the project can work at all. Updated as questions
become answered or as new ones surface.

When something here moves to "answered," it gets a brief note about the
answer and a link to the experiment that closed it, then moves to
`notes/answered-questions.md`.

## Foundational (everything depends on these)

- ~~What is the noise floor (σ/μ) of `MTLCounterSampleBuffer`
  timestamp-based timing on M1 Pro under user-interactive QoS, AC power,
  no thermal pressure?~~ **Substantively answered by 002 for the
  trivial 32-thread `write_tid` kernel:** cv depends sharply on
  inter-dispatch idle. cv ≈ 0.66 at sleep=0 (dominated by quantization +
  rare outlier), 7.0 at sleep=1ms (bimodal, nightmare zone), 2.7 at
  sleep=10ms, 2.2 at sleep=100ms, 0.21 at sleep=1s. The right answer
  to "what is the noise floor" is "depends entirely on what cadence
  you operate at." See `notes/answered-questions.md`. Still open: how
  this scales with kernel size and load.
  **Updated by 003:** at long cadences (≥10 ms), a single untimed
  warmup dispatch right before the measurement collapses cv back to
  ≈ 0.06 — the cool-cadence noise is recoverable. The 1-10 ms
  "transition zone" from 002 may be similarly recoverable, but
  003's calibration probe contaminated the K=0 baseline so we can
  not directly test that here.
- ~~What is the smallest kernel duration we can reliably distinguish from
  noise?~~ **Substantively answered by 001+002+003+004.** Timestamp tick
  is ~42 ns; the dispatch-overhead floor sits at ~8 µs back-to-back
  (K=0) and ~9 µs with K=1 warmup; cv depends on cadence (002).
  **004 reframes this:** the 8 µs / 9 µs floors are *dispatch-overhead*
  floors, not kernel floors — they show up regardless of kernel work
  below the work-dominance threshold. The smallest kernel work
  reliably distinguishable from overhead is the work that takes the
  median above the overhead floor by enough to fall outside its
  variance. For `write_tid` that means ≥ 131K-262K threads on M1 Pro;
  for `fma_loop` that means ≥ 256 iters per thread (with a +21 µs
  step at the 192→256 boundary that is itself unexplained). See
  `notes/answered-questions.md`.
- ~~**NEW from 003: What entry conditions reach the ~5.4 µs settled
  state?**~~ **Closed by 004 as non-reproducible.** Across 5 sequential
  attempts (each a fresh post-cooldown calibration burst followed by
  40 measured dispatches with no warmup), 0/200 measurements landed
  in the [5000, 6000] ns window. Every attempt converged to the
  familiar ~8 µs back-to-back floor. Filed as a single-launch artifact
  whose mechanism we cannot isolate from the available data. See
  `notes/answered-questions.md`.
- **NEW from 004: What is the +21 µs step at `fma_loop` iters
  192→256?** Median jumps from 12.6 µs to 34.2 µs — three orders of
  magnitude bigger than the actual work added by 64 extra FMAs. Looks
  like a compiler-or-hardware threshold (register spilling,
  unrolling-heuristic flip, instruction-cache crossover) but we have
  not isolated which. Affects whether kernel-complexity is a smooth
  axis we can rely on. Probably answerable cheaply by dumping Metal
  AIR / GPU assembly at the two boundary points.
- **NEW from 004: Why does `fma_loop` exhibit bimodal between-sweep
  variance at iters = 8192-16384?** Within a single script run with
  30s between-sweep cooldown, p50s vary 2.5× across sweeps in this
  duration band (~0.4-1.6 ms dispatches), but are stable everywhere
  else. Plausibly related to 002's "1-10 ms transition zone" of the
  GPU power state machine, but we have not measured directly.
- ~~**NEW from 005: Does paired-encoder ratio timing remain stable
  across separate script invocations?**~~ **Closed by 006 (2026-04-28)
  on M4 Max: YES for trials ≥ 10× the dispatch-overhead floor.**
  Across two M4 Max sessions 30 min apart, T2/T3/T4 ratios all
  agreed to ≤ 0.6 % spread (PASS). T1 (the smallest trial, ~6× the
  floor) was MARGINAL at 2.77 %. Headline finding: T2's *alone* cv
  differed by 335× between sessions while the ratio cv was
  essentially identical — exactly the variance-cancellation
  mechanism decision 003 originally claimed. Decision 005
  supersedes decision 004 on M4 Max as a result. **Still open:**
  longer idle gaps (4 hours / overnight); cross-thermal-state;
  M1 Pro cross-session test (untested).
- ~~**NEW from 005: What sets the ~42 µs inter-encoder gap inside a
  single MTLCommandBuffer, and can it be reduced?**~~ **Partial
  answer (2026-04-28):** the gap is **chip-specific** — on M4 Max
  / `applegpu_g16s` / macOS 26.4.1, the same pattern produces a
  p50 gap of **~833 ns** (50× reduction from M1 Pro's 42 µs).
  Whatever mechanism produced the M1 Pro gap was either fixed or
  vastly cheaper on the G16 front end. **Open:** what specifically
  changed between G13 and G16 to collapse the gap? Curiosity
  question more than methodology question now that we know the gap
  is recoverable on newer hardware. Also unmeasured: G14 / G15
  intermediate generations.
- **NEW from 003: Does the calibration kernel as thermometer have
  enough resolution to distinguish power states without disturbing
  them?** Partial answer: a 10-dispatch probe can tell hot vs cool
  reliably (median-of-tail), but a single probe is too noisy. And the
  10-dispatch probe is itself a 10-dispatch warmup, so observation
  affects state. Still open: probe designs that minimize
  observer-effect.
- ~~Does `device.supportsCounterSampling(at:)` return useful values for
  any sampling point besides `atStageBoundary` on M-series?~~ **Answered
  by 001 on M1 Pro / macOS 26.3.1: no.** `atDraw`, `atBlit`, `atDispatch`,
  `atTileDispatch` all return False. See `notes/answered-questions.md`.

## Methodology (depend on the foundation working)

- Does reading GPU frequency from `powermetrics` during a measurement
  window correlate with timing variance well enough to use as a
  sample-rejection signal?
  **Status (2026-04-28):** powermetrics is now available, but
  `notes/ioreport.py` (sudo-free) gives us the same GPU power signal
  with bonus per-component breakdowns (CPU, DRAM, AMCC, DCS, AFR,
  DISP). Either signal could underpin a sample-rejection rule.
  Untested as a rejection criterion.
- How well do our microbenched peak FLOPS and memory bandwidth match
  Philip Turner's published numbers on M1 Pro? (Close = our measurement
  is sound. Far = something is off and downstream work is suspect.)
- Can controlled microbench perturbations (varying threadgroup size,
  problem size, register pressure via temporary inflation) discriminate
  between bottleneck classes well enough to be useful?
- For which bottleneck classes can they discriminate, and for which can
  they not?

## Telemetry stack (new section, 2026-04-28)

- **Does IOReport's `GPU Active Time Histogram` channel give a
  usable utilization signal?** Exp 007 falsified ioreg's
  utilization fields (driver-side update too sparse). The IOReport
  histogram channel uses the consumer-side delta API and may not
  have the same problem. Untested. Same comparison shape as 008
  (vs powermetrics active residency); pre-registerable as exp 010.
- **What causes the +14 % bias of IOReport GPU power vs
  powermetrics at full saturation?** Exp 008 found IOReport reads
  consistently +14 % higher than powermetrics during step_100pct
  on a known fma_loop workload. Direction inverts at tail (-20 %).
  Mechanism not isolated. A focused channel-subset sweep would
  narrow it; pre-registerable as exp 009.
- **Can we read GPU frequency / per-P-state residency from
  IOReport?** The `GPU Stats / GPU Performance States` group
  exposes `GPUPH` (STATE format) — needs IOReportStateGetCount /
  IOReportStateGetResidency bindings (we have only SimpleGet so
  far). Useful for "what frequency was the GPU at during this
  measurement" — would address the longstanding "DVFS state is
  unobservable" gap.
- **Is there a sudo-free temp / fan-RPM source?** Exp 008's
  IOReport CSV has zero-valued temperature columns because the
  channel-name heuristic doesn't match. The agent-research
  recommended IOHID via a separate API; would need a Swift bridge
  or PyObjC ride-along since libIOHID's API isn't ctypes-friendly.

## Generalization (only matter if methodology works)

- ~~Does any of this generalize from M1 Pro to M4 Max? (M3+ Dynamic
  Caching changes occupancy behavior in ways the public docs don't
  fully describe.)~~ **Partially answered (2026-04-28):** the
  *infrastructure* generalizes — same single counter set, same single
  sampling point, same ~24 MHz timestamp tick on M4 Max
  (`applegpu_g16s`) / macOS 26.4.1 as on M1 Pro / 26.3.1. The *baseline
  numbers* do not — dispatch-overhead floor dropped ~20 % on M4 Max
  (8.0 µs → 6.4 µs at p50 for write_tid 32t back-to-back) and tail
  behavior was dramatically tighter in the one re-run we have (max
  25 µs vs 926 µs, 37× reduction; needs reproduction). Sub-questions
  still open below.

  **M4 Max status as of 2026-04-28 (end of day):**
  All five M1 Pro experiments have been re-run on M4 Max. The
  baseline numbers and several qualitative findings differ:
  - **001:** dispatch-overhead floor 8.0 µs → 6.4 µs (~20 % drop).
    Tail behavior dramatically tighter (max 25 µs vs 926 µs).
    Counter set / sampling point / 24 MHz tick all unchanged.
  - **002:** the M1 Pro "1ms nightmare zone" is GONE. sleep_1ms is
    the *cleanest* M4 Max cadence (cv = 0.07 vs M1 Pro 7.03). cv
    ordering across cadences nearly inverted; M1 Pro recipes do
    not transfer.
  - **003:** warmup-kind ranking inverted. `same K=1` (M1 Pro's
    safest) is M4 Max's destabilizer; `fma_loop K=1` (M1 Pro's
    destabilizer) is M4 Max-safe. New default: `heavy_write K=1`.
    A sub-floor ~2 µs settled state was observed in `fma_loop
    K=20 sleep_0` (mins to 2.083 µs) — single-combo finding,
    reproducibility untested.
  - **004:** work-dominance threshold for `write_tid` shifted from
    131K-262K (M1 Pro) to 262K-524K threads (M4 Max) — needs ~2×
    more threads to escape overhead. The +21 µs `fma_loop` step at
    iters 192→256 EXISTS on M4 Max as +13 µs at the same boundary
    — confirms it's a Metal-compiler/microarch threshold, not a
    chip-specific quirk. Bimodal band shifted to 4K-16K iters
    (M1 Pro: 8K-16K).
  - **005:** **inter-encoder gap collapsed from 42 µs to 833 ns
    (50× reduction).** Variance reduction now WORKS on M4 Max for
    noisy trials (T2 alone cv 0.336 → ratio cv 0.005, 63×
    improvement). Validated decision 003's original claim. Tail
    suppression (M1 Pro 005's positive surprise) does NOT broadly
    reproduce on M4 Max (only T4 saw meaningful tail reduction).

  **Open M4 Max methodology questions as of 2026-04-28:**
  - The +13 µs M4 Max step at fma_loop 192→256 — same Metal
    compiler threshold as M1 Pro's +21 µs step? AIR / GPU assembly
    inspection would distinguish. Architectural curiosity, not
    blocking methodology.
  - The ~2 µs sub-floor state from 003 M4 Max — reproducible? A
    003a-style focused re-run (fma_loop K=20 sleep_0 × 5
    attempts) would test it.
  - Cross-thermal-state ratio stability (M4 Max). 006 covered
    30-min idle gap; longer / hotter conditions untested.

- Does the methodology transfer to non-MLX kernels (raw `mx.fast.metal_kernel`
  vs PyTorch MPS vs llama.cpp shaders)?

## Out of scope right now (revisit later)

- M5 Neural Accelerators / MTLTensor / MPP integration.
- `.gputrace` bundle reverse-engineering as a counter-access alternative.
- Cross-vendor abstraction layer (Apple → AMD → Intel).
- LLM-driven kernel generation harness on top of this layer.

## What changes a question's priority

A question moves up if a downstream experiment is blocked on it. A question
moves down if we find a way to proceed without answering it definitively.
A question is removed if we decide the project doesn't need it answered at
all.
