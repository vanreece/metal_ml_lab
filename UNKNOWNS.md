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
- **NEW from 005: Does paired-encoder ratio timing remain stable
  across separate script invocations?** Within-session ratio
  stability is established (≤1% drift across 3 sweeps, per 005).
  Cross-session stability (different process, different time of
  day, different thermal state) is the strongest remaining
  justification for pair timing as a primary methodology and is
  untested. A 006 experiment running the same paired conditions
  twice with a 1-hour gap would close this. Decision 004's
  cross-session validity is contingent on it.
- **NEW from 005: What sets the ~42 µs inter-encoder gap inside a
  single MTLCommandBuffer, and can it be reduced?** Two consecutive
  `MTLComputeCommandEncoder` passes in one cb are separated by a
  p50 idle of ~42 µs (range 8-50 µs at p99). This is ~4× the
  dispatch-overhead floor and means "same buffer = same chip state"
  is approximate, not exact. Mechanism unclear: per-encoder setup
  cost, command-list reordering, GPU front-end stall? Inspecting
  the `MTLCommandBuffer` ordering semantics or trying alternative
  encoding patterns (separate cbs with different ordering, command
  buffer encoders with explicit barriers) might distinguish.
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
- How well do our microbenched peak FLOPS and memory bandwidth match
  Philip Turner's published numbers on M1 Pro? (Close = our measurement
  is sound. Far = something is off and downstream work is suspect.)
- Can controlled microbench perturbations (varying threadgroup size,
  problem size, register pressure via temporary inflation) discriminate
  between bottleneck classes well enough to be useful?
- For which bottleneck classes can they discriminate, and for which can
  they not?

## Generalization (only matter if methodology works)

- Does any of this generalize from M1 Pro to M4 Max? (M3+ Dynamic Caching
  changes occupancy behavior in ways the public docs don't fully describe.)
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
