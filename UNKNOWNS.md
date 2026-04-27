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
- What is the smallest kernel duration we can reliably distinguish from
  noise? (Partial answer from 001+002: timestamp tick is ~42 ns on M1
  Pro; per-dispatch floor is ~8 µs at sleep=0 with cv dominated by
  quantization. "Reliably distinguish from noise" depends on cadence —
  see 002. **Updated by 003:** the floor is not even 8 µs — under
  specific entry conditions hit accidentally at script start, the
  same kernel ran in ~5.4 µs with cv=0.05. There is at least one
  faster settled state than 001/002 ever observed. Reproducibility
  of this state is the highest-priority unknown right now.)
- **NEW from 003: What entry conditions reach the ~5.4 µs settled
  state?** Observed once, at the very first combo of 003: cold script
  start + caffeinate launch + 1s cooldown + 10-dispatch calibration
  burst → all 40 subsequent measured dispatches at p50=5375 ns. Never
  reproduced anywhere else in the same run. Until this is reproduced,
  every "fastest possible" timing claim has a footnote.
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
