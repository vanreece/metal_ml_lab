# What we know we don't know

Maintained list of open questions, ordered roughly by how much each one
matters for whether the project can work at all. Updated as questions
become answered or as new ones surface.

When something here moves to "answered," it gets a brief note about the
answer and a link to the experiment that closed it, then moves to
`notes/answered-questions.md`.

## Foundational (everything depends on these)

- What is the noise floor (σ/μ) of `MTLCounterSampleBuffer` timestamp-based
  timing on M1 Pro under user-interactive QoS, AC power, no thermal
  pressure?
- What is the smallest kernel duration we can reliably distinguish from
  noise?
- Does `device.supportsCounterSampling(at:)` return useful values for any
  sampling point besides `atStageBoundary` on M-series? (Strong public
  signal that the answer is no, but verify on actual hardware.)

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
