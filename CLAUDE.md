# Project context for Claude Code

This file is read by Claude Code at the start of every session in this repo.
It exists to keep the agent's behavior aligned with the project's epistemics,
not just its goals.

## What this project is

An exploration of whether Apple Silicon GPU kernel performance — bottleneck
class, roofline position, occupancy effects — can be characterized
programmatically without access to the vendor-internal counters that Xcode's
Metal Debugger exposes but the public Metal API does not.

It is also, deliberately, a case study in how foundational tooling for
underprovisioned high-performance ecosystems can be built agent-first and in
public. The methodology is as much a deliverable as the code.

## How we work

**One experiment at a time.** Each experiment answers a specific, falsifiable
question. We do not write code that anticipates future experiments. We do
not build APIs before we know what they should expose. Speculative
infrastructure is the failure mode to actively resist.

**Maximum uncertainty reduction, not maximum progress toward a product.**
The right next experiment is the one that most reduces uncertainty about
whether the overall approach can work. It is often not the experiment that
looks most like progress.

**Document decisions before outcomes.** When choosing between approaches,
write down the choice and the reasoning in `decisions/` *before* finding out
whether it worked. This is the only way the decision log stays honest.

**Negative results are first-class.** If an experiment shows the idea
doesn't work, that's a successful experiment. Write it up the same way as
a positive result. The temptation to quietly retry with different parameters
until something works is the enemy.

**Raw before robust.** When measuring something for the first time, write
code that exposes the variance, the outliers, the weird behavior. Do not
wrap things in averaging, retry logic, or smoothing until we understand
what's being smoothed away. Robustness is a later layer; instrumentation is
the first layer.

## Specific guidance for code generation

- Prefer scripts in `experiments/NNN-short-name/` over library code. Each
  experiment is self-contained, including its own small utilities if needed.
  Duplication across experiments is fine and expected at this stage.
- When asked to time something, default to showing percentiles and raw
  samples, not just means. Show the distribution.
- When unsure about a measurement methodology, say so explicitly in the
  output and in comments. "I used N=100 reps because that felt reasonable;
  we have not validated N" is better than silently picking a number.
- Do not add warmup, retries, or averaging unless the experiment is
  specifically about characterizing those things. Premature smoothing is
  the bigger risk than noisy data.
- PyObjC is the default path for talking to Metal from Python. A small
  Swift bridge is acceptable if PyObjC fights us on a specific API. Do not
  introduce a third path without writing a decision note.
- Apple Silicon DVFS cannot be locked. Every measurement should be
  considered suspect until we have a methodology for handling clock
  variation. Any code that assumes stable clocks needs a comment saying so.

## What we know we don't know

This list is intentionally maintained. Update it when something moves from
"unknown" to "known" or when a new unknown appears.

- What is the noise floor (σ/μ) of `MTLCounterSampleBuffer` timestamp-based
  timing on M1 Pro under user-interactive QoS, AC power, no thermal
  pressure? On M4 Max?
- What is the smallest kernel duration we can reliably distinguish from
  noise?
- Does reading GPU frequency from powermetrics during the measurement
  window correlate with timing variance enough to use as a sample-rejection
  signal?
- Does `device.supportsCounterSampling(at:)` actually return useful values
  for `atDispatchBoundary` on any M-series chip, or only `atStageBoundary`?
- How well does our STREAM-style bandwidth bench match Philip Turner's
  published numbers on M1 Pro?
- How well does our FFMA peak-FLOPS bench match published numbers?
- Can controlled microbench perturbations (varying threadgroup size,
  problem size, register pressure via temporary inflation) discriminate
  between bottleneck classes well enough to be useful?
- For which bottleneck classes can they discriminate, and for which can
  they not?
- Does any of this generalize across M1 Pro and M4 Max, or are we
  characterizing one chip at a time?

## What is explicitly out of scope right now

- Building a public-facing library API.
- Writing a benchmark suite (KernelBench-Metal etc.).
- Building an LLM-driven kernel generation harness.
- Reverse-engineering `.gputrace` bundle format.
- M5 Neural Accelerator work.
- Cross-vendor abstractions.
- All of these are interesting. None of them are the next experiment.

## House style

- Code: type hints where they help, no type hints where they hurt
  readability. Black/ruff defaults are fine. No premature abstraction.
- Commit messages: short imperative subject, body explains *why* if
  non-obvious. Reference experiment number where applicable.
- File naming: `experiments/NNN-kebab-case-name/` where NNN is zero-padded
  three digits. Decisions: `decisions/NNN-kebab-case-name.md`.
