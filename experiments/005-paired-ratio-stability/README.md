# 005: Paired-kernel ratio stability — does co-encoded pair timing actually reduce variance?

**Status:** pre-registered, not yet implemented
**Date pre-registered:** 2026-04-27
**Hardware target:** M1 Pro (16GB), macOS 26.3.1
**Estimated runtime:** ~5-8 min unattended (single script invocation;
cross-session validation is explicitly NOT in this experiment)

## The question

When we encode two compute passes back-to-back into the same Metal
command buffer — a fixed reference kernel followed by the
kernel-of-interest, each with its own timestamp pair — and report
`trial_delta / ref_delta` as the measurement:

**Primary 1: Does the ratio have meaningfully lower robust cv than
either kernel measured alone?** This is the entire load-bearing claim
of decision 003. If the ratio's spread is comparable to or worse than
the alone-trial spread, paired timing is just adding work without
adding signal, and decision 003 needs to be superseded.

**Primary 2: Does pairing change what the trial measures?** Compare
`trial_delta` when paired vs the same trial kernel measured alone in
the same script invocation. If the median drifts by more than ~5%,
the trial in a paired context is measuring something different from
the kernel in isolation (cache state, frontend pipelining, occupancy
sharing — any of these could plausibly shift things). The ratio is
still useful as a relative metric, but the interpretation has to
account for the perturbation.

**Primary 3: Is the ratio stable across the 3 within-session sweep
repetitions?** If sweep 0 and sweep 2 produce different ratio
medians by more than ~1%, even within a single script invocation the
chip's state is drifting in a way the ratio fails to compensate for,
which weakens the case for the technique.

## Why this is the right next experiment

Two findings from 004 push us into a methodology question rather than
the planned multi-kernel state profile:

1. The ~8 µs floor is dispatch overhead, not kernel time. Any
   single-kernel timing report that doesn't account for this is a
   characterization of overhead, not of the kernel.
2. The chip's state visibly drifts within a single script invocation
   (bimodality at fma_iters 8192-16384), and we have no way to read
   the current GPU clock from public Metal APIs. So absolute timing
   has a structural ceiling on its informativeness.

Decision 003 commits the project to ratio-based pair timing as the
primary methodology. 005 is the test of that commitment. Running 005
before any larger multi-kernel state profile experiment is critical:
if the methodology underneath does not hold, every probe-vector design
decision downstream is based on a broken assumption.

This is the cheapest possible falsification of decision 003.

## Hypothesis

Confidence: medium. Stating predictions before running:

- **Primary 1 (ratio cv reduction):** the trial/ref ratio's robust cv
  will be at least 2× lower than trial-alone robust cv for the
  compute-bound trial pairings (T1, T2 — same character as the
  reference). For mixed-character pairings (memory-bound trial vs
  compute-bound reference, T3 and T4), prediction is less confident:
  ratio cv may be lower, comparable, or in the worst case higher
  than alone, because the two halves of the ratio are responding to
  different state dimensions.

- **Primary 2 (pairing perturbation):** pairing will shift the
  trial's median by less than 5% relative to alone. Both trials will
  see the same DVFS state in either context (since the script as a
  whole is in the same warm state); pairing introduces some cache
  pollution (the reference's outputs are recently written) and some
  pipeline-pressure differences, but these should be small at the
  durations we operate at (≥ 50 µs trials).

- **Primary 3 (within-session stability):** the ratio's per-sweep
  median will agree within 1% across the 3 sweeps. This is the
  prediction with the highest confidence — by construction the ratio
  is supposed to absorb between-sweep state drift.

- **Subsidiary:** the *reference* kernel's measured delta when paired
  will be ~5-15% slower than its alone median (because the trial's
  preceding-or-following dispatch shares queue/cache resources). The
  fact that ref_delta moves doesn't break the methodology — it
  *should* move for the ratio to be useful — but quantifying the
  movement helps us understand what we're normalizing against.

## What we are NOT trying to answer

- **Cross-session ratio stability.** That requires running the script
  twice on different days / power states / ambient temperatures and
  comparing ratios across sessions. Important question, but
  separable. If 005 succeeds, a 005b can do this with minimal
  additional code.
- **What's the optimal reference kernel?** We pick one (fma_loop
  iters=1024, the tightest robust cv we measured in 004 in the
  work-dominated regime) and stick with it. Comparing reference
  candidates is an optimization that only matters once we know
  pairing works at all.
- **What's the optimal pairing pattern?** We use [ref, trial]
  same-buffer ordering. We do NOT compare to [trial, ref] or
  sandwich [ref, trial, ref] orderings. Those become interesting
  only if the basic technique works.
- **Bottleneck classification.** The whole point of 005 is to
  validate the unit of measurement. Building a probe vector and
  testing it for classification ability is the experiment-after-this
  if 005 succeeds.
- **Multi-kernel state profiling** as originally sketched. That was
  the planned 005, deprioritized to be an experiment-after this.
- **Anything about M4 Max** or other Apple Silicon variants. Same
  caveat as 004.

## Setup

### Hardware / software

- M1 Pro 16GB, macOS 26.3.1, AC power, laptop awake.
- User-interactive QoS via `pthread_set_qos_class_self_np`.
- uv + PEP 723 inline metadata, same pattern as 001-004.
- `caffeinate -d -i -m` sidecar holds display awake throughout.
- `EXP005_NO_POWERMETRICS=1` env var skips the powermetrics sidecar
  by default. (Same gate as 004.)

### Reference kernel (fixed across all paired conditions)

`fma_loop` with `FMA_ITERS = 1024` at 32 threads. Selected from 004
data because:
- p50 = 107 µs, well above the dispatch-overhead floor (~9 µs).
- Robust cv = 0.003 measured alone — the tightest we found.
- Solidly above the 192→256 step discontinuity, so the kernel is in
  the linear-with-iters regime.
- Below the 4096-iter mark, so well clear of the 8192-16384 bimodal
  band.
- Compute-bound kernel of known character.

The reference is **not parameterized in this experiment** (a single
fixed value of FMA_ITERS). One MTLLibrary, one MTLComputePipelineState
for ref.

### Trial kernels (4 conditions)

Selected to span character (compute vs memory) × duration
(shorter / similar / longer than reference):

| symbol | kernel              | params               | 004 alone p50 | 004 alone robust_cv | character    | duration vs ref |
|--------|---------------------|----------------------|--------------:|--------------------:|--------------|-----------------|
| T1     | `fma_loop`          | iters=512, 32t       |     58 500 ns |               0.014 | compute      | 0.5×            |
| T2     | `fma_loop`          | iters=4096, 32t      |    398 916 ns |               0.147 | compute      | 4×              |
| T3     | `write_tid`         | 524 288 threads      |     62 291 ns |               0.030 | memory-bound | 0.6×            |
| T4     | `write_tid`         | 1 048 576 threads    |    102 750 ns |               0.038 | memory-bound | 1×              |

T1 and T2 are same-character-as-ref (compute-bound, fma_loop in the
linear regime). T3 and T4 are different-character (memory-bound
write_tid in the work-dominated regime). T2's higher robust cv (0.147)
is mostly tail outliers from the bimodality band's edge — included
deliberately to see whether ratio-pairing helps a *noisy* trial.

### Per-trial-kernel conditions

For each of T1-T4, we run two conditions:

1. **`alone`**: trial measured by itself. Single compute pass, single
   timestamp pair, K=1 untimed warmup of the trial kernel before each
   measurement (per the 003 recipe).
2. **`paired`**: ref + trial co-encoded. Single command buffer with
   two sequential compute passes; pass 1 is the reference, pass 2 is
   the trial; each pass has its own timestamp start+end. K=1 untimed
   warmup of the trial kernel only (we do *not* untimed-warmup the
   reference; the reference's first dispatch in the paired condition
   sees a "fresh" pipeline of the kind it'll see in real use). No
   inter-pass sleep.

We additionally run **one `ref_alone` condition**: reference measured
by itself with its own K=1 untimed warmup. Used as the baseline for
ref's solo behavior.

That is 9 total conditions (4 trials × 2 modes + 1 ref_alone).

### Sweep design

300 trials per (condition, sweep). 3 sweeps with 30 s between-sweep
cooldown. 2 s per-condition cooldown within a sweep. Per-trial
cadence is sleep_0 (back-to-back) within a condition.

Total measurements per condition per sweep:
- `alone` and `ref_alone`: 300 trials × 1 timestamp pair = 300 deltas.
- `paired`: 300 trials × 2 timestamp pairs = 600 deltas (300 ref +
  300 trial), plus 300 derived ratios.

Total measurements:
- `alone` ×4 + `ref_alone` ×1 = 5 × 300 × 3 = 4 500 deltas.
- `paired` ×4 = 4 × 600 × 3 = 7 200 deltas (3 600 derived ratios).
- Grand total: 11 700 individual GPU dispatches measured.

### Encoded-pair pattern (the technical bit)

```
cb = queue.commandBuffer()

# pass 1: reference
pd_ref = MTLComputePassDescriptor.computePassDescriptor()
pd_ref.sampleBufferAttachments[0].setSampleBuffer(sample_buffer)
pd_ref.sampleBufferAttachments[0].setStartOfEncoderSampleIndex(slot_ref_start)
pd_ref.sampleBufferAttachments[0].setEndOfEncoderSampleIndex(slot_ref_end)
encoder_ref = cb.computeCommandEncoderWithDescriptor(pd_ref)
encoder_ref.setComputePipelineState(ref_pipeline)
encoder_ref.setBuffer(out_buffer, ...)
encoder_ref.dispatchThreads(...)
encoder_ref.endEncoding()

# pass 2: trial
pd_trial = MTLComputePassDescriptor.computePassDescriptor()
pd_trial.sampleBufferAttachments[0].setSampleBuffer(sample_buffer)
pd_trial.sampleBufferAttachments[0].setStartOfEncoderSampleIndex(slot_trial_start)
pd_trial.sampleBufferAttachments[0].setEndOfEncoderSampleIndex(slot_trial_end)
encoder_trial = cb.computeCommandEncoderWithDescriptor(pd_trial)
encoder_trial.setComputePipelineState(trial_pipeline)
encoder_trial.setBuffer(out_buffer, ...)
encoder_trial.dispatchThreads(...)
encoder_trial.endEncoding()

cb.commit()
cb.waitUntilCompleted()

# resolve four timestamps; compute ref_delta and trial_delta
```

Important caveat: Metal's `MTLCounterSamplingPointAtStageBoundary`
samples at the encoder boundary, not the dispatch boundary (see
`notes/answered-questions.md`). This is exactly why we use *two
encoders* — one per kernel — within the same command buffer. The
encoder-end timestamp of pass 1 and the encoder-start timestamp of
pass 2 may have a small gap; quantifying that gap is part of the
analysis.

### What we record

Per `alone` / `ref_alone` trial (one CSV per script invocation,
"alone CSV"):
- `condition` (`T1_alone`, `T2_alone`, `T3_alone`, `T4_alone`,
  `ref_alone`)
- `sweep_idx` (0..2)
- `trial_idx_within_combo` (0..299)
- `wall_clock_ns`
- `gpu_t_start_raw`, `gpu_t_end_raw`, `gpu_delta_raw`
- `cpu_encode_ns`, `cpu_commit_ns`, `cpu_wait_ns`, `cpu_total_ns`

Per `paired` trial (one CSV, "paired CSV"):
- `condition` (`T1_paired`, `T2_paired`, `T3_paired`, `T4_paired`)
- `sweep_idx`, `trial_idx_within_combo`
- `wall_clock_ns`
- `ref_t_start_raw`, `ref_t_end_raw`, `ref_delta_raw`
- `trial_t_start_raw`, `trial_t_end_raw`, `trial_delta_raw`
- `gap_ns` = `trial_t_start_raw - ref_t_end_raw` (the inter-pass gap
  inside the same command buffer)
- `ratio` = `trial_delta_raw / ref_delta_raw`
- `cpu_encode_ns`, `cpu_commit_ns`, `cpu_wait_ns`, `cpu_total_ns`

Metadata file with the same fields as 004's: device, OS, QoS, power,
display state, pmset assertions, timestamp correlation snapshots,
per-condition per-sweep summary tables.

### What we do NOT do

- No averaging in any live output. Raw deltas + ratios written to CSV.
- No outlier discarding.
- No retries, no exception swallowing.
- No varying of warmup K. Fixed at K=1 untimed warmup of the trial
  kernel (per 003 recipe), no warmup of the reference.
- No varying of pairing pattern (only `[ref, trial]` same-buffer).
- No cross-session checks (run the script once per writeup).
- No threadgroup-size variation.

## Success criterion

The experiment succeeds (in the discipline's sense) if we have:

1. Alone CSV with 5 conditions × 3 sweeps × 300 trials = 4 500 rows.
2. Paired CSV with 4 conditions × 3 sweeps × 300 trials = 3 600 rows.
3. Metadata file with per-condition per-sweep summaries.
4. Stdout log preserved alongside the CSVs.

It produces a usable answer if we can fill in:

**Question 1 — ratio variance:** for each trial T_i, compare:
- `robust_cv(T_i_alone gpu_delta)` to
- `robust_cv(T_i_paired ratio)`

The hypothesis is that the second is at least 2× lower than the
first for the compute-bound trials (T1, T2) and at least somewhat
lower for memory-bound trials (T3, T4). Outcome:
- *Clearly lower for all four (≥2×):* ratio timing is the right
  primary metric. Decision 003 is validated.
- *Lower for compute-bound, comparable for memory-bound:* ratio
  timing works for same-character pairings. The probe-vector design
  in a follow-on experiment needs to be character-matched. Decision
  003 stays but with a footnote about character-matching.
- *Not meaningfully lower in any pairing:* methodology fails.
  Supersede decision 003 with a new note that falls back to
  single-kernel timing with calibration baseline subtraction.

**Question 2 — pairing perturbation:** for each trial T_i, compute
`p50(T_i_paired trial_delta) / p50(T_i_alone gpu_delta)`. If this
is in [0.95, 1.05], pairing is benign. If it's outside that band,
the trial in a paired context is measuring a different thing.

**Question 3 — within-session ratio stability:** for each paired
condition, compute the per-sweep median ratio. The ratio of
`p50(sweep_2 ratio) / p50(sweep_0 ratio)` should be in [0.99, 1.01].

**Subsidiary — ref-alone vs ref-when-paired:** compute
`p50(ref_delta when paired) / p50(ref_alone gpu_delta)` per trial
condition. Quantifies how much the trial's presence shifts the
reference's behavior.

**Subsidiary — inter-pass gap:** report distribution of `gap_ns` from
the paired CSV. If the gap is large or variable, the same-buffer
pattern is not as tight as we hoped, and the variance reduction
hypothesis loses some of its mechanical justification.

## New questions we expect this to raise

- If ratio cv is much lower than alone cv but the ratio's *median*
  shifts non-trivially across sweeps (i.e. low intra-sweep variance
  but inter-sweep drift): the chip is in different states across
  sweeps even with 30 s cooldown, and the ratio is consistent
  *within* a state but not *across* states. We learn something about
  state dynamics but the methodology's cross-session story is in
  question.
- If the inter-pass `gap_ns` is consistently nontrivial (say, > 1 µs):
  the GPU front-end isn't as tightly coupled as we assumed, and the
  same-buffer pattern is closer to "two back-to-back commits" than
  to "atomic pair." The ratio still works but the marketing story
  changes.
- If pairing perturbs trial_delta differently for T1 vs T3 (same
  duration, different character): we learn that the perturbation is
  character-dependent, which constrains future probe-vector design.
- If T2 (the noisy one) shows enormous variance reduction: ratio
  timing does the most for the noisiest cases, which is the most
  useful possible outcome but also somewhat suspicious — would
  warrant inspecting the underlying samples to make sure we aren't
  hiding bimodality inside a stable ratio.
- If the reference kernel itself shows different cv when paired with
  T2 (long trial) vs T3 (memory-bound trial): the reference is being
  perturbed in trial-dependent ways, and the "fixed reference"
  framing has hidden state.

## After this experiment

Branches:

- **All three primary questions get clean yeses.** Decision 003 is
  validated. Pre-register a follow-on experiment (call it 006) for
  ratio-based bottleneck classification: pick 3-4 reference kernels
  of distinct character, encode each as a separate paired condition
  with the same trial, and use the *vector* of ratios as the
  classification signal. Cross-session validation (005b) is also
  worth scheduling.
- **Q1 yes, Q2 fails (perturbation > 5%).** Pairing changes what
  we're measuring, but the ratio is still a useful relative metric.
  Decision 003 stays with footnotes about interpretation. Follow-on
  is similar but has to be careful never to compare paired
  measurements to alone measurements as if they were the same thing.
- **Q1 fails for memory-bound.** Pairing only works for
  same-character pairings. Decision 003 stays but the probe-vector
  design has to be character-matched: each trial gets a reference of
  similar character. More complex but still workable.
- **Q1 fails everywhere.** Methodology fails. Supersede decision 003
  with `decisions/004-fallback-calibration-subtraction.md`. Adjust
  approach to single-kernel timing with periodic calibration
  baseline subtraction; rebuild plans accordingly.
- **Q3 fails (within-session ratio drift).** State control is worse
  than the bimodality finding suggested, and 30 s between-sweep
  cooldowns are insufficient. Investigate cooldown duration and/or
  warmup recipe before relying on the technique.

We do not plan past these branches.

## Notes for the post-run writeup

When this experiment is run and written up, the README should be
updated with sections matching the convention from 001-004:

- `## Result` (with summary tables for each trial condition)
- `## Surprises` (numbered, specific, with values)
- `## What this means operationally`
- `## New questions`
- `## After this experiment` updated to reflect actual findings

`UNKNOWNS.md` should be updated based on the outcome (the "is the
ratio stable" question moves to `notes/answered-questions.md`, plus
any new unknowns surfaced).

`decisions/003-paired-kernel-ratio-timing.md` either gets a status
update (active → validated) or a successor decision note that
supersedes it.
