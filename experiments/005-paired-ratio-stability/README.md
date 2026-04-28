# 005: Paired-kernel ratio stability — does co-encoded pair timing actually reduce variance?

**Status:** complete
**Date pre-registered:** 2026-04-27
**Date run:** 2026-04-27 (`raw/20260427T172313-*`)
**Hardware:** Apple M1 Pro 16GB, macOS 26.3.1, AC power, `applegpu_g13s`
**Actual runtime:** 125 s (~2 min — faster than estimated because
fma_loop iters=4096 trials are still well under the bimodality band,
and per-combo cooldowns dominated wall time).
**Outcome:** mixed. Headline claim falsified; secondary properties
hold. Decision 003 is superseded by decision 004 (narrowed scope).

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

## Result

**The headline claim of decision 003 — that paired-encoder ratio
timing reduces within-session variance — is falsified.** Across all
four trial kernels in the 50-400 µs range, the ratio's robust cv was
**equal to or higher than** the trial-alone robust cv, never the ≥2×
reduction the hypothesis predicted. The other two primary properties
(median preservation and within-session ratio stability) both held
cleanly. Plus two surprises: the inter-encoder gap inside a single
command buffer is much larger than expected (~42 µs at p50), and
pairing dramatically suppresses *tail outliers* in the trial
measurement even though it doesn't reduce the bulk distribution's
spread.

Pooled across 3 sweeps × 300 trials = N=900 per condition.

### Primary 1 (FAILS): ratio robust_cv vs trial-alone robust_cv

| trial             | alone p50 (ns) | alone robust_cv | paired-trial p50 | paired-trial robust_cv | ratio p50 | ratio robust_cv | verdict          |
|-------------------|---------------:|----------------:|-----------------:|-----------------------:|----------:|----------------:|------------------|
| T1 (fma_loop 512) |         58 542 |          0.0095 |           58 000 |                 0.0107 |    0.5398 |          0.0112 | ratio ≈ alone, slightly worse |
| T2 (fma_loop 4096)|        399 250 |          0.0012 |          398 750 |                 0.0013 |    3.7219 |          0.0057 | **ratio 4.75× worse than alone** |
| T3 (write_tid 524K)|        66 875 |          0.0346 |           66 417 |                 0.0363 |    0.6203 |          0.0374 | ratio ≈ alone    |
| T4 (write_tid 1M) |        110 812 |          0.0392 |          111 708 |                 0.0393 |    1.0403 |          0.0410 | ratio ≈ alone    |

For all four trials, ratio cv is at best comparable and at worst
~5× worse than alone cv. The pre-registered prediction of "≥2× lower
for compute-bound trials, somewhat lower for memory-bound" is wrong
in every cell. Mechanism: when the trial's cv is already at or near
the per-dispatch quantization floor (T2's alone cv = 0.001 is
effectively at quantization), dividing by another low-cv signal
*adds* the reference's cv into the ratio rather than canceling
anything. Variance composition for independent ratios:
`cv²(A/B) ≈ cv²(A) + cv²(B)`. The ratio's cv is dominated by
whichever of `A` or `B` has the higher cv — for our trials that's
sometimes the trial, sometimes the reference, but the result is
never a *reduction*.

### Primary 2 (PASSES): trial median paired vs alone

| trial            | alone p50 (ns) | paired-trial p50 (ns) | shift     |
|------------------|---------------:|----------------------:|----------:|
| T1 (fma 512)     |         58 542 |                58 000 |    -0.93% |
| T2 (fma 4096)    |        399 250 |               398 750 |    -0.13% |
| T3 (write 524K)  |         66 875 |                66 417 |    -0.68% |
| T4 (write 1M)    |        110 812 |               111 708 |    +0.81% |

All four trial medians shift by < 1 % when paired vs measured alone.
**Pairing does not perturb the trial's underlying behavior.** The
reference dispatch immediately preceding the trial inside the same
command buffer doesn't meaningfully change cache state, pipelining,
or anything else the trial sees. This is the cleanest pass of the
three primary questions and is itself useful: any per-trial number
read out of the paired CSV can be compared 1:1 to single-kernel
measurements.

### Primary 3 (PASSES): ratio stability across 3 within-session sweeps

| trial           | sweep 0 ratio p50 | sweep 1 ratio p50 | sweep 2 ratio p50 | spread |
|-----------------|------------------:|------------------:|------------------:|-------:|
| T1 (fma 512)    |            0.5395 |            0.5405 |            0.5393 |  0.23% |
| T2 (fma 4096)   |            3.7211 |            3.7196 |            3.7271 |  0.20% |
| T3 (write 524K) |            0.6239 |            0.6216 |            0.6177 |  0.98% |
| T4 (write 1M)   |            1.0360 |            1.0409 |            1.0423 |  0.60% |

All four ratios stay within 1 % across 3 sweeps with 30 s cooldown
between sweeps. T3 sits right at the boundary (0.98%), which mostly
reflects T3 itself being noisier (alone cv = 0.035) rather than
ratio-specific drift. **The ratio is a sweep-stable relative metric
within a session**, even though it didn't reduce within-sweep
variance.

### Subsidiary: ref-when-paired vs ref-alone

| condition         | ref p50 (ns) | ref robust_cv | shift vs ref_alone |
|-------------------|-------------:|--------------:|-------------------:|
| ref_alone         |      107 042 |        0.0079 |                 -  |
| T1_paired (ref)   |      107 333 |        0.0034 |             +0.27% |
| T2_paired (ref)   |      107 208 |        0.0061 |             +0.16% |
| T3_paired (ref)   |      107 250 |        0.0029 |             +0.19% |
| T4_paired (ref)   |      107 333 |        0.0026 |             +0.27% |

The reference is essentially undisturbed by being paired with any
trial. Its median in paired conditions is +0.16-0.27 % vs ref_alone
(directionally consistent because it's now the *first* dispatch in
a cb, not preceded by a warmup; tiny effect). Its robust cv when
paired is actually *lower* (0.003-0.006) than ref_alone (0.008),
which is a hint about why pairing's outlier suppression works (see
Surprise §3 below).

### Subsidiary: inter-encoder gap_ns distribution

| condition  | min   | p05    | p50    | p95    | p99    | max       |
|------------|------:|-------:|-------:|-------:|-------:|----------:|
| T1_paired  | 14 125 | 24 706 | 42 000 | 46 591 | 48 877 | 2 979 500 |
| T2_paired  |  7 959 | 23 415 | 40 458 | 46 792 | 49 958 |   301 333 |
| T3_paired  | 15 459 | 24 458 | 41 750 | 46 875 | 55 125 | 1 361 250 |
| T4_paired  | 11 292 | 25 039 | 42 542 | 47 791 | 73 807 | 1 447 708 |

The gap from `ref encoder end` to `trial encoder start` inside the
*same command buffer* sits at ~42 µs at p50 — about 4 × the
dispatch-overhead floor we measured in 001-004 (~9 µs) and ~13 % of
even the longest trial (T2 at 399 µs). The "same buffer = atomically
tight coupling" assumption that motivated the technique is wrong:
two consecutive compute encoders inside one cb still pay substantial
per-encoder setup time. Distribution is consistent across trials
(p50 in [40, 43] µs, p95 in [46, 48] µs, similar shape), so the gap
appears to be a property of the encoder model, not the trial kernel.

### Subsidiary: outlier suppression (alone vs paired-trial naive_cv)

Naive cv (sd / p50) catches outliers that robust cv (IQR-based)
ignores. Comparing naive cv between alone and paired-trial measures
how much pairing changes the *tails*:

| condition          | p50 (ns) | p99 (ns) | max (ns) | robust_cv | naive_cv |
|--------------------|---------:|---------:|---------:|----------:|---------:|
| ref_alone          |  107 042 |  110 889 |  733 708 |    0.0079 |   0.3067 |
| T1_alone           |   58 542 |   62 960 |  833 292 |    0.0095 |   0.5474 |
| T1_paired (trial)  |   58 000 |   62 375 |  157 917 |    0.0107 |   **0.0749** |
| T3_alone           |   66 875 |  109 318 |  796 125 |    0.0346 |   0.6849 |
| T3_paired (trial)  |   66 417 |   74 459 |  718 792 |    0.0363 |   **0.3858** |

T1_paired's tail naive_cv is **7.3× lower** than T1_alone's; T3 is
1.8× lower. The bulk distributions (robust_cv) are identical, but the
trial measurement when paired sees far fewer huge tail outliers. The
trial's worst-case max drops from 833 µs (alone) to 158 µs (paired)
for T1 — a 5× reduction. **This was not predicted.**

## Surprises

### 1. Ratios DON'T cancel variance when there's nothing to cancel

The decision-003 framing assumed that DVFS / thermal / cache state
variance is the dominant noise source, and that a co-located ratio
would cancel it. 005's data shows this is wrong for our kernels at
warm steady state inside a 2-minute experiment: the dominant noise
sources at this scale are timestamp quantization, OS scheduling /
preemption events, and per-encoder setup variability — none of
which are shared between ref and trial in a way the ratio absorbs.
DVFS *probably does* matter at longer timescales (cross-session,
cross-thermal-state), but within a hot 2-minute run, it doesn't
fluctuate enough to leave the technique anywhere to add value as a
variance reducer.

### 2. Two compute encoders in the same command buffer aren't atomically coupled

The ~42 µs inter-encoder gap is the methodological surprise of 005.
The original framing imagined that two compute passes in one cb
execute back-to-back at GPU-front-end speed (sub-µs); reality is
that the GPU front end appears to insert ~42 µs of effective idle
between consecutive compute encoders. This is roughly the same
duration as the dispatch-overhead floor for *one* dispatch (~9 µs)
multiplied by ~4-5, suggesting per-encoder setup is paid per
encoder, not amortized across same-buffer encoders. So
"same-buffer pairing" is closer to "two back-to-back command
buffers without the cb commit overhead" than to "atomic kernel
pair." The variance-cancellation argument is weakened
proportionally — the two halves of the ratio see slightly
different chip states, separated by 42 µs of mostly-idle time.

### 3. Pairing dramatically suppresses tail outliers in the trial measurement

This was not predicted and is the most interesting positive
finding. Two complementary explanations:

- The reference dispatch acts as a *second warmup* for the trial.
  Alone-mode has K=1 untimed warmup, then 1 timed measurement
  separated by a CPU-side `commit + waitUntilCompleted` cycle.
  Paired-mode has K=1 untimed warmup, then a single command buffer
  with [ref, trial] — the trial sits behind the ref in the same
  cb, so it benefits from the ref's "front-end priming." The
  trial's worst-case max drops 5× for T1 (833 µs → 158 µs).
- The trial's measurement window is shifted by ~107 µs (the ref's
  duration) inside the cb. If outliers are caused by something
  with rough alignment to OS scheduler ticks (1 ms intervals at
  user-interactive QoS), shifting the window by 107 µs pushes the
  trial out of one alignment band and into another — possibly with
  fewer collisions.

We did not isolate the mechanism; either way, the practical
consequence is real and useful: paired measurements have *both*
the same bulk distribution as alone measurements *and* far cleaner
tails. For any application where p99 / max matters, paired is the
better measurement even though it doesn't help the median's
spread.

### 4. The reference is a more stable measurement than the trial in nearly all paired conditions

ref_alone has robust_cv = 0.008. ref-when-paired has robust_cv =
0.003-0.006 (better than alone). Ref-when-paired is the *first*
dispatch in the cb after the trial's K=1 untimed warmup, so the
ref's stability is sitting in a slightly cleaner state than
ref_alone's (which is the *second* dispatch after its own K=1
warmup). The cv reduction is small (0.008 → ~0.004) but
consistent, and it goes in the opposite direction from what
"pairing perturbs the ref" would predict. The reference kernel is
well-behaved.

### 5. T2 (fma_loop iters=4096) is so quiet alone that any ratio inflates its cv

T2's alone robust_cv of 0.0012 is at or below the timestamp
quantization floor (~42 ns / ~399 µs ≈ 0.0001). It's already as
clean as it can be. Pairing it with the reference (cv ~0.005)
forces the ratio's cv up to 0.0057 — the ref's cv simply cannot
be hidden in a ratio whose numerator is already at the noise
floor. **For very-clean trial measurements, pair timing is
strictly worse than alone timing**, which is the inverse of the
hypothesis. This is a useful operating-range constraint.

## What this means operationally

For paired co-encoded ratio timing on M1 Pro at the durations 005
characterized (50-400 µs trials, 107 µs ref):

1. **Don't use it as a within-session variance reducer.** The
   mechanism doesn't exist for our kernels at warm steady state.
   Single-kernel timing with N samples and robust statistics gives
   a tighter bulk distribution than paired ratios do, by
   construction.
2. **Do use it as a within-session relative-magnitude metric.**
   Trial median is preserved, ratio is sweep-stable to <1 %, the
   ref isn't perturbed. If the question is "is kernel A 2× kernel
   B's duration?", the ratio answers it cleanly.
3. **Do use it for tail-sensitive measurements.** When p99 / max
   matter (e.g., latency analysis), paired-trial measurements have
   cleaner tails than alone measurements at the same bulk
   distribution.
4. **Avoid it for trials that are already at the quantization
   floor.** If `alone robust_cv < ~0.005`, pairing only adds
   noise. Use alone timing.
5. **Account for the ~42 µs inter-encoder gap.** Two compute
   passes in one cb run ~42 µs apart, not adjacent. For any
   analysis that depends on "same chip state" assumptions
   between ref and trial, this gap is the relevant resolution
   limit.
6. **Cross-session ratio stability remains untested and is the
   strongest remaining argument for the technique.** If ratios
   are stable across 1-hour gaps where DVFS state and ambient
   temperature drift, then pair timing's value lives in
   cross-session comparison even though it doesn't reduce
   within-session variance. 005b should test this.

Decision 003 is superseded by decision 004 (`decisions/004-narrowed-pair-timing-scope.md`),
which formalizes points 1-6 above.

## New questions

- **Cross-session ratio stability.** The strongest remaining argument
  for pair timing is that ratios may be stable across sessions where
  absolute kernel times drift. Untested. Would be a one-script
  experiment that runs the same paired conditions twice in two
  separate process invocations 1 hour apart (or across thermal
  states), then compares ratios. Could be 005b or 006.
- **What sets the 42 µs inter-encoder gap?** Is it a fixed
  per-encoder cost (in which case pair timing adds 42 µs of dead
  time but doesn't change with kernel kind), a function of pipeline
  state for the trial kernel, or something the operator can
  influence with command-buffer or queue properties? Inspecting the
  Metal documentation for `MTLCommandBuffer` ordering semantics, or
  trying alternative encoder patterns (separate command buffers,
  command buffer with `waitUntilScheduled`-style barriers, etc.),
  would distinguish.
- **Why does pairing suppress trial tails so much for T1 (8×) but
  T4 not at all (no change)?** Mechanism investigation. Could
  inform whether pair timing is uniformly safe for tail analysis or
  only some kernel kinds.
- **Does the pattern hold for kernels above 1 ms?** 005's longest
  trial is T2 at 399 µs. The 4096-16384 fma-iter bimodality from
  004 sits at 0.4-1.6 ms. We never tested whether ratio timing
  cleanly cuts through that bimodality (i.e., whether the *bimodal
  trial* paired with a *non-bimodal ref* gives a stable ratio).
  This is a follow-up worth doing if pair timing is otherwise
  worth keeping.
- **Does the same-character-vs-cross-character distinction matter
  for variance, even if it doesn't here?** Pre-registered hypothesis
  predicted compute-bound trials might fare better than memory-bound
  with a compute-bound ref. Reality: variance reduction failed
  uniformly. Either the hypothesis is wrong about character mattering
  for variance, or the failure-to-reduce-variance is so total that
  character distinctions are invisible underneath it. Cannot tell
  from this experiment.

## After this experiment

Branch we landed on: per the pre-registered branches, this is
"Q1 fails everywhere" — methodology, as primarily framed, fails.
But not as badly as that branch's name suggests, because Q2 and Q3
both passed, and the surprise findings (tail suppression, ref
stability) point at a narrower but still useful technique.

Next moves, in priority order:

1. **Supersede decision 003 with decision 004.** Done as part of
   this writeup. The narrowed claim is that paired-encoder ratio
   timing is a within-session relative-magnitude metric and a
   tail-suppression technique, not a variance reducer.
2. **Pre-register 006: cross-session ratio stability.** This is
   the question 005 explicitly did not answer and the strongest
   remaining justification for the technique. One script runs in
   morning, identical script runs in afternoon, compare ratios.
   If ratios match within ~1 % across sessions while absolutes
   drift, decision 004 gets a status update from "active" to
   "validated as cross-session metric."
3. **Investigate the +21 µs step at fma_loop 192→256** that 004
   surfaced. Tiny experiment, ~30 lines + reading Metal AIR /
   GPU assembly. Independent of 005's outcome but still
   methodology-relevant.

We do not plan past these branches.

## Notes on the post-run writeup

`UNKNOWNS.md` updated:
- "Does paired ratio timing reduce within-session variance?" closed
  as **NO** for kernels at this duration range.
- "Is the paired ratio stable across within-session sweeps?" closed
  as **YES** (within 1%).
- "Does pairing perturb the trial measurement?" closed as **NO**
  (within 1%).
- New: "Does paired ratio timing remain stable across separate
  script invocations?" (the cross-session question, untested).
- New: "What is the per-encoder gap inside a single command buffer
  on M1 Pro and what controls it?" (open).

`notes/answered-questions.md` gains entries for the three primary
questions plus the inter-encoder gap finding.

`decisions/003-paired-kernel-ratio-timing.md` status updated to
"superseded by 004." Body unchanged per the discipline (decisions
are append-only).

`decisions/004-narrowed-pair-timing-scope.md` written, formalizing
the operational consequences from above.

A small `analysis.py` next to `run.py` does the pooled analysis
used in this writeup. One-off, not a library.
