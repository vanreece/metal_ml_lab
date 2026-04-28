# 004: Narrow paired-kernel ratio timing to relative-magnitude / tail-suppression metric (supersedes 003)

**Date:** 2026-04-27
**Status:** active on M1 Pro, **superseded on M4 Max by [decision 005](005-restore-pair-timing-on-m4-max.md)** (cross-session ratio stability validated by exp 006 on G16 / macOS 26.4.1; the variance-reduction mechanism this decision wrote off works on M4 Max because the 42 µs inter-encoder gap collapses to ~833 ns there). Supersedes decision 003.
**Confidence:** medium-high on M1 Pro.

## Context

Decision 003 committed the project to paired co-encoded ratio timing
as the primary measurement methodology, contingent on experiment 005
validating the variance-reduction claim. 005 ran on 2026-04-27
(`experiments/005-paired-ratio-stability/raw/20260427T172313-*`) and
returned a clean falsification of that claim: ratio robust_cv was
equal to or worse than trial-alone robust_cv for all four trial
kernels in the 50-400 µs range, sometimes by ~5×.

005 did not, however, falsify the technique entirely. Two of the
three primary properties held cleanly:

- Pairing did not perturb the trial's median (all four trials within
  ±1 % vs alone).
- Ratios were stable within a session across 3 sweeps with 30 s
  cooldowns (all four within 1 %).

And two unpredicted positive findings emerged:

- Tail outliers in the trial measurement are dramatically reduced
  when paired (T1: alone naive_cv = 0.55 vs paired_trial naive_cv =
  0.07, an 8× reduction in tail noise without any change to bulk
  distribution).
- The reference dispatch is essentially undisturbed by pairing
  (median shift +0.16-0.27 % vs ref_alone) and is actually slightly
  *cleaner* when paired.

So pair timing is real, useful, and stable — just not for the
variance-reduction reason decision 003 invoked.

This decision narrows the scope.

## Options considered (now that 005's data is in hand)

- **Abandon pair timing entirely; revert to single-kernel timing
  with calibration baseline subtraction.** This was decision 003's
  written fallback for the "Q1 fails everywhere" branch. Has the
  benefit of staying close to what 001-004 already validated. Costs
  us the median-preservation, ratio-stability, and tail-suppression
  properties that 005 actually demonstrated. Loses the
  cross-session-stability angle entirely (untested but plausible).
  Conservative but throws away real findings.

- **Keep pair timing as primary, footnote the variance result.**
  Decision 003 plus a "yeah but actually it didn't reduce variance"
  asterisk. Rejected because the original decision's load-bearing
  argument was specifically variance reduction; keeping the
  framing while gutting its justification is not honest about
  what we learned.

- **Narrow pair timing to the properties that 005 validated**
  (median preservation, ratio stability, tail suppression) and
  *don't* claim variance reduction. Pair timing becomes a tool
  for some questions (relative magnitude, scaling, latency
  analysis) and not others (within-session cv tightening). Single
  kernel timing remains the right tool for cv-bound questions
  inside one session. This option treats 005 as a calibration of
  the technique's actual operating envelope, not as a flat
  rejection.

- **Wait for cross-session results before deciding.** Pair timing's
  strongest remaining justification is cross-session ratio
  stability, untested by 005. We could keep decision 003 in
  "pending validation" until 006 runs. Rejected because (a) 005
  *did* answer concrete questions whose answers should be reflected
  immediately, and (b) keeping a falsified decision live is
  sloppy.

## Decision

Adopt the third option: narrow pair timing to its validated and
pending properties, and write down the operating envelope.

### Pair timing IS the right primary methodology for:

1. **Within-session relative-magnitude questions.** "Is kernel A 2×
   kernel B's duration on this chip right now?" The ratio's median
   is stable to <1 % across within-session sweeps and the underlying
   trial median is preserved within ±1 % vs alone, so paired ratios
   answer this kind of question cleanly.

2. **Within-session scaling-relationship questions.** Same
   technique: pair the kernel-of-interest at two different
   parameter levels with the same reference, and the ratio of
   ratios tracks scaling behavior without inheriting whatever
   absolute drift the chip is doing. Useful for occupancy /
   threadgroup-size / problem-size sweeps.

3. **Tail-sensitive measurements (p99, max, latency analysis).**
   Pairing demonstrably reduces tail outliers in the trial
   measurement (T1 max dropped 5× vs alone). For any question
   where worst-case matters as much as median, paired-trial
   measurements are quantifiably better than alone measurements.

4. **(Pending validation by 006) Cross-session relative
   characterization.** This is the strongest remaining
   justification, untested in 005. If ratios stay stable across
   process invocations / thermal states / time-of-day differences
   while absolute kernel times drift, then pair timing solves
   the cross-session-comparability problem that absolute timing
   structurally cannot.

### Pair timing is NOT the right primary methodology for:

5. **Within-session variance reduction.** Falsified by 005.
   Ratio cv is ≥ trial-alone cv for kernels in the duration
   range we measured. Use single-kernel timing with N samples
   and robust statistics if the goal is tight cv on one number.

6. **Trials whose alone cv is already at quantization floor**
   (`alone robust_cv < ~0.005`). For these, the ratio's cv is
   dominated by the reference's cv, and pairing makes things
   strictly worse. Use alone timing.

7. **Sub-µs timing resolution between ref and trial.** The
   inter-encoder gap inside one cb is ~42 µs at p50 (~4 × the
   dispatch-overhead floor). Two encoders in one cb are not
   atomically tight. Any analysis that requires the ref and trial
   to be in identical sub-µs chip states needs a different
   encoding pattern (not yet known to exist on M1 Pro).

### Operational defaults

- **Reference kernel:** `fma_loop iters=1024` at 32 threads. This
  was the choice in 005, and the data confirmed the choice was
  good (ref_alone robust_cv = 0.008, ref-when-paired even
  cleaner). Re-evaluate per chip; not portable to M4 Max without
  re-validation.
- **Pairing pattern:** [ref, trial] in a single
  `MTLCommandBuffer` with two sequential
  `MTLComputeCommandEncoder` instances, each with its own
  `MTLComputePassDescriptor` and `sampleBufferAttachments[0]`
  start/end at distinct slot indices.
- **Warmup:** K=1 untimed warmup of the trial kernel only,
  immediately before each measured trial. No warmup of the
  reference (it sees a fresh-pipeline launch on every
  measurement, and that's fine — its variance is bounded
  empirically).
- **Reporting:** for any paired condition, report all of
  `trial_delta_alone p50 / robust_cv`,
  `trial_delta_paired p50 / robust_cv`, and
  `ratio p50 / robust_cv`. Do not collapse to one number — the
  three carry distinct information.

## What would make us revisit

- 006 (cross-session ratio stability) returns *not stable* —
  ratios drift across script invocations. That collapses the
  remaining strongest argument for pair timing and shifts the
  default for all cross-session work back to "characterize
  per-session and footnote drift." Would be superseded.
- A new pairing pattern is found that closes the 42 µs
  inter-encoder gap (e.g. some Metal encoder ordering API we
  haven't discovered, or a Swift bridge that exposes a tighter
  pattern). Would warrant updating point 7 above; technique
  could become useful for sub-µs comparisons.
- A future kernel kind we want to measure has alone cv much
  larger than 005's range (e.g. cv > 0.5 due to non-deterministic
  scheduling effects). The variance-reduction story might hold
  for these where it doesn't hold here, because the dominant
  noise source might be DVFS-correlated. Not currently
  hypothesized.
- Cross-session findings expose new failure modes (e.g. ratios
  stable in some thermal regimes but not others), forcing a
  more conditional decision.

## Note on supersession

Per the lab discipline (`CLAUDE.md`): decisions are append-only.
Decision 003 stays in place; its status header gets updated to
"superseded by 004" but its body is unchanged. The reasoning that
led to 003 (relative > absolute, ratios cancel DVFS variance)
remains valid context for understanding why 005 was the right
test; only the operationalization is being narrowed.
