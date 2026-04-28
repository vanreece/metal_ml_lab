# 003: Adopt paired-kernel ratio timing as the primary measurement methodology (pending 005 validation)

**Date:** 2026-04-27
**Status:** active, pending validation by experiment 005
**Confidence:** medium

## Context

Experiments 001-004 measured single kernels in isolation. The headline
finding from 004 reframed everything we had measured: the ~8 µs floor
the project kept hitting was *dispatch overhead*, not kernel time, and
all of 001/002/003 had been operating ~64× below the work-dominance
threshold for `write_tid`. Even at thread counts where work does
dominate, the *absolute* numbers are tied to whatever DVFS state the
chip happens to be in at that moment, which is unobservable to us and
visibly drifts (002, 003, 004's bimodality finding at 0.4-1.6 ms).

This is structural, not fixable by methodology refinement: there is no
"true kernel time" we can measure on Apple Silicon, because the chip
has no fixed clock to time against and the public Metal API gives us
no way to read the current GPU frequency. Single-kernel absolute
timing is therefore not what we should be optimizing for.

But the project's actual goals — bottleneck classification, roofline
position, occupancy effects — are nearly all *relative* questions.
"Is this kernel compute-bound or memory-bound?" "Does doubling the
problem size double the time?" "Is the kernel close to peak FLOPS or
not?" None of these need an absolute number; they all need a stable
ratio between two related measurements made in nearly the same chip
state.

This decision formalizes the move to ratio-based timing as the
primary methodology, contingent on 005 showing the technique works.

## Options considered

- **Continue with single-kernel absolute timing + post-hoc
  normalization.** Status quo from 001-004. The data we already have
  is honest about its variance (cv reported, percentiles reported,
  outliers visible), and we can keep going. But the *interpretation*
  is fragile: we are always implicitly comparing across DVFS states
  that drifted between measurements, and the published numbers we
  want to compare against (Philip Turner's M1 Pro bandwidth, peak
  FLOPS) were measured under entry conditions we cannot recreate.
  Cross-session comparisons in particular are not trustworthy.

- **Single-kernel timing + frequent calibration baseline subtraction.**
  Treat a fixed-character calibration kernel (e.g. `write_tid 32t`)
  as a thermometer; periodically measure it, subtract its current
  median from the kernel-of-interest measurements, report
  "kernel time minus current overhead floor." Cheaper than pair
  timing because the calibration doesn't run for every trial. But
  003 already showed the calibration-as-thermometer has observer
  effect (the 10-dispatch probe is itself warmup), and the
  calibration kernel is dispatch-overhead-dominated (per 004, it's
  100% overhead), so it characterizes overhead variance, not chip
  speed. Subtracting overhead from kernel time is reasonable for
  "what was the kernel's contribution," but it still leaves the
  remainder dependent on whatever DVFS state the chip is in, which
  is what we are actually trying to escape.

- **Paired co-encoded ratio timing.** For each trial, encode a
  reference kernel and the kernel-of-interest into the same Metal
  command buffer, in two sequential compute passes, each with its
  own timestamp pair. The unit of measurement is the *ratio* of
  trial duration to reference duration. Both halves of the ratio
  see nearly identical DVFS, thermal, and cache state because they
  execute microseconds apart. The ratio cancels out absolute clock
  variation by construction, and stays comparable across sessions
  as long as the reference kernel's character is constant. This is
  the proposal.

- **Statistical regression / mixed-effects model.** Treat each
  measurement as a sample from a noise model with random effects
  for sweep, session, etc., and fit a model. Could in principle
  back out a stable kernel time from many noisy measurements. Too
  sophisticated for the current state of the project — we have not
  yet established that the simple thing works, and adding modeling
  complexity before that would obscure what's signal vs assumption.
  Defer until ratio timing has been validated and we know what its
  residual variance looks like.

## Decision

Adopt paired co-encoded ratio timing. Specifically:

- A measurement is a pair `(ref_delta, trial_delta)` from two
  consecutive compute passes within the same `MTLCommandBuffer`,
  each with its own `MTLCounterSampleBuffer` start/end attachment.
- The reported metric for a kernel-of-interest is `trial_delta /
  ref_delta`, possibly aggregated across N pairs with robust
  statistics (median, IQR-based cv).
- The reference kernel is held fixed within an experiment so that
  ratios are comparable across conditions in that experiment.
  Cross-experiment comparison requires the same reference.

Single-kernel absolute timing remains a valid secondary measurement
where it is informative (e.g. for characterizing the dispatch-overhead
floor itself, or for STREAM-style absolute-bandwidth verification
against published numbers), but it stops being the primary metric for
kernel-characterization questions.

This decision is *contingent on experiment 005 validating that pairing
actually reduces variance the way the proposal predicts*. If 005 shows
that pairing perturbs the trial measurement enough to wash out the
variance reduction, or that the ratio is itself unstable across
sweeps/sessions, this decision gets superseded by a new note and we
fall back to single-kernel timing with calibration baseline.

## What would make us revisit

- 005 shows the ratio cv is not meaningfully lower than the trial-alone
  cv (i.e. pairing did not buy us the variance reduction the technique
  is supposed to give). Methodology fails; supersede.
- 005 shows that pairing changes the trial median by more than ~5%
  versus alone (i.e. pairing is *measuring something different* from
  the kernel in isolation). The ratio is still useful as a relative
  measure but the interpretation has to be footnoted carefully; may
  warrant a different reference choice or a different pairing pattern.
- A reference kernel candidate turns out to be in a non-linear
  regime we did not characterize (compiler / hardware threshold like
  the +21 µs step at fma_loop 192→256). Picking the wrong reference
  contaminates every ratio. Mitigation: keep the reference choice
  empirically validated (in the linear, work-dominated, low-cv zone
  per 004), and re-validate when porting to new hardware.
- Cross-session ratio stability turns out to be poor (a sub-question
  not directly addressed by 005, which is within-session). If
  ratios drift across script invocations or across days, the
  technique works as a within-session relative metric but cannot
  underpin published characterizations. Likely follow-up experiment
  005b would test this.
- New hardware (M4 Max, etc.) has paired-timing semantics we haven't
  characterized. The technique may work on M1 Pro and not transfer.
  Assume re-validation is required per chip.
