# 005: Restore paired-encoder ratio timing as the primary methodology on M4 Max (supersedes 004 on G16; 004 remains active on M1 Pro)

**Date:** 2026-04-28
**Status:** active on M4 Max, supersedes [decision 004](004-narrowed-pair-timing-scope.md) on M4 Max only. Decision 004 remains active on M1 Pro.
**Confidence:** medium-high (within-session validated by exp 005 M4 Max addendum, cross-session validated by exp 006 with one MARGINAL trial near the dispatch-overhead floor)

## Context

Decision 004 narrowed paired-encoder ratio timing to a within-session
relative-magnitude / tail-suppression metric on the basis of M1 Pro
experiment 005. Two weeks of M4 Max re-runs (committed 2026-04-28)
produced two findings that invalidate decision 004's load-bearing
arguments on G16:

1. **The ~42 µs inter-encoder gap collapsed to ~833 ns on M4 Max**
   (exp 005 M4 Max addendum). Decision 004's variance-reduction
   failure on M1 Pro was mechanistically driven by that 42 µs gap
   (ref and trial saw different fast-changing chip state). On M4 Max
   the gap is sub-µs, so ref and trial sit in the same chip state,
   and variance composition can actually cancel shared noise.

2. **Variance reduction now works on M4 Max for noisy trials.**
   Exp 005 M4 Max showed T2 (fma_loop iters=4096) ratio cv = 0.005
   vs alone cv = 0.336 — a 63× reduction. Exp 006 M4 Max went
   further: across two separate process invocations 30 min apart,
   T2's *alone* cv differed by 335× between sessions (0.335 vs
   0.001) while the ratio cv was 0.006 vs 0.005 and the ratio p50
   matched to 0.022 %.

3. **Cross-session ratio stability holds for trials ≥ ~10× the
   dispatch-overhead floor.** Exp 006: T2/T3/T4 (durations
   59-249 µs vs ~6.4 µs floor) all PASS the strict ≤ 1 %
   cross-session criterion. T1 (37 µs, ~6× the floor) is MARGINAL
   at 2.77 %. The rule is "trials whose duration is much greater
   than the per-dispatch overhead get clean ratios; trials close to
   the floor get noisier ratios."

The conditions decision 004 listed under "What would make us
revisit" are essentially all met for M4 Max. This decision narrows
the scope of decision 004 to M1 Pro and re-elevates pair timing on
M4 Max.

## Options considered

- **Apply decision 005 to all chips (supersede decision 004
  globally).** Rejected. Decision 004's narrowing is *correct on M1
  Pro* — the 42 µs inter-encoder gap is real on G13, the variance-
  reduction failure is real on G13, and using pair timing as a
  primary variance reducer on M1 Pro would be relying on a
  mechanism that doesn't exist there. Decision 004's evidence is
  intact; only its scope is overbroad.

- **Per-chip decision file (this option).** Decision 004 stays
  active on M1 Pro. This decision (005) restores pair timing as
  primary on M4 Max. Future chip generations get re-validated
  before their methodology is set. Avoids pretending the
  methodology question has a chip-independent answer when the
  evidence shows it doesn't.

- **Wait for longer-gap cross-session data before deciding.** Exp
  006 used a 30-min idle gap. A 4-hour or overnight gap would
  exercise more thermal / DVFS drift. Rejected because (a) 30 min
  on M4 Max already showed stable ratios across CV swings of 335×,
  (b) keeping decision 004 active on M4 Max while we have positive
  evidence is sloppy, (c) longer-gap evidence can supersede 005 if
  it shows breakage; that's the discipline working as intended.

- **Wait for the MARGINAL T1 result to be resolved.** T1 (the
  shortest trial, ~6× the dispatch-overhead floor) lands at 2.77 %
  cross-session spread, outside the strict ≤ 1 % bar but inside the
  3 % "marginal" band. Rejected because the operational rule
  ("trials should be ≥ ~10× the floor") is well-supported even
  before T1's mechanism is isolated, and waiting for a single trial
  to clear an arbitrary threshold delays a decision the bulk of the
  evidence supports.

## Decision

On Apple M4 Max (`applegpu_g16s`) running macOS 26.4.1 (Tahoe) or
later: paired co-encoded ratio timing is the **primary measurement
methodology** for any kernel-characterization question whose trial
duration is at least ~10× the dispatch-overhead floor (i.e., trial
p50 ≥ ~64 µs at the current 6.4 µs floor).

Specifically:

### Pair timing on M4 Max IS the right primary methodology for:

1. **Within-session variance reduction for noisy trials.** When
   trial-alone cv is in the noisy-but-not-quantization band (roughly
   `0.01 < alone_robust_cv < 1.0`), the ratio cancels shared chip-
   state noise and produces a tighter measurement. Demonstrated on
   T2 (alone cv 0.336 → ratio cv 0.005, exp 005 M4 Max).

2. **Cross-session relative-magnitude characterization** for trials
   ≥ 10× the dispatch-overhead floor. Demonstrated on T2/T3/T4 in
   exp 006 (≤ 0.6 % spread across 30-min-gap sessions).

3. **All within-session questions** previously listed in decision 004
   (relative magnitude, scaling, ratio of ratios). These were never
   in question on M4 Max.

### Pair timing on M4 Max is NOT the right primary methodology for:

4. **Trials at quantization floor.** When `alone_robust_cv < ~0.005`
   (the trial is already as clean as the timestamp tick allows),
   pair timing adds the reference's noise to the ratio. Single-
   kernel timing is strictly better. Same constraint as decision
   004 point 6, unchanged on M4 Max.

5. **Trials with duration close to the dispatch-overhead floor.**
   Trials ≤ ~5× the floor (i.e., < ~32 µs on M4 Max) show degraded
   ratio stability (T1 case, 2.77 % cross-session spread). Use
   single-kernel timing with N samples and robust statistics, or
   redesign the kernel to operate above the floor.

6. **Sub-µs comparisons between ref and trial.** The inter-encoder
   gap on M4 Max is ~833 ns at p50 (vs ~42 µs on M1 Pro). Sub-µs
   alignment is now achievable for most purposes but still finite —
   any analysis that requires ref and trial in *literally* identical
   chip state needs a different pattern.

### Operational defaults on M4 Max

- **Reference kernel:** `fma_loop iters=1024` at 32 threads. Same
  as decision 004; reconfirmed by exp 005/006 M4 Max (ref alone cv
  ~0.004-0.007, very stable).
- **Pairing pattern:** [ref, trial] in a single `MTLCommandBuffer`
  with two sequential `MTLComputeCommandEncoder` instances, each
  with its own timestamp pair. Same as decision 004; the pattern
  is unchanged, only the wait-time-between-encoders differs by
  chip.
- **Warmup:** K=1 untimed warmup of the trial kernel only,
  immediately before each measured trial. Same as decision 004;
  exp 003 M4 Max addendum confirmed K=1 still recovers cool-cadence
  medians on G16 (with `heavy_write` as the most robust warmup
  kind, but `same` and `fma_loop` also acceptable).
- **Trial duration target:** ≥ ~64 µs (10× the M4 Max
  dispatch-overhead floor of 6.4 µs). Trials below this threshold
  may use pair timing but should treat the ratio's cross-session
  spread as ≤ 3 %, not ≤ 1 %.
- **Reporting:** for any paired condition, report all of
  `trial_delta_alone p50 / robust_cv`,
  `trial_delta_paired p50 / robust_cv`, and
  `ratio p50 / robust_cv`. Same as decision 004; the three carry
  distinct information and combining loses signal.

### Per-chip applicability summary

| chip | dispatch-overhead floor | inter-encoder gap | pair timing primary methodology? |
|------|------------------------:|------------------:|----------------------------------|
| M1 Pro `applegpu_g13s` | ~8.0 µs | ~42 000 ns | NO — decision 004 narrows it |
| M4 Max `applegpu_g16s` | ~6.4 µs | ~833 ns    | YES — this decision (005) elevates it |
| Other M-series        | unknown | unknown    | re-validate per chip before assuming |

## What would make us revisit

- **006a (longer idle gap, e.g. 4 hours or overnight) shows ratio
  drift > 3 % on M4 Max.** Decision 005 was made on a 30-min-gap
  test; if longer gaps expose drift, the cross-session claim is
  scoped down to "stable for short gaps."
- **A new M4 Max experiment shows ratio drift across thermal-state
  changes** (e.g. running heavy GPU work in between sessions).
  Decision 005 covers idle-gap state drift; thermal-state drift is
  a separate axis we haven't tested.
- **The macOS 27 release changes the inter-encoder gap on M4 Max**
  (back up toward M1 Pro's 42 µs, or further down). Decision 005's
  mechanism depends on the small gap; if it changes, re-test.
- **A future chip generation (G17+) reverts to a large
  inter-encoder gap.** Each chip is its own validation problem.
- **The MARGINAL T1 result is shown to be a real cross-session
  drift, not a "trial too close to floor" artifact.** Would prompt
  re-narrowing point 5 above.

## Note on supersession scope

Per the lab discipline, decisions are append-only: decision 004's
body is unchanged. Its status header was updated when the M4 Max
exp 005 evidence first appeared ("active on M1 Pro, *under review
on M4 Max*"); now that exp 006 closes the cross-session question,
its status is updated to "active on M1 Pro, superseded by 005 on
M4 Max."

This is the first time the lab has a per-chip decision split. The
project is now structurally multi-hardware (per the user-memory
"hardware tagging discipline"); the methodology will need to track
that. Future decisions should default to per-chip scope unless an
explicit cross-chip claim is being made and validated.
