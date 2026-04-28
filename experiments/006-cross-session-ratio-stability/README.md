# 006: Does paired-encoder ratio timing stay stable across separate process invocations?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-28 (M4 Max session)
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** 4 min compute + 30 min idle gap = ~34 min wall

## The question

Experiment 005 established that paired-encoder ratio timing is stable
*within a single script invocation* (≤ 1 % spread across 3 sweeps with
30 s between-sweep cooldowns) on both M1 Pro and M4 Max. The much
stronger stability claim — needed to make pair timing a credible
cross-experiment / cross-session methodology — is whether the same
ratio reproduces across **separate process invocations**.

Specifically, on M4 Max, run experiment 005's exact protocol twice
in two separate Python processes with a 30-minute idle gap between
them. The trial kernels (T1-T4), the reference kernel, the warmup
recipe, the sample counts, and the encoder pattern are identical
across the two sessions. The MTLDevice handle is fresh in each
session, the MTLLibrary is recompiled, the MTLComputePipelineStates
are rebuilt — everything that happens at process startup gets
re-done.

**Primary question:** for each of T1, T2, T3, T4, is the *ratio
median* (`p50` of `trial_delta / ref_delta` pooled across the
session's 3 sweeps × 300 trials = 900 paired measurements) consistent
between session A and session B?

- **Pass:** spread ≤ 1 % between session A and session B for all four
  trials. This is the same threshold used for within-session sweep
  stability in 005, applied at the next level up. If pair timing is
  truly a stable relative metric, the same threshold should hold.
- **Marginal:** spread in [1 %, 3 %]. Ratio is approximately stable
  but not at the within-session tightness. Suggests there's a
  cross-session drift component that needs characterizing.
- **Fail:** spread > 3 %. Pair timing is a within-session technique
  only; cross-session comparability needs a different mechanism
  (recalibration probe, normalization to a session-anchored
  reference, etc.).

## Why this question, now

Experiment 005's M4 Max re-run (committed 2026-04-28) reversed two
key M1 Pro findings:

- The ~42 µs inter-encoder gap collapsed to ~833 ns, restoring the
  variance-cancellation mechanism that decision 003 originally
  hypothesized.
- Variance reduction WORKS for noisy trials on M4 Max (T2: ratio cv
  0.005 vs alone cv 0.336, a 63× reduction).

Decision 004's status is now "active on M1 Pro, under review on
M4 Max." The single biggest factor in deciding whether decision 004
should be superseded for G16 is **cross-session ratio stability**.
If 006 passes on M4 Max, pair timing should be re-elevated to the
primary methodology for all the kernel-characterization questions
the project cares about (bottleneck class, roofline position,
occupancy effects). If 006 fails, decision 004 stays active even
though within-session variance reduction works.

This is the single question whose answer most reduces uncertainty
about whether the pair-timing methodology is sound on M4 Max. Per
the project discipline, that's the right next experiment.

## Hypothesis

Confidence: medium. Stating predictions to prevent post-hoc
retrofitting:

- **Cross-session ratio stability ≤ 1 %** for compute-bound trials
  (T1 = fma_loop iters=512, T2 = fma_loop iters=4096) — the kernel
  character is fully determined by the FMA loop itself, no
  external state, and 005 already showed within-session ratio
  stability is ≤ 0.86 % on M4 Max for these trials. The fresh-process
  setup adds variance, but a fresh MTLDevice and recompiled pipeline
  shouldn't shift the *steady-state* per-dispatch behavior of an
  arithmetic-bound kernel.
- **Cross-session ratio stability ≤ 1 % for write_tid trials**
  (T3, T4) is less confident. These are memory-bound and may be
  more sensitive to MTLBuffer placement, system-wide memory
  pressure, and OS-level GPU scheduling state that differs across
  sessions.
- **The M4 Max sub-floor state from 003 (the ~2 µs `fma_loop K=20
  sleep_0` finding) will NOT appear** in either session because
  005's protocol uses K=1 warmup, not K=20.
- **Wall-clock for each session ≈ 120 s** matching the M4 Max 005
  baseline within ±5 s.

## What we are NOT trying to answer

- **Cross-day, cross-power-state, cross-thermal-state stability.**
  This is a 30-min idle-gap test, both sessions on AC power, both at
  similar ambient temperature, both on the same calendar day. If
  this passes, a follow-on (006a) can extend to longer gaps and
  state changes.
- **What happens if the trial parameters are tuned differently.**
  We use the M1 Pro-derived T1-T4 specifically so the cross-session
  question is asked on the same kernels 005 already characterized.
- **Cross-chip ratio stability** (running 006 on M1 Pro and comparing
  the *ratios themselves* between chips). The ratios encode chip-
  specific kernel-relative-cost; they're not expected to be cross-
  chip-stable.
- **Whether the inter-encoder gap drifts across sessions.** Subsidiary
  observation only; the gap was tight (~833 ns p50) within 005 on
  M4 Max and we'd notice if it changed by a factor.
- **Mechanism for any drift we observe.** If ratios drift, this
  experiment surfaces the fact; characterizing why is a follow-on.
- **Anything about M1 Pro.** This is M4 Max-specific. The M1 Pro
  cross-session question is a separate experiment if/when we want
  to run it on that machine.

## Setup

### Session protocol

Each session is one full execution of `experiments/005-paired-ratio-
stability/run.py`, unchanged. The pre-existing M4 Max 005 run
(timestamp `20260428T115757`, ~12:00 today) was a development
shake-out and is NOT one of the two sessions for this experiment.
Both 006 sessions are fresh runs with the same M4 Max 005 protocol.

- **Session A:** launch immediately after pre-registering 006.
- **Idle gap:** 30 minutes (1800 s). No GPU work between sessions
  beyond what the OS does in the background.
- **Session B:** launch after the 30-min idle.
- Both sessions run under `caffeinate -d -i -m`. Both write to
  `experiments/005-paired-ratio-stability/raw/` with the standard
  005 timestamp prefix (no modification to 005's output paths).
- `EXP005_NO_POWERMETRICS` environment variable left UNSET (the
  default in 005 already gates powermetrics behind the env var per
  the 003/004/005 convention; we do not need powermetrics for this
  question).

### What 006 specifically writes

- `006/raw/{ts}-cross-session.txt`: a single-purpose linkage file
  recording (a) session A's 005 timestamp prefix, (b) gap duration,
  (c) session B's 005 timestamp prefix, (d) hardware/OS metadata,
  (e) wall-clock start/end of each session.
- `006/raw/{ts}-stdout.log`: the 006 orchestrator's stdout (mostly
  status messages — both 005 sessions have their own stdout logs in
  005's `raw/`).

The actual measurement data lives in 005's `raw/` (one
`{tsA}-alone.csv`, one `{tsA}-paired.csv`, one `{tsB}-alone.csv`,
one `{tsB}-paired.csv`). 006's analysis script reads from there
using the linkage file's prefixes.

### Analysis

`006/analysis.py` computes, for each of T1-T4:
- session A pooled ratio p50 (p_A), session A pooled ratio robust_cv
- session B pooled ratio p50 (p_B), session B pooled ratio robust_cv
- relative spread `|p_B - p_A| / mean(p_A, p_B)`
- pass/marginal/fail verdict per the success criteria above

Plus subsidiary tracking:
- session A vs session B trial-alone p50 (does the *absolute* number
  also drift, or does only the ratio absorb drift?)
- session A vs session B inter-encoder gap p50 (does the gap shift
  across sessions?)
- session A vs session B ref kernel p50 (does the reference itself
  drift?)

## Success criterion

The experiment succeeds (in the discipline's sense) if:
1. Both sessions complete cleanly with all expected CSV outputs.
2. Linkage file is written.
3. Analysis can fill in the per-trial cross-session-spread table.

It produces a usable answer if we can categorize each of T1-T4 as
pass / marginal / fail per the thresholds above, and decide whether
decision 004 should be:
- **Superseded** (all four pass): pair timing is the primary
  methodology on M4 Max for both within-session variance reduction
  AND cross-session relative-magnitude characterization.
- **Conditionally narrowed** (some pass, some marginal): pair timing
  works for some trial classes but not others; the operating envelope
  needs character-aware narrowing.
- **Kept as-is** (any fail): cross-session pair timing is not viable
  even on M4 Max; decision 004's narrowing is the right call.

## New questions we expect this to raise

- If ratios are cross-session stable but absolute trial p50s drift:
  by how much? This quantifies the actual benefit of pair timing.
- If T2's ratio cv stays as low as in within-session 005 (0.005),
  but the ratio p50 itself drifts: separates "the ratio is a stable
  measurement of a varying thing" from "the ratio is a stable
  measurement of a stable thing."
- If the inter-encoder gap drifts from session to session even
  though both sessions are the same protocol: implicates a
  process-initialization-state-dependent component to the gap.
- If only ONE of the two sessions exhibits anomalies (a transient
  state from cold start, etc.), we cannot tell from N=2 sessions.
  A 006b might extend to N=3 or N=4 sessions to disambiguate.

## After this experiment

Branches:

- **All pass.** Write decision 005 (superseding 004) re-elevating
  pair timing to the primary methodology on M4 Max for any
  kernel-characterization question. Pre-register 007: a
  bottleneck-classification experiment using the (now validated)
  pair-timing protocol on M4 Max.
- **Mixed results.** Write decision 005 narrowing the M4 Max pair-
  timing scope per which trial characters pass; pre-register a
  small focused investigation of why the failing cases failed.
- **All fail.** Decision 004 is correct on M4 Max too. Pre-register
  the kernel-state-profiling work that decision 003 originally
  envisioned, but with absolute-timing methodology rather than
  pair-timing.
- **Inconclusive (N=2 not sufficient).** Run 006b at N=4 sessions
  before deciding.
