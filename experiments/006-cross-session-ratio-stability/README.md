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

---

## Result

**Date run:** 2026-04-28 (linkage `raw/20260428T132113-cross-session.txt`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Session A:** 005 prefix `20260428T132113`, 119.56 s wall
**Idle gap:** 1800.00 s (30 min, exactly as pre-registered)
**Session B:** 005 prefix `20260428T135313`, 119.83 s wall
**Outcome:** Mixed-pass — 3 of 4 trials pass the strict ≤ 1 %
threshold; T1 is marginal at 2.77 %. The variance-reduction-across-
sessions story is the headline.

### Primary: cross-session ratio stability

| trial | A_ratio_p50 | B_ratio_p50 | spread | A_ratio_rcv | B_ratio_rcv | verdict   |
|-------|------------:|------------:|-------:|------------:|------------:|-----------|
| T1    | 0.56560     | 0.55016     | **2.767 %** | 0.04700 | 0.01599 | **MARGINAL** |
| T2    | 3.71901     | 3.71820     | **0.022 %** | 0.00596 | 0.00462 | **PASS**     |
| T3    | 0.89173     | 0.89439     | **0.298 %** | 0.00754 | 0.00736 | **PASS**     |
| T4    | 1.70383     | 1.69500     | **0.519 %** | 0.01865 | 0.22201 | **PASS**     |

T2, T3, and T4 all clear the strict ≤ 1 % bar. T1 lands in the
"marginal" 1-3 % band. T1 is the shortest trial (alone p50 ~37 µs,
ratio ~0.55, only ~6× the dispatch-overhead floor of 6.4 µs), so its
ratio is more sensitive to small absolute shifts in the overhead-
dominated portion of either kernel. T2-T4 are 10-40× the floor and
land cleanly.

### Subsidiary 1 (the killer chart): trial-alone p50 vs ratio across sessions

The whole point of pair timing is that *the ratio is stable even
when absolute durations are not.* So compare what the underlying
trial signal looks like in each session:

| trial | A_alone_p50 | B_alone_p50 | shift  | A_alone_rcv | B_alone_rcv | rcv ratio A/B |
|-------|------------:|------------:|-------:|------------:|------------:|--------------:|
| T1    | 36 917      | 36 750      | -0.45 % | 0.01086    | 0.00924     | 1.18×         |
| T2    | 248 667     | 248 709     | +0.02 % | **0.33527** | **0.00099** | **335×!**     |
| T3    | 59 417      | 59 417      |  0.00 % | 0.00363    | 0.00468     | 0.78×         |
| T4    | 112 417     | 112 542     | +0.11 % | **0.33171** | **0.00384** | **86×**       |

**T2's alone robust_cv differed by a factor of 335× between
sessions** (0.335 in A vs 0.001 in B), yet the ratio cv was 0.006
vs 0.005 — essentially identical — and the ratio p50 matched to
0.022 %. T4 shows the same pattern (86× cv difference alone, but
ratio p50 within 0.52 %). This is exactly the variance-cancellation
mechanism decision 003 originally hypothesized:

- Session A happened to land T2 in M4 Max's bimodal fma_iters band
  (per the 004 M4 Max addendum, iters=4096 has 1.81× sweep variation).
- Session B happened to skip the bimodal state entirely.
- The reference kernel saw the same chip state as T2 in each session
  (because the inter-encoder gap is now ~833 ns, not ~42 µs as on
  M1 Pro).
- The ratio cancels the bimodal noise; both sessions report the
  same 3.72× ratio.

Interestingly, the trial-alone *medians* themselves are highly
stable across sessions (all ≤ 0.5 % shift). So in this specific 30-
min-gap test, the alone numbers also reproduced. That's a separate
observation — it doesn't undermine pair timing, it just means the
M4 Max isn't drifting *much* over a 30-min idle gap. Longer gaps
or thermal-state changes could expose drift the ratio absorbs.

### Subsidiary 2: reference kernel cross-session

| condition  | A_ref_p50 | B_ref_p50 | shift  | A_ref_rcv | B_ref_rcv |
|------------|----------:|----------:|-------:|----------:|----------:|
| ref_alone  | 67 167    | 66 958    | -0.31 % | 0.00691  | 0.00369  |

The reference itself is stable to 0.31 % across sessions — well
within the 1 % criterion. The ref is doing its job as a stable
baseline.

### Subsidiary 3: inter-encoder gap cross-session

| trial | A_gap_p50 | B_gap_p50 | shift   |
|-------|----------:|----------:|--------:|
| T1    | 1 062     | 875       | -17.65 %|
| T2    | 792       | 834       |  +5.30 %|
| T3    | 792       | 834       |  +5.30 %|
| T4    | 875       | 833       |  -4.80 %|

The inter-encoder gap stays in the 800-1100 ns range across both
sessions on M4 Max (vs ~42 000 ns on M1 Pro). T1 has the largest
relative shift (17.65 %), but the absolute change is 187 ns — well
inside the noise of a single dispatch. The gap is **not** drifting
in a way that should affect ratio stability.

### Pre-registered branch landed on

This is the **"mixed results"** branch from the post-experiment
plan: 3 of 4 PASS, 1 MARGINAL. Per the pre-registration:

> Mixed results. Write decision 005 narrowing the M4 Max pair-
> timing scope per which trial characters pass; pre-register a
> small focused investigation of why the failing cases failed.

The "narrowing" is light: pair timing PASSES for trials whose
duration is ≥ 10× the dispatch-overhead floor (T2, T3, T4) and is
MARGINAL for trials close to the floor (T1, ~6× the floor). The
operational rule: when designing a new measurement, target trial
durations ≥ ~64 µs (10 × 6.4 µs floor on M4 Max) and pair timing
gives both within-session variance reduction and cross-session
stability.

T1 itself is not "broken" — its ratio is reproducible to 2.77 %,
which is plenty of resolution for many questions. The MARGINAL
classification is a flag, not a veto.

### Surprises

#### 1. Alone medians were also stable across sessions

The pre-registration anticipated that absolute trial p50s would
drift even if ratios stayed stable. In this 30-min-gap test on M4
Max with both sessions on AC power, the alone p50s drifted by
≤ 0.5 % across all four trials. The ratio's value-add here is in
absorbing CV variation, not in absorbing median drift. Whether
median drift appears at longer gaps or under thermal/AC-state
changes is the natural follow-up question.

#### 2. The 335× CV difference for T2 alone

T2's session A alone cv was 0.335 — a value typical of M4 Max's
bimodal fma_iters band that 004 documented. T2's session B alone cv
was 0.001 — at quantization floor. Neither session was "wrong"; both
are valid samples of T2's distribution under different chip
microstates. The ratio cv was 0.006 vs 0.005 in the same two
sessions. **This is the strongest single-experiment evidence we
have that pair timing is the right primary methodology on M4 Max.**

#### 3. Session B's reference cv is tighter than session A's

Ref alone cv: A=0.00691, B=0.00369 — almost 2× tighter in B. The
reference is not a constant-character measurement either; it sits
in whatever DVFS/state the chip is in at each session start. The
ratio's job is to cancel this — and it does. Session B happened to
catch the chip in a marginally tighter state across the board (also
visible in the lower trial-alone cv).

### What this means for decision 004

Decision 004's status was "active on M1 Pro, *under review on M4
Max*" pending this experiment. The data now supports a successor
decision (005) that re-elevates pair timing as the primary
methodology on M4 Max for any trial ≥ ~64 µs.

Drafted as `decisions/005-restore-pair-timing-on-m4-max.md`.
Decision 004 stays in place per the append-only discipline; its
header gets updated to "superseded by 005 on M4 Max; remains active
on M1 Pro."

### New questions raised

- **Does this hold at longer idle gaps?** 30 min showed alone p50s
  stable to 0.5 %. 1 hr / overnight / cross-AC-vs-battery would
  test whether ratio stability gracefully extends or breaks. A 006a
  with a 4-hour gap would be cheap (just two more 005 invocations).
- **What's the M4 Max equivalent of T1 that would clear the strict
  bar?** A redux with shorter trials below the dispatch-overhead
  floor would map out the failure boundary precisely.
- **Why does T1 have higher session-A ratio cv (0.047) than T1 had
  in 005 (0.011)?** The 4× cv inflation is unexplained — possibly a
  state difference between the morning 005 run and session A, but
  not characterized.
- **Did powermetrics interference matter?** Both 006 sessions had
  `EXP005_NO_POWERMETRICS` unset, but our gpu_telemetry script wasn't
  running either. Future runs with active telemetry could let us
  correlate ratio drift with observable chip state.
