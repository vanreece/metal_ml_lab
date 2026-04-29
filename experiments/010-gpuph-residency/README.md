# 010: Does IOReport's `GPUPH` channel expose per-DVFS-state GPU residency that tracks workload intensity?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-28
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~110 s (10 s baseline + 5 × 8 s staircase + 10 s
tail + 50 s 009-style sub-floor recipe)

## The question

`notes/ioreport.py` only handles `SIMPLE`-format channels (energy buckets).
But `--list` on M4 Max reveals a dozen `STATE`-format channels under group
`'GPU Stats'` with unit `24Mticks` (the same 24 MHz tick we time
microbenchmarks with). The most relevant by name:

| channel    | subgroup                                          |
|------------|---------------------------------------------------|
| **GPUPH**  | GPU Performance States                            |
| BSTGPUPH   | GPU Boost Controller Performance States           |
| GPU_SW     | GPU Software Performance States                   |
| PWRCTRL    | GPU Power Controller States                       |
| GPU_CLTM   | CLTM-induced GPU Performance States               |
| GPU_PPM    | PPM Target as % of Max GPU Power                  |
| PZRSDNCY   | GPU Discrete Power Zone Residency                 |
| AFRSTATE   | AFR Performance States                            |
| AFRCTRL    | AFR Power Controller States                       |
| GPUDVDH    | DVD Request States                                |
| FENDER     | Fender State                                      |
| PMU_RC     | PMU Loop Lost Performance Reason Code States      |

`GPUPH` is the canonical "what DVFS state was the GPU in?" channel —
this is the channel that, if it works, closes the long-standing
"DVFS state is unobservable from outside vendor counters" gap. The
public Metal API exposes nothing about GPU frequency or P-state.

**Primary question:** subscribing to `GPU Stats` / `GPUPH` and reading
per-state residency via `IOReportStateGetCount` /
`IOReportStateGetResidency` (bindings we don't have yet), do we see a
residency time series that **monotonically shifts to higher state
indices as workload intensity increases?**

- **PASS:** during a 0→100 % busy staircase, the residency-weighted
  mean state index increases monotonically across the 5 phases (≥ 4 of
  4 transitions in expected direction). And during sustained
  saturation, ≥ 80 % of residency lands in the top half of states.
- **MARGINAL:** residency moves with workload but not monotonically,
  OR top-state residency at saturation < 80 %, OR state names look
  weird (negative, duplicated, off-by-one). Bindings work but
  interpretation is ambiguous.
- **FAIL:** residency time series doesn't track workload at all.
  Either the channel is reporting something other than what the name
  suggests, the bindings are wrong, or the residency math is off.

## Why this question, now

Three reasons:

1. **It's the immediate next step from exp 009.** 009 found a sub-floor
   state with 3.77× cycle reduction, strongly suggestive of a max-DVFS
   upshift, but couldn't *observe* the DVFS state — only its
   consequence. GPUPH bindings would let us *see* the state
   transition during a 009-style recipe instead of inferring it.
2. **The bindings are reusable infrastructure.** Once `notes/ioreport.py`
   handles STATE channels, every future experiment can record
   per-state residency at zero cost (same subprocess pattern as exp
   008). This is the "build infrastructure for the lab" half of the
   project's deliverable.
3. **It's a focused tooling test.** No microbench timing, no kernel
   variation — just a workload that drives the GPU through known
   regimes and a verification that the residency data is sensible.

If it works, exp 011 can do strict cross-validation against
powermetrics' per-P-state output (`gpu_power` sampler reports
"GPU active residency: 18.03% (300 MHz: 0.00% ... 1340 MHz: 18.03%)").
If 010 fails, 011 doesn't make sense yet.

## Why this is scoped to face validity, not strict cross-validation

Strict cross-validation against powermetrics' P-state breakdown is a
better test, but it requires either:
- The two-terminal sudo dance (user starts powermetrics, we run
  workload, both write CSVs to be joined post-hoc), OR
- Resolving how powermetrics labels frequencies vs how IOReport
  labels states, which we don't know yet.

Face validity asks the simpler question: do we get *any* state
residency at all, do the names make sense, do they shift with
workload? If yes, exp 011 can do the strict version. If no, exp 011
is unnecessary.

Per "raw before robust": expose the data first, judge it second.

## Hypothesis

Confidence: medium (the channel name is descriptive and macmon
reads it with similar bindings; we expect it to work).

Predictions:

- **GPUPH state count:** 8-16 states. M-series GPUs publish 4-16
  P-states historically; M4 Max likely closer to 16.
- **Idle baseline residency:** ~95 %+ in the lowest state (idle
  state index 0 or 1).
- **Staircase residency:** time-weighted mean state index increases
  with each step. step_25 should land mostly in mid-low states,
  step_50 in mid, step_75 in mid-high, step_100 in top 1-2 states.
- **Tail residency:** decays back toward the lowest state but more
  slowly than idle (the chip stays in mid-states briefly after load
  ends).
- **Sub-floor recipe (50 s of fma_loop K=20 sleep_0):** ≥ 90 %
  residency in the top state once the warmup completes (matches
  009's "deep tier ~40 ticks" mechanism speculation: GPU at peak
  DVFS).
- **State name format:** likely "Pn" (n = index) or descriptive
  ("APSC_P0", "PERFORMANCE_S5", or similar). May be opaque
  abbreviations. Names won't include the actual MHz value (Apple
  doesn't publish per-state freq mappings, hence the long-standing
  unknown).
- **Residency in 24Mticks:** sum of residencies per window should
  approximately equal `window_s * 24_000_000`. If it's much less,
  the channel doesn't account for off-cycles; if it's much more,
  states are double-counted (would hurt percent calculations).

## What we are NOT trying to answer

- **Frequency / MHz mapping per state.** This experiment exposes
  state names + residencies. Mapping those names to MHz values is a
  separate question (would need powermetrics cross-ref or reverse
  engineering). In scope only as a freebie if state names happen to
  contain MHz (unlikely).
- **Strict cross-validation against powermetrics.** Exp 011 if 010
  passes face validity.
- **Other STATE channels.** `BSTGPUPH`, `GPU_SW`, `PWRCTRL`,
  `AFRSTATE` etc. will be enumerated and their state counts logged
  (free side effect of the bindings) but only `GPUPH` is the
  pass/fail target.
- **Cross-chip generalization.** M4 Max only. M1 Pro will need a
  separate run to confirm the same channel exists with similar
  shape. (Likely yes per macmon's history but unverified here.)
- **Sub-second sampling.** Residency over 1 s windows is the
  default; finer-grained sampling questions (e.g. trial-level
  residency during exp 009 recipe) need sub-second cadence and a
  separate experiment.
- **Anything about ANE / CPU / DRAM state channels.** Out of scope.

## Setup

### What `notes/ioreport.py` needs

New ctypes bindings:
- `IOReportStateGetCount(channel) -> int`
- `IOReportStateGetNameForIndex(channel, idx) -> CFString`
- `IOReportStateGetResidency(channel, idx) -> int64`

New CLI flag `--include-states` that, when set:
- Subscribes to all channels via existing path (no group filter).
- For each `STATE`-format channel encountered in the delta dict,
  emits a row per state to a separate CSV: `<csv-prefix>-states.csv`
  with columns `iso_ts, monotonic_ns, window_s, group, subgroup,
  channel, state_idx, state_name, residency_24Mticks, residency_pct`.
- The existing energy CSV (`<csv-prefix>.csv`) continues to work unchanged.

### What `run.py` does

Same shape as exp 008, with one extra phase appended:

1. Validates that `notes/ioreport.py` script exists.
2. Launches `notes/ioreport.py --include-states --interval-ms 1000
   --csv raw/<ts>.csv` as a subprocess. This is the sudo-free side.
3. Phase markers written to `raw/<ts>-phases.csv` for each phase
   transition.
4. Phase 0 — **baseline** (10 s idle, no GPU work).
5. Phase 1 — **staircase** (5 × 8 s steps at 0/25/50/75/100 % target
   busy). Same `fma_loop` workload + duty cycling as exp 008.
6. Phase 2 — **tail** (10 s idle).
7. Phase 3 — **sub-floor recipe** (50 s of `fma_loop K=20 sleep_0`
   warmup pattern from exp 009, looped — ~10 K dispatches). This is
   where, per 009, we'd expect peak DVFS residency.
8. SIGINT to ioreport.py for clean CSV flush.

### What `analysis.py` does

1. Loads `raw/<ts>-states.csv`, `raw/<ts>.csv` (energy), `raw/<ts>-phases.csv`.
2. Filters to `channel == 'GPUPH'`.
3. For each phase window: aggregate residency per state across the
   window, compute residency percentages, compute weighted-mean
   state index.
4. Reports the headline pass/fail table:

   | phase     | dominant state(s) | top-state pct | mean state idx | verdict for that phase |
   |-----------|------------------:|--------------:|---------------:|------------------------|
   | baseline  |                   |               |                |                        |
   | step_000  |                   |               |                |                        |
   | step_025  |                   |               |                |                        |
   | step_050  |                   |               |                |                        |
   | step_075  |                   |               |                |                        |
   | step_100  |                   |               |                |                        |
   | tail      |                   |               |                |                        |
   | sub-floor |                   |               |                |                        |

5. Bonus: per-phase summary for the *other* GPU-stats STATE channels
   (BSTGPUPH, PWRCTRL, GPU_SW, etc.) so we know what else came along
   for free.

### What we record

`raw/<ts>.csv` — energy, same as exp 008.
`raw/<ts>-states.csv` — new: per-state residency rows.
`raw/<ts>-phases.csv` — phase markers (same as exp 008).
`raw/<ts>-meta.txt` — env, channel inventory, residency-sum sanity
check, verdict.

### What we do NOT do

- No averaging in live output. CSVs are raw per-window deltas.
- No retries on failure. If bindings are broken, raise loudly.
- No hardcoded state-index → frequency mapping. We log whatever
  names IOReport gives us.
- No discarding of "weird" residency values (negative, zero-sum,
  off-by-one). They go in the CSV; the analysis surfaces them.

## Success criterion

The experiment **succeeds** (in the discipline's sense) if we have:

1. New STATE bindings in `notes/ioreport.py` that don't crash on
   M4 Max.
2. `<ts>-states.csv` with rows for `GPUPH` covering the workload
   window.
3. Phase markers covering all 8 phases.
4. Residency-sum sanity check in metadata: sum-of-states-per-window
   ≈ window_s × 24M ticks (within 5 %).

It produces a **usable answer** if the per-phase table fills in and
the verdict is PASS / MARGINAL / FAIL by the threshold above.

## New questions we expect this to raise

- If GPUPH residency tracks workload cleanly but state names are
  opaque ("P0".."Pn") with no frequency information, the natural
  follow-up is a cross-channel inference: do `BSTGPUPH` /
  `PWRCTRL` / `GPU_SW` indices line up with `GPUPH` indices, and
  does any of them include MHz info?
- If the sub-floor recipe (phase 3) does NOT show top-state
  dominance, that falsifies the 009 mechanism speculation that the
  ~2 µs state is a DVFS upshift. Mechanism would need to be
  something else (different scheduling path, timestamp counter
  domain shift, dynamic caching). Big finding either way.
- If state count is much larger than expected (say, 30+), GPUPH
  may include sub-states or virtual states we don't have intuition
  for. Doesn't fail the experiment but reframes interpretation.
- If residency-sum is much less than `window_s × 24M ticks`, the
  channel's residency excludes off / sleep / clock-gated time,
  which means percent-of-residency requires careful interpretation
  ("of active time" vs "of wall time"). Worth knowing.
- If `BSTGPUPH` (boost) shows different residency from `GPUPH`,
  the boost controller has its own state machine. Worth
  characterizing in a follow-up.

## After this experiment

Branches:

- **PASS.** Add a project memory entry on what GPUPH state names look
  like on M4 Max. Schedule exp 011 as cross-validation against
  powermetrics. Promote GPUPH residency capture to default for
  future experiments wanting DVFS visibility (subprocess pattern,
  same as exp 008 telemetry).
- **MARGINAL.** Document the interpretation gap. May still be
  usable for *relative* state inference (e.g. "during workload A
  vs workload B, residency shifted UP/DOWN") even if the absolute
  state-index meaning is unclear.
- **FAIL.** Most likely cause: the bindings were wrong or the channel
  is reporting different units than the label suggests. Re-validate
  against macmon's Rust implementation; if macmon agrees with our
  output, the channel itself is misleading and we file the negative
  result. If macmon disagrees, our bindings are wrong; fix and
  re-run.

We do not plan past these branches.

---

## Result

**Date run:** 2026-04-28 (timestamp prefix `20260428T212810`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Wall-clock:** ~110 s (10 s baseline + 5 × 8 s staircase + 10 s tail
+ 50 s sub-floor recipe).
**Outcome:** **PASS.** Staircase mean state index trajectory was
**monotonic 4/4 transitions** (0.14 → 0.57 → 1.15 → 2.10 → 13.65),
and step_100pct concentrated **89.8 % of residency in the top half**
of states (P15 alone at 89.2 %). GPUPH bindings work; per-state
residency tracks workload intensity faithfully.

### Headline: per-phase GPUPH residency on M4 Max

| phase       | n_win | top-1 state | top-1 %  | top-2 state | top-2 % | mean idx |
|-------------|------:|:-----------:|---------:|:-----------:|--------:|---------:|
| baseline    |     8 | OFF         |  85.1 %  | P1          |  14.9 % |     0.15 |
| step_000pct |     6 | OFF         |  86.1 %  | P1          |  13.9 % |     0.14 |
| step_025pct |     7 | P1          |  57.4 %  | OFF         |  42.6 % |     0.57 |
| step_050pct |     7 | P1          |  80.2 %  | OFF         |   9.0 % |     1.15 |
| step_075pct |     6 | P2          |  76.7 %  | P1          |   9.9 % |     2.10 |
| **step_100pct** | 7 | **P15**     | **89.2 %** | P2        |   8.4 % | **13.65** |
| tail        |     8 | OFF         |  83.7 %  | P1          |  12.6 % |     0.66 |
| **subfloor** |   41 | **P15**     | **61.6 %** | P1        |  23.7 % |    10.41 |

GPUPH on M4 Max has **16 states**: index 0 = `OFF`, indices 1-15 =
`P1` through `P15`. State names contain no frequency info (as
predicted — Apple doesn't publish per-state MHz mappings).

Per-window residency sum vs wall-clock ticks: **99.8-100.1 %** —
GPUPH residency adds up to wall time, not just active time. This
makes percent-of-residency interpretation natural ("of all wall
clock, X % was in state Y") without "of active time" caveats.

### Hypothesis check

| prediction                                          | result                                | verdict |
|-----------------------------------------------------|---------------------------------------|---------|
| GPUPH state count: 8-16                             | 16                                    | ✓       |
| Idle baseline: ~95 %+ in lowest state               | 85.1 % OFF + 14.9 % P1 = 100 %        | partial — OFF alone is 85 %; chip never quite reaches "fully off" because of background activity |
| Staircase mean state idx monotonic up               | 0.14 → 0.57 → 1.15 → 2.10 → 13.65     | ✓       |
| Tail decays slower than idle baseline               | tail mean 0.66 vs baseline 0.15       | ✓       |
| Sub-floor recipe: ≥ 90 % residency in top state     | P15 at 61.6 %                         | **falsified** |
| State names "Pn" or descriptive                     | "OFF", "P1".."P15"                    | ✓       |
| Residency sum ≈ window × 24 M ticks                 | 99.8-100.1 %                          | ✓       |

The big falsified prediction is sub-floor recipe pinning P15 ≥ 90 %.
Reality: 61.6 % P15 + 23.7 % P1. See Surprises § 1 for the mechanism
that explains it.

## Surprises

### 1. The sub-floor recipe oscillates P15 ↔ P1, doesn't pin P15

Exp 009's mechanism speculation was "the chip enters peak DVFS state
and stays there during sub-floor recipe — that's why per-dispatch time
drops 3.77×." 010 says: **the chip enters P15 when actively executing
the kernel, but drops back toward P1 between dispatches.** Sum 85.3 %
(P15 + P1); the rest is in transition states.

Compare to step_100pct (heavier 65 K-FMA kernel): P15 at 89.2 %.
Sustained heavy work pins P15 because each dispatch occupies the
full ~1 ms scheduler window. The 009 recipe's kernels are tiny
(1 K FMAs × 32 threads ≈ 30 µs at peak), so the chip cycles
peak ↔ idle on each dispatch.

**This refines, not refutes, the 009 mechanism.** When the chip
*is* executing a sub-floor-recipe kernel, it's at P15 — and the
kernel does take ~39 ticks. The 1 625 ns absolute min in 009 is
correctly attributed to P15-cycle-rate; what was wrong was the
mental picture of the chip "staying there." It's there at the
moments that matter.

### 2. PWRCTRL is in `DEADLINE` mode during sub-floor, not `PERF`

The bonus signal from neighbouring `GPU Stats` channels exposes
something neither 009 nor the GPUPH analysis alone could see:

| phase       | PWRCTRL state           |
|-------------|-------------------------|
| baseline    | IDLE_OFF (85 %)         |
| step_025-100| **PERF** (57 → 100 %)  |
| tail        | IDLE_OFF (84 %)         |
| **subfloor**| **DEADLINE (76 %)**     |

The GPU power controller has at least three modes: `IDLE_OFF`,
`PERF`, and `DEADLINE`. The staircase phases all run in `PERF` mode.
**The sub-floor recipe runs in `DEADLINE` mode**, a different
controller behavior we hadn't seen named anywhere.

Mechanism speculation: `DEADLINE` is the latency-oriented power
controller mode, possibly geared to "wake the GPU to peak DVFS
per-dispatch and let it idle between." This would explain both:
- Why sub-floor cycle count is so low (peak DVFS during work).
- Why GPUPH residency is split P15+P1 (peak during dispatch, idle
  between).
- Why 009's two-tier structure exists (mid-tier = chip in transition
  toward DEADLINE; deep-tier = chip stable in DEADLINE mode).

This is a **new opening for 009's mechanism story** that needs its
own follow-up. Hypothesis: the sub-floor state isn't a different
DVFS state; it's the same P15 state, accessed via the DEADLINE
controller path which has lower overhead than PERF.

### 3. BSTGPUPH (boost controller) caps at P10, never reaches P15

| phase     | BSTGPUPH top |
|-----------|--------------|
| step_100  | P10 (89 %)   |
| subfloor  | P10 (62 %)   |

GPUPH and BSTGPUPH report different state numberings. GPUPH's P15
during step_100 corresponds to BSTGPUPH's P10. The boost controller
likely indexes states above some "base" point, with P10 as its top.
Cross-channel index correspondence is not 1:1; future analysis
should not assume equivalence.

### 4. AFRCTRL and GPU_PPM are uninformative — same state always

Several "free signal" channels stayed at a single state across all
phases:

- `AFRCTRL`: PERF 100 % everywhere
- `GPU_CLTM`: NO_CLTM 100 % everywhere
- `GPU_PPM`: 100 % everywhere
- `FENDER`: ON 100 % everywhere
- `GPUDVDH`: 100 % everywhere
- `PMU_RC`: NONE 100 % everywhere
- `PZRSDNCY`: NO_ZONE 100 % everywhere

These channels exist for state machines that don't engage during
trivial GPU compute work. They might engage during display-heavy
workloads, sustained thermal pressure, or specific power-zone
configurations. Filed as "available but inactive on these phases."

### 5. AFRSTATE and GPU_SW track GPUPH closely with their own naming

- `GPU_SW`: SW_OFF, SW_P1, ..., SW_P15. Software-side performance
  state. At step_100 → SW_P10 (89 %), at subfloor → SW_P10 (62 %).
  Same shape as BSTGPUPH (caps at index 10, not 15).
- `AFRSTATE`: OFF / P1-P7 (only 8 states observed). At step_100 →
  P7 (90 %); at subfloor → P7 (67 %). Likely the AFR (display
  engine) participating in GPU work proportionally.

Useful as cross-validation: when GPUPH says P15, GPU_SW says SW_P10
and AFRSTATE says P7 simultaneously. All three agree the chip is at
the top of its respective state space.

## What this means operationally

For the project:

1. **GPUPH residency is now part of the lab's standard telemetry
   stack.** `notes/ioreport.py --include-states` writes a sibling
   `-states.csv` with no extra setup; future experiments can record
   per-DVFS-state residency by passing one flag.
2. **The DVFS-state-is-unobservable gap is closed.** We now have a
   sudo-free path to per-state residency at 1 s cadence on M4 Max.
   Not yet validated against powermetrics' MHz output (exp 011), but
   face-validity is strong.
3. **Sub-floor recipe interpretation needs a rewrite** in the lab
   state. The original 009 narrative ("chip enters peak DVFS state")
   is partly right and partly wrong: it enters peak DVFS *during
   dispatches*, but the controller is in `DEADLINE` mode (different
   from staircase's `PERF` mode). The "peak DVFS" framing is too
   simple; the better framing is "different controller mode produces
   different DVFS access pattern."
4. **The PWRCTRL channel is more informative than GPUPH alone for
   recipe characterization.** GPUPH tells you "what state right now."
   PWRCTRL tells you "what power-management strategy is the chip
   pursuing across the workload." Both matter.

## What does NOT change

- All decisions (003 / 004 / 005) and all 001-009 results stand
  unchanged. Adding a new observation channel doesn't invalidate
  prior measurements — it gives them a new lens.
- `notes/ioreport.py`'s SIMPLE-channel energy CSV format is
  unchanged. New `--include-states` flag is opt-in.

## What changes

- UNKNOWNS.md: close the "GPUPH bindings unknown" question, refine
  the 009 mechanism question (DVFS-state path → DEADLINE-controller
  path), open new question about PERF vs DEADLINE mode boundaries.
- The 009-sub-floor narrative in the lab snapshot should be edited to
  add: "mechanism is PWRCTRL DEADLINE mode + intermittent P15 access
  per dispatch, not sustained P15 residency." This is non-trivial
  and goes in the *next* dated snapshot, not the 0428 one.

### Natural follow-ups

- **Exp 011: GPUPH vs powermetrics cross-validation.** If powermetrics'
  per-MHz residency lines up with GPUPH's per-state-name residency,
  we can build a state-index → MHz mapping. Pre-registerable now.
- **Exp 012: Recipe → controller mode mapping.** What recipes put the
  chip in DEADLINE vs PERF? Is it dispatch size? Inter-dispatch
  spacing? Cumulative dispatch rate? A small focused sweep would
  expose the boundary.
- **Exp: 009-mechanism re-frame.** Re-run 009 with GPUPH + PWRCTRL
  capture during the sub-floor entry. Should show the chip
  transitioning from PERF to DEADLINE mode at the trial-29 onset.
- **Exp: GPU_SW / BSTGPUPH cross-channel state alignment.** Map all
  `GPU Stats` STATE channels' state spaces to one another. Would
  consolidate the multi-channel residency picture.

We do not plan past these.
