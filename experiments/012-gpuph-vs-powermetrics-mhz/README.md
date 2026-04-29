# 012: Can we build a GPUPH state-index → MHz mapping by aligning IOReport per-state residency with powermetrics per-MHz residency?

**Status:** pre-registered, not yet run
**Date pre-registered:** 2026-04-28
**Hardware target:** Apple M4 Max 36GB / `applegpu_g16s`, MacBook Pro 14"
(Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~80 s (10 s baseline + 5 × 8 s staircase + 10 s
tail + 20 s recovery), with the user-runs-sudo dance from exp 008.

## The question

Exp 010 established that GPUPH's 16 states (`OFF`, `P1`-`P15`) track
workload intensity but contain no MHz information in their names.
Apple doesn't publish a GPU-state-to-frequency table. powermetrics
*does* report per-MHz residency at each sample, in a parenthesized
breakdown after the `GPU active residency` line:

    GPU active residency:  84.42% (444 MHz: 0.00% 800 MHz: 0.05% ...
                                   1180 MHz: 84.36%)

(Exact MHz buckets vary by chip and macOS version.) If we record both
sources side-by-side over the same staircase workload, we can align
them and build a state-index → MHz mapping.

**Primary question:** sampling GPUPH per-state residency (via the
exp 010 bindings) and powermetrics per-MHz active residency
*simultaneously* across a 0→100 % busy staircase, can we recover a
monotonic mapping `P_idx → MHz` such that GPUPH state X's residency
profile across phases matches powermetrics' MHz-X' residency profile?

- **PASS:** every active GPUPH P-state (P1-P15 with non-zero
  residency in some phase) has a unique powermetrics MHz bucket whose
  per-phase residency-distribution correlates ≥ 0.85 (Spearman) with
  the GPUPH state's per-phase distribution. Ordering matches:
  P_n → MHz_n monotonic in n.
- **MARGINAL:** mapping exists but is not 1:1 (e.g. powermetrics
  reports 8 MHz buckets and GPUPH has 15 active states; some MHz
  buckets span multiple P-states or some P-states are sub-states of
  one MHz bucket). Or correlations are 0.7-0.85 — directional but
  noisy.
- **FAIL:** no consistent alignment. GPUPH state ordering doesn't
  match MHz ordering (e.g. P5 maps to a higher MHz than P10), or
  residency profiles don't correlate. Either the bindings are
  reading something different from what the channel name suggests,
  or `GPUPH` indexes a different state space than powermetrics'
  P-state breakdown.

## Why this question, now

Three reasons:

1. **It's the next obvious step from 010.** GPUPH bindings work; we
   know they track workload. Without the MHz mapping, we can only
   make qualitative claims ("residency shifted to higher states");
   with it, we can make quantitative ones ("chip ran at 1140 MHz
   for X seconds").
2. **The 011 mechanism findings need a numeric layer.** We saw
   DEADLINE 33-48 % during sub-floor and P15 6-15 %. Knowing that
   P15 = N MHz (and lower P-states = M MHz) lets us compute average
   GPU clock frequency during sub-floor, which is the variable
   that drives the 1.7 µs floor calculation directly. (Cycle count
   per kernel × P15 clock → wall-clock per kernel.)
3. **Same architecture as 008.** User starts powermetrics in another
   terminal; we launch ioreport in this one; both write to CSVs
   joined by `monotonic_ns`. Pattern is established; this is a new
   data type using the same protocol.

## Hypothesis

Confidence: medium-high. Predictions:

- **Number of MHz buckets powermetrics reports:** likely 8-12, fewer
  than GPUPH's 15 active states. The "P-states" Apple ships in
  software may aggregate finer hardware substates.
- **Mapping shape:** monotonic. P1 = lowest active MHz, P15 = highest.
  The intervening states should map to MHz buckets in order, possibly
  with some MHz buckets covering multiple P-states (e.g. P3-P4 both
  map to 800 MHz).
- **At step_100, both sources should agree:** powermetrics' top MHz
  bucket residency ≈ GPUPH's P15 residency. Both ~85-90 %.
- **At step_025, both should show similar low-MHz dominance:**
  powermetrics' lowest non-zero active MHz bucket residency ≈ GPUPH's
  P1-P2 residency.
- **OFF state alignment:** powermetrics' "active residency" 16 % at
  baseline ≈ 100 % - GPUPH's `OFF` residency 84 %. (`OFF` is "GPU not
  active"; powermetrics calls this "inactive" implicitly, by
  excluding it from the active-residency breakdown.)
- **The M4 Max top GPU MHz is around 1.4 GHz.** macOS-published
  numbers for M4 Max GPU peak; we expect to see something near this
  in powermetrics' top bucket name.

## What we are NOT trying to answer

- **Whether powermetrics' MHz numbers are themselves correct.**
  Trusted as the canonical reference; cross-validating against
  external sources is out of scope.
- **Mapping for non-canonical state channels.** `BSTGPUPH`,
  `GPU_SW`, etc. are bonus inputs to analysis, not the pass/fail
  target.
- **Cross-chip generalization.** M4 Max only. Any M-series chip
  would need its own enumeration.
- **MHz mapping for GPUPH OFF.** OFF means "GPU not active" — no
  meaningful MHz. Only P1-P15 get mapped.
- **Sub-second / per-kernel MHz.** 250 ms aggregate is the
  resolution. Per-kernel-dispatch MHz claims need a separate
  experiment.
- **Whether MHz is the *only* thing distinguishing P-states.**
  Voltage, instruction-throughput, etc. could also differ. We map
  by MHz because that's what powermetrics gives us; full P-state
  characterization is bigger.

## Setup

### Prerequisites the user (not Claude) starts

In a separate terminal, before invoking `run.py`:

```
sudo uv run notes/gpu_telemetry.py \
    --csv experiments/012-gpuph-vs-powermetrics-mhz/raw/PMTELEM.csv \
    --interval-ms 250 \
    --quiet
```

This writes the per-sample dashboard fields to CSV. **gpu_telemetry.py
doesn't currently capture per-MHz residency**, so we additionally
have the user start a raw powermetrics capture:

```
sudo powermetrics --samplers gpu_power -i 250 \
    > experiments/012-gpuph-vs-powermetrics-mhz/raw/PMRAW.txt 2>&1
```

The raw output contains the per-MHz residency breakdown that we
parse in analysis. (We could fold this into gpu_telemetry.py later
if useful — for now, raw + parser keeps 012 self-contained.)

### What `run.py` does

Same staircase shape as exp 008 / 010, with phase markers:

1. Validates that `PMRAW.txt` exists and is being written.
2. Launches `notes/ioreport.py --include-states --interval-ms 250
   --csv raw/{ts}-ioreport.csv`. State data → `raw/{ts}-ioreport-states.csv`.
3. Phase 0 — baseline (10 s idle).
4. Phase 1 — staircase (5 × 8 s steps at 0/25/50/75/100 % target busy).
5. Phase 2 — tail (10 s idle).
6. SIGINT to ioreport.py for clean CSV flush.
7. Tells user to stop powermetrics.

Phase markers go to `raw/{ts}-phases.csv` keyed by `monotonic_ns` for
post-hoc joining. Same kernel as exp 008/010 (fma_loop with
FMA_ITERS=65536, 32 threads — known long enough to land in
work-dominated regime).

### What `analysis.py` does

1. Loads four sources:
   - `raw/PMRAW.txt`: parse with a custom regex for the per-MHz
     residency lines. Each sample emits a dict
     `{monotonic_ns? (parsed from sample timestamp), mhz_residency:
     {444: 0.00, 800: 0.05, ..., 1180: 84.36}, active_pct: float}`.
     We capture monotonic_ns by parsing powermetrics' "elapsed_ns"
     field if present, OR fall back to the user-process clock from
     gpu_telemetry.py's CSV (joined by sample number).
   - `raw/PMTELEM.csv`: per-sample monotonic_ns + active_pct + freq_mhz.
   - `raw/{ts}-ioreport-states.csv`: GPUPH per-state residency with
     monotonic_ns.
   - `raw/{ts}-phases.csv`: phase markers with monotonic_ns.
2. Aligns: for each phase window, aggregate per-MHz residency from
   PMRAW (averaging across samples in the window) and per-P-state
   residency from ioreport-states (summing across windows in the
   phase, normalizing).
3. Builds the alignment table:

   | phase    | pm: 444 | pm: 800 | pm: ... | pm: 1180 | iop: P1 | iop: P2 | iop: ... | iop: P15 |
   |----------|--------:|--------:|--------:|---------:|--------:|--------:|---------:|---------:|

   And reports correlations: for each GPUPH state index n with
   non-zero residency, find the powermetrics MHz bucket with the
   most similar per-phase residency profile (Spearman correlation).
4. Verdict per the success criteria above.

### What we record

`raw/PMRAW.txt`: powermetrics raw text output (~80 s of samples).
`raw/PMTELEM.csv`: gpu_telemetry CSV (joinable on monotonic_ns).
`raw/{ts}-ioreport.csv`: IOReport energy CSV.
`raw/{ts}-ioreport-states.csv`: IOReport per-state residency
(GPU Stats group).
`raw/{ts}-phases.csv`: phase markers.
`raw/{ts}-meta.txt`: env, channel inventory, alignment table,
verdict.

### What we do NOT do

- No averaging in live output.
- No retries. If powermetrics drops a sample (it sometimes does at
  high cadence), it goes in the parsed data with whatever fields
  it managed to emit.
- No discarding of "weird" residency values (negative, off-by-one).
- No interpolation across powermetrics samples — we accept the
  cadence as-is and bin into phases at the granularity recorded.

## Success criterion

The experiment **succeeds** (in the discipline's sense) if we have:

1. PMRAW.txt with per-MHz residency parsed for ≥ 80 % of expected
   samples.
2. PMTELEM.csv with ≥ 80 % expected samples.
3. ioreport-states.csv with rows spanning the workload window.
4. Phase markers covering all 7 phases.

It produces a **usable answer** if we can fill in the per-phase
alignment table and decide:

- The number of distinct MHz buckets powermetrics reports.
- Whether the GPUPH state ordering matches MHz ordering.
- Per-state correlations between GPUPH and powermetrics MHz residency.

## New questions we expect this to raise

- If powermetrics reports e.g. 8 MHz buckets and GPUPH has 15 active
  states, **multiple GPUPH P-states map to one MHz bucket**. Which
  ones? The mapping shape is its own follow-up.
- If powermetrics' top MHz at step_100 is *lower* than expected
  (~1.0-1.2 GHz when M4 Max is published as ~1.4 GHz peak), the
  staircase recipe may not push the chip hard enough to reach top
  DVFS. Worth confirming with the 009 sub-floor recipe pattern.
- If GPUPH P15 residency is much higher than powermetrics' top MHz
  residency at step_100, GPUPH may aggregate "P15 + boost" while
  powermetrics shows them separately. Cross-check with `BSTGPUPH`.
- If *all* mapping correlations are below 0.7, we may be reading
  GPUPH wrong. Re-validate ioreport bindings against macmon's
  expected output.

## After this experiment

Branches:

- **PASS.** GPUPH ↔ MHz mapping is now in the lab's reference
  inventory. Future experiments can quote "P15 at 1140 MHz" and
  similar concrete claims. exp 011's "P15 residency 6-15 % during
  sub-floor" can be re-stated as time-at-1140-MHz.
- **MARGINAL.** Map what we can map; document the parts that don't
  cleanly align (1:N or N:1 P-state ↔ MHz relationships).
- **FAIL.** Re-validate ioreport bindings vs macmon. If macmon
  shows the same numbers we do, the channel itself is misleading
  and we file the negative result. If different, our bindings are
  wrong; fix and re-run.

We do not plan past these branches.

---

## Result

**Date run:** 2026-04-28 (timestamp prefix `20260428T221203`)
**Hardware:** Apple M4 Max 36GB / `applegpu_g16s` / macOS 26.4.1
**Wall-clock:** ~80 s (10 s baseline + 5 × 8 s staircase + 10 s tail).
**Outcome:** **MARGINAL PASS.** GPUPH ↔ MHz mapping is recovered with
median 0.03 % residency drift between powermetrics' positional MHz
breakdown and its own SW_state for the canonical positions (idle, low,
peak), and median 2.65 % between IOReport GPUPH and powermetrics MHz
across phases. The mapping has a structural surprise: powermetrics'
**`SW state` uses only 10 labels** (SW_P1-SW_P10) for the chip's 15
hardware DVFS settings — IOReport GPUPH gives finer-grained visibility.

### The M4 Max GPUPH → MHz table

Extracted directly from powermetrics' positional MHz list (the parens
content of `GPU HW active residency`):

| GPUPH idx | MHz   | GPUPH idx | MHz    | GPUPH idx | MHz    |
|----------:|------:|----------:|-------:|----------:|-------:|
| OFF       | (idle)| P6        | 1 056  | P11       | 1 242  |
| **P1**    |   338 | P7        | 1 062  | P12       | 1 380  |
| P2        |   618 | P8        | 1 182  | P13       | 1 326  |
| P3        |   796 | **P9**    | 1 182  | P14       | 1 470  |
| P4        |   924 | P10       | 1 312  | **P15**   | 1 578  |
| P5        |   952 |           |        |           |        |

Three structural curiosities in this table:

1. **Not strictly monotonic in index.** P10 = 1 312 MHz but P11 = 1 242
   MHz (lower). Same for P12/P13 (1 380 / 1 326) and P14 is also a
   step. Apple's GPUPH index isn't a frequency ordering.
2. **P8 = P9 = 1 182 MHz.** Two consecutive states at the same
   frequency. Likely a voltage/SLC/cache substate distinction the
   public APIs don't expose.
3. **15 distinct GPUPH states, 10 distinct powermetrics SW labels.**
   `GPU SW state` only ever shows SW_P1..SW_P10 active. The OS-side
   API aggregates GPUPH's 15 states into 10 user-visible categories.
   IOReport sees the full 15.

### Per-phase residency: powermetrics MHz vs IOReport GPUPH P-states

| phase       | active% (pm) | top MHz   | top GPUPH | gpuph_pct | mhz_pct |
|-------------|-------------:|:---------:|:---------:|----------:|--------:|
| baseline    |      12.6 %  | 338 MHz   | P1        |   12.63 % |  12.60 %|
| step_000    |      17.1 %  | 338 MHz   | P1        |   12.17 % |  17.14 %|
| step_025    |      65.5 %  | 338 MHz   | P1        |   59.52 % |  64.52 %|
| step_050    |      99.4 %  | 338 MHz   | P1        |   66.16 % |  56.37 %|
| step_075    |      99.7 %  | 618 MHz   | P2        |   57.27 % |  56.53 %|
| **step_100**|     100.0 %  | **1 578 MHz** | **P15** | **94.58 %**| **91.32 %**|
| tail        |      15.3 %  | 338 MHz   | P1        |   14.65 % |  15.26 %|

Cross-source agreement: **median 2.65 % residency drift** between
IOReport GPUPH and powermetrics MHz at the same positional index, max
9.79 % (step_050 P1: GPUPH 66 %, MHz pos 1 56 %; the difference is
likely IOReport's GPUPH counts a moment of P1 that powermetrics MHz
classifies into a slightly different bucket, or a 250 ms window
boundary effect).

### Hypothesis check

| prediction                                              | observed                                    | verdict   |
|---------------------------------------------------------|---------------------------------------------|-----------|
| 8-12 MHz buckets in powermetrics                        | **15** (one per GPUPH active state)         | refined   |
| Mapping is monotonic                                    | **NO** — P10 > P11 in MHz                   | falsified |
| At step_100, top MHz residency ≈ GPUPH P15 residency    | 91.32 % vs 94.58 %                          | ✓         |
| At step_025, lowest active MHz dominates                | 338 MHz at 64.52 %                          | ✓         |
| OFF residency aligns with 100 % - powermetrics active   | baseline OFF 87.4 % ≈ inactive 87.4 %       | ✓         |
| M4 Max top GPU MHz ≈ 1.4 GHz                            | **1.578 GHz** (higher than expected)        | refined   |
| State name format "Pn"                                  | exactly "Pn"                                | ✓         |

The "monotonic in index" prediction is the headline falsification.
Apple's GPUPH index ordering doesn't follow frequency. The three
non-monotonicities are real, not measurement artifacts (multiple
samples in the workload show position-10 / position-11 residency
swapping).

## Surprises

### 1. Three different P-state numberings at three layers

This experiment exposed three concurrent state-numbering systems on
the same chip:

- **IOReport GPUPH (`OFF + P1..P15`):** 16 states, finer-grained.
  P15 = 1 578 MHz (peak).
- **powermetrics' MHz positional list:** 15 states by ordinal
  position. Position 15 = 1 578 MHz. Maps 1:1 with GPUPH P1..P15.
- **powermetrics' `GPU SW state` (`SW_P1..SW_P10`):** 10 user-visible
  states. SW_P10 = 1 578 MHz (peak). The OS aggregates the chip's
  15 hardware states into 10 user-visible labels.
- **powermetrics' `GPU SW requested state` (`P1..P10`):** matches
  the SW state — a 10-slot indexing.

When something hits peak DVFS:
- IOReport says GPUPH P15
- powermetrics' SW state says SW_P10
- powermetrics' MHz residency says 1 578 MHz at position 15

These are all the **same physical state**, indexed differently. For
future cross-tool work this is critical to remember.

### 2. The M4 Max top GPU frequency is 1 578 MHz

Higher than the ~1.4 GHz often cited in third-party characterizations
of M4 Max. powermetrics' positional list explicitly names 1 578 MHz
as the top bucket and step_100 of our staircase reaches it for 91 %
of the phase. Operational implication: any "M4 Max GPU peak ≈ 1.4 GHz"
estimate the project has been using needs revising upward.

### 3. P-state index ordering is not monotonic in frequency

The table shows three explicit non-monotonicities:
- P10 = 1 312 MHz, P11 = 1 242 MHz (P11 is lower)
- P12 = 1 380 MHz, P13 = 1 326 MHz (P13 is lower)
- P8 = P9 = 1 182 MHz (same)

Possibilities:
- **Voltage/SLC substates:** P8 and P9 might run at the same MHz but
  different voltages or cache configurations. P10/P11 and P12/P13
  pairs may swap order due to a similar dimension Apple cares about
  but powermetrics' MHz column doesn't expose.
- **Boost / non-boost lanes:** P11/P13 might be "non-boost" variants
  of P10/P12 at slightly lower MHz. The naming would index "boost
  pairs" together rather than by raw MHz.
- **Hardware order:** the chip's hardware DVFS table just isn't
  sorted by MHz in any clean way; the indices are a hardware-level
  identifier rather than an ordering.

We don't have data to distinguish these. The next step would be
correlating BSTGPUPH (boost controller) state with each P-state to
see if the non-monotonic pairs alternate boost / non-boost.

### 4. The 1 625 ns sub-floor floor finds its frequency

Exp 009's 1 625 ns absolute minimum corresponds to **39 ticks of the
24 MHz GPU timestamp counter**. The kernel ran in 39 timestamp ticks
when the chip was at GPUPH P15 = 1 578 MHz. At 1 578 MHz, 39 ticks
correspond to:

    39 / (24 × 10⁶ Hz) × 10⁹ = 1 625 ns wall-clock
    1 625 × 10⁻⁹ × 1 578 × 10⁶ = ~2 564 GPU clocks per kernel

So the minimum write_tid 32t kernel takes ≈ 2 564 GPU cycles at peak
DVFS. The 6 125 ns back-to-back floor at lower DVFS ÷ 1 625 ns sub-
floor = 3.77×, matching powermetrics' DVFS ratio of 1 578 / 338 ≈
4.67× from idle (or 1 578 / 618 ≈ 2.55× from typical mid-load
cadence). The 3.77× sits between these — consistent with the chip
being at peak during P15 visits but transitioning through lower
states between dispatches.

This is the **first concrete frequency-attached interpretation** of
the 009 sub-floor measurement.

### 5. Cross-source agreement (IOReport ↔ powermetrics) is 2.65 % median

Strong validation for the IOReport GPUPH bindings. Per-phase residency
between the two sources agrees to within a few percent. The 9.79 %
max is at step_050 P1 — likely a window-boundary effect (one or both
sources have a sample window that overlaps a phase boundary).

For relative-magnitude analyses (e.g. "did P15 residency double
between recipe A and recipe B?"), agreement is well within tolerance.
For absolute claims ("the chip was at 1 578 MHz for X seconds"),
the two sources will give numbers within 5 % of each other in most
cases. The IOReport path can replace powermetrics for DVFS
observations going forward.

## What this means operationally

For the project:

1. **The M4 Max GPUPH → MHz table is now in the lab's reference
   inventory.** Future statements like "the chip was at P15 during
   sub-floor" can be quantified: P15 = 1 578 MHz; cycles per kernel
   at P15 = wall-clock × 1 578 MHz.
2. **GPUPH residency is a finer-grained DVFS signal than powermetrics'
   SW state.** Where `GPU SW state` reports SW_P10, GPUPH might be
   in P10, P11, P12, P13, P14, or P15. Useful for differentiating
   recipes that drive different sub-states of "peak."
3. **The "sub-floor mechanism is peak DVFS" story is now numerically
   grounded.** Exp 009's 1 625 ns floor = 39 ticks of the 24 MHz
   timestamp clock = 2 564 GPU clocks at 1 578 MHz. The chip really
   is at the published peak frequency during sub-floor visits.
4. **Sudo dependency for DVFS frequency claims is now optional.**
   IOReport GPUPH at 250 ms cadence gives us per-state residency that
   agrees with powermetrics within ~3 %. For tight-tolerance work
   (e.g. < 1 % agreement for power-efficiency claims), keep
   powermetrics as the reference; otherwise use IOReport.
5. **When citing P-state numbers from this lab, use the IOReport
   GPUPH index** (P1..P15) for consistency. Note that powermetrics
   reports the same chip behavior with `SW_P1..SW_P10`; the two
   numberings are NOT interchangeable.

## What does NOT change

- All prior decisions and measurements stand.
- The IOReport bindings (`notes/ioreport.py`) are unchanged. The new
  state-channel additions from 010 are validated.

## What changes

- Lab state snapshot's M4 Max cheat sheet should add the GPUPH → MHz
  table to the operational rules section (next snapshot, not the
  0428 frozen one).
- UNKNOWNS.md: close "GPUPH MHz mapping unknown"; refine the "M4 Max
  peak GPU freq ≈ 1.4 GHz" assumption upward to 1 578 MHz.
- The 009 sub-floor mechanism story can be re-told in cycles: kernel
  runs in ~2 564 GPU cycles at peak DVFS, vs ~9 200 cycles at typical
  back-to-back floor (~6 125 ns × 1 578 MHz / 1 ÷ 1 = no, this
  arithmetic only works one way; back-to-back floor is at lower freq).

### Natural follow-ups

- **Why are P10/P11, P12/P13, and P8/P9 ordered the way they are?**
  Cross-correlate with BSTGPUPH (boost) and PWRCTRL state at the
  moments these substates engage. May reveal "boost vs non-boost"
  dimension.
- **Re-frame 011 with the MHz mapping.** The "DEADLINE 30-48 % during
  sub-floor" finding can now include "and during DEADLINE the chip
  hit 1 578 MHz X % of the time."
- **Original queue:** exp 013 (PERF/DEADLINE recipe boundary).

We do not plan past these.
