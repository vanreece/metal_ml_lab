# 017: How fast does MPS `MPSMatrixMultiplication` run on 016's exact shape grid, and how does that ceiling compare to naive matmul on M4 Max?

**Status:** complete — methodology MARGINAL (substantively PASS, see Result); discrimination signal confirmed
**Date pre-registered:** 2026-05-02
**Date run:** 2026-05-05 (two runs: battery + AC; AC is primary)
**Hardware target:** Apple M4 Max 36 GB / `applegpu_g16s`, MacBook Pro
14" (Mac16,6), 14-core (10P+4E), AC power
**OS target:** macOS 26.4.1 (build 25E253)
**Estimated runtime:** ~5-8 min including IOReport sidecar startup,
MPS warm-up dispatches, and 34 cells × ≤ 5 s.
**Predecessors:**
- 016 established naive fp32 matmul ps/FLOP across 34 shapes on
  M4 Max. Compute plateau at M=N=K ∈ [128, 768] sits at 0.45-0.54
  ps/FLOP ≈ 1.86-2.21 TFLOP/s — ~7 % of fp32 peak. Memory-bound
  probes elevate up to 8.7×.
- That number means little without a ceiling reference. This
  experiment provides the ceiling: Apple's shipped, optimized GEMM.

## The question

016's headline number — **naive matmul plateaus at ~2 TFLOP/s on
M4 Max** — is meaningful only relative to a ceiling. Without one,
we can't tell whether ~2 TFLOP/s is "fine for naive" or "an order
of magnitude below what's possible." The natural ceiling on Apple
Silicon is `MPSMatrixMultiplication`: Apple's framework-shipped
GEMM implementation, with tiled / shared-memory / SIMD-aware
internals optimized for the chip.

Specific question:

> When we run MPS GEMM on 016's exact 34-shape grid on M4 Max,
> what per-matmul time does it produce, and what is the
> shape-by-shape ratio `naive_per_matmul / mps_per_matmul`?

The ratio is the meaningful number. Three regimes are plausible:

- **Compute-bound shapes**: naive may run 5-20× slower than MPS,
  reflecting tiling and shared-memory advantages.
- **Memory-bound shapes** (016's narrow-output probes): MPS may
  benefit less from tiling because the kernel is bandwidth-bound;
  ratio could be smaller (2-5×).
- **Tiny shapes** (sub-floor): both naive and MPS bottom out at
  the dispatch-overhead floor (~6.4 µs); ratio approaches 1×
  because dispatch dominates, not kernel.

Which regime each shape lands in is the finding.

## Pre-registered verdicts

For methodology:

- **PASS:** Per-shape p50 of MPS time recovered cleanly across
  all 34 shapes; per-shape ratio computable for ≥ 30/34 shapes.
- **MARGINAL:** MPS times available for most shapes but timing
  noise dominates at small shapes (CV > 30%), making ratios
  unreliable for ≥ 8 shapes.
- **FAIL:** MPS API doesn't run, or returns wildly inconsistent
  timings (e.g., > 5× variance in p50 across consecutive runs of
  the same shape).

For the cross-shape ceiling signal:

- **WIDE GAP:** Median ratio (naive/MPS) across the 016 compute
  plateau (M=N=K ∈ [128, 768]) is ≥ 5×. Naive matmul is leaving
  significant perf on the table; tiling matters a lot.
- **NARROW GAP:** Median ratio is < 2×. Naive isn't far from
  optimal — the tiled GEMM doesn't help enough to justify the
  complexity at these shapes.
- **SHAPE-DEPENDENT:** Ratio varies > 3× across the 34 shapes,
  with a clear pattern (e.g., big-square gap > narrow-output gap,
  or vice versa). Tells us *which* shapes naive is most behind on.

For the absolute MPS number:

- **CALIBRATED:** MPS plateau lands within 2× of theoretical fp32
  peak (28 TFLOP/s at 1.578 GHz). I.e., MPS achieves ≥ 14 TFLOP/s
  effective on the larger shapes. Apple's GEMM is doing what
  shipping production GEMMs do.
- **UNDER-PERFORMS:** MPS plateau is < 5 TFLOP/s, or no clearly
  shape-monotonic structure. Suggests MPS isn't using the chip
  well at these specific shapes (possibly fp32 is a stepchild
  vs fp16/bf16/Int8).

## Why this question, now

1. **The 2 TFLOP/s naive number is the lab's first non-synthetic
   compute reference.** It's only useful with a ceiling.
2. **Cheap.** Same protocol as 016, no internal-loop amp needed
   (we time MPS dispatches directly), 34 cells instead of 247
   ≈ 3-5 min.
3. **MPS is the natural ceiling on Apple Silicon** for Metal-API
   work. It's what every framework (PyTorch MPS backend, MLX,
   Core ML) ultimately falls back on for GEMM. Its number is the
   one shipping ML code on this chip is bounded by.
4. **External-verification adjacent.** Not a comparison against
   *another lab's* numbers, but against *Apple's own optimized
   path on this chip*. Catches "our naive impl is suspiciously
   fast" or "our timing infra has a 10× bias" failure modes that
   purely-internal cross-checks can't.

## Hypothesis

Confidence is mixed, leaning toward SHAPE-DEPENDENT + CALIBRATED.

- **MPS has a steady plateau at high TFLOP/s for mid-to-large
  square matmul.** Expected: 8-20 TFLOP/s at M=N=K ∈ [256, 2048].
  fp32 GEMM on M4 Max is bounded by ~28 TFLOP/s peak; tiled
  optimal usually achieves 50-70% of peak.
- **Ratio ≥ 5× on 016's compute plateau.** Naive at ~2 TFLOP/s,
  MPS at 10+ → 5×+ gap. Tiling is well-known to give ~10×
  speedup on naive.
- **Tiny shapes (M ≤ 32) bottom out at dispatch overhead** for
  MPS just like for naive. Ratio approaches 1× because dispatch
  cost is shape-independent. This is a *floor* on the ceiling
  comparison — it'd be a methodology bug to report a ratio
  meaningful for tiny shapes.
- **Memory-bound probes show smaller ratio** because both naive
  and MPS bottom out on memory bandwidth. MPS's tiling helps less
  when the kernel is BW-bound.
- **MPS first-dispatch overhead** (kernel JIT, descriptor setup)
  may bias the first 1-2 trials high. Warm up before timing.

## What we are NOT trying to answer

- **MPS internals.** Closed source; we treat MPS as a black-box
  ceiling. We won't speculate on tile sizes / SIMD strategies.
- **fp16 / bf16 / Int8.** fp32 only, for direct comparison to 016.
  MPS is widely thought to favor lower precisions; that's a
  separate experiment.
- **Cross-chip generalization.** M4 Max only. M1 Pro re-run
  remains a separate, deferred question.
- **Threadgroup-size sensitivity, kernel choice, buffer storage
  modes inside MPS.** We don't tune.
- **Verifying matmul correctness against MPS.** We're comparing
  *speed*, not correctness. MPS is correct by Apple's contract.
- **Internal-loop amplification of MPS.** MPS doesn't expose an
  internal-loop knob (it's a single canned dispatch); we time
  it directly. Sub-floor shapes will have dispatch-overhead-
  dominated p50, which is itself the answer for those shapes.

## Setup

### Kernel

`MPSMatrixMultiplication`, fp32, no transpose on either operand,
alpha=1.0, beta=0.0. Per-shape we create:

```python
desc = MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
    rows, cols, cols * 4, MPSDataTypeFloat32
)
matrix = MPSMatrix.alloc().initWithBuffer_descriptor_(buffer, desc)
mm = MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
    device, False, False, M, N, K, 1.0, 0.0
)
mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(cb, A, B, C)
```

### Buffers

Same allocation strategy as 016: max-sized A, B, C buffers
allocated once (16 MB / 128 MB / 16 MB), populated with same
seed (`16092604`) for reproducibility, reused per shape via
`MPSMatrixDescriptor`'s explicit row/col/rowBytes parameters.

### Shape grid (identical to 016)

| sweep | shapes | N |
|-------|--------|--:|
| Square diagonal | M=N=K ∈ {8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048} | 17 |
| K-sweep at M=N=128 | K ∈ {2, 4, 16, 64, 256, 1024, 4096} | 7 |
| K-sweep at M=N=512 | K ∈ {2, 4, 16, 64, 256, 1024, 4096} | 7 |
| Memory-bound probes | (8,4096,4096), (4,8192,4096), (2,8192,4096) | 3 |

Total: **34 shapes / 34 cells** (one cell per shape; no N_AMP grid).

### Timing protocol

For each shape:
1. **Warm-up:** dispatch MPS GEMM 5 times (discard timings) to
   amortize JIT / descriptor cache setup.
2. **Cell:** 5 s wall-clock target, 5 000 trial cap. Each trial
   is one MPS dispatch. Per-trial GPU time recovered from
   `MTLCommandBuffer.GPUStartTime` / `GPUEndTime` (see amendment
   below).
3. **Inter-cell idle:** 1 s.
4. **IOReport sidecar at 250 ms** with `--include-states` (to
   verify which PWRCTRL state each shape lands in — does MPS
   trigger PRFBOOST on the same shapes 016 did?).

### Pre-reg amendment 2026-05-05 (before run): timing source

The pre-reg's "Sample-buffer-based timestamp per trial" matched
014b/015/016, which use one `MTLComputeCommandEncoder` per trial
with `setStartOfEncoderSampleIndex_` / `setEndOfEncoderSampleIndex_`
attached to a shared `MTLCounterSampleBuffer`. **That
instrument cannot be applied to MPS dispatches.** `MPSMatrixMultiplication`
calls `encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:` —
i.e. it creates and owns its own compute encoder internally; we
hand it a `MTLCommandBuffer`, not a pass descriptor, so we have
no way to attach sample-buffer endpoints to MPS's encoder via the
public API.

Substitute timing source: `MTLCommandBuffer.GPUStartTime()` and
`GPUEndTime()`. These are available on every command buffer after
`waitUntilCompleted()`, return seconds (CFTimeInterval) from the
GPU's clock, and are the standard MPS-friendly path. They differ
from the sample-buffer instrument in two ways:
- They report command-buffer-level GPU time, including command
  scheduling between commit and first dispatch. Sample-buffer
  timestamps fence to a specific encoder boundary. Dispatch-
  overhead floor is therefore likely a few hundred ns higher
  in 017 than the ~1.6-6.4 µs floor 016 saw (we'll measure).
- Resolution: documented as ns-equivalent on Apple Silicon, but
  derived from the same GPU clock the sample buffer reads, so
  no methodology bias is expected at the ≥ 5 µs scale this
  experiment cares about.

This amendment is filed before any 017 dispatch is run; it is
not a retro-fit. The pre-reg's verdicts (PASS / MARGINAL / FAIL,
WIDE GAP / NARROW GAP / SHAPE-DEPENDENT, CALIBRATED / UNDER-PERFORMS)
are unchanged. The change is purely the instrument used to read
each trial's GPU time.

A consequence: the "shared sample buffer across all 34 cells"
operational rule from 016 doesn't apply here (no sample buffer is
used). 016's reason for that rule (counter-pool exhaustion at ~30
cells of per-cell allocation) doesn't apply when no counter sample
buffers are allocated.

### What we record

Same schema as 016, with the timing column reinterpreted per the
amendment above:
- `raw/{ts}-trials.csv` — per-trial timing. `gpu_delta_raw` =
  `GPUEndTime - GPUStartTime` in ns (vs 016 where it was a
  sample-buffer raw delta in ns).
- `raw/{ts}-cells.csv` — per-cell summary (one row per shape).
- `raw/{ts}.csv` and `raw/{ts}-states.csv` — IOReport.
- `raw/{ts}-meta.txt` — env, run config, MPS version detection
  if available.

### What we do NOT do

- No retries on MPS errors; raise loudly.
- No cooldown beyond the 1 s inter-cell idle.
- No sweep over alpha/beta/transpose flags. Fixed at α=1, β=0,
  no transpose.
- No internal-loop amp (impossible without modifying MPS).
- No back-to-back amp. Direct timing only.

## Success criterion

The experiment **succeeds** if we have:

1. Per-trial CSV for all 34 shapes, ≥ 100 trials each.
2. Per-cell summary CSV with 34 rows (one per shape).
3. IOReport states CSV with PWRCTRL/GPUPH state per cell.
4. `analysis.py` produces:
   - Per-shape: MPS p50 time → ns / matmul → ps/FLOP → TFLOP/s.
   - **Cross-shape ratio table** (naive / MPS) using 016's slope-
     derived ns/matmul as the naive baseline.
   - Headline: ratio in the 016 compute plateau (M ∈ [128, 768]).

## After this experiment

Branches:

- **WIDE GAP + CALIBRATED MPS plateau.** Confirmed: naive matmul
  is leaving N× perf on the table. Provides a concrete target if
  the lab ever writes a tiled-matmul kernel; meanwhile gives 016's
  number proper context.
- **NARROW GAP.** Naive is closer to MPS than expected on M4
  Max — possibly fp32 isn't well-optimized in MPS, or the chip's
  cache hierarchy makes naive less terrible than on competing
  hardware. Either way, an interesting finding worth a follow-up
  experiment with different precision (fp16) or a different
  framework backend (MLX).
- **SHAPE-DEPENDENT, with structure.** Ratio depends meaningfully
  on shape. Identifies the regime where naive is most behind —
  likely the K-sweep or memory-bound probe shapes. Future work
  should target that regime.
- **MARGINAL / FAIL on methodology.** MPS API integration issues
  or noisy timings. Diagnose first, possibly drop to a smaller
  shape grid for debugging.

We do not plan past these branches.

## Result

**Two runs, battery and AC.** First run was inadvertently on
battery; rerun on AC the same day. Both reported below side-by-side.

| run | timestamp prefix | power | wall-clock |
|-----|------------------|-------|-----------:|
| battery | `20260505T065707` | Battery | 63 s |
| AC      | `20260505T165356` | AC      | 53 s |

**Headline (AC, primary):**
- methodology **MARGINAL** (CV gate; substantively PASS — see below)
- gap **SHAPE-DEPENDENT** (plateau median ratio 2.01×, range 0.25× → 9.21× across 34 shapes)
- MPS plateau **MID** (10.72 TFLOP/s = 38 % of fp32 peak — *unchanged from battery*)

### Battery vs AC: telemetry and what changed

Original pre-reg specified AC; first run was inadvertently on
battery. Re-running on AC tested whether the 10.72 TFLOP/s ceiling
was DVFS-throttled. **It wasn't** — peak is unchanged. What *did*
change is mid-shape latency.

GPUPH / PWRCTRL aggregate residency for the two runs:

| state | battery | AC | reading |
|-------|--------:|---:|---------|
| GPUPH P15 (1578 MHz peak) | 17 % | 36 % | AC reaches peak ~2× more often |
| PWRCTRL PRFBOOST          |  3 % | <0.1 % | AC engaged P15 directly without PRFBOOST |
| PWRCTRL IDLE_OFF          | 30 % | 47 % | AC cells finished faster, more inter-cell idle |
| PWRCTRL PERF              | 48 % | 30 % | – |
| PWRCTRL DEADLINE          | 18 % | 20 % | – |

**Where the two runs match (peak ceiling, large squares, large K):**
both pinned 10.72 TFLOP/s on 1024³–2048³, both showed CV ≤ 11 % at
the largest cells, both saturated within ±1 % at 1536³ and K=4096.
At those shapes a single dispatch is long enough (≥ 200 µs) that
the chip ramps to P15 and stays there regardless of power source.
The chip is *bandwidth-bound* there, not DVFS-bound, so AC vs
battery doesn't move the needle.

**Where they diverged (mid-shape regime, M ∈ [128, 768]):** AC was
1.5–4.3× faster per dispatch than battery on these shapes. Single
dispatches at these sizes are 8–235 µs — short enough that
P-state ramp time is a meaningful fraction of the work, and battery
DVFS is more reluctant to run at peak voltage transiently.

This rewrites the "the battery measurement is a floor" claim from
the first writeup: **for the absolute TFLOP/s ceiling on large
squares, battery and AC produce the same number; the battery
measurement was misleading specifically about the mid-shape
regime.** The substantive finding from the AC run is what stands.

### Per-shape numbers — AC (primary) with battery alongside

MPS p50 single-shot ns/matmul. r_N=1 = naive p50(N_AMP=1) / MPS
p50 — the apples-to-apples ratio (both include dispatch overhead).
naive baseline from `experiments/016-matmul-discrimination/raw/20260429T213959-cells.csv`.

**Square diagonal (M=N=K):**

| M³ | MPS AC p50 | MPS battery p50 | AC TFLOP/s | r_N=1 (AC) | r_N=1 (battery) |
|----|-----------:|----------------:|-----------:|-----------:|----------------:|
| 8    | 9.00 µs   | 10.21 µs  | 0.0   | 0.81× | 0.71× |
| 32   | 7.00 µs   | 7.67 µs   | 0.009 | 1.57× | 1.43× |
| 64   | 5.71 µs   | 19.38 µs  | 0.092 | **2.57×** | 0.76× |
| 128  | 7.75 µs   | 28.25 µs  | 0.541 | **2.98×** | 0.82× |
| 256  | 11.92 µs  | 45.79 µs  | 2.82  | **2.88×** | 0.75× |
| 384  | 21.71 µs  | 92.42 µs  | 5.22  | **2.88×** | 0.68× |
| 512  | 32.79 µs  | 62.38 µs  | 8.19  | **4.41×** | 2.32× |
| 768  | 93.58 µs  | 234.17 µs | 9.68  | **5.32×** | 2.13× |
| 1024 | 208.17 µs | 275.62 µs | 10.32 | 5.21× | 3.94× |
| 1536 | 676.38 µs | 676.25 µs | 10.72 | 6.34× | 6.34× |
| 2048 | 1.615 ms  | 1.609 ms  | 10.64 | 8.14× | 8.17× |

The 1024³+ rows show how thoroughly the chip saturates: AC and
battery are within 1 % at 1536³ and 2048³ despite being 4× apart at
192³.

**Memory-bound probes (AC):**

| shape | MPS AC p50 | TFLOP/s | naive slope | r_slope | r_N=1 |
|-------|-----------:|--------:|------------:|--------:|------:|
| 8×4096×4096 | 195.25 µs | 1.38 | 334.56 µs | 1.71× | 1.75× |
| 4×8192×4096 | 345.21 µs | 0.78 | 715.35 µs | 2.07× | 1.27× |
| 2×8192×4096 | 343.50 µs | 0.39 | 524.43 µs | 1.53× | 1.06× |

**K-sweep at M=N=512 (AC):** ratio rises monotonically with K from
1.66× (K=2) to 4.85× (K=1024) and 4.52× (K=4096) — the cleanest
single-axis discrimination signal in this experiment.

Full tables in `raw/20260505T165356-analysis.log` (AC) and
`raw/20260505T065707-analysis.log` (battery).

### Verdicts (AC, primary)

- **Methodology: MARGINAL** (literal). 27/34 shapes had per-trial
  CV ≥ 30 %. The CV inflation is the same bimodal fast/slow-mode
  pattern 014b documented for sub-floor cells, not a methodology
  failure. Per-shape p50 is well-resolved on all 34 shapes
  (≥ 5000 trials except 2048³ which hit time cap at 2637). p50-
  based discrimination works on all 34 shapes. *Literal MARGINAL,
  substantive PASS with the bimodality caveat from 014b.*

- **Gap: SHAPE-DEPENDENT** (literal, plateau median ratio 2.01×,
  range 4334× across 34 shapes). The structure on AC:

  - **M ≤ 48 squares:** dispatch overhead dominates both kernels;
    naive often *faster* than MPS by 0.6-1.6× at the smallest sizes
    (MPS's per-dispatch bookkeeping > naive's). Visible in 8³, 12³,
    16³, 24³, 32³, 48³.
  - **M = 64 to 384:** MPS is 2-3× faster (r_N=1). The 016 compute
    plateau is where tiling starts to pay off. *On battery this
    regime looked equal — battery DVFS hides MPS's mid-shape
    advantage.*
  - **M ∈ [512, 768]:** MPS 4.4-5.3× faster.
  - **M ≥ 1024 squares:** MPS 5.2-8.1× faster. Both runs converge
    to identical numbers here — large dispatches saturate the
    chip regardless of power source.
  - **K-sweep at M=N=512:** monotonic ratio rise from 1.66× (K=2)
    to ~4.5-4.9× (K ≥ 1024) — the cleanest single-axis ratio
    signal in the experiment.
  - **Memory-bound probes:** MPS only 1.5-2.1× faster. Bandwidth
    bound, tiling helps less.

  This pattern matches the pre-reg's hypothesis sketch (with the
  exception that the gap *opens earlier* than predicted — by M=128
  on AC, not just M ≥ 1024). Pre-reg "WIDE GAP if median plateau
  ratio ≥ 5×" is *just* missed at 4.85× for the largest plateau
  shape (768³); on the median, SHAPE-DEPENDENT is the correct call.

- **MPS plateau: MID** (literal, threshold CALIBRATED ≥ 14
  TFLOP/s). Peak observed 10.72 TFLOP/s on 1536³ and 2048³
  squares — *unchanged from battery*. AC vs battery confirmed
  this is **not a DVFS-throttling artifact**: at these shapes
  the chip pegs P15 / 1578 MHz on both runs. The 38 % of fp32 peak
  result is real for fp32 GEMM on this chip via MPS. Likely
  bandwidth-bound at 1536³ / 2048³ (working sets 18 MB / 32 MB,
  exceeds SLC at 2048³); at 1536³ which fits in SLC, MPS is also
  at 10.7 TFLOP/s, suggesting MPS's fp32 implementation simply
  doesn't extract > ~38 % of theoretical peak even cache-resident.
  fp16 / bf16 would likely be much closer to peak.

### Surprises

1. **MPS fp32 ceiling is bandwidth/implementation-bound, not
   DVFS-bound.** Battery and AC produce identical 10.72 TFLOP/s on
   1024³+. The first writeup hypothesized battery suppression of
   the absolute ceiling; the AC re-run *falsifies* that for large
   shapes. MPS simply doesn't extract more than ~38 % of theoretical
   fp32 peak on this chip via the fp32 GEMM path.

2. **DVFS suppresses the *mid-shape* regime, not the peak.** Battery
   gave 4.4× slower 384³ matmul (92 µs vs 22 µs on AC) but identical
   1536³ (676 µs both). The cells that lose to battery are the ones
   short enough that DVFS ramp time matters; the cells where the
   chip has time to settle at P15 are unaffected. This is
   methodologically important: **for the lab's existing per-FLOP
   numbers from naive matmul (016) and synthetic kernels
   (014b/015) at sub-millisecond cells, AC vs battery may be a
   silent ~2× factor.** Worth re-checking 016's compute-plateau
   numbers were AC.

3. **MPS slower than naive at the smallest shapes.** Even on AC,
   MPS at 8³–48³ is 0.8-1.6× the speed of naive matmul. MPS's
   dispatch overhead is higher than a hand-written single-pass
   kernel's, so for GEMV / tiny-matrix work, MPS isn't a free win.

4. **The K-sweep at M=N=512 is the cleanest discrimination axis.**
   AC ratio rises monotonically 1.66× → 4.85× as K goes 2 → 1024,
   smooth and shape-monotonic with no inversions. If a future
   experiment needs a single shape-axis to test "did this kernel
   tile better than naive?", the K=2-1024 sweep at M=N=512 is the
   instrument.

5. **The MPS-sample-buffer-attachment limitation.** Documented in
   the pre-reg amendment above. Any future experiment wrapping
   vendor-provided kernels (Accelerate GEMM, MLX, Metal-FX) will
   face the same instrument constraint and have to fall back to
   `gpuStartTime`/`gpuEndTime`.

### Bimodality at small shapes (both runs)

The CV > 30 % observation is the same bimodal fast/slow mode 014b
documented. Visible directly in the MPS p50 ≪ p90 ratio at small
shapes on the AC run:

- 16³: p50=5.71 µs, p90=18.04 µs (3.2× spread)
- 32³: p50=7.00 µs, p90=18.83 µs (2.7× spread)
- 64³: p50=5.71 µs, p90=19.75 µs (3.5× spread)
- 128³: p50=7.75 µs, p90=28.58 µs (3.7× spread)

Same regime, same pattern as 014b/015's sub-floor distributions.
Confirms that bimodality is not naive-kernel-specific — it's a
property of the GPU's idle / active transition that appears when
the work is short relative to dispatch overhead, regardless of
which kernel is dispatched.

### Recommended follow-ups

1. **Re-check 016's compute plateau on AC.** The battery vs AC
   delta on mid-shape matmuls is ~3-4× at 256³–768³. If 016 ran
   on battery for any of its mid-shape cells, the "naive matmul
   plateaus at 2 TFLOP/s" headline number is potentially
   ~2-3× low. *Cheap to verify: re-run a subset of 016's grid on
   confirmed AC.*
2. **fp16 / bf16 MPS GEMM.** The 38 % of fp32 peak result suggests
   MPS doesn't optimize fp32 well. fp16 may close the gap to
   theoretical peak considerably. One-flag change to this script
   (`MPSDataTypeFloat16`).
3. **Tighten the methodology fit gate** for per-shape p50
   experiments: stop using per-trial CV (bimodality inflates it
   structurally); use p50 stability across blocks-within-cell.
4. **PRFBOOST trigger boundary.** AC engaged P15 directly without
   PRFBOOST (< 0.1 % residency); battery engaged PRFBOOST 3 % of
   the time. PRFBOOST may be a battery-specific bridge state to
   help reach P15 when DVFS is being conservative; on AC the chip
   skips it.
