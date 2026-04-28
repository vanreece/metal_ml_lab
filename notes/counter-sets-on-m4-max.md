# Counter sets on M4 Max / `applegpu_g16s` — what the public Metal API exposes

**Date:** 2026-04-28
**Hardware:** Apple M4 Max 36GB, 14-core (10P + 4E), MacBook Pro 14" (Mac16,6)
**Architecture string:** `applegpu_g16s`
**OS:** macOS 26.4.1 (build 25E253)
**Probe:** `notes/probe-counter-sets.py`
**Raw output:** captured below in this file

This is the M4 Max counterpart to `counter-sets-on-m1-pro.md`. Same probe,
new chip generation, point-release-newer macOS (M1 Pro was on 26.3.1).
Question: did Apple open up additional counter sets in the G16 family or
in the macOS 26.4 driver?

## Headline: no, the gap is the same

One counter set (`timestamp`), one counter inside it (`GPUTimestamp`).
Identical to M1 Pro. Apple has **not** populated the
`StageUtilization` or `Statistic` sets on M4 Max in the public Metal
API; the constants exist in the headers and PyObjC reads them, but the
sets are absent from `device.counterSets()`.

The project's "without vendor-internal counters" thesis is therefore
not just an M1-Pro constraint — it's the working environment on the
G16-family hardware too. Multi-kernel state inference remains the only
"free" path forward.

## Sampling-point support — identical to M1 Pro

```
atStageBoundary:        True
atDrawBoundary:         False
atBlitBoundary:         False
atDispatchBoundary:     False
atTileDispatchBoundary: False
```

Per-dispatch sampling (`atDispatchBoundary`) is still unsupported. Multi-
dispatch encoders cannot be timed per-dispatch; one compute-pass-encoder
remains the minimum timing window.

## End-to-end sanity: timestamp resolve

A real compute pass with stage-boundary timestamp sampling produces
two `GPUTimestamp` values; the delta for a trivial 32-thread kernel
was 8500 ns (single sample, n=1, no statistics). Same order of
magnitude as M1 Pro's ~8 µs floor — but **whether the tick resolution
is the same ~24 MHz / ~42 ns** that M1 Pro uses needs experiment-001
quantization analysis to confirm. A single delta of 8500 isn't
inconsistent with either 42 ns ticks (~204 ticks) or some other tick
size that happens to give a near-multiple value.

## What this changes for the project

Nothing structural. The G16 family on macOS 26.4.1 looks like the G13
family on macOS 26.3.1 from the public-Metal API surface:

- Same single counter set
- Same single sampling point
- Same opacity past timing

What may still differ on M4 Max (open):

- Timestamp tick resolution (M1 Pro: ~24 MHz / 41.67 ns). M4 Max could
  be different — newer chips have changed timestamp infrastructure in
  other Apple platforms.
- Dispatch-overhead floor exact value.
- Cadence-vs-cv structure (M1 Pro had a 1-10 ms "nightmare zone").
- Work-dominance thresholds (M4 Max has materially higher peak FLOPS
  and bandwidth, so the threshold thread counts likely shift).
- The +21 µs step at fma_loop iters 192→256 (M1 Pro). Plausibly a
  G13-specific compiler/hardware threshold.
- The ~42 µs inter-encoder gap (M1 Pro). Could be smaller or larger on
  G16 depending on front-end design.

These are the questions exp 001-005 answered for M1 Pro and that we have
to re-answer for M4 Max. The cheapest first re-run is exp 001
(quantization + dispatch-overhead floor + back-to-back vs spaced-1s),
which calibrates the units for everything else.

## Raw probe output

For the historical record, the full output of
`uv run notes/probe-counter-sets.py` on this date:

```
# Device: Apple M4 Max
# RegistryID: 4294968451
# Architecture: applegpu_g16s

## device.counterSets()
  1 counter set(s) exposed:

  ### counter set: 'timestamp'
    1 counter(s):
      - 'GPUTimestamp'

## Common counter-set name constants in PyObjC
  - MTLCommonCounterSetTimestamp = 'timestamp' (matches a counter set above)
  - MTLCommonCounterSetStageUtilization = 'stageutilization' (NOT in available sets)
  - MTLCommonCounterSetStatistic = 'statistic' (NOT in available sets)

## Common counter name constants in PyObjC
  - MTLCommonCounterTimestamp = 'GPUTimestamp' (PRESENT)
  - MTLCommonCounterTotalCycles = 'TotalCycles' (not in any available counter set)
  - MTLCommonCounterVertexCycles = 'VertexCycles' (not in any available counter set)
  - MTLCommonCounterFragmentCycles = 'FragmentCycles' (not in any available counter set)
  - MTLCommonCounterRenderTargetWriteCycles = 'RenderTargetWriteCycles' (not in any available counter set)
  - MTLCommonCounterVertexInvocations = 'VertexInvocations' (not in any available counter set)
  - MTLCommonCounterClipperInvocations = 'ClipperInvocations' (not in any available counter set)
  - MTLCommonCounterClipperPrimitivesOut = 'ClipperPrimitivesOut' (not in any available counter set)
  - MTLCommonCounterFragmentInvocations = 'FragmentInvocations' (not in any available counter set)
  - MTLCommonCounterFragmentsPassed = 'FragmentsPassed' (not in any available counter set)
  - MTLCommonCounterComputeKernelInvocations = 'KernelInvocations' (not in any available counter set)
  - MTLCommonCounterPostTessellationVertexInvocations = 'PostTessellationVertexInvocations' (not in any available counter set)
  - MTLCommonCounterTessellationInputPatches = 'TessellationInputPatches' (not in any available counter set)

## Sample-buffer alloc trial per counter set
  - 'timestamp': OK  (sample buffer allocated)

## Sampling-point support
  - atStageBoundary: True
  - atDrawBoundary: False
  - atBlitBoundary: False
  - atDispatchBoundary: False
  - atTileDispatchBoundary: False

## End-to-end trial: dispatch + resolve for each set with sample-buffer support
  ### 'timestamp': resolved 16 bytes (1 counters × 2 samples × 8 = 16 expected)
    sample 0: GPUTimestamp = 3657632721333
    sample 1: GPUTimestamp = 3657632729833
    deltas (sample1 - sample0): GPUTimestamp = 8500
```
