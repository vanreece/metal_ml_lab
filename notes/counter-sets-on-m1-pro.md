# Counter sets on M1 Pro / `applegpu_g13s` — what the public Metal API actually exposes

**Date:** 2026-04-27
**Hardware:** Apple M1 Pro 16GB (architecture string: `applegpu_g13s`)
**OS:** macOS 26.3.1
**Probe:** `notes/probe-counter-sets.py`
**Raw output:** captured below in this file

This is exploratory enumeration, not a pre-registered experiment. The
question was: "before we keep building methodology on top of timing
alone, is there free counter data we've been missing?" Answer: no.

## What `device.counterSets()` actually returns

```
1 counter set(s) exposed:

  ### counter set: 'timestamp'
    1 counter(s):
      - 'GPUTimestamp'
```

That's it. **One** counter set, **one** counter inside it.

## What Metal *defines* (and isn't there)

PyObjC happily exposes all these constants (they live in the Metal
headers), but only `MTLCommonCounterSetTimestamp` is actually present
on the M1 Pro:

| Constant                                   | Value          | On M1 Pro? |
|--------------------------------------------|----------------|------------|
| `MTLCommonCounterSetTimestamp`             | `timestamp`    | **YES**    |
| `MTLCommonCounterSetStageUtilization`      | `stageutilization` | NOT IN AVAILABLE SETS |
| `MTLCommonCounterSetStatistic`             | `statistic`    | NOT IN AVAILABLE SETS |

And the defined per-counter names — none of which are exposed because
the sets they would belong to aren't exposed:

| Constant                                              | Value                              | On M1 Pro? |
|-------------------------------------------------------|------------------------------------|------------|
| `MTLCommonCounterTimestamp`                           | `GPUTimestamp`                     | **YES**    |
| `MTLCommonCounterTotalCycles`                         | `TotalCycles`                      | not present |
| `MTLCommonCounterVertexCycles`                        | `VertexCycles`                     | not present |
| `MTLCommonCounterFragmentCycles`                      | `FragmentCycles`                   | not present |
| `MTLCommonCounterRenderTargetWriteCycles`             | `RenderTargetWriteCycles`          | not present |
| `MTLCommonCounterVertexInvocations`                   | `VertexInvocations`                | not present |
| `MTLCommonCounterClipperInvocations`                  | `ClipperInvocations`               | not present |
| `MTLCommonCounterClipperPrimitivesOut`                | `ClipperPrimitivesOut`             | not present |
| `MTLCommonCounterFragmentInvocations`                 | `FragmentInvocations`              | not present |
| `MTLCommonCounterFragmentsPassed`                     | `FragmentsPassed`                  | not present |
| `MTLCommonCounterComputeKernelInvocations`            | `KernelInvocations`                | not present |
| `MTLCommonCounterPostTessellationVertexInvocations`   | `PostTessellationVertexInvocations`| not present |
| `MTLCommonCounterTessellationInputPatches`            | `TessellationInputPatches`         | not present |

This is consistent with what the public Metal documentation hints at —
"vendors may choose which common counter sets to support" — but seeing
it spelled out on actual hardware confirms there is no path to these
counters via the public API on M1 Pro.

## Sampling points (re-verified)

Same result as 001 and 003 — only `atStageBoundary` is supported:

```
atStageBoundary:        True
atDrawBoundary:         False
atBlitBoundary:         False
atDispatchBoundary:     False
atTileDispatchBoundary: False
```

## End-to-end sanity: timestamp resolve

A real compute pass with stage-boundary timestamp sampling produces
two `GPUTimestamp` values; the delta is sensible (11917 ns for a
trivial 32-thread kernel ≈ ~9 µs warm). Confirms the resolved bytes
decode the way we've been assuming in 001-003.

## What this means for the project

1. **The "free counters we've been missing" hypothesis is dead.** No
   StageUtilization, no Statistic, no occupancy reading, no per-stage
   cycle counts. Anything beyond timing requires either:
   - powermetrics (sudo, sliding window, no per-kernel detail)
   - Instruments / `xctrace` -> `.gputrace` (sudo, opaque format,
     reverse-engineering explicitly out of scope per CLAUDE.md)
   - Asahi-style direct AGX command submission (Linux only)
   - private Apple SPIs / entitlement-gated APIs (no)

2. **The project's whole thesis lives in this gap, and it's now
   confirmed to be a real gap, not a "we just haven't looked hard
   enough" gap.** Apple's GPU driver does NOT populate the per-stage
   counter sets that the Metal API conceptually supports — even though
   the constant names are present in the headers PyObjC reads.

3. **Multi-kernel state inference is the only "free" path forward.**
   If timing is all we get, we have to make timing smarter:
   - Multiple thermometer kernels of different character
     (memory-bound, compute-bound, latency-bound) run as a probe burst
     should give us a *vector* of timings that distinguishes power
     states more reliably than one kernel's timing
   - Differential timing across deliberately-perturbed kernel parameters
     (varying threadgroup size, memory access pattern, register
     pressure) should let us infer bottleneck class without per-stage
     counters
   - Calibration probe design needs to minimize observer effect, since
     we can't read state without running work

4. **The architecture string `applegpu_g13s` is worth noting.** This
   is M1 Pro specifically. M4 Max will have a different architecture
   string and may expose different counter sets — worth re-running
   this probe on M4 Max when the hardware is available, and on every
   future macOS release on M1 Pro since Apple could in principle add
   counter set support in a driver update.

## Raw probe output

For the historical record, the full output of
`uv run notes/probe-counter-sets.py` on this date:

```
# Device: Apple M1 Pro
# RegistryID: 4294969592
# Architecture: applegpu_g13s

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
    sample 0: GPUTimestamp = 133933959868208
    sample 1: GPUTimestamp = 133933959880125
    deltas (sample1 - sample0): GPUTimestamp = 11917
```
