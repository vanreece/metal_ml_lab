# Answered questions

Questions that started in `UNKNOWNS.md` and got definitively (or
substantively) answered by an experiment. Kept here so that the answer
travels with the project even after the question leaves the active list.

Format: question, answer, hardware/software it applies to, and the
experiment that closed it.

## Does `MTLDevice.supportsCounterSampling(at:)` return True for any sampling point besides `atStageBoundary` on M-series?

**Answer:** No, on M1 Pro / macOS 26.3.1.

`supportsCounterSampling_` returns:
- `atStageBoundary`: True
- `atDrawBoundary`: False
- `atBlitBoundary`: False
- `atDispatchBoundary`: False
- `atTileDispatchBoundary`: False

This matches the public-docs signal but is now verified on actual hardware.
Practical consequence: any per-dispatch timing must be done at the
compute-pass-encoder level (one pass = one timing window) rather than
sampling individual `dispatchThreads` calls. Multi-dispatch encoders
cannot be timed per-dispatch.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, PyObjC via
`pyobjc-framework-Metal`.
**Closed by:** experiment 001.
**Still open for:** M4 Max, future macOS versions, all other Apple Silicon
variants.

## Can we read GPU timestamps from Python via PyObjC at all?

**Answer:** Yes. `MTLCounterSampleBuffer` with the `timestamp` counter
set, attached to a `MTLComputePassDescriptor`'s
`sampleBufferAttachments[0]` with start/end indices, then resolved with
`resolveCounterRange_((start, count))`, returns nanosecond-scaled GPU
timestamps as raw uint64 little-endian bytes in NSData. No Swift bridge
needed.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, PyObjC.
**Closed by:** experiment 001.

## What is the noise floor (σ/μ) of MTLCounterSampleBuffer timing on M1 Pro?

**Answer:** It is *not* a single number — it depends sharply on how
much idle time sits between consecutive dispatches. For the trivial
`write_tid` kernel (32 threads, 1 SIMD width) on M1 Pro with macOS
26.3.1 at user-interactive QoS on AC power:

| inter-dispatch sleep | median gpu_delta_raw (ns) | cv (σ/p50) |
|----------------------|---------------------------|------------|
| 0                    | 8083                      | 0.66       |
| 1 ms                 | 8291                      | **7.03**   |
| 10 ms                | 11292                     | 2.71       |
| 100 ms               | 13791                     | 2.23       |
| 1 s                  | 15229                     | **0.21**   |

The non-monotonic cv is the headline. The 1 ms condition is by far the
worst — bimodal with most dispatches near the back-to-back floor and
~5/200 dispatches taking 30–80x longer because the GPU is in a
transition zone of its power-state machine. The 1 s condition is by
far the *best* (tightest distribution), even though the absolute median
is highest. "More idle = more noise" is wrong; "more idle = different
power state, possibly tighter" is right.

**Practical consequence:** any microbench design must explicitly choose
its cadence and inherit that cadence's noise structure. There is no
"pin to one frequency" workaround on Apple Silicon. The 1–10 ms range
is poison and should be avoided unless characterizing it is the point.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power,
laptop awake, no other heavy processes.
**Closed by:** experiment 002.
**Still open for:** how this scales with kernel duration / size / load,
and whether the same picture holds on M4 Max.

## What is the GPU timestamp counter's hardware tick resolution on M1 Pro?

**Answer:** ~24 MHz (one tick ≈ 41.67 ns). Apparent in 001's back-to-back
distribution as quantization at 8000, 8041, 8083, 8125, 8167, ... raw
units. Metal's `resolveCounterRange_` returns these raw counter values
already scaled to nanoseconds — i.e. each integer "nanosecond" we read
is actually one of every ~42 contiguous nanoseconds. The
`device.sampleTimestamps:gpuTimestamp:` API returns matching values
(elapsed-ratio across 100s = 1.000000 in 001), confirming the two paths
expose the same clock at the same granularity.

**Practical consequence:** smallest distinguishable GPU duration delta is
~42 ns. Anything reported as a sub-42-ns delta would be impossible.

**Hardware/software:** M1 Pro / macOS 26.3.1.
**Closed by:** experiment 001.
**Still open for:** M4 Max (likely different — newer chips have changed
timestamp infrastructure).
