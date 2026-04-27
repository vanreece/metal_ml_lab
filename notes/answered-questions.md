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
