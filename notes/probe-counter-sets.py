# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Quick exploratory probe (NOT an experiment):
- Enumerate device.counterSets() on M1 Pro
- For each set, list its counters by name
- For each common counter-set name (timestamp / stage_utilization /
  statistic), try to allocate a sample buffer and report success/failure
- Try a small compute dispatch with each working sample buffer and
  see if resolveCounterRange returns plausible data

Output is a markdown-ish text dump suitable for pasting into
counter-sets-on-m1-pro.md.

Run: `uv run probe-counter-sets.py`
"""
import Metal


COMPUTE_KERNEL = """
#include <metal_stdlib>
using namespace metal;
kernel void noop(device uint *out [[buffer(0)]],
                 uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
"""

COMMON_COUNTER_SET_NAMES = [
    ("MTLCommonCounterSetTimestamp",        "timestamp"),
    ("MTLCommonCounterSetStageUtilization", "stageutilization"),
    ("MTLCommonCounterSetStatistic",        "statistic"),
]

# Common counter names defined in Metal headers — try to find them
# in the available counter sets.
COMMON_COUNTERS = [
    "MTLCommonCounterTimestamp",
    "MTLCommonCounterTotalCycles",
    "MTLCommonCounterVertexCycles",
    "MTLCommonCounterFragmentCycles",
    "MTLCommonCounterRenderTargetWriteCycles",
    "MTLCommonCounterVertexInvocations",
    "MTLCommonCounterClipperInvocations",
    "MTLCommonCounterClipperPrimitivesOut",
    "MTLCommonCounterFragmentInvocations",
    "MTLCommonCounterFragmentsPassed",
    "MTLCommonCounterComputeKernelInvocations",
    "MTLCommonCounterPostTessellationVertexInvocations",
    "MTLCommonCounterTessellationInputPatches",
]


def main():
    device = Metal.MTLCreateSystemDefaultDevice()
    print(f"# Device: {device.name()}")
    print(f"# RegistryID: {device.registryID()}")
    if hasattr(device, "architecture") and device.architecture():
        print(f"# Architecture: {device.architecture().name()}")
    print()

    print("## device.counterSets()")
    sets = device.counterSets()
    if not sets:
        print("  (empty / nil)")
        return
    print(f"  {len(sets)} counter set(s) exposed:")
    for cs in sets:
        cs_name = str(cs.name())
        counters = cs.counters() or []
        print(f"\n  ### counter set: {cs_name!r}")
        print(f"    {len(counters)} counter(s):")
        for c in counters:
            print(f"      - {str(c.name())!r}")
    print()

    print("## Common counter-set name constants in PyObjC")
    print("  (does Metal module expose the constant, and what value?)")
    for sym, expected in COMMON_COUNTER_SET_NAMES:
        if hasattr(Metal, sym):
            val = getattr(Metal, sym)
            match = " (matches a counter set above)" if str(val) in [str(cs.name()) for cs in sets] else " (NOT in available sets)"
            print(f"  - {sym} = {val!r}{match}")
        else:
            print(f"  - {sym} = <not exposed by PyObjC>")
    print()

    print("## Common counter name constants in PyObjC")
    found_in_sets = {str(c.name()) for cs in sets for c in (cs.counters() or [])}
    for sym in COMMON_COUNTERS:
        if hasattr(Metal, sym):
            val = getattr(Metal, sym)
            in_set = " (PRESENT)" if str(val) in found_in_sets else " (not in any available counter set)"
            print(f"  - {sym} = {val!r}{in_set}")
        else:
            print(f"  - {sym} = <not exposed by PyObjC>")
    print()

    print("## Sample-buffer alloc trial per counter set")
    print("  (can we allocate a 2-slot buffer for each set?)")
    for cs in sets:
        cs_name = str(cs.name())
        desc = Metal.MTLCounterSampleBufferDescriptor.alloc().init()
        desc.setCounterSet_(cs)
        desc.setSampleCount_(2)
        desc.setStorageMode_(Metal.MTLStorageModeShared)
        desc.setLabel_(f"probe-{cs_name}")
        buf, err = device.newCounterSampleBufferWithDescriptor_error_(desc, None)
        if buf is None:
            print(f"  - {cs_name!r}: FAILED  err={err}")
        else:
            print(f"  - {cs_name!r}: OK  (sample buffer allocated)")
    print()

    print("## Sampling-point support")
    sampling_points = [
        ("atStageBoundary",        "MTLCounterSamplingPointAtStageBoundary"),
        ("atDrawBoundary",         "MTLCounterSamplingPointAtDrawBoundary"),
        ("atBlitBoundary",         "MTLCounterSamplingPointAtBlitBoundary"),
        ("atDispatchBoundary",     "MTLCounterSamplingPointAtDispatchBoundary"),
        ("atTileDispatchBoundary", "MTLCounterSamplingPointAtTileDispatchBoundary"),
    ]
    for label, attr in sampling_points:
        if hasattr(Metal, attr):
            v = device.supportsCounterSampling_(getattr(Metal, attr))
            print(f"  - {label}: {v}")
        else:
            print(f"  - {label}: <constant {attr} not in PyObjC>")
    print()

    print("## End-to-end trial: dispatch + resolve for each set with sample-buffer support")
    print("  (does a real sampled compute pass produce non-zero data?)")

    queue = device.newCommandQueue()
    options = Metal.MTLCompileOptions.alloc().init()
    library, err = device.newLibraryWithSource_options_error_(
        COMPUTE_KERNEL, options, None
    )
    if library is None:
        print(f"  Could not compile noop kernel: {err}")
        return
    fn = library.newFunctionWithName_("noop")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    out_buffer = device.newBufferWithLength_options_(
        4 * 32, Metal.MTLResourceStorageModeShared
    )

    for cs in sets:
        cs_name = str(cs.name())
        n_counters = len(cs.counters() or [])
        # 2 samples (start + end), each holds n_counters values
        desc = Metal.MTLCounterSampleBufferDescriptor.alloc().init()
        desc.setCounterSet_(cs)
        desc.setSampleCount_(2)
        desc.setStorageMode_(Metal.MTLStorageModeShared)
        desc.setLabel_(f"trial-{cs_name}")
        sample_buffer, err = device.newCounterSampleBufferWithDescriptor_error_(desc, None)
        if sample_buffer is None:
            print(f"\n  ### {cs_name!r}: skipping (alloc failed: {err})")
            continue

        pass_desc = Metal.MTLComputePassDescriptor.computePassDescriptor()
        att = pass_desc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
        att.setSampleBuffer_(sample_buffer)
        att.setStartOfEncoderSampleIndex_(0)
        att.setEndOfEncoderSampleIndex_(1)

        cb = queue.commandBuffer()
        encoder = cb.computeCommandEncoderWithDescriptor_(pass_desc)
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(32, 1, 1),
            Metal.MTLSizeMake(32, 1, 1),
        )
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        data = sample_buffer.resolveCounterRange_((0, 2))
        if data is None:
            print(f"\n  ### {cs_name!r}: resolveCounterRange returned None")
            continue
        raw = bytes(data)
        # Each sample = n_counters * 8 bytes (uint64)
        print(f"\n  ### {cs_name!r}: resolved {data.length()} bytes "
              f"({n_counters} counters × 2 samples × 8 = "
              f"{n_counters * 2 * 8} expected)")
        if n_counters == 0 or data.length() < n_counters * 2 * 8:
            print(f"    raw hex: {raw.hex()}")
            continue
        # Decode
        counter_names = [str(c.name()) for c in cs.counters()]
        for sample_idx in range(2):
            print(f"    sample {sample_idx}:")
            for c_idx, c_name in enumerate(counter_names):
                offset = sample_idx * n_counters * 8 + c_idx * 8
                val = int.from_bytes(raw[offset:offset+8], "little", signed=False)
                print(f"      {c_name:>40s} = {val}")
        # Compute deltas (sample 1 - sample 0) for each counter
        print("    deltas (sample1 - sample0):")
        for c_idx, c_name in enumerate(counter_names):
            v0 = int.from_bytes(raw[c_idx*8 : c_idx*8+8], "little", signed=False)
            v1 = int.from_bytes(raw[n_counters*8 + c_idx*8 : n_counters*8 + c_idx*8+8],
                                "little", signed=False)
            print(f"      {c_name:>40s} = {v1 - v0}")


if __name__ == "__main__":
    main()
