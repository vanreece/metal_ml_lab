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

## What counter sets does Metal actually expose on M1 Pro?

**Answer:** Exactly one — `timestamp`, with one counter inside it
(`GPUTimestamp`). No `StageUtilization`, no `Statistic`, no per-stage
cycle counts, no `ComputeKernelInvocations`, no occupancy. The Metal
headers define all of these as constants and PyObjC happily exposes
them, but the Apple GPU driver does not populate them on M1 Pro
(`applegpu_g13s`).

This means **timing is the only "free" GPU-side signal Metal gives
us.** Anything else (occupancy, stage activity, cycle counts) requires
either powermetrics (sudo, sliding window, no per-kernel detail),
Instruments / `xctrace` -> `.gputrace` (sudo, opaque format,
reverse-engineering out of scope), or private Apple SPIs we cannot
reach.

The project's "without vendor-internal counters" thesis is therefore
not a self-imposed constraint but a real architectural gap that Apple
has chosen to keep closed in the public API.

**Hardware/software:** Apple M1 Pro 16GB (`applegpu_g13s`),
macOS 26.3.1, PyObjC.
**Closed by:** `notes/counter-sets-on-m1-pro.md` (probe), 2026-04-27.
**Still open for:** M4 Max (different `applegpu_g*` arch string,
likely different driver behavior); future macOS versions on M1 Pro
(Apple could add support in a driver update — worth re-running the
probe yearly).

## Does a warmup prefix recover the "cool cadence" noise from 002?

**Answer:** Yes, with caveats, for cadences ≥ 10 ms. **No, and don't
try, at sleep_0.**

For inter-dispatch sleep ≥ 10 ms, K=0 puts p50 at 14–17 µs (the
"cool" regime from 002). One untimed warmup dispatch (K=1) of the
same kernel kind drops p50 back to ~9.7 µs and leaves cv ≤ 0.16.
Bigger K offers no obvious benefit; K=20 occasionally introduces tail
risk (one combo at cv=5).

At sleep_0, the chip is already in (or below) the warm steady state.
Adding warmup pushes it to a slightly slower settled state (p50 climbs
from 5.4–8 µs to ~9 µs).

Warmup *kind* matters in unexpected ways: a single arithmetic-heavy
warmup dispatch (`fma_loop K=1`) is **strictly worse** than no warmup
at all (cv 1.4–3.8 vs 0.1–0.3 at the same cadences). K≥5 of the same
kind recovers; one is destabilizing. The destabilization is specific
to switching kernel character before the measured (memory-write)
dispatch.

**Practical recipe for `write_tid` 32-thread microbenches:** at
cadence ≥ 10 ms, `K=1` of the same kernel kind right before each
measurement is the right warmup. At sleep_0, no warmup. Avoid one
dispatch of an arithmetic-heavy kernel as a "light" warmup.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power,
display held awake via `caffeinate -d -i -m`.
**Closed by:** experiment 003.
**Still open:** whether the recipe transfers to non-trivial kernels
(planned for 004); how to escape the 1-10 ms transition zone (003
contaminated the K=0 baseline at this cadence via the calibration
probe and could not test cleanly).

## At what `write_tid` thread count does measured time start scaling with work rather than reflecting dispatch overhead?

**Answer:** T\* ≈ 131 072 - 262 144 threads on M1 Pro (threadgroup size 32).
Below ~32K threads the median sits in 9-11 µs regardless of thread count
— this is dispatch-overhead-dominated and timing tells you about overhead
variance, not the kernel. From 32K to 131K threads the work begins to
contribute (`p50_ratio / level_ratio` rises from 0.6 to 0.67). At 131K →
262K threads, doubling threads multiplies median time by 1.78× —
work-dominated. Linear scaling continues through 1M threads, then goes
sublinear at 8M (8× threads → 3.83× time) because the 32 MiB write hits
DRAM bandwidth saturation (~85 GB/s observed, in line with M1 Pro's
realistic write-only bandwidth).

**Practical consequence:** any `write_tid`-class memory-bound benchmark
on M1 Pro must operate at ≥ 262K threads to be measuring the kernel,
not the dispatch overhead. The 524K-1M range is the cleanest operating
point. **Retroactively this means experiments 001, 002, 003 all
measured `write_tid` 32t — i.e., were measuring dispatch overhead, not
write_tid kernel time.** Their cv numbers describe *overhead* variance,
not write-kernel variance.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power, K=1
warmup recipe (per 003).
**Closed by:** experiment 004.
**Still open for:** M4 Max; other access patterns (strided, scatter);
threadgroup sizes other than 32; whether the bandwidth-saturation knee
sits between 1M and 8M threads (no fine-grained sweep yet).

## At what `fma_loop` per-thread iter count does measured time start scaling with work?

**Answer:** Sharp step at iters = 192 → 256 on M1 Pro (32 threads,
threadgroup 32). Below 192 iters: median climbs slowly from 9 to 12.6
µs (overhead-dominated). At 192 → 256 iters: a discrete **+21 µs jump**
in median (12.6 → 34.2 µs) — far too large to be explained by the
extra 64 FMA instructions doing useful work. From 256 onward: clean
linear scaling (`p50_ratio / level_ratio` ∈ [0.90, 1.00] up to 65K
iters), with one bimodal-variance band at iters = 8192-16384 where
between-sweep p50s vary 2.5× even with 30 s between-sweep cooldowns.

**Practical consequence:** any `fma_loop`-class compute benchmark must
use ≥ 256 iters per thread to be measuring the kernel. The 96-256
range is non-monotonic and unsuitable for any "increase work,
measure proportional increase in time" methodology. The 4096-16384
range has bimodal between-run state effects and should also be
avoided for between-run-comparable measurements.

**Open mechanism:** the +21 µs step at 192→256 has not been isolated.
Most likely explanation is a compiler / hardware threshold (register
pressure, loop-unrolling heuristic flip, instruction-cache crossover)
rather than work actually appearing — kept as an open question in
`UNKNOWNS.md`.

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power, K=1
warmup recipe (per 003).
**Closed by:** experiment 004.
**Still open for:** M4 Max; the mechanism behind the +21 µs step;
whether the bimodal band at 8192-16384 iters is thermal, DVFS, or a
queue-depth effect.

## Is the ~5.4 µs settled state observed in 003's first combo reproducible by the cold-start + cooldown + calibration recipe?

**Answer:** No. Across 5 sequential attempts with the protocol that
recreated the 003 first-combo conditions (5 s cooldown → 10-dispatch
calibration burst → 40 measured dispatches, no warmup, sleep_0
between trials), 0 of 200 measured dispatches landed in [5000, 6000]
ns. Every attempt converged to the ~8 µs back-to-back floor that
001/002/003 always observed elsewhere (per-attempt p50: 8146, 8333,
8084, 8166, 8334 ns). Per the 004 success criterion's threshold (0-1
reproductions = "non-reproducible artifact"), this state is filed.

**Practical consequence:** the operational floor for `write_tid`
32-thread back-to-back dispatches on M1 Pro is **8 µs (K=0) / 9 µs
(K=1)** for downstream calibration-probe work. The 5.4 µs observation
in 003 is a single-launch artifact whose trigger we cannot reconstruct
from logged state. Future experiments do not get to claim sub-8-µs
steady-state without independently reproducing it.

**What we ruled out:** that the recipe of "cold-start + caffeinate +
short cooldown + calibration burst" reaches the state. What we did
not test: that *the very first GPU activity in a fresh process tree
that has had no recent prior MLX/Metal activity at all* reaches it.
But this is unfalsifiable in practice (we cannot confidently produce
a "no recent prior GPU activity" environment without a full reboot
and a measurement protocol that does not contaminate it).

**Hardware/software:** Apple M1 Pro 16GB, macOS 26.3.1, AC power.
**Closed by:** experiment 004.

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
