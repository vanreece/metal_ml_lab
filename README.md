# metal_ml_lab

An open exploration into whether Apple Silicon GPU kernel performance can be
characterized — bottleneck class, roofline position, occupancy effects —
without access to the vendor-internal counters that NVIDIA's CUPTI exposes
and Apple's Metal framework does not.

This repo is also a deliberate case study in agent-first foundational tooling
work: building observability infrastructure for an underprovisioned
high-performance computing ecosystem, in public, with the process documented
as carefully as the code.

## Status

Early but moving. As of 2026-04-29, seventeen experiments are
complete (001-008 on both M1 Pro and M4 Max; 009-016 M4 Max only).
**The methodology arc 001-016 is closed**: timing infrastructure
(001-006), telemetry (007-008), DVFS state-machine (009-013),
amplification methodology validated on synthetic compute & memory
(014-015), and discrimination methodology validated on a real ML
kernel (naive fp32 matmul, exp 016).

For the current snapshot of what's known, what's open, and what
methodology applies on each chip, see
[`notes/state-2026-04-29-post016.md`](notes/state-2026-04-29-post016.md).

Headline status:

- **Pair timing is the primary methodology on M4 Max** for trials
  ≥ ~64 µs (decision 005). Within-session variance reduction works
  (T2 alone cv 0.336 → ratio cv 0.005, 63× improvement), and
  cross-session ratios are stable to ≤ 0.6 % spread (exp 006).
- **Single-kernel timing remains correct on M1 Pro** (decision 004).
  The 42 µs inter-encoder gap on G13 broke the variance-cancellation
  mechanism that decision 003 hypothesized; on G16 the gap collapses
  to ~833 ns and the mechanism works.
- **Sudo-free Apple Silicon DVFS observability** is available via
  `notes/ioreport.py --include-states` (exp 010 PASS). M4 Max GPUPH
  has 16 states (`OFF + P1..P15`) with a positional MHz mapping
  recovered (peak 1 578 MHz; non-monotonic in P-index — exp 012).
  PWRCTRL has 4 modes; DEADLINE entry rule characterized (exp 013).
- **The M4 Max sub-floor regime** at ~1.7 µs / 39 timestamp ticks is
  reproducible 5/5 under `fma_loop K=20 sleep_0` (exp 009), driven
  by PWRCTRL `DEADLINE` mode + brief P15 visits per dispatch
  (exp 011).
- **Sudo-free power telemetry** stays available via `notes/ioreport.py`
  (exp 008 MARGINAL — known +14 % full-load bias). The
  powermetrics-via-sudo workflow stays available as
  `notes/gpu_telemetry.py` for tight-tolerance work.

If you found this and want to talk about it, open an issue.

## What's here

- `experiments/` — numbered, dated, self-contained scripts. Each
  answers a specific question. Most of them will be wrong or
  incomplete; that's the point. See `experiments/README.md`.
- `decisions/` — append-only log of decisions and their reasoning,
  including the ones that get reversed later. See
  `decisions/README.md`.
- `notes/` — working notes, references, things being figured out,
  and a small set of user-run utility scripts (`probe-counter-sets.py`,
  `gpu_telemetry.py`, `ioreport.py`).
- `UNKNOWNS.md` — the maintained list of open questions, ordered
  by how much each one matters.

## What's not here yet, and why

No library API. No package structure. No `metal_ml_lab.measure()`. The
shape of the eventual library should emerge from "I keep writing this same
code in every experiment" — not from speculation about what it should look
like before the experiments exist.

No tests. The experiments themselves are the tests of whether the ideas
work; tests of code stability come once code is stable enough to be worth
stabilizing.

No CI, no packaging, no docs site. Premature scaffolding.

## License

Apache-2.0. See [LICENSE](LICENSE).
