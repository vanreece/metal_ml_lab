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

Early but moving. As of 2026-04-28, eight experiments are complete on
Apple M1 Pro (`applegpu_g13s` / macOS 26.3.1) and seven on Apple M4 Max
(`applegpu_g16s` / macOS 26.4.1). The lab is now structurally
multi-hardware; methodology decisions are per-chip.

For the current snapshot of what's known, what's open, and what
methodology applies on each chip, see
[`notes/state-2026-04-28.md`](notes/state-2026-04-28.md).

Headline status:

- **Pair timing is the primary methodology on M4 Max** for trials
  ≥ ~64 µs (decision 005). Within-session variance reduction works
  (T2 alone cv 0.336 → ratio cv 0.005, 63× improvement), and
  cross-session ratios are stable to ≤ 0.6 % spread (exp 006).
- **Single-kernel timing remains correct on M1 Pro** (decision 004).
  The 42 µs inter-encoder gap on G13 broke the variance-cancellation
  mechanism that decision 003 hypothesized; on G16 the gap collapses
  to ~833 ns and the mechanism works.
- **Sudo-free Apple Silicon power telemetry is available** via
  `notes/ioreport.py` (exp 008 MARGINAL pass — usable with a known
  +14 % bias at full saturation). The previous powermetrics-via-sudo
  workflow stays available as `notes/gpu_telemetry.py`.

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
