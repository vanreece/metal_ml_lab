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

Very early. The current goal is to characterize the noise floor of timing
measurement on Apple Silicon GPUs and figure out what kinds of objective
signals are actually reachable from a Python harness. Nothing here is
stable, nothing here is recommended for use, and the API does not exist yet.

If you found this and want to talk about it, open an issue.

## What's here

- `experiments/` — numbered, dated, self-contained scripts. Each answers a
  specific question. Most of them will be wrong or incomplete; that's the
  point. See `experiments/README.md`.
- `decisions/` — append-only log of decisions and their reasoning, including
  the ones that get reversed later. See `decisions/README.md`.
- `notes/` — working notes, references, things being figured out.

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
