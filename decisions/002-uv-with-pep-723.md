# 002: uv + PEP 723 inline script metadata as the default Python tooling

**Date:** 2026-04-27
**Status:** active
**Confidence:** medium-high

## Context

Experiment 001's README left "Python 3.11+ via uv or pyenv" as a pending
decision. When 001 was actually implemented, the path of least resistance
was uv with [PEP 723](https://peps.python.org/pep-0723/) inline script
metadata — the script declares its own dependencies in a header comment,
and `uv run run.py` does the rest with no project-level pyproject.toml,
no manual venv activation, and no per-experiment shared state. The
decision is being formalized retroactively; per the discipline this means
explicit confidence and explicit revisit conditions, since the "write
before acting" rule was bent.

## Options considered

- **System Python + manual venv per experiment.** Most reproducible from
  zero, but high friction at the start of every experiment, and the venv
  state is invisible from the script itself. New collaborators have to
  read the README to know how to run anything. The script doesn't
  document its own dependencies in a machine-readable way.

- **pyenv + a top-level requirements.txt or pyproject.toml.** Familiar
  shape, but creates a project-wide dependency surface that fights the
  "each experiment is self-contained" principle from CLAUDE.md. If
  experiment N needs a different PyObjC version than experiment N-3, the
  shared file becomes a battleground. Also, pyenv shims add latency to
  every `python` invocation in a way uv does not.

- **uv + PEP 723 inline metadata in each experiment's run.py.** Every
  script declares its own dependencies in a header block, runnable as
  `uv run run.py` from a clean machine with only uv installed. uv handles
  Python version selection, dependency resolution, venv management, and
  caching transparently. Deps are scoped to the script. Different
  experiments can pin different versions without conflict. The
  reproduction story is one line in every experiment's README.

- **uv + project-level pyproject.toml.** All the uv benefits but loses
  per-experiment dependency scoping. Still tempting later if many
  experiments end up depending on the same things, but premature now.

## Decision

Use uv with PEP 723 inline script metadata. The `# /// script ... # ///`
block at the top of each experiment's run.py is the canonical declaration
of its dependencies. Reproduction from a clean machine: install uv, then
`uv run run.py` from the experiment directory.

No project-level pyproject.toml. No requirements.txt. No shared venv.
Duplication of `pyobjc-framework-Metal` across experiments' headers is
expected and acceptable.

## What would make us revisit

- If we end up with five+ experiments all pinning the exact same set of
  dependencies, and a security/compat update to one of them would need to
  be applied identically to each, the duplication starts costing more than
  the per-experiment isolation buys. At that point introduce a
  pyproject.toml at the repo root and let scripts opt in.
- If `uv run` ever fails in a way that's hard to diagnose because the
  dependency resolution is hidden inside uv's machinery. (PEP 723 trades
  visibility of resolution for convenience; if that trade hurts, undo it.)
- If we need to support a contributor or CI environment where uv is not
  installable for some reason. Unlikely but possible.
- If a future experiment needs C-level toolchain integration (e.g. linking
  against a Metal-cpp shim, building a custom Python extension) that PEP
  723 cannot express. Then that experiment specifically gets a
  pyproject.toml; the rest stay on the inline pattern.
