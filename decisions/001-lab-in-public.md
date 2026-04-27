# 001: Lab in public, with a decision log and pre-registered experiments

**Date:** 2026-04-27
**Status:** active
**Confidence:** high

## Context

This project is being deliberately structured as a case study in agent-first
foundational tooling work, not just as a library. The methodology of how it
gets built is intended to be as transferable as the code. That goal puts
specific demands on the structure of the repo from day one — demands that
would be hard to retrofit later.

## Options considered

- **Standard open-source project structure** (src/, tests/, docs/, CI from
  day one). Familiar, but assumes the shape of the eventual artifact in
  ways we don't yet have evidence for. Encourages premature API design.
  Doesn't naturally produce the documentation-of-process that is half the
  point.

- **Private exploration, open-source later when ready.** Lower
  embarrassment risk, but loses the entire methodology contribution. The
  whole thesis is that working in public on foundational tooling, with the
  process visible, is itself the contribution. Can't validate that thesis
  by working in private.

- **Public but loosely structured ("blog + repo, figure it out").** Tempting
  because it's flexible, but flexibility this early is a euphemism for
  "won't actually capture the decisions and reasoning" — those things
  require structural pressure to record consistently. Past experience says
  loosely-structured projects produce loosely-structured artifacts.

- **Public lab structure with experiments/, decisions/, notes/, and an
  explicit norm of pre-registering experiments before running them and
  publishing periodic writeups regardless of outcome.** Highest overhead
  per experiment, but the overhead is exactly the thing that produces the
  transferable methodology artifact. Forces honesty about negative results
  by making them structurally indistinguishable from positive ones.

## Decision

Public lab structure. Experiments are numbered, self-contained, and
pre-registered (question + hypothesis + setup committed before result).
Decisions are append-only and written before action. Periodic writeups go
out on a cadence regardless of whether there's a "result worth publishing"
— the cadence is the discipline that prevents lab-in-public from becoming
lab-in-private.

Outreach to potential collaborators (Asankhaya Sharma, Awni Hannun, Philip
Turner, Stanford KernelBench team) is done one-on-one and in private, not
through the public artifacts. Public artifacts describe what's being done;
they don't enroll specific people in the narrative without consent.

## What would make us revisit

- If after ~10 experiments the decision-log overhead is consistently
  preventing experiments from happening rather than improving them. (The
  log should accelerate work by sharpening thinking, not slow it. If it
  doesn't, the format is wrong.)
- If pre-registration starts producing experiments that are too
  conservative — only running things that are obviously going to produce
  publishable results. The point is to remove uncertainty, including by
  trying things that might not work.
- If we discover collaborators want to engage but the public lab structure
  is making coordination harder than it needs to be. Adjust the
  public/private boundary, don't abandon the structure.
