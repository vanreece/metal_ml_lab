# Decisions

Append-only log of decisions and the reasoning behind them. Inspired by
ADRs (Architecture Decision Records) but with a stronger emphasis on
honesty about uncertainty.

## Rules

1. **Write the decision down before acting on it**, not after. Retrospective
   decision logs flatter the writer and lose the actual reasoning.

2. **Include what you considered and rejected**, not just what you chose.
   The rejected options are the most useful part for anyone reading later
   (including future you).

3. **State your confidence level honestly.** "I'm 60% sure this is right
   but want to revisit if X happens" is more useful than false certainty.

4. **Never delete or rewrite a decision.** When you change your mind, write
   a new decision that supersedes the old one and link them. The trail is
   the point.

5. **Decisions about what NOT to do are valid decisions** and worth
   recording, especially when there's a tempting alternative you're
   actively choosing against.

## Format

Filename: `NNN-kebab-case-name.md`, zero-padded.

Body:

```markdown
# NNN: Title

**Date:** YYYY-MM-DD
**Status:** active | superseded by NNN | reversed
**Confidence:** low | medium | high

## Context

What's the situation? What forced the decision?

## Options considered

- Option A: brief description, pros, cons
- Option B: ...
- Option C: ...

## Decision

What we're doing and why.

## What would make us revisit

The specific observations or events that should trigger reconsidering this.
This is the most important section. If you can't think of what would
change your mind, you may be more committed than the evidence warrants.
```

## What goes here vs. notes/

Decisions: choices that close off paths or commit time. "We're using PyObjC
not Swift bridges." "We're not building a library API yet."

Notes: observations, references, things being figured out. "Here's what I
learned about MTLCounterSampleBuffer's enum semantics."
