# Experiments

Each experiment lives in its own directory, numbered sequentially. The
structure is intentionally minimal:

```
experiments/
  001-can-we-time-anything/
    README.md       # The question, the setup, the result, the surprise
    run.py          # The actual experiment
    raw/            # Raw output, never overwritten
    notes.md        # Working notes during the experiment (optional)
```

## What a good experiment looks like

It answers **one question** that can be answered yes/no/numerically. "Can we
read `MTLCounterSampleBuffer` timestamps from Python?" is a good question.
"Build the timing layer" is not — it's a project, not an experiment.

It is **runnable from a single command** with no required arguments. If it
needs setup, the setup is in the README and is reproducible from a clean
machine.

Its README is written **before** the experiment runs (the question, the
hypothesis, the setup) and **updated after** (the result, the surprise, the
new questions). The before-version gets committed before the result. This
is the only way to avoid retrofitting hypotheses to outcomes.

It saves **raw data** to `raw/` with timestamps. Never overwrite raw data;
add new files. Analysis is reproducible from raw data.

Its README ends with **"new questions"** — what we didn't know before that
we now know we don't know. These feed the next experiment selection.

## What a bad experiment looks like

- Tries to answer too many questions at once
- Has a "successful" outcome that's vague enough to be unfalsifiable
- Skips the raw data because the result "looked obvious"
- Quietly tweaks parameters until it works without recording the tweaks
- Generates a clean library API as a side effect
- Has no "new questions" section because everything went as expected
  (everything going as expected is itself suspicious and worth noting)

## How experiments are selected

The next experiment is the one that most reduces uncertainty about whether
the overall approach can work — *not* the one that makes the most progress
toward a hypothetical product. When two experiments would reduce similar
uncertainty, prefer the cheaper one.

If you can't articulate what you'd learn from running an experiment, don't
run it yet.
