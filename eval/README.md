# Human-grounded validation set

This is the external ground truth that turns the project's central claim from
"the energy detects my own perturbations" into "the energy tracks human
judgement of coherence". Every other number in the project is measured against
the synthetic perturbation benchmark — a proxy. This set is the check on that
proxy.

## What this file is

`human_drinks.jsonl` — one drink per line, each labeled `coherent` or
`incoherent` **by a human** (you). Scored by `scripts/human_eval.py`, which
reports AUROC of the energy against these labels.

The file currently holds 8 coherent + 4 incoherent **seed examples** — obvious
cases to establish the format. It is NOT yet large enough or balanced enough
to be the reported result. The critique asked for 30–50 labeled drinks; extend
this file to roughly 20–25 coherent and 20–25 incoherent before treating its
AUROC as reportable.

## The labeling discipline (this is the whole point)

The value of this set is that the labels are an **external** judgement, not the
model's and not an LLM's. An LLM labeling drinks to validate the model would be
circular. So:

- **Coherent** = real, well-regarded drinks. IBA classics (Negroni, Daiquiri,
  Old Fashioned…), well-reviewed craft-bar drinks. Things a bartender would
  serve without comment.
- **Incoherent** = genuinely bad combinations. Sourced from "worst cocktail"
  lists, or constructed by you and confirmed bad by tasting/judgement. The bar
  is "a person would call this a bad drink", not "the model dislikes it".
- **Avoid the easy-corruption trap.** Don't just take a real drink and swap one
  ingredient for motor oil — that recreates the synthetic benchmark. The
  incoherent set is strongest when the drinks are *plausible-looking but bad*:
  real ingredients, wrong together. That is the hard, honest test.
- **Keep proportions realistic.** They need not sum to 1 (the harness
  normalizes), but use sensible relative amounts so the proportion channel
  isn't carrying the signal by itself.

## Ingredient strings

Write ingredients naturally ("fresh lime juice", "Angostura bitters"); the
harness runs each through the same `canonicalize()` the corpus was built with,
so they map to the model's vocabulary. The harness prints an `[oov]` warning
for any ingredient that falls outside the vocabulary — if a drink shows OOV
tokens, it is being scored on partial input, so either rephrase the ingredient
or drop the drink.

## Running it

After retraining, against the new best seed:

```
uv run --no-sync python scripts/human_eval.py \
    --ckpt runs/jepa-long-final-v2/jepa-long-s1.ckpt \
    --labels eval/human_drinks.jsonl
```

Compare the printed human-AUROC to the synthetic benchmark's overall AUROC.
Close agreement vindicates the proxy; a large gap is a finding to report
honestly, not to hide.
