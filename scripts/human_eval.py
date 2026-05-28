"""
human_eval.py -- validate the energy against HUMAN coherence labels.

Every other number in the project is measured against the model's own
perturbation benchmark: AUROC for "can the energy detect the corruptions I
injected".  That is a proxy.  The claim the project actually wants to make
is "the energy tracks what a person would call a bad drink" -- and the only
way to test THAT is against human judgement.

This harness does exactly that.  It loads a hand-labeled set of real drinks
(coherent) and genuinely-bad combinations (incoherent), scores each with a
trained checkpoint's energy, and reports AUROC of human-incoherent
(positive, should be HIGH energy) vs human-coherent (negative, LOW energy).

  * If this AUROC is high, the synthetic perturbation benchmark is
    VINDICATED as a proxy -- the energy tracks human judgement, not just
    its own corruption artifacts.
  * If it is low, that is a finding worth knowing BEFORE an interviewer
    finds it: the energy detects the perturbations but does not generalize
    to human-judged coherence.

The labels are HUMAN.  This script does not generate or label drinks -- it
reads a file the project author wrote and verified.  An LLM labeling drinks
to validate the model would be circular; the whole point is an external
ground truth.

Label file format (JSONL, one drink per line):
    {"name": "Negroni", "label": "coherent",
     "ingredients": [{"ingredient": "gin", "proportion": 0.33},
                     {"ingredient": "campari", "proportion": 0.33},
                     {"ingredient": "sweet vermouth", "proportion": 0.34}]}
    {"name": "Gin & Ketchup", "label": "incoherent",
     "ingredients": [{"ingredient": "gin", "proportion": 0.5},
                     {"ingredient": "ketchup", "proportion": 0.5}]}

`label` must be "coherent" or "incoherent".  Proportions need not sum to 1;
they are normalized by the same pipeline the corpus uses.

Run:
    uv run --no-sync python scripts/human_eval.py \
        --ckpt runs/jepa-long-final-v2/jepa-long-s1.ckpt \
        --labels eval/human_drinks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, _stack
from cocktail_jepa.data.vocab import Vocabulary
from cocktail_jepa.energy.energy import energy_over_loader
from cocktail_jepa.energy.evaluate import auroc
from cocktail_jepa.train.checkpoint import load_checkpoint

# the canonicalizer the corpus was built with -- so a human-entered
# "fresh lime juice" maps to the same token the model was trained on.
sys.path.insert(0, str(CONFIG.paths.root / "scripts"))
try:
    from build_corpus import canonicalize  # type: ignore
except Exception:  # pragma: no cover - build_corpus must be importable
    canonicalize = None


def load_labeled(path) -> tuple[list[dict], list[int], list[str]]:
    """
    Load the human label file.

    Returns (recipes, labels, names) where label is 1 for incoherent
    (positive: should score HIGH energy) and 0 for coherent.
    """
    recipes, labels, names = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lab = d["label"].strip().lower()
            if lab not in ("coherent", "incoherent"):
                raise ValueError(
                    f"{d.get('name', '?')}: label must be "
                    f"'coherent'/'incoherent', got {d['label']!r}"
                )
            recipes.append({"ingredients": d["ingredients"]})
            labels.append(1 if lab == "incoherent" else 0)
            names.append(d.get("name", "?"))
    return recipes, labels, names


def canonicalize_recipes(recipes: list[dict], vocab: Vocabulary) -> list[int]:
    """
    Canonicalize every ingredient string in place and report how many fall
    outside the model's vocabulary.

    A human drink scored on ingredients the model never saw is being judged
    on partial input (OOV tokens map to [MASK]) -- the caller must KNOW
    that, so we count and warn rather than silently drop.  Returns a list
    of per-recipe OOV counts, aligned with `recipes`, so the caller can
    build an in-vocab-only subset for a clean read.
    """
    per_recipe_oov = []
    for r in recipes:
        oov = 0
        for ing in r["ingredients"]:
            raw = ing["ingredient"]
            canon = canonicalize(raw) if canonicalize else raw.lower().strip()
            ing["ingredient"] = canon
            if canon not in vocab.token_to_id:
                oov += 1
                print(f"  [oov] {raw!r} -> {canon!r} not in vocabulary")
        per_recipe_oov.append(oov)
    return per_recipe_oov


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate energy against human coherence labels.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True,
                    help="JSONL of hand-labeled drinks")
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    paths = CONFIG.paths
    device = CONFIG.device
    print(f"device: {device}")

    ckpt_path = (paths.root / args.ckpt
                 if not args.ckpt.startswith("/") else args.ckpt)
    ck = load_checkpoint(ckpt_path, map_location=device)
    model = ck["model"]
    print(f"loaded checkpoint: step {ck['step']}")

    vocab = Vocabulary.from_file(paths.vocabulary)
    recipes, labels, names = load_labeled(args.labels)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"labeled drinks: {len(labels)}  "
          f"({n_neg} coherent, {n_pos} incoherent)")
    if n_pos == 0 or n_neg == 0:
        print("[FAIL] need at least one of each label to compute AUROC")
        return 1

    print("canonicalizing ingredients against the model vocabulary ...")
    per_oov = canonicalize_recipes(recipes, vocab)
    total_oov = sum(per_oov)
    if total_oov:
        print(f"  {total_oov} out-of-vocab tokens across "
              f"{sum(1 for o in per_oov if o)} drinks "
              f"(OOV tokens map to [MASK]; those drinks scored on partial input)")

    # Pre-filter exactly as CocktailDataset will (2 <= n_ingredients <=
    # max_len), so recipes, labels, names, and oov stay parallel by
    # construction -- no fragile post-hoc realignment against the dataset.
    keep = [2 <= len(r["ingredients"]) <= args.max_len for r in recipes]
    dropped = keep.count(False)
    if dropped:
        for r, k, nm in zip(recipes, keep, names):
            if not k:
                print(f"  [drop] {nm!r}: {len(r['ingredients'])} ingredients "
                      f"(outside 2..{args.max_len})")
    recipes = [r for r, k in zip(recipes, keep) if k]
    labels = [l for l, k in zip(labels, keep) if k]
    names = [n for n, k in zip(names, keep) if k]
    per_oov = [o for o, k in zip(per_oov, keep) if k]

    if sum(labels) == 0 or (len(labels) - sum(labels)) == 0:
        print("[FAIL] after filtering, need at least one of each label")
        return 1

    # score with the real energy path -- same as evaluate.py
    ds = CocktailDataset(recipes, vocab, max_len=args.max_len,
                         n_frequencies=args.n_frequencies)
    assert len(ds) == len(recipes), (
        "pre-filter must match CocktailDataset's filter exactly"
    )
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=_stack)
    energies, _ = energy_over_loader(model, loader, device=device)

    lab_t = torch.tensor(labels, dtype=torch.long)
    oov_t = torch.tensor(per_oov, dtype=torch.long)

    def report_subset(tag: str, mask: torch.Tensor) -> None:
        n = int(mask.sum())
        sub_lab = lab_t[mask]
        sub_e = energies[mask]
        n_pos = int((sub_lab == 1).sum())
        n_neg = int((sub_lab == 0).sum())
        print(f"\n[{tag}]  {n} drinks ({n_neg} coherent, {n_pos} incoherent)")
        if n_pos == 0 or n_neg == 0:
            print("  (need at least one of each label -- AUROC undefined)")
            return
        pos_e = sub_e[sub_lab == 1]
        neg_e = sub_e[sub_lab == 0]
        a = auroc(neg_e, pos_e)
        print(f"  AUROC (energy vs human labels): {a:.4f}")
        print(f"  coherent mean energy  : {neg_e.mean():.4f}")
        print(f"  incoherent mean energy: {pos_e.mean():.4f}")

    print("\n" + "=" * 60)
    print("HUMAN-GROUNDED VALIDATION")
    print("=" * 60)

    # Full set: every labeled drink, OOV tokens included (as [MASK]).
    report_subset("full set", torch.ones_like(lab_t, dtype=torch.bool))

    # In-vocab-only: drinks with ZERO out-of-vocab tokens. This is the
    # clean read -- the energy judges these on their actual ingredients,
    # not on a masked slot standing in for an unknown one. The two numbers
    # together show whether OOV masking is helping or hurting, instead of
    # hiding the confound in a single figure.
    report_subset("in-vocab only", oov_t == 0)

    # Per-drink energies, sorted, so the writeup can show WHICH drinks the
    # energy ranks as most/least coherent and whether that matches the human
    # label -- honest, inspectable, not a single opaque number.
    print("\nper-drink energy (low = coherent), * marks OOV drinks:")
    order = torch.argsort(energies)
    for i in order.tolist():
        star = "*" if per_oov[i] else " "
        mark = "OK " if (labels[i] == 1) == (energies[i] > energies.median()) \
            else "?? "
        print(f"  {energies[i]:7.4f} {star} [{('incoh' if labels[i] else 'coher'):>5}] "
              f"{names[i]}")

    print("\nInterpretation:")
    print("  Compare the in-vocab-only AUROC to the synthetic benchmark's")
    print("  overall AUROC. Close agreement upgrades the evaluation from")
    print("  'detects my perturbations' to 'tracks human coherence'. The")
    print("  full-set number includes drinks scored partly on a [MASK] slot")
    print("  (non-cocktail ingredients the vocab never saw); report both and")
    print("  say which is which -- do not collapse them into one figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
