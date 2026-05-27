"""
analyze_scramble.py -- diagnostic for the scramble-vs-over_dilution puzzle.

NOT part of the pipeline.  A one-off investigation: the aux-weight sweep
showed over_dilution AUROC swing 0.68->0.58->0.78 while scramble stayed
flat at ~0.61.  Hypothesis: the #13 proportion head regresses ONE masked
slot's proportion, so it can flag an EXTREME single value (over_dilution
inflates one slot) but not a PERMUTATION (scramble reshuffles the same
multiset of values -- every individual slot value stays plausible).

This script tests that hypothesis on an existing checkpoint, no retrain:

  TEST 1 -- energy delta.  For each test recipe, energy(real) vs
            energy(scramble) vs energy(over_dilution).  If the model is
            blind to permutations, scramble's delta (pert - real) is
            small / often negative; over_dilution's is large / positive.

  TEST 2 -- what the aux head keys on.  On the masked slot of scramble
            and over_dilution negatives, compare the proportion head's
            prediction to (a) the slot's TRUE original proportion and
            (b) the slot's PERTURBED value.  If the head tracks the
            perturbed value on over_dilution but is uninformative on
            scramble, the hypothesis holds.

Run on the pod:
    uv run --no-sync python scripts/analyze_scramble.py --ckpt runs/jepa-05-aux20/best.ckpt
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from cocktail_jepa.config import CONFIG
from cocktail_jepa.data.dataset import CocktailDataset, _stack, load_recipes
from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.energy.energy import recipe_energy
from cocktail_jepa.train.checkpoint import load_checkpoint


def _by_source(perturbed: list[dict], kind: str) -> dict[str, dict]:
    """Perturbed recipes of one type, keyed by the source recipe id."""
    return {r["source_id"]: r for r in perturbed
            if r.get("perturbation") == kind}


@torch.no_grad()
def _energy_of(model, recipes: list[dict], vocab, device, max_len: int,
               n_freq: int) -> dict[str, float]:
    """Energy per recipe, keyed by recipe_id (or source_id if present)."""
    ds = CocktailDataset(recipes, vocab, max_len=max_len, n_frequencies=n_freq)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=_stack)
    out: dict[str, float] = {}
    idx = 0
    for batch in loader:
        e = recipe_energy(model, batch, device=device)
        for i, rid in enumerate(batch["recipe_id"]):
            out[rid] = float(e[i].item())
        idx += len(batch["recipe_id"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-frequencies", type=int, default=6)
    args = ap.parse_args()

    paths = CONFIG.paths
    device = CONFIG.device
    splits = paths.corpus / "splits"
    print(f"device: {device}")

    ck = load_checkpoint(args.ckpt, map_location=device)
    model = ck["model"]
    model.to(device).eval()
    print(f"checkpoint: {args.ckpt}  (val_loss "
          f"{ck['extra'].get('val_loss', 'n/a')})")

    vocab = Vocabulary.from_file(paths.vocabulary)
    test = load_recipes(splits / "test.jsonl")
    perturbed = load_recipes(splits / "perturbations.jsonl")

    # index test recipes by id; perturbations by source id, per type
    test_by_id = {r.get("recipe_id", ""): r for r in test}
    scramble = _by_source(perturbed, "scramble")
    dilution = _by_source(perturbed, "over_dilution")
    print(f"test {len(test)} | scramble {len(scramble)} | "
          f"over_dilution {len(dilution)}")

    # -- TEST 1: energy deltas -------------------------------------------
    # score reals and each perturbation type with the SAME max_len budget
    # (+2 headroom on perturbations, matching evaluate.py)
    real_e = _energy_of(model, test, vocab, device, args.max_len,
                        args.n_frequencies)
    scr_e = _energy_of(model, list(scramble.values()), vocab, device,
                       args.max_len + 2, args.n_frequencies)
    dil_e = _energy_of(model, list(dilution.values()), vocab, device,
                       args.max_len + 2, args.n_frequencies)

    def deltas(pert_recipes: dict[str, dict],
               pert_e: dict[str, float]) -> list[float]:
        d = []
        for sid, prec in pert_recipes.items():
            r = test_by_id.get(sid)
            if r is None:
                continue
            re = real_e.get(r.get("recipe_id", ""))
            pe = pert_e.get(prec.get("recipe_id", ""))
            if re is None or pe is None:
                continue
            d.append(pe - re)
        return d

    scr_d = torch.tensor(deltas(scramble, scr_e))
    dil_d = torch.tensor(deltas(dilution, dil_e))

    def summ(name: str, d: torch.Tensor) -> None:
        pos = float((d > 0).float().mean().item())
        print(f"  {name:14s} n={len(d):4d}  "
              f"mean delta={d.mean().item():+.4f}  "
              f"std={d.std().item():.4f}  "
              f"frac(pert>real)={pos:.2%}")

    print("\nTEST 1 -- energy delta (perturbed energy minus real energy):")
    print("  a healthy negative should LOWER below... no: perturbed should")
    print("  be HIGHER, so positive delta = the energy noticed the corruption.")
    summ("scramble", scr_d)
    summ("over_dilution", dil_d)

    # -- TEST 2: what the proportion head predicts -----------------------
    # for each perturbation type, mask the slot the perturbation changed
    # and read the proportion head's output.  Compare to the slot's
    # perturbed value.  We mask the LAST real slot (deterministic, matches
    # the collator) and compare the head's prediction to that slot's value.
    @torch.no_grad()
    def head_probe(pert_recipes: dict[str, dict], max_len: int) -> dict:
        ds = CocktailDataset(list(pert_recipes.values()), vocab,
                             max_len=max_len, n_frequencies=args.n_frequencies)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=128, shuffle=False,
                            collate_fn=_stack)
        abs_err, n = 0.0, 0
        for batch in loader:
            ids = batch["ingredient_ids"].to(device)
            props = batch["proportions"].to(device)
            pad = batch["pad_mask"].to(device)
            raw = batch["raw_proportion"].to(device)
            n_ing = batch["n_ingredients"]
            B = ids.shape[0]
            bidx = torch.arange(B, device=device)
            mask_index = (n_ing - 1).to(device).long()
            tokens = model.tokens(ids, props)
            ctx = model.context_encoder(
                model.tokens.apply_mask(tokens, mask_index), pad)
            query = ctx[bidx, mask_index].unsqueeze(1)
            predicted = model.predictor(query, ctx, pad)
            pred_p = torch.sigmoid(
                model.proportion_head(predicted).squeeze(-1))
            true_p = raw[bidx, mask_index]
            known = true_p >= 0  # -1.0 sentinel = unknown
            if known.any():
                abs_err += float(
                    (pred_p[known] - true_p[known]).abs().sum().item())
                n += int(known.sum().item())
        return {"mae": abs_err / max(1, n), "n": n}

    print("\nTEST 2 -- proportion head: |predicted - perturbed slot value|")
    print("  (lower MAE = head tracks the perturbed value at that slot)")
    sp = head_probe(scramble, args.max_len + 2)
    dp = head_probe(dilution, args.max_len + 2)
    print(f"  scramble       MAE={sp['mae']:.4f}  (n={sp['n']})")
    print(f"  over_dilution  MAE={dp['mae']:.4f}  (n={dp['n']})")

    print("\nInterpretation guide:")
    print("  Hypothesis HOLDS if: scramble energy delta is small / often")
    print("  negative AND over_dilution delta is clearly positive -- i.e.")
    print("  the energy sees inflation but not permutation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
