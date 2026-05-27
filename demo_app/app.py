"""
app.py -- local web app serving the real Cocktail-JEPA model.

This is NOT a pre-computed demo.  It loads the trained jepa-long
checkpoint and runs the actual model on every request: energy descent
for recipe completion, masked-prediction energy for scoring.

Run it on a machine that has the checkpoint and the cocktail_jepa
package (your Mac):

    uv run python demo_app/app.py --ckpt runs/jepa-long-final/jepa-long-s4.ckpt

then open http://localhost:8000

Endpoints
  GET  /                  -- the single-page frontend (index.html)
  POST /api/complete       -- {have:[...], n_generate:int} -> completed recipe
  POST /api/score          -- {ingredients:[{name,proportion}]} -> energy
  GET  /api/ingredients    -- the known-ingredient list (for autocomplete)
  GET  /api/health         -- model loaded?  used by the frontend on boot

Everything the model does here is the real model.  The only thing the
backend adds is input validation -- unknown ingredients are caught and
reported rather than silently mapped to [MASK] (which encode() would do).
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cocktail_jepa.data.vocab import Vocabulary, proportion_encoding_dim
from cocktail_jepa.energy.energy import recipe_energy
from cocktail_jepa.generate.generate import GenConfig, generate
from cocktail_jepa.train.checkpoint import load_checkpoint

# instruction-phrase patterns -- matched as whole words, never substrings
# (so "peach" is never mistaken for the word "each").  An ingredient name
# hitting any of these is a corpus parsing artifact, not a real ingredient.
_JUNK_PAT = re.compile(
    r"\b(garnish|juice of|if needed|fill|top up|to taste|cubes?|wedge|"
    r"wheel|twist|peel|zest|dash(es)?|sprig|slices?|halves|diced|"
    r"skinned|chilled)\b", re.I)


def _is_junk_name(name: str) -> bool:
    """
    True if an ingredient string is a corpus parsing artifact rather than
    a real ingredient -- used to keep the demo's autocomplete clean.

    NOTE: this filters only the AUTOCOMPLETE SUGGESTIONS.  The model's
    full vocabulary is still accepted for scoring; we are cleaning the
    demo's front door, not pretending the junk tokens do not exist.
    """
    if _JUNK_PAT.search(name):
        return True
    if name.startswith((",", "(")):       # broken-parse leftovers
        return True
    if re.search(r"\d", name):            # 'juice of 1/2 lemon'
        return True
    if len(name.split()) > 4:             # over-long => parse error
        return True
    return False

# ---------------------------------------------------------------------------
# global state -- the model is loaded ONCE at startup, then reused
# ---------------------------------------------------------------------------

STATE: dict = {
    "model": None,
    "vocab": None,
    "prop_dim": None,
    "device": "cpu",
    "energy_stats": None,   # {min,max,mean} of real recipes, for percentiles
    "known": set(),         # ALL known ingredient names (accepted for scoring)
    "suggest": [],          # clean, count-ranked subset (autocomplete only)
}

HERE = Path(__file__).resolve().parent


def _load(ckpt_path: str) -> None:
    """Load the checkpoint, vocab, and a small reference energy distribution."""
    from cocktail_jepa.config import CONFIG
    import json as _json

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocabulary.from_file(CONFIG.paths.vocabulary)
    model = load_checkpoint(ckpt_path, map_location=device)["model"]
    model.to(device).eval()

    prop_dim = proportion_encoding_dim(6)

    # the AUTOCOMPLETE list: clean, well-supported ingredients only,
    # ranked by how many recipes use them.  The corpus has a long tail of
    # parse-artifact strings ('garnish with lime', 'juice of 1/2 lemon');
    # those are dropped here so the demo's input box shows real
    # ingredients.  The model's FULL vocabulary is still accepted for
    # scoring -- see STATE["known"].
    suggest: list[str] = []
    try:
        vj = _json.load(open(CONFIG.paths.vocabulary, encoding="utf-8"))
        ranked = sorted(vj["ingredients"], key=lambda it: -it["count"])
        suggest = [it["name"] for it in ranked
                   if it["count"] >= 5 and not _is_junk_name(it["name"])]
    except Exception as exc:
        print(f"[warn] could not build ranked suggest list ({exc})")
        suggest = sorted(vocab.token_to_id.keys())

    STATE.update(
        model=model, vocab=vocab, prop_dim=prop_dim, device=device,
        known=set(vocab.token_to_id.keys()),
        suggest=suggest,
    )

    # reference energy distribution -- used to turn a raw energy into a
    # "more coherent than X% of real recipes" percentile.  Computed once
    # from the test split if available; falls back to a fixed range.
    try:
        from torch.utils.data import DataLoader

        from cocktail_jepa.data.dataset import (CocktailDataset, _stack,
                                                load_recipes)
        from cocktail_jepa.energy.energy import energy_over_loader
        test = load_recipes(CONFIG.paths.corpus / "splits" / "test.jsonl")
        ds = CocktailDataset(test, vocab, max_len=12, n_frequencies=6)
        loader = DataLoader(ds, batch_size=128, shuffle=False,
                            collate_fn=_stack)
        e, _ = energy_over_loader(model, loader, device=device)
        STATE["energy_stats"] = {"min": float(e.min()), "max": float(e.max()),
                                 "mean": float(e.mean()),
                                 "sorted": sorted(e.tolist())}
    except Exception as exc:  # demo still works without percentiles
        print(f"[warn] no reference energy distribution ({exc})")
        STATE["energy_stats"] = None

    print(f"model loaded on {device}; {len(STATE['known'])} known ingredients")


def _percentile(energy: float) -> float | None:
    """
    Energy -> percentile among real recipes.

    Returns the % of real test recipes that have HIGHER energy (= worse,
    less coherent) than this one.  Low energy => most real recipes are
    worse => high percentile => 'more coherent than X% of real recipes'.

    Returns None if no reference distribution is available, so the caller
    can omit the claim entirely rather than show a wrong number.
    """
    stats = STATE["energy_stats"]
    if not stats or not stats.get("sorted"):
        return None
    s = stats["sorted"]
    higher = sum(1 for x in s if x > energy)
    return round(100.0 * higher / len(s), 1)


def _coherence_band(energy: float) -> str:
    """
    A verdict derived DIRECTLY from energy (not from the percentile), so
    the dial and the verdict can never disagree -- both key off energy.

    Thresholds are anchored to the real-recipe energy distribution when
    one is available (mean +- spread); otherwise a fixed fallback.
    """
    stats = STATE["energy_stats"]
    if stats and stats.get("sorted"):
        s = stats["sorted"]
        lo = s[len(s) // 20]            # ~5th percentile of real recipes
        hi = s[-len(s) // 20]           # ~95th percentile
    else:
        lo, hi = 0.47, 0.63
    if energy <= lo:
        return "high"          # at or below the most-coherent real recipes
    if energy <= (lo + hi) / 2:
        return "good"
    if energy <= hi:
        return "fair"
    return "low"               # worse than almost all real recipes


def _suggest_matches(name: str, k: int = 3) -> list[str]:
    """
    Suggest known ingredients for an unrecognized input.

    Combines two signals so multi-word user input still matches:
      - word overlap: shared words between the input and a vocab entry
        (so 'fresh squeezed lime' -> 'lime juice')
      - character similarity: difflib ratio, for typos ('whiskey'/'wiskey')
    Suggestions are drawn from the clean `suggest` list, ranked by the
    combined score.  Returns at most k names.
    """
    q_words = set(re.findall(r"[a-z]+", name.lower()))
    scored: list[tuple[float, str]] = []
    for cand in STATE["suggest"]:
        c_words = set(re.findall(r"[a-z]+", cand.lower()))
        overlap = (len(q_words & c_words) / len(q_words | c_words)
                   if (q_words | c_words) else 0.0)
        char = difflib.SequenceMatcher(None, name.lower(),
                                       cand.lower()).ratio()
        score = 0.65 * overlap + 0.35 * char
        if score > 0.3:
            scored.append((score, cand))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


def _validate(names: list[str]) -> tuple[list[str], list[dict]]:
    """
    Split typed names into (known, unknown-with-suggestions).

    Validation accepts the model's FULL vocabulary -- a real but rare
    ingredient is still scoreable.  Only the suggestion list draws from
    the cleaned set.  encode() would silently map an unknown name to
    [MASK]; catching it here is the bulletproofing.
    """
    known, unknown = [], []
    for raw in names:
        name = raw.strip().lower()
        if not name:
            continue
        if name in STATE["known"]:
            known.append(name)
        else:
            unknown.append({"input": raw,
                            "suggestions": _suggest_matches(name)})
    return known, unknown


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------

class CompleteReq(BaseModel):
    have: list[str] = []
    n_generate: int = 3


class ScoreItem(BaseModel):
    name: str
    proportion: float | None = None


class ScoreReq(BaseModel):
    ingredients: list[ScoreItem]


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

app = FastAPI(title="Cocktail-JEPA demo")


@app.get("/api/health")
def health():
    # the energy scale, derived from the real-recipe distribution, so the
    # frontend dial uses the SAME anchors as _coherence_band -- no
    # hardcoded range that could drift from the actual model.
    scale = None
    stats = STATE["energy_stats"]
    if stats and stats.get("sorted"):
        s = stats["sorted"]
        scale = {"lo": round(s[len(s) // 20], 4),       # ~5th percentile
                 "hi": round(s[-len(s) // 20], 4),      # ~95th percentile
                 "median": round(s[len(s) // 2], 4)}
    return {"ready": STATE["model"] is not None,
            "device": STATE["device"],
            "n_ingredients": len(STATE["known"]),
            "scale": scale}


@app.get("/api/ingredients")
def ingredients():
    """The autocomplete list -- clean, well-supported ingredients only,
    already ranked by recipe frequency (most common first)."""
    return {"ingredients": STATE["suggest"]}


@app.post("/api/complete")
def complete(req: CompleteReq):
    """Energy-descent recipe completion -- the real generate() call."""
    if STATE["model"] is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    known, unknown = _validate(req.have)
    if unknown:
        return JSONResponse(
            {"error": "unknown_ingredients", "unknown": unknown},
            status_code=400)

    n_gen = max(1, min(int(req.n_generate), 8))   # clamp to a sane range
    if len(known) + n_gen > 12:
        n_gen = max(1, 12 - len(known))

    try:
        out = generate(
            STATE["model"], STATE["vocab"],
            fixed_ingredients=known, n_generate=n_gen,
            cfg=GenConfig(restarts=6, steps=140),
            max_len=12, prop_dim=STATE["prop_dim"], device=STATE["device"],
        )
    except Exception as exc:
        return JSONResponse({"error": f"generation failed: {exc}"},
                            status_code=500)

    energy = float(out["energy"])
    return {
        "have": known,
        "generated": out["generated"],
        "full_recipe": out["full_recipe"],
        "energy": round(energy, 4),
        "percentile": _percentile(energy),
        "coherence": _coherence_band(energy),
        "is_generated": True,   # flags that this came from energy descent
    }


@app.post("/api/score")
def score(req: ScoreReq):
    """Score a complete recipe -- the real recipe_energy() call."""
    if STATE["model"] is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    names = [it.name for it in req.ingredients]
    known, unknown = _validate(names)
    if unknown:
        return JSONResponse(
            {"error": "unknown_ingredients", "unknown": unknown},
            status_code=400)
    if len(known) < 2:
        return JSONResponse(
            {"error": "need at least 2 known ingredients to score"},
            status_code=400)

    # build the single-recipe batch recipe_energy expects
    vocab = STATE["vocab"]
    device = STATE["device"]
    prop_dim = STATE["prop_dim"]
    n = len(known)

    ids = torch.full((1, 12), 0, dtype=torch.long)        # 0 = PAD
    for i, name in enumerate(known):
        ids[0, i] = vocab.encode(name)
    pad_mask = torch.zeros(1, 12, dtype=torch.bool)
    pad_mask[0, :n] = True

    # proportions: use given values where present, else uniform
    given = [it.proportion for it in req.ingredients]
    props = torch.zeros(1, 12, prop_dim)
    raw = torch.zeros(1, 12)
    for i in range(n):
        p = given[i] if (i < len(given) and given[i] is not None) else 1.0 / n
        raw[0, i] = float(p)
    props[:, :, -1] = 1.0   # "known" flag channel

    batch = {"ingredient_ids": ids, "proportions": props,
             "raw_proportion": raw, "pad_mask": pad_mask,
             "n_ingredients": torch.tensor([n]),
             "recipe_id": ["user"]}

    try:
        e = recipe_energy(STATE["model"], batch, device=device)
        energy = float(e[0])
    except Exception as exc:
        return JSONResponse({"error": f"scoring failed: {exc}"},
                            status_code=500)

    return {
        "ingredients": known,
        "energy": round(energy, 4),
        "percentile": _percentile(energy),
        "coherence": _coherence_band(energy),
        "is_generated": False,
    }


# static frontend -- index.html + assets live next to this file
@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def main() -> int:
    ap = argparse.ArgumentParser(description="Cocktail-JEPA local demo.")
    ap.add_argument("--ckpt", required=True,
                    help="path to the jepa-long checkpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"loading model from {args.ckpt} ...")
    _load(args.ckpt)
    print(f"\n  Cocktail-JEPA demo running -> http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
