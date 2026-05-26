"""
contrastive.py -- an InfoNCE contrastive baseline, the contrastive row of
the #43 comparison table.

WHY THIS EXISTS
---------------
The MAE baseline asks "would a RECONSTRUCTION objective give as good an
energy as the JEPA's latent-prediction objective?".  This asks the other
obvious question: "would a CONTRASTIVE objective?".  Like the MAE, it is
built to be identical to the JEPA in every way EXCEPT the objective:

  same TokenEncoder (hierarchical coarse+fine + proportion),
  same SetEncoder backbone (same class and size),
  same training budget.

THE OBJECTIVE
-------------
For a masked slot, the encoder produces a contextualized "query"
embedding.  A small projection maps it, and every ingredient's fine
embedding, into a shared space.  InfoNCE then asks: among a set of
candidate ingredients, pick the one that truly fills this slot.

  positive : the slot's TRUE ingredient
  negatives: the true ingredients of the OTHER recipes' masked slots in
             the same batch -- IN-BATCH negatives.  These are real
             ingredients appearing at their true frequency, so the task
             is "which real ingredient fits THIS context", not the
             trivial "is this a real ingredient at all".

False negatives: if two recipes in a batch mask the SAME ingredient, that
"negative" is really a positive.  The InfoNCE logits are masked to
exclude any in-batch candidate whose ingredient id equals the positive's
(except the positive's own column), so a false negative never penalises
the model.

ENERGY
------
The energy of a recipe is the mean InfoNCE loss over its masked slots --
the direct analog of the JEPA's mean latent-prediction error and the
MAE's mean reconstruction error.  All three #43 model energies are thus
"mean masked-slot error", differing only in WHICH error, which keeps the
comparison clean.  For energy we score the positive against a FIXED
candidate set (the full vocabulary) rather than in-batch negatives, so a
recipe's energy does not depend on whatever else shares its batch.

This is a BASELINE: minimal, matched to the JEPA, not tuned to win.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocktail_jepa.model.encoder import SetEncoder
from cocktail_jepa.model.tokens import TokenEncoder


@dataclass
class ContrastiveConfig:
    """Architecture for the contrastive baseline -- mirrors JEPAConfig."""
    vocab_size: int
    prop_dim: int
    d_model: int = 192
    enc_layers: int = 3
    enc_heads: int = 6
    dropout: float = 0.1
    # InfoNCE temperature -- the one contrastive-specific hyperparameter.
    # 0.1 is the common default; left fixed (not tuned) to keep the
    # baseline fair.
    temperature: float = 0.1
    proj_dim: int = 128
    # hierarchical vocabulary (#4) -- same as the JEPA
    coarse_size: int | None = None
    coarse_ids: list[int] | None = None


class CocktailContrastive(nn.Module):
    """
    InfoNCE contrastive model over cocktail recipes.

    Encoder backbone identical to the JEPA's; the head is a pair of
    projections (one for the context query, one for candidate ingredient
    embeddings) into a shared space where InfoNCE is computed.  No target
    encoder, no EMA, no predictor -- a contrastive objective needs none.
    """

    def __init__(self, cfg: ContrastiveConfig):
        super().__init__()
        self.cfg = cfg

        # shared embedding layer -- the SAME hierarchical TokenEncoder
        self.tokens = TokenEncoder(
            cfg.vocab_size, cfg.prop_dim, cfg.d_model,
            coarse_size=cfg.coarse_size, coarse_ids=cfg.coarse_ids,
        )
        # encoder backbone -- same class and size as the JEPA encoder
        self.encoder = SetEncoder(
            d_model=cfg.d_model, n_layers=cfg.enc_layers,
            n_heads=cfg.enc_heads, dropout=cfg.dropout,
        )

        # two projection heads into the shared InfoNCE space:
        #  query_proj     -- maps the masked slot's context embedding
        #  candidate_proj -- maps an ingredient's fine embedding
        self.query_proj = nn.Linear(cfg.d_model, cfg.proj_dim)
        self.candidate_proj = nn.Linear(cfg.d_model, cfg.proj_dim)

    def _candidate_embeddings(self, ingredient_ids: torch.Tensor
                              ) -> torch.Tensor:
        """
        Projected candidate embeddings for a set of ingredient ids.

        A candidate ingredient is represented by its hierarchical
        (coarse+fine) embedding -- the same representation the encoder is
        fed -- then projected.  Using the TokenEncoder's ingredient
        embedding (not a separate table) keeps the candidate side tied to
        the same learned vocabulary.
        """
        emb = self.tokens.ingredient_embedding(ingredient_ids)  # [..., d]
        return self.candidate_proj(emb)                         # [..., proj]

    def forward(self, batch: dict) -> dict:
        """
        InfoNCE masked-slot task on a batch, with in-batch negatives.

        `batch` is what JEPAMaskCollator produces (reused unchanged).
        Returns the loss dict (total + the contrastive accuracy, handy
        for monitoring).
        """
        ids = batch["ingredient_ids"]
        props = batch["proportions"]
        pad_mask = batch["pad_mask"]
        mask_index = batch["mask_index"]
        B = ids.shape[0]
        batch_idx = torch.arange(B, device=ids.device)

        # embed, mask the chosen slot, encode -> the query embedding
        tokens = self.tokens(ids, props)
        masked_tokens = self.tokens.apply_mask(tokens, mask_index)
        enc = self.encoder(masked_tokens, pad_mask)             # [B, L, d]
        query = self.query_proj(enc[batch_idx, mask_index])     # [B, proj]

        # the true ingredient of each masked slot -> the in-batch
        # candidate set.  candidate j is recipe j's true masked ingredient;
        # for recipe i the positive is candidate i.
        true_ids = ids[batch_idx, mask_index]                   # [B]
        candidates = self._candidate_embeddings(true_ids)       # [B, proj]

        # InfoNCE logits: cosine-style similarity / temperature
        q = F.normalize(query, dim=-1)
        c = F.normalize(candidates, dim=-1)
        logits = (q @ c.t()) / self.cfg.temperature             # [B, B]

        # false-negative mask: a non-diagonal candidate whose ingredient
        # id equals the positive's is really a positive -> set its logit
        # to -inf so it never contributes as a negative.  The diagonal
        # (the true positive) is always kept.
        same = true_ids.unsqueeze(0) == true_ids.unsqueeze(1)   # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=ids.device)
        false_neg = same & ~eye
        logits = logits.masked_fill(false_neg, float("-inf"))

        # the positive for row i is column i
        labels = torch.arange(B, device=ids.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()

        return {"loss": loss, "contrastive_loss": loss,
                "contrastive_acc": acc}

    def num_parameters(self) -> dict[str, int]:
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "tokens": count(self.tokens),
            "encoder": count(self.encoder),
            "query_proj": count(self.query_proj),
            "candidate_proj": count(self.candidate_proj),
            "trainable_total": count(self),
        }


def build_contrastive(vocab_size: int, prop_dim: int,
                      **overrides) -> CocktailContrastive:
    """Convenience constructor, mirroring build_jepa / build_mae."""
    cfg = ContrastiveConfig(vocab_size=vocab_size, prop_dim=prop_dim,
                            **overrides)
    return CocktailContrastive(cfg)


def load_contrastive_checkpoint(path, map_location: str = "cpu") -> dict:
    """
    Load a contrastive checkpoint and rebuild the model.

    Counterpart of mae.load_mae_checkpoint: train.checkpoint.save_checkpoint
    writes it fine (generic), but load_checkpoint rebuilds a CocktailJEPA
    and cannot load this.  Returns {"model","config","step","extra"}.
    """
    import dataclasses

    import torch as _torch

    blob = _torch.load(path, map_location=map_location, weights_only=False)
    valid = {f.name for f in dataclasses.fields(ContrastiveConfig)}
    config = {k: v for k, v in blob["config"].items() if k in valid}
    cfg = ContrastiveConfig(**config)
    model = CocktailContrastive(cfg)
    missing, unexpected = model.load_state_dict(blob["model_state"],
                                                strict=False)
    if missing:
        print(f"[contrastive checkpoint] absent keys: {list(missing)}")
    if unexpected:
        print(f"[contrastive checkpoint] unknown keys: {list(unexpected)}")
    return {
        "model": model,
        "config": cfg,
        "step": blob.get("step", 0),
        "extra": blob.get("extra", {}),
    }


# ---------------------------------------------------------------------------
# contrastive energy -- the counterpart of energy.recipe_energy
# ---------------------------------------------------------------------------

@torch.no_grad()
def recipe_energy_contrastive(
    model: CocktailContrastive,
    batch: dict,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Energy for each recipe: mean masked-slot InfoNCE loss.

    The contrastive analog of energy.recipe_energy.  For a recipe of n
    ingredients, mask each slot in turn and score the true ingredient
    against a FIXED candidate set -- the full vocabulary -- with the
    InfoNCE cross-entropy.  Using the full vocabulary (not in-batch
    negatives) makes a recipe's energy independent of its batch-mates, so
    energy is deterministic and reproducible.

    High InfoNCE loss = the encoder cannot identify the slot's true
    ingredient from context = the slot does not fit = incoherent.

    `batch` is a plain (non-masking) _stack batch.  Returns FloatTensor [B].
    """
    model.eval()
    ids = batch["ingredient_ids"].to(device)
    props = batch["proportions"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    n_ing = batch["n_ingredients"]
    B, L = ids.shape
    batch_idx = torch.arange(B, device=device)

    tokens = model.tokens(ids, props)

    # fixed candidate set: every ingredient id in the vocabulary,
    # projected once.  [vocab, proj], normalized.
    vocab_ids = torch.arange(model.cfg.vocab_size, device=device)
    cand = model._candidate_embeddings(vocab_ids)               # [vocab, proj]
    cand = F.normalize(cand, dim=-1)

    err_sum = torch.zeros(B, device=device)
    slot_count = torch.zeros(B, device=device)

    max_n = int(n_ing.max().item())
    for slot in range(max_n):
        active = (slot < n_ing).to(device)
        if not active.any():
            break

        mask_index = torch.full((B,), slot, device=device, dtype=torch.long)
        masked_tokens = model.tokens.apply_mask(tokens, mask_index)
        enc = model.encoder(masked_tokens, pad_mask)
        query = model.query_proj(enc[batch_idx, mask_index])    # [B, proj]
        q = F.normalize(query, dim=-1)

        logits = (q @ cand.t()) / model.cfg.temperature         # [B, vocab]
        true_ids = ids[batch_idx, mask_index]                   # [B]
        err = F.cross_entropy(logits, true_ids, reduction="none")  # [B]
        err_sum += err * active
        slot_count += active.float()

    return err_sum / slot_count.clamp(min=1.0)


@torch.no_grad()
def contrastive_energy_over_loader(
    model: CocktailContrastive,
    loader,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """Energy for every recipe from a DataLoader -- mirrors
    energy.energy_over_loader."""
    model.to(device)
    energies, ids = [], []
    for batch in loader:
        e = recipe_energy_contrastive(model, batch, device=device)
        energies.append(e.cpu())
        ids.extend(batch["recipe_id"])
    return torch.cat(energies), ids
