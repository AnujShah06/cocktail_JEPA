import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    from cocktail_jepa.config import CONFIG
    return CONFIG, json, mo


@app.cell
def _(mo):
    mo.md(
        """
        # Cocktail JEPA -- corpus exploration

        This notebook is the cockpit for **Stage 1**: looking at the data
        before any encoding code is written. marimo notebooks are plain
        `.py` files, so this versions in git like real code.

        Nothing here trains anything -- it's for *seeing* the corpus.
        """
    )
    return


@app.cell
def _(CONFIG, json):
    # load the corpus
    recipes = [
        json.loads(line)
        for line in open(CONFIG.paths.recipes, encoding="utf-8")
    ]
    vocab = json.load(open(CONFIG.paths.vocabulary, encoding="utf-8"))
    len(recipes), vocab["size"]
    return recipes, vocab


@app.cell
def _(mo, recipes):
    sizes = [r["n_ingredients"] for r in recipes]
    mo.md(
        f"""
        **Corpus loaded:** {len(recipes)} recipes
        Recipe size -- min {min(sizes)}, mean {sum(sizes)/len(sizes):.1f},
        max {max(sizes)}
        """
    )
    return (sizes,)


@app.cell
def _(mo, recipes):
    # browse a single recipe
    idx = mo.ui.slider(0, len(recipes) - 1, value=0, label="recipe index")
    idx
    return (idx,)


@app.cell
def _(idx, mo, recipes):
    r = recipes[idx.value]
    rows = "\n".join(
        f"- {i['ingredient']}  ({i['category']})  "
        f"prop={i['proportion']}"
        for i in r["ingredients"]
    )
    mo.md(f"### {r['name']}  \n*source: {r['source']}*\n\n{rows}")
    return r, rows


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ### Next

        Stage 1 proper: build the ingredient embedding table, the Fourier
        proportion encoding, the leakage-controlled splits, and the
        perturbation set. Those go in `src/cocktail_jepa/data/`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
