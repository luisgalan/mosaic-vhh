import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import json
    import os
    import gemmi
    from vhh.utils import pdb_viewer
    from vhh.af3 import af3
    import marimo as mo
    from scipy import stats

    return af3, gemmi, json, mo, os, pdb_viewer, px, stats


@app.cell
def _(gemmi, pdb_viewer):
    def read_pdb(path, chain=0):
        structure = gemmi.read_structure(path)
        sequence = gemmi.one_letter_code([r.name for r in structure[0][chain]])
        return sequence, structure


    target_sequence, target_structure = read_pdb("il3.pdb")
    print(target_sequence)
    pdb_viewer(target_structure)
    return read_pdb, target_sequence


@app.cell
def _(af3, target_sequence):
    HISTOGRAM_KWARGS = {"nbins": 30, "range_x": (-0.02, 1.02)}
    SCATTER_KWARGS = {
        "range_x": (-0.02, 1.02),
        "range_y": (-0.02, 1.02),
        "trendline": "ols",
    }
    ONLY_USE_CACHED_AF3_PREDS = True


    def get_af3_fold(binder_sequence):
        try:
            return af3.fold(
                binder_sequence=binder_sequence,
                target_sequence=target_sequence,
                require_cached=ONLY_USE_CACHED_AF3_PREDS,
            )
        except Exception as e:
            print(f"Warning: skipping binder: {e}")
            return None

    return HISTOGRAM_KWARGS, SCATTER_KWARGS, get_af3_fold


@app.cell
def _(HISTOGRAM_KWARGS, get_af3_fold, mo, os, px, read_pdb):
    def get_germinal_seqs(pdb_dir):
        pdbs = os.listdir(pdb_dir)
        print(pdbs)
        seqs = [read_pdb(f"{pdb_dir}/{pdb}", chain=1)[0] for pdb in pdbs]
        print(seqs)
        return list(set(seqs))


    seqs = get_germinal_seqs("pdbs/baseline_pdbs/germinal_new/structures")
    # seqs = get_germinal_seqs("pdbs/baseline_pdbs/germinal_early_filters_only")
    # seqs = get_germinal_seqs("vhh/baseline_pdbs/germinal_unfiltered")

    germinal_folds = [get_af3_fold(binder) for binder in seqs]
    germinal_folds = [fold for fold in germinal_folds if fold is not None]

    mo.vstack(
        [
            mo.md(f"#Germinal AF3 metrics (n={len(germinal_folds)})"),
            px.histogram(
                x=[fold.ipsae_min for fold in germinal_folds],
                title="Germinal AF3 ipsae_min",
                **HISTOGRAM_KWARGS,
            ),
            px.histogram(
                x=[fold.ipsae for fold in germinal_folds],
                title="Germinal AF3 ipsae",
                **HISTOGRAM_KWARGS,
            ),
            px.histogram(
                x=[fold.iptm for fold in germinal_folds],
                title="Germinal AF3 iptm",
                **HISTOGRAM_KWARGS,
            ),
        ]
    )
    return (germinal_folds,)


@app.cell
def _(HISTOGRAM_KWARGS, SCATTER_KWARGS, get_af3_fold, json, mo, os, px, stats):
    my_runs_dir = "results/abmpnn_test6"
    # my_runs_dir = "results/results1"
    my_runs = []
    my_folds = []
    for filename in os.listdir(my_runs_dir):
        with open(f"{my_runs_dir}/{filename}", "r") as file:
            run = json.load(file)
        binder = run["binder_sequence"]
        fold = get_af3_fold(binder)
        run["af3"] = fold
        my_runs.append(run)
        my_folds.append(fold)


    def scatter(x, y, **kwargs):
        r_spearman, p_spearman = stats.spearmanr(x, y)
        fig = px.scatter(x=x, y=y, **(SCATTER_KWARGS | kwargs))
        fig.add_annotation(
            text=f"spearman: {r_spearman:.3f}, p-value = {p_spearman:.3g}",
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.95,
            showarrow=False,
        )
        return fig


    mo.vstack(
        [
            mo.md(f"#{my_runs_dir} AF3 metrics (n={len(my_folds)})"),
            scatter(
                x=[run["metadata"]["final_loss"] for run in my_runs],
                y=[fold.iptm for fold in my_folds],
                title=f"{my_runs_dir} loss vs af3 ipTM",
                labels={"x": "Loss (lower is better)", "y": "AF3 ipTM"},
                range_x=None,
            ),
            scatter(
                x=[run["metadata"]["metrics"]["iptm"] for run in my_runs],
                y=[fold.iptm for fold in my_folds],
                title=f"{my_runs_dir} boltz2 vs af3 ipTM",
                labels={"x": "Boltz2 ipTM", "y": "AF3 ipTM"},
            ),
            scatter(
                x=[fold.iptm for fold in my_folds],
                y=[fold.ipsae for fold in my_folds],
                title=f"{my_runs_dir} AF3 ipTM / ipSAE",
                labels={"x": "AF3 ipTM", "y": "AF3 ipSAE"},
            ),
            px.histogram(
                x=[run["metadata"]["pssm_sharpness"] for run in my_runs],
                title=f"{my_runs_dir} PSSM sharpness (%)",
                **(HISTOGRAM_KWARGS | {"range_x": (95, 100.1)}),
            ),
            px.histogram(
                x=[fold.ipsae_min for fold in my_folds],
                title=f"{my_runs_dir} AF3 ipSAE_min",
                **HISTOGRAM_KWARGS,
            ),
            px.histogram(
                x=[fold.ipsae for fold in my_folds],
                title=f"{my_runs_dir} AF3 ipSAE",
                **HISTOGRAM_KWARGS,
            ),
            px.histogram(
                x=[fold.iptm for fold in my_folds],
                title=f"{my_runs_dir} AF3 ipTM",
                **HISTOGRAM_KWARGS,
            ),
        ]
    )
    return my_folds, my_runs_dir


@app.cell
def _(germinal_folds, my_folds, my_runs_dir, px):
    ys_germinal = [fold.iptm for fold in germinal_folds]
    ys_other = [fold.iptm for fold in my_folds]

    labels = [f"Germinal (n={len(ys_germinal)})"] * len(ys_germinal) + [
        f"{my_runs_dir} (n={len(ys_other)})"
    ] * len(ys_other)

    fig = px.violin(
        x=labels,
        y=ys_germinal + ys_other,
        color=labels,
        labels={"x": "Group", "y": "ipTM"},
    )
    fig.update_traces(scalemode="count")
    fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
