import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import plotly.express as px
    import json
    import os
    import gemmi
    from vhh.utils import pdb_viewer
    from vhh.af3 import af3
    import marimo as mo

    return af3, gemmi, json, mo, os, pdb_viewer, px


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
def _(af3, mo, os, px, read_pdb, target_sequence):
    def get_germinal_seqs(pdb_dir):
        pdbs = os.listdir(pdb_dir)
        seqs = [read_pdb(f"{pdb_dir}/{pdb}", chain=1)[0] for pdb in pdbs]
        return list(set(seqs))


    seqs = get_germinal_seqs("vhh/baseline_pdbs/germinal_early_filters_only")
    # seqs = get_germinal_seqs("vhh/baseline_pdbs/germinal_unfiltered")

    germinal_folds = [
        af3.fold(
            binder_sequence=binder,
            target_sequence=target_sequence,
            require_cached=True,
        )
        for binder in seqs
    ]

    mo.vstack(
        [
            mo.md(f"#Germinal AF3 metrics (n={len(germinal_folds)})"),
            px.histogram(
                x=[fold.ipsae_min for fold in germinal_folds],
                title="Germinal AF3 ipsae_min",
                range_x=(0, 1),
            ),
            px.histogram(
                x=[fold.ipsae for fold in germinal_folds],
                title="Germinal AF3 ipsae",
                range_x=(0, 1),
            ),
            px.histogram(
                x=[fold.iptm for fold in germinal_folds],
                title="Germinal AF3 iptm",
                range_x=(0, 1),
            ),
        ]
    )
    return


@app.cell
def _(af3, json, mo, os, px, target_sequence):
    my_runs_dir = "results1"
    my_runs = []
    my_folds = []
    for filename in os.listdir(my_runs_dir):
        with open(f"{my_runs_dir}/{filename}", "r") as file:
            run = json.load(file)
        binder = run["binder_sequence"]
        try:
            fold = af3.fold(
                binder_sequence=binder,
                target_sequence=target_sequence,
                require_cached=True,
            )
        except FileNotFoundError:
            print(f"Warning: skipping {filename}")
        run["af3"] = fold
        my_runs.append(run)
        my_folds.append(fold)


    mo.vstack(
        [
            mo.md(f"#{my_runs_dir} AF3 metrics (n={len(my_folds)})"),
            px.histogram(
                x=[fold.ipsae_min for fold in my_folds],
                title=f"{my_runs_dir} AF3 ipsae_min",
                range_x=(0, 1),
            ),
            px.histogram(
                x=[fold.ipsae for fold in my_folds],
                title=f"{my_runs_dir} AF3 ipsae",
                range_x=(0, 1),
            ),
            px.histogram(
                x=[fold.iptm for fold in my_folds],
                title=f"{my_runs_dir} AF3 iptm",
                range_x=(0, 1),
            ),
            px.scatter(
                x=[fold.iptm for fold in my_folds],
                title=f"{my_runs_dir} AF3 iptm",
                range_x=(0, 1),
            
            )
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
