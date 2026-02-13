import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from vhh.af3 import af3
    import os
    from vhh.utils import pdb_viewer
    import gemmi

    return af3, gemmi, os, pdb_viewer


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
def _(af3, os, read_pdb, target_sequence):
    dir = "vhh/baseline_pdbs/germinal_early_filters_only"
    pdbs = os.listdir(dir)
    sequences = list(set([read_pdb(f"{dir}/{pdb}", chain=1)[0] for pdb in pdbs]))

    for sequence in sequences:
        output = af3.get_fold(
            binder_sequence=sequence, target_sequence=target_sequence
        )
        print("af3 iptm:", output.iptm)
        print("af3 ipsae:", output.ipsae)
        print("af3 ipsae_min:", output.ipsae_min)
    return


if __name__ == "__main__":
    app.run()
