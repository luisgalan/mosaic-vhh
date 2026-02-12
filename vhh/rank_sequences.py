import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    from jaxtyping import Float, Array
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mosaic.models.boltz2 import Boltz2
    from mosaic.models.protenix import Protenix2025
    from mosaic.structure_prediction import TargetChain
    from mosaic.common import TOKENS
    import gemmi
    from vhh.utils import pdb_viewer
    from vhh.metrics import calculate_metrics
    import os

    return Protenix2025, TOKENS, TargetChain, gemmi, jax, np, os, pdb_viewer


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
def _(Protenix2025):
    # fold_model = Boltz2()
    fold_model = Protenix2025()
    return (fold_model,)


@app.cell
def _(TOKENS, TargetChain, fold_model, jax, np, os, read_pdb, target_sequence):
    # EVAL_RNG_KEY = jax.random.key(np.random.randint(34726893745))

    my_sequences = [
        "YVQLVESGGGLVQPGGSLRLSCAASGDTFRGSRDTCLGWFRQAPGQGLEAVAAIWNDENQEYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCCAWILCSSDRYYWYWQAYWGQGTLVTVS",
        "YVQLVESGGGLVQPGGSLRLSCAASGVTFRPNTQTSLGWVRQAPGQGLEWVAAITVSKNKEYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCFVLLAESSNSCYWDWLVSWGQGTLVTVS",
        "QVQLVESGGGLVQPGGSLRLSCAASEAYSRPSAHTNLGWFRQAPGQGLEAVAAIWHDGTYQYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCFRYQFSSSRDLWNDLEHAWGQGTLVTVS",
    ]

    dir = "vhh/baseline_pdbs/germinal_early_filters_only"
    pdbs = os.listdir(dir)
    sequences = list(set([read_pdb(f"{dir}/{pdb}", chain=1)[0] for pdb in pdbs]))
    # sequences = my_sequences

    for sequence in sequences:
        pssm = jax.nn.one_hot([TOKENS.index(c) for c in sequence], 20)

        features, writer = fold_model.target_only_features(
            chains=[
                # TargetChain(sequence=sequence, use_msa=True),
                TargetChain(sequence=sequence, use_msa=False),
                TargetChain(sequence=target_sequence, use_msa=True),
            ]
        )

        pred = fold_model.predict(
            PSSM=pssm,
            writer=writer,
            features=features,
            # recycling_steps=10,
            recycling_steps=1,
            key=jax.random.key(np.random.randint(10000)),
        )

        print(pred.iptm)

    # print(pred)
    # pdb_viewer(pred.st)


    # dir = "vhh/baseline_pdbs/germinal_early_filters_only"
    # pdbs = os.listdir(dir)
    # sequences = list(set([read_pdb(f"{dir}/{pdb}", chain=1)[0] for pdb in pdbs]))


    # for i, sequence in enumerate(sequences):
    #     print(f"\n--- SEQUENCE {i} ---\n")
    #     features, writer = fold_model.target_only_features(
    #         chains=[
    #             # TargetChain(sequence=sequence, use_msa=False),
    #             TargetChain(sequence=sequence, use_msa=True),
    #             TargetChain(sequence=target_sequence, use_msa=True),
    #         ]
    #     )
    #     # print(f'Features: {features}')
    #     # print(f"restype: {features['restype']}")
    #     # print(f"jaxified: {jnp.array(features['restype'])}")
    #     features["restype"] = jnp.array(features["restype"])
    #     features["profile"] = jnp.array(features["profile"])
    #
    #     pssm = jax.nn.one_hot([TOKENS.index(c) for c in sequence], 20)
    #     print(pssm.shape[0])
    #
    #     metrics = calculate_metrics(
    #         sequence,
    #         pssm,
    #         fold_model,
    #         features,
    #         EVAL_RNG_KEY,
    #     )
    #     targets = {"ipsae_min": ">0.61", "iptm": ">0.75"}
    #
    #     for k, v in metrics.items():
    #         line = f"{k}: {v:.2f}"
    #         if k in targets:
    #             line += f" (target: {targets[k]})"
    #         print(line)
    return


if __name__ == "__main__":
    app.run()
