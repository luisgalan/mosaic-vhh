import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mosaic.optimizers import (
        simplex_APGM,
        gradient_MCMC,
    )
    import mosaic.losses.structure_prediction as sp
    from mosaic.models.boltz1 import Boltz1
    from mosaic.models.boltz2 import Boltz2

    from mosaic.common import TOKENS
    from mosaic.losses.transformations import SoftClip
    from jaxtyping import Float, Array
    from mosaic.common import LossTerm
    from mosaic.structure_prediction import TargetChain
    from mosaic.models.af2 import AlphaFold2
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    from mosaic.losses.protein_mpnn import FixedStructureInverseFoldingLL, InverseFoldingSequenceRecovery
    import ablang
    import jablang
    import gemmi
    from ipymolstar import MolViewSpec
    import molviewspec
    from mosaic.losses.ablang import AbLangPseudoLikelihood
    import requests
    import base64
    return (
        AbLangPseudoLikelihood,
        Boltz2,
        MolViewSpec,
        TOKENS,
        TargetChain,
        ablang,
        base64,
        gemmi,
        jablang,
        jax,
        molviewspec,
        np,
        plt,
        simplex_APGM,
        sp,
    )


@app.cell
def _(MolViewSpec, base64, gemmi, molviewspec):
    # IL3 pdb to fasta
    target_sequence = "KTSWVNCSNMIDEIITHLKQPPLPLLDFNNLNGEDQDILMENNLRRPNLEAFNRAVKSLQNASAIESILKNLLPCLPLATAAPTRHPIHIKDGDWNEFRRKLTFYLKTLENA"

    # Germinal nanobody scaffold pdb to fasta
    scaffold_sequence = "QVQLVESGGGLVQPGGSLRLSCAASGGSEYSYSTFSLGWFRQAPGQGLEAVAAIASMGGLTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAAVRGYFMRLPSSHNFRYWGQGTLVTVSSRGR"
    binder_length = len(scaffold_sequence)

    # From germinal nanobody config
    cdr_lengths = [11, 8, 18]
    fw_lengths = [25, 17, 38, 14]

    # germinal/utils/utils.py
    def compute_cdr_positions(
        cdr_lengths: list[int], framework_lengths: list[int]
    ) -> list[int]:
        cumulative = 0
        positions = []
        for i, cdr_length in enumerate(cdr_lengths):
            fw_len = framework_lengths[i] + cumulative
            positions.extend(range(fw_len, fw_len + cdr_length))
            cumulative = fw_len + cdr_length
        return positions


    cdr_positions = compute_cdr_positions(
        cdr_lengths, fw_lengths
    )

    def pdb_viewer(st: gemmi.Structure):
        # Get PDB string uri
        pdb_str = st.make_pdb_string()
        pdb_bytes = pdb_str.encode('utf-8')
        data_uri = f"data:text/plain;base64,{base64.b64encode(pdb_bytes).decode('utf-8')}"

        # Create molviewspec scene
        builder = molviewspec.create_builder()
        structure_node = builder.download(url=data_uri).parse(format="pdb").model_structure()

        structure_repr = structure_node.component().representation(type='cartoon')
        structure_repr.color(custom={"molstar_color_theme_name": "plddt-confidence"})

        cdr_component = structure_node.component(
            selector=[{'auth_seq_id': pos, 'label_asym_id': 'A'} for pos in cdr_positions]
        )
        cdr_repr = cdr_component.representation(type='ball_and_stick', size_factor=0.5)
        cdr_repr.color(custom={"molstar_color_theme_name": "element-symbol"})

        # Return a viewer widget for the scene
        viewer = MolViewSpec()
        viewer.msvj_data = builder.get_state().dumps()

        return viewer
    return binder_length, pdb_viewer, scaffold_sequence, target_sequence


@app.cell
def _(Boltz2):
    fold_model = Boltz2()
    # fold_model = AlphaFold2()
    return (fold_model,)


@app.cell
def _(
    TOKENS,
    TargetChain,
    binder_length,
    fold_model,
    jax,
    pdb_viewer,
    scaffold_sequence,
    target_sequence,
):
    def predict(sequence, features, writer):
        pred = fold_model.predict(PSSM=sequence, features=features, writer=writer, key = jax.random.key(11))
        return pred, pdb_viewer(pred.st)
        # return pred, mosaic_pdb_viewer(pred.st)

    fold_features, fold_writer = fold_model.binder_features(
        binder_length=binder_length,
        # chains=[TargetChain(sequence=target_sequence, use_msa=False)],
        chains=[TargetChain(sequence=target_sequence)],
    )

    scaffold_PSSM=jax.nn.one_hot([TOKENS.index(c) for c in scaffold_sequence], 20)

    pred, _viewer = predict(scaffold_PSSM, fold_features, fold_writer)
    _viewer
    return fold_features, fold_writer, pred, predict


@app.cell
def _(plt, pred):
    plt.imshow(pred.pae)
    return


@app.cell
def _(ablang):
    # mpnn = ProteinMPNN.from_pretrained()
    heavy_ablang = ablang.pretrained("heavy")
    heavy_ablang.freeze()
    return (heavy_ablang,)


@app.cell
def _(
    AbLangPseudoLikelihood,
    binder_length,
    fold_features,
    fold_model,
    heavy_ablang,
    jablang,
    jax,
    np,
    simplex_APGM,
    sp,
):
    print("Making loss function...")
    ab_log_likelihood = AbLangPseudoLikelihood(
        model=jablang.from_torch(heavy_ablang.AbLang),
        tokenizer=heavy_ablang.tokenizer,
        stop_grad=True,
    )
    structure_loss = fold_model.build_loss(
        loss=2 * sp.BinderTargetContact()
        + sp.WithinBinderContact(),
        # + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.01)),
        features=fold_features,
    )
    loss = 25.0 * ab_log_likelihood + structure_loss

    print("Starting optimizer")
    _, PSSM = simplex_APGM(
        loss_function=loss,
        n_steps=75,
        x=jax.nn.softmax(
            0.5*jax.random.gumbel(
                key=jax.random.key(np.random.randint(100000)),
                shape=(binder_length, 20),
            )
        ),
        stepsize=0.1,
        momentum=0.0,
    )
    return PSSM, loss


@app.cell
def _(PSSM, fold_features, fold_writer, predict):
    print("Predicting structure...")
    _o, _viewer = predict(
        PSSM, fold_features, fold_writer
    )
    _viewer
    return


@app.cell
def _(PSSM, loss, simplex_APGM):
    print("Starting optimizer...")
    PSSM_sharper, _ = simplex_APGM(
        loss_function=loss,
        n_steps=50,
        x=PSSM,
        stepsize = 0.5,
        scale = 1.5,
        momentum=0.0
    )
    return (PSSM_sharper,)


@app.cell
def _(PSSM_sharper, fold_features, fold_writer, predict):
    print("Predicting structure...")
    _o, _viewer = predict(
        PSSM_sharper, fold_features, fold_writer
    )
    _viewer
    return


if __name__ == "__main__":
    app.run()
