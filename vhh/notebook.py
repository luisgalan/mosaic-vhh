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
    from mosaic.optimizers import (
        simplex_APGM,
        gradient_MCMC,
    )
    import mosaic.losses.structure_prediction as sp
    from mosaic.models.boltz2 import Boltz2
    from mosaic.common import TOKENS, LossTerm
    from mosaic.losses.transformations import SoftClip
    from mosaic.structure_prediction import TargetChain
    from mosaic.models.af2 import AlphaFold2
    import ablang
    import jablang
    import gemmi
    from ipymolstar import MolViewSpec
    import molviewspec
    from mosaic.losses.ablang import AbLangPseudoLikelihood
    import base64
    from vhh.utils import compute_cdr_positions, pdb_viewer

    return (
        AbLangPseudoLikelihood,
        Array,
        Boltz2,
        Float,
        LossTerm,
        TOKENS,
        TargetChain,
        ablang,
        gemmi,
        jablang,
        jax,
        jnp,
        mo,
        np,
        pdb_viewer,
        simplex_APGM,
        sp,
    )


@app.cell
def _(jax, np):
    # RNG_KEY = jax.random.key(np.random.randint(42))
    RNG_KEY = jax.random.key(np.random.randint(100))
    return (RNG_KEY,)


@app.cell
def _(gemmi, np):
    target_structure = gemmi.read_structure('il3.pdb')
    target_sequence = gemmi.one_letter_code(
        [r.name for r in target_structure[0][0]]
    )

    # Framework sequence from protenij_vhh example - this is the same scaffold germinal uses
    masked_framework_seq = "QVQLVESGGGLVQPGGSLRLSCAASXXXXXXXXXXXLGWFRQAPGQGLEAVAAXXXXXXXXYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCXXXXXXXXXXXXXXXXXXWGQGTLVTVS"
    cdr_positions = np.array(list([i for i, aa in enumerate(masked_framework_seq) if aa == 'X']))
    framework_positions = np.array(list([i for i, aa in enumerate(masked_framework_seq) if aa != 'X']))

    binder_length = len(masked_framework_seq)
    cdr_length = len(cdr_positions)
    return (
        binder_length,
        cdr_length,
        cdr_positions,
        masked_framework_seq,
        target_sequence,
    )


@app.cell
def _(Boltz2):
    fold_model = Boltz2()
    # fold_model = AlphaFold2()
    return (fold_model,)


@app.cell
def _(
    RNG_KEY,
    TOKENS,
    TargetChain,
    binder_length,
    cdr_length,
    cdr_positions,
    fold_model,
    jax,
    masked_framework_seq,
    pdb_viewer,
    target_sequence,
):
    from numpy.ma import masked
    def predict(sequence, features, writer):
        pred = fold_model.predict(PSSM=sequence, features=features, writer=writer, key = jax.random.key(11))
        return pred, pdb_viewer(pred.st, cdr_positions=cdr_positions.tolist())

    fold_features, fold_writer = fold_model.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(sequence=target_sequence)],
    )

    init_PSSM = jax.nn.one_hot([0 if c == 'X' else TOKENS.index(c) for c in masked_framework_seq], 20)
    init_PSSM = init_PSSM.at[cdr_positions].set(
        0.5 * jax.random.gumbel(
            key=RNG_KEY,
            shape=(cdr_length, 20),
        )
    )

    pred, _viewer = predict(init_PSSM, fold_features, fold_writer)
    _viewer
    return fold_features, fold_writer, init_PSSM, predict


@app.cell
def _(ablang):
    # mpnn = ProteinMPNN.from_pretrained()
    heavy_ablang = ablang.pretrained("heavy")
    heavy_ablang.freeze()
    return (heavy_ablang,)


@app.cell
def _(
    AbLangPseudoLikelihood,
    Array,
    Float,
    LossTerm,
    TOKENS,
    binder_length,
    fold_features,
    fold_model,
    heavy_ablang,
    init_PSSM,
    jablang,
    jax,
    jnp,
    masked_framework_seq,
    np,
    simplex_APGM,
    sp,
):
    # cross-entropy loss to preserve framework residues
    def create_framework_loss(masked_framework_sequence):
        framework_positions = jnp.array([i for i, c in enumerate(masked_framework_sequence) if c != 'X'])
        framework_aas = [TOKENS.index(c) for i, c in enumerate(masked_framework_sequence) if c != 'X']
        framework_targets = jax.nn.one_hot(framework_aas, 20)

        class FrameworkCELoss(LossTerm):
            def __call__(self, pssm: Float[Array, "N 20"], key=None):
                # eps = 1e-7
                eps = 1e-5
                framework_probs = pssm[framework_positions]
                framework_probs = jnp.clip(framework_probs, eps, 1.0 - eps)
                # Sum over 20 AAs (axis=-1), then mean over framework positions
                ce = -jnp.mean(jnp.sum(framework_targets * jnp.log(framework_probs), axis=-1))
                return ce, {"framework_ce": ce}

        return FrameworkCELoss()

    ab_log_likelihood = AbLangPseudoLikelihood(
        model=jablang.from_torch(heavy_ablang.AbLang),
        tokenizer=heavy_ablang.tokenizer,
        stop_grad=True,
    )
    structure_loss = fold_model.build_loss(
        loss=sp.BinderTargetContact(
            paratope_idx=np.array(
                [
                    i for (i, c) in enumerate(masked_framework_seq) if c == "X"
                ]  # encourage binding with the CDRs rather than the framework.
            ))
            + 0.05 * sp.TargetBinderPAE()
            + 0.05 * sp.BinderTargetPAE()
            + 0.025 * sp.IPTMLoss()
            + 0.4 * sp.WithinBinderPAE()
            + 0.025 * sp.pTMEnergy()
            + 0.1 * sp.PLDDTLoss(),
        features=fold_features,
    )
    framework_loss = create_framework_loss(masked_framework_seq)

    loss = ab_log_likelihood + structure_loss + framework_loss

    print("Starting optimizer")
    _, PSSM = simplex_APGM(
        loss_function=loss,
        n_steps=75,
        x=init_PSSM,
        stepsize=0.2 * np.sqrt(binder_length),
        momentum=0.3,
    )
    PSSM, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(PSSM + 1e-5),
        n_steps=30,
        stepsize=0.5 * np.sqrt(binder_length),
        momentum=0.0,
        scale=1.4,
        logspace=True,
        max_gradient_norm=1.0,
    )
    return (PSSM,)


@app.cell
def _(Array, Float, TOKENS, fold_model, jnp, masked_framework_seq, sp):
    def framework_similarity(masked_framework_seq, pssm: Float[Array, "N 20"]) -> float:
        framework_positions = jnp.array([i for i, c in enumerate(masked_framework_seq) if c != 'X'])
        framework_aas = jnp.array([TOKENS.index(c) for i, c in enumerate(masked_framework_seq) if c != 'X'])
        framework_probs = pssm[framework_positions]  # Shape: (num_framework, 20)
        # Extract probability of correct AA at each position
        correct_probs = framework_probs[jnp.arange(len(framework_aas)), framework_aas]
        # Average to get similarity percentage
        value = jnp.mean(correct_probs) * 100.0
        return value, {'framework_pct': value}

    def sequence_sharpness(pssm: Float[Array, "N 20"]) -> float:
        # Get the probability of the argmax AA at each position
        max_probs = jnp.max(pssm, axis=-1)
        # Average over all positions
        value = jnp.mean(max_probs) * 100.0
        return value, {'sharpness_pct': value}

    ranking_metrics = [
        ('ipsae_min', -sp.IPSAE_min(), 0.61), # https://www.biorxiv.org/content/10.1101/2025.08.14.670059v2
        ('ipsae', -0.5 * (sp.BinderTargetIPSAE() + sp.TargetBinderIPSAE()), None),
        ('iptm', -sp.IPTMLoss(), 0.75), # germinal iptm filter
        ('framework_pct', (lambda pssm, output, key: framework_similarity(masked_framework_seq, pssm)), None),
        ('sharpness_pct', (lambda pssm, output, key: sequence_sharpness(pssm)), None),
    ]

    def log_metrics(PSSM, features, key, prefix=''):
        output = fold_model.model_output(PSSM=PSSM, features=features, key=key)
        for (name, metric, target) in ranking_metrics:
            value, aux = metric(PSSM, output, key=key)
            line = f'{prefix}{name}: {value}'
            if target is not None:
                line += f' (target: {target})'
            print(line)

    return (log_metrics,)


@app.cell
def _(PSSM, RNG_KEY, fold_features, fold_writer, log_metrics, predict):
    print("Predicting structure of relaxed sequence...")

    log_metrics(PSSM, fold_features, RNG_KEY, prefix='relaxed_')

    _o, _viewer = predict(
        PSSM, fold_features, fold_writer
    )

    _viewer
    return


@app.cell
def _(
    PSSM,
    RNG_KEY,
    TOKENS,
    TargetChain,
    fold_model,
    jax,
    log_metrics,
    predict,
    target_sequence,
):
    # Compute final sequence
    final_PSSM = jax.nn.one_hot(PSSM.argmax(-1), 20)
    final_seq = "".join(TOKENS[i] for i in PSSM.argmax(-1))
    print(final_seq)

    # Repredict
    final_features, final_writer = fold_model.target_only_features(
        chains=[
            TargetChain(sequence=final_seq, use_msa=False),
            TargetChain(sequence=target_sequence, use_msa=True),
        ]
    )

    log_metrics(final_PSSM, final_features, RNG_KEY, prefix='argmax_')

    print("Predicting structure...")
    final_structure, _viewer = predict(
        PSSM, final_features, final_writer
    )
    _viewer
    return (final_structure,)


@app.cell
def _(final_structure, mo):
    mo.download(data=final_structure.st.make_pdb_string(), filename="vhh.pdb")
    return


if __name__ == "__main__":
    app.run()
