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
    from mosaic.models.boltz2 import Boltz2, Boltz2Loss
    from mosaic.models.protenix import Protenix2025
    from mosaic.common import TOKENS, LossTerm
    from mosaic.losses.transformations import SoftClip
    from mosaic.structure_prediction import TargetChain
    from mosaic.models.af2 import AlphaFold2
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    import ablang
    import jablang
    import gemmi
    from ipymolstar import MolViewSpec
    import molviewspec
    from mosaic.losses.ablang import AbLangPseudoLikelihood
    import base64
    from vhh.utils import pdb_viewer
    from vhh.metrics import calculate_metrics, sequence_sharpness
    import optax
    import equinox as eqx
    import wandb

    return (
        AbLangPseudoLikelihood,
        Array,
        Boltz2,
        Boltz2Loss,
        Float,
        LossTerm,
        ProteinMPNN,
        TOKENS,
        TargetChain,
        ablang,
        calculate_metrics,
        eqx,
        gemmi,
        jablang,
        jax,
        jnp,
        np,
        optax,
        pdb_viewer,
        sequence_sharpness,
        sp,
        wandb,
    )


@app.cell
def _(jax, np):
    # RNG_KEY = jax.random.key(np.random.randint(42))
    RNG_KEY = jax.random.key(np.random.randint(100))
    return (RNG_KEY,)


@app.cell
def _(gemmi, np):
    target_structure = gemmi.read_structure("il3.pdb")
    target_sequence = gemmi.one_letter_code(
        [r.name for r in target_structure[0][0]]
    )

    # Framework sequence from protenij_vhh example - this is the same scaffold germinal uses
    masked_framework_seq = "QVQLVESGGGLVQPGGSLRLSCAASXXXXXXXXXXXLGWFRQAPGQGLEAVAAXXXXXXXXYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCXXXXXXXXXXXXXXXXXXWGQGTLVTVS"
    cdr_positions = np.array(
        list([i for i, aa in enumerate(masked_framework_seq) if aa == "X"])
    )
    framework_positions = np.array(
        list([i for i, aa in enumerate(masked_framework_seq) if aa != "X"])
    )

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
    def predict(sequence, features, writer, key):
        pred = fold_model.predict(
            PSSM=sequence, features=features, writer=writer, key=key
        )
        return pred, pdb_viewer(pred.st, cdr_positions=cdr_positions.tolist())


    fold_features, fold_writer = fold_model.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(sequence=target_sequence, use_msa=True)],
    )

    # fold_features, fold_writer = fold_model.target_only_features(
    #     chains=[
    #         TargetChain(sequence=masked_framework_seq, use_msa=False),
    #         TargetChain(sequence=target_sequence, use_msa=True),
    #     ]
    # )

    init_PSSM = jax.nn.one_hot(
        [0 if c == "X" else TOKENS.index(c) for c in masked_framework_seq], 20
    )
    init_PSSM = init_PSSM.at[cdr_positions].set(
        0.5
        * jax.random.gumbel(
            key=RNG_KEY,
            shape=(cdr_length, 20),
        )
    )

    pred, _viewer = predict(init_PSSM, fold_features, fold_writer, key=RNG_KEY)
    _viewer
    return fold_features, init_PSSM, predict


@app.cell
def _(ProteinMPNN, ablang):
    mpnn = ProteinMPNN.from_pretrained()
    heavy_ablang = ablang.pretrained("heavy")
    heavy_ablang.freeze()
    return (heavy_ablang,)


@app.cell
def _(
    AbLangPseudoLikelihood,
    Array,
    Boltz2Loss,
    Float,
    LossTerm,
    TOKENS,
    fold_features,
    fold_model,
    heavy_ablang,
    jablang,
    jax,
    jnp,
    masked_framework_seq,
    np,
    sp,
):
    # cross-entropy loss to preserve framework residues
    def create_framework_loss(masked_framework_sequence):
        framework_positions = jnp.array(
            [i for i, c in enumerate(masked_framework_sequence) if c != "X"]
        )
        framework_aas = [
            TOKENS.index(c)
            for i, c in enumerate(masked_framework_sequence)
            if c != "X"
        ]
        framework_targets = jax.nn.one_hot(framework_aas, 20)

        class FrameworkCELoss(LossTerm):
            def __call__(self, pssm: Float[Array, "N 20"], key=None):
                # eps = 1e-7
                eps = 1e-5
                framework_probs = pssm[framework_positions]
                framework_probs = jnp.clip(framework_probs, eps, 1.0 - eps)
                # Sum over 20 AAs (axis=-1), then mean over framework positions
                ce = -jnp.mean(
                    jnp.sum(framework_targets * jnp.log(framework_probs), axis=-1)
                )
                return ce, {"framework_ce": ce}

        return FrameworkCELoss()


    ab_log_likelihood = AbLangPseudoLikelihood(
        model=jablang.from_torch(heavy_ablang.AbLang),
        tokenizer=heavy_ablang.tokenizer,
        stop_grad=True,
    )

    sp_loss = (
        sp.BinderTargetContact(
            paratope_idx=np.array(
                [
                    i for (i, c) in enumerate(masked_framework_seq) if c == "X"
                ]  # encourage binding with the CDRs rather than the framework.
            )
        )
        # + 0.5 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.001))
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
    )

    structure_loss = Boltz2Loss(
        joltz2=fold_model.model,
        features=fold_features,
        recycling_steps=1,
        sampling_steps=25,
        # sampling_steps=17,
        loss=sp_loss,
        deterministic=False,
    )

    # structure_loss = fold_model.build_loss(
    #     loss=sp_loss,
    #     features=fold_features,
    # )

    framework_loss = create_framework_loss(masked_framework_seq)

    loss = ab_log_likelihood + structure_loss + framework_loss
    # loss = 0.2 * ab_log_likelihood + structure_loss + framework_loss
    return (loss,)


@app.cell
def _(Array, Float, eqx, init_PSSM, jax, jnp, loss, np, optax, wandb):
    @eqx.filter_jit
    def _compute_loss_fn(loss_function, logits, key, eps, entropy_weight):
        probs = jax.nn.softmax(logits)
        value, aux = loss_function(probs, key=key)
        entropy = -jnp.mean(jnp.sum(probs * jnp.log(probs + eps), axis=-1))
        return value + entropy_weight * entropy, (value, entropy, aux)


    def adam_logit_optimizer(
        *,
        loss_function,
        x: Float[Array, "N 20"],
        n_steps: int,
        lr: float,
        entropy_weight: float,
        entropy_warmup_steps: int = 0,
        lr_end: float = None,
        clip_grads: float = 1.0,
        key=None,
        wandb_run=None,
    ):
        if key is None:
            key = jax.random.key(np.random.randint(0, 10000))

        eps = 1e-7
        logits = jnp.log(jnp.clip(x, eps, 1.0 - eps))

        if lr_end is not None:
            # Linear decay schedule
            schedule = optax.linear_schedule(
                init_value=lr, end_value=lr_end, transition_steps=n_steps
            )
        else:
            # Constant lr
            schedule = lr

        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_grads),
            # optax.adam(schedule, b1=0.95)
            optax.adam(schedule),
        )
        opt_state = optimizer.init(logits)

        best_total_loss = np.inf
        best_logits = logits

        initial_temp = 0.5
        final_temp = 0.001

        for i in range(n_steps):
            key, noise_key, loss_key = jax.random.split(key, 3)

            temp = initial_temp * (final_temp / initial_temp) ** (i / n_steps)
            noisy_logits = logits + temp * jax.random.normal(
                noise_key, shape=logits.shape
            )

            # cur_entropy_weight = jnp.array((i / n_steps) * entropy_weight)
            if i < entropy_warmup_steps:
                w_ent = jnp.array(0.0)
            else:
                progress = (i - entropy_warmup_steps) / (
                    n_steps - entropy_warmup_steps
                )
                w_ent = entropy_weight * (1 - jnp.cos(jnp.pi * progress)) / 2

            (value, (loss, seq_entropy, aux)), grads = eqx.filter_value_and_grad(
                # lambda l, k: _compute_loss_fn(loss_function, l, k, eps, entropy_weight),
                lambda l, k: _compute_loss_fn(loss_function, l, k, eps, w_ent),
                has_aux=True,
            )(logits, loss_key)
            # )(noisy_logits, loss_key)

            total_loss = loss + entropy_weight * seq_entropy

            metrics = {}

            for k, v in jax.tree_util.tree_leaves_with_path(aux):
                name = jax.tree_util.keystr(k, simple=True, separator=".")
                dict_keys = [
                    key.key for key in k if isinstance(key, jax.tree_util.DictKey)
                ]
                name = ".".join(dict_keys)
                metrics[name] = v

            metrics["loss"] = loss
            metrics["seq_entropy"] = seq_entropy
            metrics["total_loss"] = total_loss
            metrics["temperature"] = temp

            updates, opt_state = optimizer.update(grads, opt_state)
            logits = optax.apply_updates(logits, updates)

            if total_loss < best_total_loss:
                best_total_loss = total_loss
                best_logits = logits

            metrics["best_total_loss"] = best_total_loss
            metrics["grad_norm"] = jnp.linalg.norm(grads)

            print(
                f"{i} loss: {loss:.2f} seq_entropy: {seq_entropy:.2f} -- {'  '.join(f'{k}: {v:.2f}' for k, v in metrics.items())}"
            )
            if wandb_run is not None:
                wandb_run.log(metrics)

            key = jax.random.fold_in(key, 0)

        return jax.nn.softmax(logits), jax.nn.softmax(best_logits)


    wandb_run = wandb.init(
        project="mosaic-vhh",
        settings=wandb.Settings(code_dir="."),
    )
    wandb_run.log_code("./vhh")

    _, PSSM = adam_logit_optimizer(
        loss_function=loss,
        x=init_PSSM,
        n_steps=75,
        lr=1.0,
        entropy_weight=20.0,
        entropy_warmup_steps=30,
        # clip_grads=1.0,
        clip_grads=0.2,
        wandb_run=wandb_run,
    )
    return PSSM, wandb_run


@app.cell
def _(
    PSSM,
    TOKENS,
    TargetChain,
    calculate_metrics,
    fold_model,
    jax,
    masked_framework_seq,
    np,
    predict,
    sequence_sharpness,
    target_sequence,
    wandb,
    wandb_run,
):
    EVAL_RNG_KEY = jax.random.key(np.random.randint(34726893745))

    # Compute final sequence
    final_PSSM = jax.nn.one_hot(PSSM.argmax(-1), 20)
    final_seq = "".join(TOKENS[i] for i in PSSM.argmax(-1))
    print(final_seq)
    print(f"Sharpness: {float(sequence_sharpness(pssm=PSSM)[0]):.2f}%")

    # Repredict
    # final_features, final_writer = fold_features, fold_writer
    final_features, final_writer = fold_model.target_only_features(
        chains=[
            TargetChain(sequence=final_seq, use_msa=False),
            # TargetChain(sequence=final_seq, use_msa=True),
            TargetChain(sequence=target_sequence, use_msa=True),
        ]
    )

    metrics = calculate_metrics(
        masked_framework_seq,
        final_PSSM,
        fold_model,
        final_features,
        EVAL_RNG_KEY,
    )
    targets = {"ipsae_min": ">0.61", "iptm": ">0.75"}


    for k, v in metrics.items():
        line = f"{k}: {v:.2f}"
        if k in targets:
            line += f" (target: {targets[k]})"
        print(line)

    wandb.log({f"argmax_{k}": v for k, v in metrics.items()})

    print("Predicting structure...")
    _o, _viewer = predict(PSSM, final_features, final_writer, key=EVAL_RNG_KEY)

    wandb_run.finish()

    _viewer
    return


if __name__ == "__main__":
    app.run()
