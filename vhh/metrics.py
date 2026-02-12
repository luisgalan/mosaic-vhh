from ipymolstar import MolViewSpec
import molviewspec
import gemmi
import base64
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from mosaic.common import TOKENS
# from mosaic.losses.boltz2 import set_binder_sequence
import mosaic.losses.structure_prediction as sp

def framework_similarity(
    masked_framework_seq, pssm: Float[Array, "N 20"]
):
    framework_positions = jnp.array(
        [i for i, c in enumerate(masked_framework_seq) if c != "X"]
    )
    framework_aas = jnp.array(
        [
            TOKENS.index(c)
            for i, c in enumerate(masked_framework_seq)
            if c != "X"
        ]
    )
    framework_probs = pssm[framework_positions]  # Shape: (num_framework, 20)
    # Extract probability of correct AA at each position
    correct_probs = framework_probs[
        jnp.arange(len(framework_aas)), framework_aas
    ]
    # Average to get similarity percentage
    value = jnp.mean(correct_probs) * 100.0
    return value, {"framework_pct": value}


def sequence_sharpness(pssm: Float[Array, "N 20"]):
    # Get the probability of the argmax AA at each position
    max_probs = jnp.max(pssm, axis=-1)
    # Average over all positions
    value = jnp.mean(max_probs) * 100.0
    return value, {"sharpness_pct": value}

def calculate_metrics(masked_framework_sequence, PSSM, fold_model, features, key, num_samples=6):
    # Set binder sequence once
    try:
        from mosaic.losses.boltz2 import set_binder_sequence
        updated_features = set_binder_sequence(PSSM, features)
    except:
        from mosaic.losses.protenix import set_binder_sequence
        updated_features = set_binder_sequence(PSSM, features)

    ranking_metrics = [
        (
            "ipsae_min",
            -sp.IPSAE_min(),
            0.61,
        ),  # https://www.biorxiv.org/content/10.1101/2025.08.14.670059v2
        ("ipsae", -0.5 * (sp.BinderTargetIPSAE() + sp.TargetBinderIPSAE()), None),
        ("iptm", -sp.IPTMLoss(), 0.75),  # germinal iptm filter
        ("plddt", -sp.PLDDTLoss(), None),
        (
            "framework_pct",
            (
                lambda pssm, output, key: framework_similarity(
                    masked_framework_sequence, pssm
                )
            ),
            None,
        ),
        (
            "sharpness_pct",
            (lambda pssm, output, key: sequence_sharpness(pssm)),
            None,
        ),
    ]

    def compute_metrics_single_sample(sample_key):
        output = fold_model.model_output(
            PSSM=PSSM, features=updated_features, key=sample_key
        )
        return {
            name: metric(PSSM, output, key=sample_key)[0]
            for (name, metric, _) in ranking_metrics
        }

    # Generate keys for all samples
    sample_keys = jax.random.split(key, num_samples)

    # Use lax.map - returns {name: array([val1, val2, ...], shape=(num_samples,))}
    all_metrics = jax.lax.map(compute_metrics_single_sample, sample_keys)

    return {name: jnp.mean(all_metrics[name]) for name, _, _ in ranking_metrics}


# def log_metrics(PSSM, fold_model, features, key, prefix="", num_samples=6, wandb_run=None):
#
#     # Set binder sequence once
#     updated_features = set_binder_sequence(PSSM, features)
#
#     def compute_metrics_single_sample(sample_key):
#         output = fold_model.model_output(
#             PSSM=PSSM, features=updated_features, key=sample_key
#         )
#         return {
#             name: metric(PSSM, output, key=sample_key)[0]
#             for (name, metric, _) in ranking_metrics
#         }
#
#     # Generate keys for all samples
#     sample_keys = jax.random.split(key, num_samples)
#
#     # Use lax.map - returns {name: array([val1, val2, ...], shape=(num_samples,))}
#     all_metrics = jax.lax.map(compute_metrics_single_sample, sample_keys)
#
#     # Print averaged results
#     log_line = {}
#     for name, _, target in ranking_metrics:
#         value = jnp.mean(all_metrics[name])
#         line = f"{prefix}{name}: {value}"
#         log_line[f"{prefix}{name}"] = value
#
#         if target is not None:
#             line += f" (target: {target})"
#         print(line)
#
#     if wandb_run is not None:
#         wandb_run.log(log_line)
#
