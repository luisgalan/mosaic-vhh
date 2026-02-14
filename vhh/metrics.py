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
import numpy as np
import equinox as eqx

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
    return jnp.mean(correct_probs) * 100.0


def sequence_sharpness(pssm: Float[Array, "N 20"]):
    # Get the probability of the argmax AA at each position
    max_probs = jnp.max(pssm, axis=-1)
    # Average over all positions
    return jnp.mean(max_probs) * 100.0

def calculate_metrics(PSSM, fold_model, features, key, num_samples=6):
    dummy_loss = eqx.filter_jit(fold_model.build_loss(
        loss=sp.IPSAE_min()
        + sp.BinderTargetIPSAE()
        + sp.TargetBinderIPSAE()
        + sp.IPTMLoss()
        + sp.PLDDTLoss(),
        features=features
    ))

    # Generate keys for all samples
    sample_keys = jax.random.split(key, num_samples)
    all_results = []
    for sk in sample_keys:
        # Just make a loss function with all the metrics we care about, then extract the metrics from aux.
        # Not pretty, but seems to be the easiest way to do it
        _, aux = dummy_loss(PSSM, key=sk)
        model_name, metrics = [(k, v) for k, v in aux.items() if isinstance(v, list)][0]
        # Flatten list of dicts into dict
        metrics = {k: v for d in metrics for k, v in d.items()}
        all_results.append({
            f'{model_name}.ipsae_min': metrics['ipsae_min'],
            f'{model_name}.ipsae': 0.5 * (metrics['bt_ipsae'] + metrics['tb_ipsae']),
            f'{model_name}.iptm': metrics['iptm'],
            f'{model_name}.plddt': metrics['plddt']
        })
    return {name: np.mean([r[name] for r in all_results]) for name in all_results[0]}
