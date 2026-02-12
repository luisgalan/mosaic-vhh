from ipymolstar import MolViewSpec
import molviewspec
import gemmi
import base64
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from mosaic.common import TOKENS
from mosaic.losses.boltz2 import set_binder_sequence

# def framework_similarity(
#     masked_framework_seq, pssm: Float[Array, "N 20"]
# ):
#     framework_positions = jnp.array(
#         [i for i, c in enumerate(masked_framework_seq) if c != "X"]
#     )
#     framework_aas = jnp.array(
#         [
#             TOKENS.index(c)
#             for i, c in enumerate(masked_framework_seq)
#             if c != "X"
#         ]
#     )
#     framework_probs = pssm[framework_positions]  # Shape: (num_framework, 20)
#     # Extract probability of correct AA at each position
#     correct_probs = framework_probs[
#         jnp.arange(len(framework_aas)), framework_aas
#     ]
#     # Average to get similarity percentage
#     value = jnp.mean(correct_probs) * 100.0
#     return value, {"framework_pct": value}
#
#
# def sequence_sharpness(pssm: Float[Array, "N 20"]):
#     # Get the probability of the argmax AA at each position
#     max_probs = jnp.max(pssm, axis=-1)
#     # Average over all positions
#     value = jnp.mean(max_probs) * 100.0
#     return value, {"sharpness_pct": value}

def pdb_viewer(st: gemmi.Structure, cdr_positions: list[int] | None = None):
    """
    Create a PDB viewer widget for a given structure, highlighting its CDR positions.

    Args:
        st: The structure to view.
        cdr_positions: The positions of the CDRs in the structure.

    Returns:
        A viewer widget for the structure.
    """
    # Get PDB string uri
    pdb_str = st.make_pdb_string()
    pdb_bytes = pdb_str.encode('utf-8')
    data_uri = f"data:text/plain;base64,{base64.b64encode(pdb_bytes).decode('utf-8')}"

    # Create molviewspec scene
    builder = molviewspec.create_builder()
    structure_node = builder.download(url=data_uri).parse(format="pdb").model_structure()

    structure_repr = structure_node.component().representation(type='cartoon')
    structure_repr.color(custom={"molstar_color_theme_name": "plddt-confidence"})

    if cdr_positions is not None:
        cdr_component = structure_node.component(
            selector=[{'auth_seq_id': pos, 'label_asym_id': 'A'} for pos in cdr_positions]
        )
        cdr_repr = cdr_component.representation(type='ball_and_stick', size_factor=0.5)
        cdr_repr.color(custom={"molstar_color_theme_name": "element-symbol"})

    # Return a viewer widget for the scene
    viewer = MolViewSpec()
    viewer.msvj_data = builder.get_state().dumps()

    return viewer
