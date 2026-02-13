from ipymolstar import MolViewSpec
import molviewspec
import gemmi
import base64
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from mosaic.common import TOKENS
from mosaic.losses.boltz2 import set_binder_sequence
import time
import os
import json
import wandb
import hashlib

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

def save_run_metadata(output_dir: str, binder_sequence: str, target_sequence: str, notebook_path: str | None=None, wandb_run=None, metadata: dict = {}):
    # Get wandb run metadata
    wandb_run_id = wandb_run.id if wandb_run else None
    wandb_run_url = wandb_run.url if wandb_run else None
    wandb_run_logs = None
    if wandb_run:
        try:
            api_run = wandb.Api().run(f"{wandb_run.entity}/{wandb_run.project}/{wandb_run.id}")
            wandb_run_logs = api_run.history().to_dict(orient="records")
        except Exception as e:
            print(f"Failed to fetch wandb run logs: {e}")

    notebook_hash = None
    if notebook_path:
        try:
            with open(notebook_path, 'rb') as f:
                notebook_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            print(f"Failed to compute notebook hash: {e}")


    # Save json output
    filename = time.strftime("%m%d-%H%M%S") + ".json"
    output = {
        'timestamp': time.time(),
        'notebook_path': notebook_path,
        'notebook_hash': notebook_hash,
        'binder_sequence': binder_sequence,
        'target_sequence': target_sequence,
        'metadata': metadata,
        'wandb_run_id': wandb_run_id,
        'wandb_run_url': wandb_run_url,
        'wandb_run_logs': wandb_run_logs,
    }
    print(f"Saving run metadata to {os.path.join(output_dir, filename)}...")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(output, f, indent=4)
