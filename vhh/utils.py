from ipymolstar import MolViewSpec
import molviewspec
import gemmi
import base64

# From germinal/utils/utils.py
def compute_cdr_positions(
    cdr_lengths: list[int], framework_lengths: list[int]
) -> list[int]:
    """
    Compute the positions of the CDRs in a protein structure using values from germinal.

    Args:
        cdr_lengths: A list of lengths of the CDRs.
        framework_lengths: A list of lengths of the framework regions.

    Returns:
        A list of positions of the CDRs.
    """
    cumulative = 0
    positions = []
    for i, cdr_length in enumerate(cdr_lengths):
        fw_len = framework_lengths[i] + cumulative
        positions.extend(range(fw_len, fw_len + cdr_length))
        cumulative = fw_len + cdr_length
    return positions

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
