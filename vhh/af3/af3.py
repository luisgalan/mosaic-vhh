from dataclasses import dataclass
import os
from dotenv import load_dotenv
import subprocess
import json
from pathlib import Path
import hashlib
import random
import pandas as pd
import gemmi
import ipsae
from vhh.af3.ipsae import IpSAEMetrics, compute_ipsae

load_dotenv()

AF3_COMMAND = os.environ.get('AF3_COMMAND')
AF3_MODEL_DIR = Path(os.environ.get('AF3_MODEL_DIR') or '').resolve()
AF3_DB_DIR = Path(os.environ.get('AF3_DB_DIR') or '').resolve()
AF3_OUT_DIR = Path('.af3').resolve()
JAX_CACHE_DIR = AF3_OUT_DIR / "jax_cache"
MSA_DIR = AF3_OUT_DIR / "msa"
FOLD_DIR = AF3_OUT_DIR / "fold"

def get_output_id(*inputs):
    hasher = hashlib.sha256()
    for input in inputs:
        hasher.update(str(input).encode())
    return hasher.hexdigest()[:16]

def get_msa(sequence, chain_id):
    id = get_output_id(sequence)
    output_file = MSA_DIR / f"{id}/{id}_data.json"

    # Check if cached MSA is available
    if os.path.exists(output_file):
        print(f"Using cached MSA at {output_file}")
        with open(output_file, "r") as f:
            protein = json.load(f)['sequences'][0]['protein']
            protein['id'] = [chain_id] # In case it's cached under a different ID
            return protein

    print(f"No cached MSA found for id {id}, computing MSA... (this will take a while)")

    # Create AF3 input json for MSA
    job_input = {
        "name": f"{id}",
        "sequences": [
            {
                "protein": {
                    "id": [chain_id],
                    "sequence": sequence
                }
            }
        ],
        "modelSeeds": [1], # Not used since we're only computing MSA here, but still required by AF3
        "dialect": "alphafold3",
        "version": 1
    }
    msa_json_path = MSA_DIR / f"{id}/job_input.json"
    os.makedirs(MSA_DIR / id, exist_ok=True)
    with open(msa_json_path, "w") as f:
        json.dump(job_input, f, indent=2)

    # Run AF3 to compute MSA
    cmd = f"{os.environ['AF3_COMMAND']} --json_path {msa_json_path} --output_dir {MSA_DIR} --force_output_dir --norun_inference --model_dir {AF3_MODEL_DIR} --db_dir {AF3_DB_DIR}"
    subprocess.run(cmd, shell=True, check=True)

    # Return MSA
    with open(output_file, "r") as f:
        return json.load(f)['sequences'][0]['protein']

@dataclass
class AF3Output:
    model: gemmi.cif.Document
    ranking_scores: pd.DataFrame
    summary_confidences: dict
    ipsae_metrics: IpSAEMetrics

def get_fold_output(id) -> AF3Output:
    assert(os.path.exists(FOLD_DIR / id))

    cif_path = str(FOLD_DIR / f"{id}/{id}_model.cif")
    ranking_path = str(FOLD_DIR / f"{id}/{id}_ranking_scores.csv")
    confidences_path = str(FOLD_DIR / f"{id}/{id}_confidences.json")
    summary_path = str(FOLD_DIR / f"{id}/{id}_summary_confidences.json")

    model = gemmi.cif.read_file(cif_path)
    ranking_scores = pd.read_csv(ranking_path)
    with open(summary_path) as f:
        summary_confidences = json.load(f)
    ipsae_metrics = compute_ipsae(cif_path, confidences_path, summary_path)

    return AF3Output(model, ranking_scores, summary_confidences, ipsae_metrics)


def get_fold(binder_sequence, target_sequence, num_seeds=3) -> AF3Output:
    id = get_output_id(binder_sequence, target_sequence, num_seeds)

    # Check if cached output exists
    try:
        output = get_fold_output(id)
        print(f"Using cached fold output at {FOLD_DIR / id}")
        return output
    except Exception as e:
        print(f"No cached fold output found for id {id}, running AF3 inference...")

    target_chain = get_msa(target_sequence, chain_id="B")

    # Create AF3 input json for inference using precomputed MSA
    fold_json = {
        "name": f"{id}",
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": binder_sequence,
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": []
                },
            },
            {
                "protein": target_chain,
            }
        ],
        # AF3 outputs are not guaranteed to be deterministic even when seeds are fixed, so using random seeds on purpose
        "modelSeeds": [int.from_bytes(os.urandom(3)) for _ in range(num_seeds)],
        "dialect": "alphafold3",
        "version": 1
    }
    os.makedirs(FOLD_DIR / id, exist_ok=True)
    fold_json_path = Path(FOLD_DIR / f"{id}/job_input.json").resolve()
    with open(fold_json_path, "w") as f:
        json.dump(fold_json, f, indent=2)

    # Run AF3
    cmd = f"{os.environ['AF3_COMMAND']} --json_path {fold_json_path} --output_dir {FOLD_DIR} --force_output_dir --norun_data_pipeline --model_dir {AF3_MODEL_DIR} --db_dir {AF3_DB_DIR} --jax_compilation_cache_dir {JAX_CACHE_DIR}"
    subprocess.run(cmd, shell=True, check=True)

    return get_fold_output(id)

target_sequence = "KTSWVNCSNMIDEIITHLKQPPLPLLDFNNLNGEDQDILMENNLRRPNLEAFNRAVKSLQNASAIESILKNLLPCLPLATAAPTRHPIHIKDGDWNEFRRKLTFYLKTLENA"
binders = [
    # "YVQLVESGGGLVQPGGSLRLSCAASGDTFRGSRDTCLGWFRQAPGQGLEAVAAIWNDENQEYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCCAWILCSSDRYYWYWQAYWGQGTLVTVS",
    "YVQLVESGGGLVQPGGSLRLSCAASGVTFRPNTQTSLGWVRQAPGQGLEWVAAITVSKNKEYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCFVLLAESSNSCYWDWLVSWGQGTLVTVS",
    # "QVQLVESGGGLVQPGGSLRLSCAASEAYSRPSAHTNLGWFRQAPGQGLEAVAAIWHDGTYQYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCFRYQFSSSRDLWNDLEHAWGQGTLVTVS",
]

for binder in binders:
    fold = get_fold(binder, target_sequence)
    print(fold)
