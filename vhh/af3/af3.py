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
import sys
from vhh.af3.ipsae import IpSAEMetrics, compute_ipsae

load_dotenv()

AF3_COMMAND = os.environ.get('AF3_COMMAND')
AF3_MODEL_DIR = Path(os.environ.get('AF3_MODEL_DIR') or '').resolve()
AF3_DB_DIR = Path(os.environ.get('AF3_DB_DIR') or '').resolve()
AF3_OUT_DIR = Path('.af3').resolve()
JAX_CACHE_DIR = AF3_OUT_DIR / "jax_cache"
TRITON_CACHE_DIR = AF3_OUT_DIR / "triton_cache"
MSA_DIR = AF3_OUT_DIR / "msa"
FOLD_DIR = AF3_OUT_DIR / "fold"

def _get_output_id(*inputs):
    hasher = hashlib.sha256()
    for input in inputs:
        hasher.update(str(input).encode())
    return hasher.hexdigest()[:16]

def _get_or_compute_msa(sequence, chain_id):
    output_id = _get_output_id(sequence)
    output_file = MSA_DIR / f"{output_id}/{output_id}_data.json"

    # Check if cached MSA is available
    if os.path.exists(output_file):
        print(f"Using cached MSA at {output_file}")
        with open(output_file, "r") as f:
            protein = json.load(f)['sequences'][0]['protein']
            protein['id'] = [chain_id] # In case it's cached under a different ID
            return protein

    print(f"No cached MSA found for id {output_id}, computing MSA... (this will take a while)")

    # Create AF3 input json for MSA
    job_input = {
        "name": f"{output_id}",
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
    msa_json_path = MSA_DIR / f"{output_id}/job_input.json"
    os.makedirs(MSA_DIR / output_id, exist_ok=True)
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
    iptm: float
    ipsae: float
    ipsae_min: float

def _get_fold_output(output_id) -> AF3Output:
    cif_path = str(FOLD_DIR / f"{output_id}/{output_id}_model.cif")
    ranking_path = str(FOLD_DIR / f"{output_id}/{output_id}_ranking_scores.csv")
    confidences_path = str(FOLD_DIR / f"{output_id}/{output_id}_confidences.json")
    summary_path = str(FOLD_DIR / f"{output_id}/{output_id}_summary_confidences.json")

    model = gemmi.cif.read_file(cif_path)
    ranking_scores = pd.read_csv(ranking_path)
    with open(summary_path) as f:
        summary_confidences = json.load(f)
    ipsae_metrics = compute_ipsae(cif_path, confidences_path, summary_path)

    iptm = summary_confidences["iptm"]
    ipsae = sum(m.ipsae for m in ipsae_metrics.max_pairs) / len(ipsae_metrics.max_pairs)
    ipsae_min = sum(m.ipsae_min for m in ipsae_metrics.max_pairs) / len(ipsae_metrics.max_pairs)

    return AF3Output(
        model=model,
        ranking_scores=ranking_scores,
        summary_confidences=summary_confidences,
        ipsae_metrics=ipsae_metrics,
        iptm=iptm,
        ipsae=ipsae,
        ipsae_min=ipsae_min
    )

def _archive_unused_fold_files(output_id):
    """Archive all files and directories except those used by _get_fold_output()"""
    fold_path = FOLD_DIR / output_id
    archive_path = fold_path / f"{output_id}_unused.tar.xz"

    # Skip if archive already exists
    if archive_path.exists():
        return

    # Files to keep
    keep_files = {
        f"{output_id}_model.cif",
        f"{output_id}_ranking_scores.csv",
        f"{output_id}_confidences.json",
        f"{output_id}_summary_confidences.json",
    }

    # Find all files and directories to archive
    items_to_archive = []
    for item in fold_path.iterdir():
        if item.is_file() and item.name not in keep_files:
            items_to_archive.append(item)
        elif item.is_dir():
            items_to_archive.append(item)

    if not items_to_archive:
        return

    # Create tar.xz archive
    import tarfile
    import shutil
    with tarfile.open(archive_path, "w:xz") as tar:
        for item_path in items_to_archive:
            tar.add(item_path, arcname=item_path.name)

    # Remove archived files and directories
    for item_path in items_to_archive:
        if item_path.is_file():
            item_path.unlink()
        elif item_path.is_dir():
            shutil.rmtree(item_path)

    print(f"Archived {len(items_to_archive)} unused items to {archive_path}")

def fold(binder_sequence, target_sequence, num_seeds=3, require_cached=False) -> AF3Output:
    """Fold a binder-target pair using AF3, caching results and re-using cached outputs.

    Args:
        binder_sequence (str): The binder sequence.
        target_sequence (str): The target sequence.
        num_seeds (int): The number of seeds to use.
        require_cached (bool): If true, raise a FileNotFoundError if no cached output is found.

    Returns:
        AF3Output: The AF3 output.
    """

    output_id = _get_output_id(binder_sequence, target_sequence, num_seeds)

    # Check if cached output exists
    try:
        output = _get_fold_output(output_id)
        if not require_cached:
            print(f"Using cached fold output at {FOLD_DIR / output_id}")
        return output
    except Exception as e:
        if require_cached:
            raise FileNotFoundError(f"No cached fold output found for id {output_id}")
        else:
            print(f"No cached fold output found for id {output_id}, running AF3 inference...")


    target_msa = _get_or_compute_msa(target_sequence, chain_id="B")

    # Create AF3 input json for inference using precomputed MSA
    fold_json = {
        "name": f"{output_id}",
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
                "protein": target_msa,
            }
        ],
        # AF3 outputs are not guaranteed to be deterministic even when seeds are fixed, so using random seeds on purpose
        "modelSeeds": [int.from_bytes(os.urandom(3)) for _ in range(num_seeds)],
        "dialect": "alphafold3",
        "version": 1
    }
    os.makedirs(FOLD_DIR / output_id, exist_ok=True)
    fold_json_path = Path(FOLD_DIR / f"{output_id}/job_input.json").resolve()
    with open(fold_json_path, "w") as f:
        json.dump(fold_json, f, indent=2)

    # Run AF3
    cmd = f"{os.environ['AF3_COMMAND']} --json_path {fold_json_path} --output_dir {FOLD_DIR} --force_output_dir --norun_data_pipeline --model_dir {AF3_MODEL_DIR} --db_dir {AF3_DB_DIR} --jax_compilation_cache_dir {JAX_CACHE_DIR}"
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Filter out annoying autotuning messages
    for line in process.stdout or []:
        if 'Autotuning cache miss' not in line:
            print(line.rstrip('\n'))

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    _archive_unused_fold_files(output_id)

    return _get_fold_output(output_id)
