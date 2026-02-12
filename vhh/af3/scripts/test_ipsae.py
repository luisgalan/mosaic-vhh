"""
Test script for ipsae_af3.py
Usage: python test_ipsae.py <af3_output_directory>

Finds the CIF, full_data/confidences JSON, and summary JSON in the directory,
runs compute_ipsae, and prints all metrics for eyeball comparison with the
original ipsae.py output.
"""

import glob
import os
import subprocess
from vhh.af3.ipsae import compute_ipsae

# ---- CONFIG ----
DIRECTORY = ".af3/fold/f40f8b7ceca09914"
PAE_CUTOFF = 10.0
DIST_CUTOFF = 15.0
# ----------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_IPSAE = os.path.join(SCRIPT_DIR, "original_ipsae.py")


def find_files(directory):
    """Find the CIF, PAE/confidences JSON, and summary JSON in an AF3 output directory.
    Expects: {hash}_model.cif, {hash}_confidences.json, {hash}_summary_confidences.json
    """
    cifs = glob.glob(os.path.join(directory, "*_model.cif"))
    assert len(cifs) >= 1, f"No *_model.cif files found in {directory}"
    cif = cifs[0]
    stem = cif.replace("_model.cif", "")

    pae_json = stem + "_confidences.json"
    summary_json = stem + "_summary_confidences.json"

    assert os.path.exists(pae_json), f"PAE JSON not found: {pae_json}"
    if not os.path.exists(summary_json):
        summary_json = None

    return cif, pae_json, summary_json


def main():
    cif, pae_json, summary_json = find_files(DIRECTORY)
    print(f"CIF:     {cif}")
    print(f"PAE:     {pae_json}")
    print(f"Summary: {summary_json}")
    print(f"Cutoffs: PAE={PAE_CUTOFF}, dist={DIST_CUTOFF}")
    print()

    result = compute_ipsae(cif, pae_json, summary_json, PAE_CUTOFF, DIST_CUTOFF)

    header = f"{'C1':>3} {'C2':>3} {'Type':>5}  {'ipSAE':>8}  {'ipSAE_min':>9}  {'ipSAE_avg':>9}  {'min_calc':>9}  {'ipSAE_chn':>9}  {'ipSAE_dom':>9}  {'ipTM_af3':>8}  {'ipTM_chn':>8}  {'pDockQ':>8}  {'pDockQ2':>8}  {'LIS':>8}  {'ipae':>8}  {'n0res':>5}  {'n0chn':>5}  {'n0dom':>5}"
    print(header)
    print("-" * len(header))

    for m in result.pairs:
        print(
            f"{m.chain1:>3} {m.chain2:>3}  asym  "
            f"{m.ipsae:8.6f}  {m.ipsae_min:9.6f}  {m.ipsae_avg:9.6f}  {m.ipsae_min_in_calc:9.6f}  {m.ipsae_d0chn:9.6f}  {m.ipsae_d0dom:9.6f}  "
            f"{m.iptm_af3:8.3f}  {m.iptm_d0chn:8.6f}  "
            f"{m.pdockq:8.4f}  {m.pdockq2:8.4f}  {m.lis:8.4f}  {m.ipae:8.4f}  "
            f"{m.n0res:5d}  {m.n0chn:5d}  {m.n0dom:5d}"
        )

    print()
    for m in result.max_pairs:
        print(
            f"{m.chain1:>3} {m.chain2:>3}   max  "
            f"{m.ipsae:8.6f}  {m.ipsae_min:9.6f}  {m.ipsae_avg:9.6f}  {m.ipsae_min_in_calc:9.6f}  {m.ipsae_d0chn:9.6f}  {m.ipsae_d0dom:9.6f}  "
            f"{m.iptm_af3:8.3f}  {m.iptm_d0chn:8.6f}  "
            f"{m.pdockq:8.4f}  {m.pdockq2:8.4f}  {m.lis:8.4f}  {m.ipae:8.4f}  "
            f"{m.n0res:5d}  {m.n0chn:5d}  {m.n0dom:5d}"
        )

    # --- Run original ipsae.py for comparison ---
    print("\n" + "=" * 80)
    print("ORIGINAL ipsae.py output:")
    print("=" * 80 + "\n")

    cmd = [
        "python", ORIGINAL_IPSAE,
        pae_json, cif,
        str(int(PAE_CUTOFF)), str(int(DIST_CUTOFF)),
    ]
    print(f"$ {' '.join(cmd)}\n")
    subprocess.run(cmd)

    # Print the .txt output file
    pae_str = f"{int(PAE_CUTOFF):02d}"
    dist_str = f"{int(DIST_CUTOFF):02d}"
    txt_path = cif.replace(".cif", f"_{pae_str}_{dist_str}.txt")
    if os.path.exists(txt_path):
        print(f"\n--- {txt_path} ---")
        with open(txt_path) as f:
            print(f.read())
    else:
        print(f"Expected output file not found: {txt_path}")


if __name__ == "__main__":
    main()
