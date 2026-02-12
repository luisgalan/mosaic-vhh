"""
Test script for comparing ipSAE metrics between the new compute_ipsae
and the reference implementations.

Uses:
  - reference_ipsae_w_ipae.py: produces .txt with ipSAE_avg, ipSAE_min_in_calculation, ipae columns
  - reference_min_ipsae.py: get_ipsae_min_max() to parse the .txt

Runs:
  1. reference_ipsae_w_ipae.py to produce the .txt file
  2. reference_min_ipsae.py's get_ipsae_min_max to parse that .txt
  3. compute_ipsae from vhh.af3.ipsae
  4. Prints both side by side
"""

import os
import glob
import subprocess
import sys
from vhh.af3.ipsae import compute_ipsae

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from reference_min_ipsae import get_ipsae_min_max

# ---- CONFIG ----
DIRECTORY = ".af3/fold/f40f8b7ceca09914"
PAE_CUTOFF = 10.0
DIST_CUTOFF = 10.0
REFERENCE_IPSAE = os.path.join(SCRIPT_DIR, "reference_ipsae_w_ipae.py")
# ----------------


def find_files(directory):
    cifs = glob.glob(os.path.join(directory, "*_model.cif"))
    assert cifs, f"No *_model.cif in {directory}"
    cif = cifs[0]
    stem = cif.replace("_model.cif", "")
    pae_json = stem + "_confidences.json"
    summary_json = stem + "_summary_confidences.json"
    assert os.path.exists(pae_json), f"Not found: {pae_json}"
    if not os.path.exists(summary_json):
        summary_json = None
    return cif, pae_json, summary_json


def find_txt(cif, pae_cutoff, dist_cutoff):
    stem = cif.replace(".cif", "")
    pae_str = f"{int(pae_cutoff):02d}"
    dist_str = f"{int(dist_cutoff):02d}"
    txt = f"{stem}_{pae_str}_{dist_str}.txt"
    if os.path.exists(txt):
        return txt
    return None


def main():
    cif, pae_json, summary_json = find_files(DIRECTORY)
    print(f"CIF:     {cif}")
    print(f"PAE:     {pae_json}")
    print(f"Summary: {summary_json}")
    print(f"Cutoffs: PAE={PAE_CUTOFF}, dist={DIST_CUTOFF}")

    # --- Step 1: run reference ipsae_w_ipae.py ---
    print("\n" + "=" * 60)
    print("Running reference_ipsae_w_ipae.py ...")
    print("=" * 60)
    cmd = [
        sys.executable, REFERENCE_IPSAE,
        pae_json, cif,
        str(int(PAE_CUTOFF)), str(int(DIST_CUTOFF)),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # --- Step 2: parse .txt with reference get_ipsae_min_max ---
    txt = find_txt(cif, PAE_CUTOFF, DIST_CUTOFF)
    assert txt, "reference_ipsae_w_ipae.py didn't produce expected .txt"
    print(f"\nParsing: {txt}")

    orig_min, orig_max, orig_avg, orig_lis, orig_min_calc, orig_d0chn, orig_d0dom, orig_ipae = get_ipsae_min_max(txt)

    # --- Step 3: run new implementation ---
    print("\n" + "=" * 60)
    print("Running compute_ipsae ...")
    print("=" * 60)
    result = compute_ipsae(cif, pae_json, summary_json, PAE_CUTOFF, DIST_CUTOFF)

    # --- Step 4: compare ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print("\n--- Asymmetric pairs (new) ---")
    for m in result.pairs:
        print(f"  {m.chain1}->{m.chain2}: ipSAE={m.ipsae:.6f}  avg={m.ipsae_avg:.6f}  min_calc={m.ipsae_min_in_calc:.6f}  ipae={m.ipae:.4f}")

    print("\n--- Max pairs (new) ---")
    for m in result.max_pairs:
        print(f"  {m.chain1}-{m.chain2}: ipSAE_max={m.ipsae:.6f}  ipSAE_min={m.ipsae_min:.6f}  avg={m.ipsae_avg:.6f}  min_calc={m.ipsae_min_in_calc:.6f}  ipae={m.ipae:.4f}")

    # The batch script averages per-partner mins/maxes across partners of chain A.
    # For a two-chain case this is just the single pair value.
    if result.max_pairs:
        n = len(result.max_pairs)
        new_min = sum(m.ipsae_min for m in result.max_pairs) / n
        new_max = sum(m.ipsae for m in result.max_pairs) / n
        new_avg = sum(m.ipsae_avg for m in result.max_pairs) / n
        new_lis = sum(m.lis for m in result.max_pairs) / n
        new_min_calc = sum(m.ipsae_min_in_calc for m in result.max_pairs) / n
        new_d0chn = sum(m.ipsae_d0chn for m in result.max_pairs) / n
        new_d0dom = sum(m.ipsae_d0dom for m in result.max_pairs) / n
        new_ipae = sum(m.ipae for m in result.max_pairs) / n
    else:
        new_min = new_max = new_avg = new_lis = new_min_calc = 0.0
        new_d0chn = new_d0dom = new_ipae = 0.0

    print(f"\n{'Metric':<25} {'Reference':>12} {'New':>12} {'Match':>12}")
    print("-" * 64)

    rows = [
        ("ipSAE_min",              orig_min,      new_min),
        ("ipSAE_max",              orig_max,      new_max),
        ("ipSAE_avg",              orig_avg,      new_avg),
        ("ipSAE_min_in_calc",      orig_min_calc, new_min_calc),
        ("LIS",                    orig_lis,      new_lis),
        ("ipSAE_d0chn",            orig_d0chn,    new_d0chn),
        ("ipSAE_d0dom",            orig_d0dom,    new_d0dom),
        ("ipae",                   orig_ipae,     new_ipae),
    ]
    for name, ov, nv in rows:
        ov_s = f"{ov:.6f}" if ov is not None else "N/A"
        nv_s = f"{nv:.6f}"
        if ov is not None:
            diff = abs(ov - nv)
            mark = "✓" if diff < 1e-4 else f"Δ{diff:.6f}"
        else:
            mark = "?"
        print(f"{name:<25} {ov_s:>12} {nv_s:>12} {mark:>12}")


if __name__ == "__main__":
    main()
