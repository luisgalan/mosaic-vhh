"""
Simplified ipSAE scoring for AF3 models.
Based on ipsae.py v4 by Roland Dunbrack, Fox Chase Cancer Center.
https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2

ipSAE_min metric from:
Overath, Rygaard, Jacobsen, Brasas, Morell, Sormanni, Jenkins.
"Predicting Experimental Success in De Novo Binder Design:
A Meta-Analysis of 3,766 Experimentally Characterised Binders"
https://doi.org/10.1101/2025.08.14.670059
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field


NUC_RESIDUES = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
} | NUC_RESIDUES


@dataclass
class ChainPairMetrics:
    chain1: str
    chain2: str
    ipsae: float          # ipSAE (d0res, max over both directions for max_pairs)
    ipsae_min: float      # ipSAE (d0res, min over both directions for max_pairs)
    ipsae_avg: float      # mean of nonzero by-residue ipSAE (d0res)
    ipsae_min_in_calc: float  # min of nonzero by-residue ipSAE (d0res)
    ipsae_d0chn: float
    ipsae_d0dom: float
    iptm_af3: float       # from summary_confidences
    iptm_d0chn: float
    pdockq: float
    pdockq2: float
    lis: float
    ipae: float           # symmetric interaction PAE: 0.5*(mean_pae(c1→c2) + mean_pae(c2→c1))
    n0res: int
    n0chn: int
    n0dom: int


@dataclass
class IpSAEMetrics:
    """All pairwise metrics. `pairs` has asymmetric entries (A->B and B->A).
    `max_pairs` has one entry per unordered pair with the max over both directions."""
    pairs: list[ChainPairMetrics] = field(default_factory=list)
    max_pairs: list[ChainPairMetrics] = field(default_factory=list)


def _ptm(x, d0):
    return 1.0 / (1.0 + (x / d0) ** 2.0)

_ptm_vec = np.vectorize(_ptm)


def _calc_d0(L, is_nuc=False):
    L = float(L)
    min_val = 2.0 if is_nuc else 1.0
    d0 = 1.24 * (max(L, 26) - 15) ** (1.0 / 3.0) - 1.8 if L > 27 else 1.0
    return max(min_val, d0)


def _calc_d0_array(L, is_nuc=False):
    L = np.maximum(np.asarray(L, dtype=float), 26)
    min_val = 2.0 if is_nuc else 1.0
    return np.maximum(min_val, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


def _parse_cif(path):
    """Parse AF3 mmCIF, return list of residue dicts (one per CA/C1' for polymers) and token_mask."""
    fields = {}
    field_count = 0
    residues = []
    cb_residues = []
    token_mask = []

    with open(path) as f:
        for line in f:
            if line.startswith("_atom_site."):
                name = line.strip().split(".")[1]
                fields[name] = field_count
                field_count += 1
                continue

            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            parts = line.split()
            res_name = parts[fields["label_comp_id"]]
            seq_num = parts[fields["label_seq_id"]]

            if seq_num == ".":  # ligand
                token_mask.append(0)
                continue

            atom_name = parts[fields["label_atom_id"]]
            chain_id = parts[fields.get("auth_asym_id", fields["label_asym_id"])]
            atom_num = int(parts[fields["id"]])
            seq_num = int(seq_num)
            x, y, z = (float(parts[fields[c]]) for c in ("Cartn_x", "Cartn_y", "Cartn_z"))

            is_ca = atom_name == "CA" or "C1" in atom_name
            is_cb = atom_name == "CB" or "C3" in atom_name or (res_name == "GLY" and atom_name == "CA")

            if is_ca:
                token_mask.append(1)
                residues.append(dict(atom_num=atom_num, res=res_name, chain=chain_id,
                                     resnum=seq_num, coor=np.array([x, y, z])))
            if is_cb:
                cb_residues.append(dict(atom_num=atom_num, res=res_name, chain=chain_id,
                                        resnum=seq_num, coor=np.array([x, y, z])))

            # non-CA atoms in non-standard residues (PTMs etc.)
            if not is_ca and res_name not in STANDARD_RESIDUES:
                token_mask.append(0)

    return residues, cb_residues, np.array(token_mask)


def compute_ipsae(
    cif_path: str,
    pae_json_path: str,
    summary_json_path: str | None = None,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> IpSAEMetrics:
    """
    Compute ipSAE and related metrics for an AF3 model.

    Args:
        cif_path: Path to AF3 mmCIF structure file.
        pae_json_path: Path to AF3 full_data / confidences JSON (must contain 'pae').
        summary_json_path: Path to AF3 summary_confidences JSON (for chain-pair ipTM).
            If None, attempts to infer from pae_json_path.
        pae_cutoff: PAE cutoff for ipSAE calculation (default 10).
        dist_cutoff: Distance cutoff for interface residue counting (default 15).

    Returns:
        IpSAEResult with asymmetric and max chain-pair metrics.
    """
    # --- Parse structure ---
    residues, cb_residues, token_mask = _parse_cif(cif_path)
    numres = len(residues)
    chains = np.array([r["chain"] for r in residues])
    res_types = np.array([r["res"] for r in residues])
    ca_idx = np.array([r["atom_num"] - 1 for r in residues])
    cb_idx = np.array([r["atom_num"] - 1 for r in cb_residues])
    cb_coords = np.array([r["coor"] for r in cb_residues])

    _, first = np.unique(chains, return_index=True)
    unique_chains = chains[np.sort(first)]

    if len(unique_chains) < 2:
        return IpSAEMetrics()

    # chain type classification
    def is_nuc_pair(c1, c2):
        for c in (c1, c2):
            if any(r in NUC_RESIDUES for r in res_types[chains == c]):
                return True
        return False

    # distance matrix (CB/C3')
    distances = np.sqrt(((cb_coords[:, None, :] - cb_coords[None, :, :]) ** 2).sum(axis=2))

    # --- Load PAE / pLDDT ---
    with open(pae_json_path) as f:
        data = json.load(f)

    pae_full = np.array(data["pae"])
    pae_matrix = pae_full[np.ix_(token_mask.astype(bool), token_mask.astype(bool))]

    if "atom_plddts" in data:
        atom_plddts = np.array(data["atom_plddts"])
        plddt = atom_plddts[ca_idx]
        cb_plddt = atom_plddts[cb_idx]
    else:
        plddt = np.zeros(numres)
        cb_plddt = np.zeros(numres)

    # --- Load chain-pair ipTM from summary ---
    if summary_json_path is None:
        for old, new in [("confidences", "summary_confidences"), ("full_data", "summary_confidences")]:
            candidate = pae_json_path.replace(old, new)
            if candidate != pae_json_path:
                summary_json_path = candidate
                break

    iptm_af3 = {}
    if summary_json_path is not None:
        try:
            with open(summary_json_path) as f:
                s = json.load(f)
            cp = s["chain_pair_iptm"]
            for i, c1 in enumerate(unique_chains):
                for j, c2 in enumerate(unique_chains):
                    if c1 != c2:
                        iptm_af3[(c1, c2)] = cp[i][j]
        except (FileNotFoundError, KeyError):
            pass

    # --- Compute metrics per chain pair ---
    # helpers
    def _pdockq(c1, c2):
        cutoff = 8.0
        npairs = 0
        contact_res = set()
        for i in range(numres):
            if chains[i] != c1:
                continue
            mask = (chains == c2) & (distances[i] <= cutoff)
            n = mask.sum()
            if n > 0:
                npairs += n
                contact_res.add(i)
                contact_res.update(np.where(mask)[0].tolist())
        if npairs == 0:
            return 0.0, contact_res
        mean_pl = cb_plddt[list(contact_res)].mean()
        x = mean_pl * math.log10(npairs)
        return 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018, contact_res

    def _pdockq2(c1, c2, contact_res):
        cutoff = 8.0
        npairs = 0
        s = 0.0
        for i in range(numres):
            if chains[i] != c1:
                continue
            mask = (chains == c2) & (distances[i] <= cutoff)
            if mask.any():
                npairs += mask.sum()
                s += _ptm_vec(pae_matrix[i][mask], 10.0).sum()
        if npairs == 0:
            return 0.0
        mean_pl = cb_plddt[list(contact_res)].mean()
        x = mean_pl * (s / npairs)
        return 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005

    def _lis(c1, c2):
        mask = (chains[:, None] == c1) & (chains[None, :] == c2)
        vals = pae_matrix[mask]
        valid = vals[vals <= 12]
        if valid.size == 0:
            return 0.0
        return float(np.mean((12 - valid) / 12))

    asym_metrics = {}  # (c1,c2) -> ChainPairMetrics

    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue

            nuc = is_nuc_pair(c1, c2)
            n_chn = int((chains == c1).sum() + (chains == c2).sum())
            d_chn = _calc_d0(n_chn, nuc)
            ptm_d0chn = _ptm_vec(pae_matrix, d_chn)

            mask_c1 = chains == c1
            mask_c2 = chains == c2
            valid_matrix = np.outer(mask_c1, mask_c2) & (pae_matrix < pae_cutoff)

            # by-residue arrays
            iptm_chn_byres = np.zeros(numres)
            ipsae_chn_byres = np.zeros(numres)
            uniq_r1 = set()
            uniq_r2 = set()
            dist_r1 = set()
            dist_r2 = set()

            for i in range(numres):
                if not mask_c1[i]:
                    continue
                iptm_chn_byres[i] = ptm_d0chn[i, mask_c2].mean()
                vp = valid_matrix[i]
                if vp.any():
                    ipsae_chn_byres[i] = ptm_d0chn[i, vp].mean()
                    uniq_r1.add(residues[i]["resnum"])
                    uniq_r2.update(residues[j]["resnum"] for j in np.where(vp)[0])
                # dist-filtered contacts
                dvp = vp & (distances[i] < dist_cutoff)
                if dvp.any():
                    dist_r1.add(residues[i]["resnum"])
                    dist_r2.update(residues[j]["resnum"] for j in np.where(dvp)[0])

            # d0dom
            n_dom = len(uniq_r1) + len(uniq_r2)
            d_dom = _calc_d0(n_dom, nuc)
            ptm_d0dom = _ptm_vec(pae_matrix, d_dom)

            # d0res (per-residue)
            n0res_arr = valid_matrix.sum(axis=1)
            d0res_arr = _calc_d0_array(n0res_arr, nuc)

            ipsae_dom_byres = np.zeros(numres)
            ipsae_res_byres = np.zeros(numres)

            for i in range(numres):
                if not mask_c1[i]:
                    continue
                vp = valid_matrix[i]
                if vp.any():
                    ipsae_dom_byres[i] = ptm_d0dom[i, vp].mean()
                    ipsae_res_byres[i] = _ptm_vec(pae_matrix[i, vp], d0res_arr[i]).mean()

            # asymmetric: take max over residues
            best = int(np.argmax(ipsae_res_byres))
            valid_vals = ipsae_res_byres[ipsae_res_byres > 0.0]
            ipsae_avg_val = float(np.mean(valid_vals)) if valid_vals.size else 0.0
            ipsae_min_val = float(np.min(valid_vals)) if valid_vals.size else 0.0

            # ipae: symmetric interaction PAE = 0.5*(mean_pae(c1→c2) + mean_pae(c2→c1))
            idx1 = np.where(chains == c1)[0]
            idx2 = np.where(chains == c2)[0]
            if idx1.size and idx2.size:
                ipae_val = 0.5 * (pae_matrix[np.ix_(idx1, idx2)].mean() +
                                  pae_matrix[np.ix_(idx2, idx1)].mean())
            else:
                ipae_val = 0.0

            pdq, contact = _pdockq(c1, c2)

            asym_metrics[(c1, c2)] = ChainPairMetrics(
                chain1=c1, chain2=c2,
                ipsae=float(ipsae_res_byres[best]),
                ipsae_min=float(ipsae_res_byres[best]),
                ipsae_avg=ipsae_avg_val,
                ipsae_min_in_calc=ipsae_min_val,
                ipsae_d0chn=float(ipsae_chn_byres[int(np.argmax(ipsae_chn_byres))]),
                ipsae_d0dom=float(ipsae_dom_byres[int(np.argmax(ipsae_dom_byres))]),
                iptm_af3=float(iptm_af3.get((c1, c2), 0.0)),
                iptm_d0chn=float(iptm_chn_byres[int(np.argmax(iptm_chn_byres))]),
                pdockq=pdq,
                pdockq2=_pdockq2(c1, c2, contact),
                lis=_lis(c1, c2),
                ipae=ipae_val,
                n0res=int(n0res_arr[best]),
                n0chn=n_chn,
                n0dom=n_dom,
            )

    # max pairs (one per unordered pair)
    max_pairs = []
    seen = set()
    for (c1, c2), m1 in asym_metrics.items():
        key = tuple(sorted((c1, c2)))
        if key in seen:
            continue
        seen.add(key)
        m2 = asym_metrics.get((c2, c1))
        if m2 is None:
            max_pairs.append(m1)
            continue
        best = m1 if m1.ipsae >= m2.ipsae else m2
        lis_avg = (m1.lis + m2.lis) / 2.0
        max_pairs.append(ChainPairMetrics(
            chain1=key[0], chain2=key[1],
            ipsae=max(m1.ipsae, m2.ipsae),
            ipsae_min=min(m1.ipsae, m2.ipsae),
            ipsae_avg=max(m1.ipsae_avg, m2.ipsae_avg),
            ipsae_min_in_calc=max(m1.ipsae_min_in_calc, m2.ipsae_min_in_calc),
            ipsae_d0chn=max(m1.ipsae_d0chn, m2.ipsae_d0chn),
            ipsae_d0dom=max(m1.ipsae_d0dom, m2.ipsae_d0dom),
            iptm_af3=max(m1.iptm_af3, m2.iptm_af3),
            iptm_d0chn=max(m1.iptm_d0chn, m2.iptm_d0chn),
            pdockq=max(m1.pdockq, m2.pdockq),
            pdockq2=max(m1.pdockq2, m2.pdockq2),
            lis=lis_avg,
            ipae=max(m1.ipae, m2.ipae),
            n0res=best.n0res,
            n0chn=m1.n0chn,
            n0dom=max(m1.n0dom, m2.n0dom),
        ))

    return IpSAEMetrics(
        pairs=list(asym_metrics.values()),
        max_pairs=max_pairs,
    )
