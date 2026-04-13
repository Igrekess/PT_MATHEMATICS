#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRITICAL AUDIT V4: Independent Numerical Verification of pt_constants.py
Cross-checks all PT predictions against PDG 2024 / CODATA 2022 / NuFit 6.0
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import numpy as np

# Import pt_constants properly (module isolation)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pt_constants import *

# ============================================================
# INDEPENDENT PDG 2024 REFERENCE VALUES
# Cross-checked against PDG RPP 2024 (Phys. Rev. D 110, 030001)
# CODATA 2022 for alpha_EM, NuFit 6.0 for neutrinos
# ============================================================

pdg_audit = {
    # key: (PDG_value, 1sigma_uncertainty, unit, source_note)
    '1/alpha_EM':     (137.035999177, 0.000000021, '', 'CODATA 2022'),
    'sin2_thetaW':    (0.23121, 0.00004, '', 'PDG 2024 MS-bar Z-pole'),
    'alpha_s':        (0.1180, 0.0009, '', 'PDG 2024'),
    'm_mu':           (105.6583755, 0.0000023, 'MeV', 'PDG 2024'),
    'm_tau':          (1776.86, 0.12, 'MeV', 'PDG 2024'),
    'm_u':            (2.16, 0.38, 'MeV', 'PDG 2024 MS-bar 2GeV'),
    'm_d':            (4.67, 0.33, 'MeV', 'PDG 2024 MS-bar 2GeV'),
    'm_s':            (93.4, 8.6, 'MeV', 'PDG 2024 MS-bar 2GeV'),
    'm_c':            (1270.0, 20.0, 'MeV', 'PDG 2024 MS-bar m_c'),
    'm_b':            (4180.0, 25.0, 'MeV', 'PDG 2024 MS-bar m_b'),
    'm_t':            (172760.0, 300.0, 'MeV', 'PDG 2024 pole mass'),
    'm_W':            (80.3692, 0.0133, 'GeV', 'PDG 2024'),
    'm_Z':            (91.1876, 0.0021, 'GeV', 'PDG 2024'),
    'm_H':            (125.25, 0.17, 'GeV', 'PDG 2024 combined'),
    'V_ud':           (0.97373, 0.00031, '', 'PDG 2024'),
    'V_us':           (0.2243, 0.0008, '', 'PDG 2024'),
    'V_ub':           (0.00382, 0.00020, '', 'PDG 2024'),
    'V_cd':           (0.221, 0.004, '', 'PDG 2024'),
    'V_cs':           (0.975, 0.006, '', 'PDG 2024'),
    'V_cb':           (0.0408, 0.0014, '', 'PDG 2024'),
    'V_td':           (0.0080, 0.0003, '', 'PDG 2024'),
    'V_ts':           (0.0388, 0.0011, '', 'PDG 2024'),
    'V_tb':           (0.99910, 0.00035, '', 'PDG 2024'),
    'J_CKM':          (3.08e-5, 1.5e-6, '', 'PDG 2024'),
    'delta_CKM':      (67.0, 4.0, 'deg', 'PDG 2024'),
    'sin2_th12':      (0.304, 0.012, '', 'PDG 2024'),
    'sin2_th13':      (0.02220, 0.00068, '', 'PDG 2024'),
    'sin2_th23':      (0.573, 0.016, '', 'PDG 2024'),
    'delta_CP_PMNS':  (197.0, 25.0, 'deg', 'PDG 2024'),
    'J_PMNS':         (0.00990, 0.003, '', 'derived from PMNS'),
    'm_nu3_eV':       (0.0507, 0.002, 'eV', 'PDG 2024 indirect'),
    'Dm31_sq':        (2.51e-3, 0.03e-3, 'eV^2', 'PDG 2024'),
    'Dm21_sq':        (7.42e-5, 0.21e-5, 'eV^2', 'PDG 2024'),
    'sigma_QCD':      (0.194, 0.020, 'GeV^2', 'lattice'),
    'regge_slope':    (0.88, 0.03, 'GeV^-2', 'pheno'),
    'G_F':            (1.1663788e-5, 3.0e-11, 'GeV^-2', 'PDG 2024'),
}

# NuFit 6.0 alternative values for comparison
nufit60 = {
    'sin2_th12':     (0.307, 0.012),
    'sin2_th13':     (0.02195, 0.00058),
    'sin2_th23':     (0.561, 0.015),
    'Dm21_sq':       (7.49e-5, 0.19e-5),
    'Dm31_sq':       (2.534e-3, 0.025e-3),
    'delta_CP_PMNS': (177.0, 20.0),
}

# PT values from the script
pt_vals = {
    '1/alpha_EM': 1.0 / alpha_EM,
    'sin2_thetaW': sin2_thetaW,
    'alpha_s': alpha_s,
    'm_mu': m_mu,
    'm_tau': m_tau,
    'm_u': m_u,
    'm_d': m_d,
    'm_s': m_s,
    'm_c': m_c,
    'm_b': m_b,
    'm_t': m_t,
    'm_W': m_W,
    'm_Z': m_Z,
    'm_H': m_H,
    'V_ud': V_ud,
    'V_us': V_us,
    'V_ub': V_ub,
    'V_cd': V_cd,
    'V_cs': V_cs,
    'V_cb': V_cb,
    'V_td': V_td,
    'V_ts': V_ts,
    'V_tb': V_tb,
    'J_CKM': J_CKM,
    'delta_CKM': delta_CKM,
    'sin2_th12': sin2_th12,
    'sin2_th13': sin2_th13,
    'sin2_th23': sin2_th23,
    'delta_CP_PMNS': delta_CP_PMNS,
    'J_PMNS': J_PMNS,
    'm_nu3_eV': m_nu3,
    'Dm31_sq': Dm31_sq,
    'Dm21_sq': Dm21_sq,
    'sigma_QCD': sigma_QCD,
    'regge_slope': regge_slope,
    'G_F': G_F,
}

print("=" * 130)
print("  CRITICAL AUDIT V4: Independent Numerical Verification of pt_constants.py")
print("  PDG 2024 (Phys. Rev. D 110, 030001) + CODATA 2022 + NuFit 6.0")
print("=" * 130)
print()

# Header
fmt = "{:<18} {:>16} {:>16} {:>10} {:>8} {:>6}  {}"
print(fmt.format("Observable", "PT Value", "PDG Value", "Err%", "n_sig", "PASS?", "Note"))
print(fmt.format("-" * 18, "-" * 16, "-" * 16, "-" * 10, "-" * 8, "-" * 6, "-" * 30))

n_total = 0
n_pass_05 = 0
n_pass_2sig = 0
n_between_2_3sig = 0
n_beyond_3sig = 0
errs = []
issues = []

for key in pt_vals:
    if key not in pdg_audit:
        continue

    pdg_val, pdg_sig, unit, note = pdg_audit[key]
    pt_val = pt_vals[key]

    err_pct = abs(pt_val - pdg_val) / abs(pdg_val) * 100
    n_sig = abs(pt_val - pdg_val) / pdg_sig if pdg_sig > 0 else float("inf")

    n_total += 1
    errs.append(err_pct)

    pass_05 = err_pct < 0.5
    if pass_05:
        n_pass_05 += 1

    if n_sig < 2:
        n_pass_2sig += 1
    elif n_sig < 3:
        n_between_2_3sig += 1
    else:
        n_beyond_3sig += 1

    pass_str = "YES" if pass_05 else "**NO**"

    # Format values
    if abs(pt_val) < 0.001:
        pt_str = f"{pt_val:.6e}"
        pdg_str = f"{pdg_val:.6e}"
    elif abs(pt_val) < 1:
        pt_str = f"{pt_val:.7f}"
        pdg_str = f"{pdg_val:.7f}"
    elif abs(pt_val) < 100:
        pt_str = f"{pt_val:.6f}"
        pdg_str = f"{pdg_val:.6f}"
    elif abs(pt_val) < 10000:
        pt_str = f"{pt_val:.3f}"
        pdg_str = f"{pdg_val:.3f}"
    else:
        pt_str = f"{pt_val:.1f}"
        pdg_str = f"{pdg_val:.1f}"

    extra = ""
    if n_sig >= 2:
        extra += " <<<"
        issues.append((key, pt_val, pdg_val, err_pct, n_sig, note))

    print(fmt.format(key, pt_str, pdg_str, f"{err_pct:.4f}%", f"{n_sig:.1f}", pass_str, extra))

print()
print("=" * 130)
print("  SUMMARY")
print("=" * 130)
print(f"  Total scored observables:     {n_total}")
print(f"  Sub-0.5% relative error:      {n_pass_05}/{n_total}  ({100 * n_pass_05 / n_total:.0f}%)")
print(f"  Sub-2.1% (incl. J_PMNS):      {sum(1 for e in errs if e < 2.1)}/{n_total}")
print(f"  Within 2-sigma:               {n_pass_2sig}/{n_total}  ({100 * n_pass_2sig / n_total:.0f}%)")
print(f"  Between 2-3 sigma:            {n_between_2_3sig}/{n_total}")
print(f"  Beyond 3-sigma:               {n_beyond_3sig}/{n_total}")
print(f"  Mean relative error:          {np.mean(errs):.4f}%")
print(f"  Median relative error:        {np.median(errs):.4f}%")

max_idx = np.argmax(errs)
keys_list = [k for k in pt_vals if k in pdg_audit]
print(f"  Max relative error:           {max(errs):.4f}% ({keys_list[max_idx]})")
print()

if issues:
    print("  VALUES OUTSIDE 2-SIGMA:")
    for k, ptv, pdgv, ep, ns, nt in issues:
        print(f"    {k}: PT={ptv:.10g}, PDG={pdgv:.10g}, err={ep:.4f}%, n_sig={ns:.1f} ({nt})")
    print()

# Check CODATA 2018 vs CODATA 2022 for alpha_EM
print("  NOTE ON alpha_EM:")
codata_2018 = 137.035999084
codata_2022 = 137.035999177
pt_inv_alpha = 1.0 / alpha_EM
print(f"    Script PDG dict uses 1/alpha = {codata_2018} (CODATA 2018)")
print(f"    Latest CODATA 2022:          1/alpha = {codata_2022}(21)")
print(f"    PT prediction:               1/alpha = {pt_inv_alpha:.9f}")
err_2018 = abs(pt_inv_alpha - codata_2018) / codata_2018 * 100
sig_2018 = abs(pt_inv_alpha - codata_2018) / 0.000000021
err_2022 = abs(pt_inv_alpha - codata_2022) / codata_2022 * 100
sig_2022 = abs(pt_inv_alpha - codata_2022) / 0.000000021
print(f"    vs CODATA 2018:  err = {err_2018:.6f}%, n_sig = {sig_2018:.1f}")
print(f"    vs CODATA 2022:  err = {err_2022:.6f}%, n_sig = {sig_2022:.1f}")
print()

# Check NuFit 6.0 vs PDG for neutrino observables
print("  NOTE ON NEUTRINO PARAMETERS (NuFit 6.0 vs PDG 2024):")
for k in nufit60:
    nf_val, nf_sig = nufit60[k]
    pdg_val = pdg_audit[k][0]
    pdg_sig = pdg_audit[k][1]
    pt_val = pt_vals[k]
    print(f"    {k}: PT={pt_val:.6g}, PDG={pdg_val:.4g}(+/-{pdg_sig:.2g}), NuFit6={nf_val:.4g}(+/-{nf_sig:.2g})")
    print(f"      vs PDG:   {abs(pt_val - pdg_val) / pdg_sig:.2f} sig, err={abs(pt_val - pdg_val) / pdg_val * 100:.3f}%")
    print(f"      vs NuFit: {abs(pt_val - nf_val) / nf_sig:.2f} sig, err={abs(pt_val - nf_val) / nf_val * 100:.3f}%")
print()

# ============================================================
# CHECK CLAIM: "39/39 sub-0.5%"
# ============================================================
print("=" * 130)
print("  CLAIM VERIFICATION: '39/39 sub-0.5%'")
print("=" * 130)

# Count what the script scores: it excludes m_e, v_higgs (input), N_c/N_gen/theta_QCD (exact),
# and gluon_condensate (not scored). That leaves the scored observables.
# The script counts 39 = 36 scored + 3 exact (N_c, N_gen, theta_QCD)
# The 36 scored values all need to be < 0.5% OR within tolerance
# But wait, J_PMNS is 2.053% -- how does the script report PASS?
# Answer: script uses tol=5.0 for most, tol=10.0 for QCD NP
# The script counts PASS as err < 5.0% (or 10% for QCD), not 0.5%

print()
print("  The script reports 39/39 PASS with tolerance 5% (not 0.5%).")
print("  The MEMORY.md claims '39/39 sub-0.5%' and 'ALL 39 sub-0.5%'")
print()
print("  Checking each observable against the 0.5% threshold:")
over_05 = []
under_05 = 0
for key in pt_vals:
    if key not in pdg_audit:
        continue
    pdg_val = pdg_audit[key][0]
    pt_val = pt_vals[key]
    err_pct = abs(pt_val - pdg_val) / abs(pdg_val) * 100
    if err_pct >= 0.5:
        over_05.append((key, err_pct))
    else:
        under_05 += 1

print(f"  Sub-0.5%: {under_05}/{n_total}")
print(f"  >= 0.5%:  {len(over_05)}/{n_total}")
if over_05:
    print("  Observables >= 0.5%:")
    for k, e in over_05:
        print(f"    {k}: {e:.3f}%")
print()

# ============================================================
# G_F cross-check
# ============================================================
print("=" * 130)
print("  G_F CROSS-CHECK")
print("=" * 130)
gf_pt = G_F
gf_pdg = 1.1663788e-5
gf_err = abs(gf_pt - gf_pdg) / gf_pdg * 100
print(f"  G_F (PT, from v_higgs): {gf_pt:.10e} GeV^-2")
print(f"  G_F (PDG muon decay):   {gf_pdg:.10e} GeV^-2")
print(f"  Relative error:         {gf_err:.4f}%")
print(f"  NOTE: v_higgs is NOW DERIVED (line 523-532) via y_t = 1 - gamma[7]*eps,")
print(f"        v = sqrt(2)*m_t/y_t/1000. (R51, 0.002%)")
print(f"        G_F = 1/(sqrt(2)*v^2) follows from the derived v_higgs.")
print(f"        No hardcoded experimental value remains.")
print()

print("=" * 130)
print("  END OF AUDIT")
print("=" * 130)
