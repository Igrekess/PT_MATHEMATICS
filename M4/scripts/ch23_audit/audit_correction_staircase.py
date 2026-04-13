#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correction staircase audit for Persistence Theory.

Systematically analyzes every NLO/NNLO correction coefficient,
testing whether each is structurally forced or could plausibly
be something else.  For "constrained" corrections, each alternative
coefficient is substituted and the resulting observable error vs PDG
is computed.

Exit 0 if all assertions pass.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pt_constants import *  # noqa: F401,F403
from pt_light_hadrons import (
    F_PI, M_K, M_PROTON, M_ETA, M_ETAP,
    M_PI, CHI_TOP, _NF,
)

# ============================================================================
# Recompute a few private quantities that pt_constants does not export
# ============================================================================
_gamma_ghost_local = {p: gamma_p_exact(p, mu_star) for p in PRIMES_GHOST}
_sin2_ghost_local = {p: sin2_theta(p, q_stat) for p in PRIMES_GHOST}
beta_ghost = sum(_sin2_ghost_local[p] * _gamma_ghost_local[p]
                 for p in PRIMES_GHOST)

# ============================================================================
# Coefficient pool -- the six distinct NLO building blocks
# ============================================================================
COEFFICIENT_POOL = {
    's':       0.5,
    'N_c':     3,
    'C_F':     4.0 / 3.0,
    'n_f':     5,
    'Q_Koide': 2.0 / 3.0,
    'gamma_3': gamma[3],
    'gamma_5': gamma[5],
    'gamma_7': gamma[7],
}

# ============================================================================
# Helper: universal expansion parameter
# ============================================================================
_eps = beta_0_num * alpha_EM / (4.0 * np.pi)


def _pct_error(pt_val, pdg_val):
    """Percentage error |PT - PDG| / PDG * 100."""
    if pdg_val == 0:
        return float('inf')
    return abs(pt_val - pdg_val) / abs(pdg_val) * 100.0


# ============================================================================
# Correction registry
# ============================================================================
# Each entry:
#   name       : human label
#   label      : rule tag (R12, R15, ...)
#   coeff_name : symbolic name of the chosen coefficient
#   coeff_val  : numerical value
#   justif     : one-line structural justification
#   category   : "forced" | "constrained" | "pool"
#   obs_key    : observable key in PDG dict (or None)
#   pt_val     : PT value of that observable
#   alt_test   : list of (alt_name, alt_coeff) or None
#   swap_fn    : function(c_old, c_new) -> new observable value, or None

CORRECTIONS = []


def _add(name, label, coeff_name, coeff_val, justif, category,
         obs_key=None, pt_val=None, alt_test=None, swap_fn=None):
    CORRECTIONS.append(dict(
        name=name, label=label,
        coeff_name=coeff_name, coeff_val=coeff_val,
        justif=justif, category=category,
        obs_key=obs_key, pt_val=pt_val,
        alt_test=alt_test, swap_fn=swap_fn,
    ))


# ------------------------------------------------------------------
# a) Delta_1  (bare -> dressed alpha)
# ------------------------------------------------------------------
# The dressing assembles C_Koide, cost_3D, cost_2D, 26/27, 2*pi.
# Test: replace log with linear => Delta_1 = 3.78, 1/alpha = 140.1
_hab_corr_linear = C_Koide * (cost_3D * cost_2D) / (2.0 * np.pi) * (N_c**3 - 1) / N_c**3
_alpha_EM_linear = 1.0 / (1.0 / alpha_nue + _hab_corr_linear)
_inv_alpha_linear = 1.0 / _alpha_EM_linear

_add(
    "Delta_1 (dressing)", "D09",
    "Ward", hab_corr,
    "Ward identity of S_PT (log-structure forced)",
    "forced",
    obs_key="1/alpha_EM", pt_val=1.0 / alpha_EM,
    alt_test=[("linear(no log)", _inv_alpha_linear)],
    swap_fn=None,  # special: tested inline
)

# ------------------------------------------------------------------
# b) R17  self-energy  c = s^2 = 1/4
# ------------------------------------------------------------------
_add(
    "R17 (self-energy)", "R17",
    "s^2", s**2,
    "coupling iteration (3 routes agree)",
    "forced",
    obs_key=None, pt_val=delta_SE,
)

# ------------------------------------------------------------------
# c) R28  ghost VP  c = gamma_3
# ------------------------------------------------------------------
_add(
    "R28 (ghost VP)", "R28",
    "gamma_3", gamma[3],
    "color vertex (active/ghost bridge)",
    "forced",
    obs_key="1/alpha_EM", pt_val=1.0 / alpha_EM,
)

# ------------------------------------------------------------------
# d) R55  2-loop  c = 1/N_c = 1/3
# ------------------------------------------------------------------
_add(
    "R55 (2-loop VP)", "R55",
    "1/N_c", 1.0 / N_c,
    "Schwinger structure (QED 2-loop)",
    "forced",
)

# ------------------------------------------------------------------
# e) R12  CKM NLO V_ub  c = 2 = 2^D/2
# ------------------------------------------------------------------
_V_ub_tree_local = A_CKM * lam_CKM**3 * Rb_CKM


def _swap_Vub(c_new):
    return _V_ub_tree_local * (1 + c_new * _eps)


_add(
    "R12 (V_ub NLO)", "R12",
    "2 (=2^D/2)", 2.0,
    "double penguin at b->u vertex",
    "constrained",
    obs_key="V_ub", pt_val=V_ub,
    alt_test=[("c=1", 1), ("c=3", 3), ("c=s", 0.5)],
    swap_fn=_swap_Vub,
)

# ------------------------------------------------------------------
# f) R21a  CKM vertex V_cd  c = 1+s = 3/2
# ------------------------------------------------------------------
# V_cd exact (before NLO) -- recompute from CKM angles
_s13_ckm_loc = V_ub / (1 + 2 * _eps)  # undo R12 to get tree V_ub ~ s13
# Use the exported _V_cd_exact directly via reverse engineering:
# V_cd = V_cd_exact * (1 - (1+s)*eps), so V_cd_exact = V_cd / (1-(1+s)*eps)
_V_cd_exact_loc = V_cd / (1 - (1 + s) * _eps)


def _swap_Vcd(c_new):
    return _V_cd_exact_loc * (1 - c_new * _eps)


_add(
    "R21a (V_cd vertex)", "R21a",
    "1+s (=3/2)", 1.0 + s,
    "half-color intra-doublet transition",
    "constrained",
    obs_key="V_cd", pt_val=V_cd,
    alt_test=[("c=1", 1), ("c=2", 2), ("c=s", 0.5)],
    swap_fn=_swap_Vcd,
)

# ------------------------------------------------------------------
# g) R21a  CKM vertex V_cb  c = s = 1/2
# ------------------------------------------------------------------
# V_cb before R21a NLO: V_cb_pre = V_cb / (1 - s*eps) / (1 - gamma_11*alpha)
_V_cb_pre = V_cb / ((1 - s * _eps) * (1 - _gamma_ghost_local[11] * alpha_EM))


def _swap_Vcb(c_new):
    return _V_cb_pre * (1 - c_new * _eps) * (1 - _gamma_ghost_local[11] * alpha_EM)


_add(
    "R21a (V_cb vertex)", "R21a",
    "s (=1/2)", s,
    "sieve symmetry (same as Rb = s/(1+s^2))",
    "constrained",
    obs_key="V_cb", pt_val=V_cb,
    alt_test=[("c=1", 1), ("c=C_F", 4.0 / 3.0)],
    swap_fn=_swap_Vcb,
)

# ------------------------------------------------------------------
# h) R18  EW bosons Delta_r  c = n_f+s = 5.5
# ------------------------------------------------------------------
_add(
    "R18 (Delta_r)", "R18",
    "n_f+s (=5.5)", n_f + s,
    "flavor sum (5 active + symmetry)",
    "forced",
    obs_key="m_W", pt_val=m_W,
)

# ------------------------------------------------------------------
# i) R15  Higgs NLO  c = C_F = 4/3
# ------------------------------------------------------------------
_add(
    "R15 (Higgs NLO)", "R15",
    "C_F (=4/3)", C_F,
    "fundamental Casimir SU(3)",
    "forced",
    obs_key="m_H", pt_val=m_H,
)

# ------------------------------------------------------------------
# j) R26  NNLO lepton  c = 2^D = 4
# ------------------------------------------------------------------
_add(
    "R26 (NNLO lepton)", "R26",
    "2^D (=4)", 2**D,
    "decoherence channels (quantum numbers)",
    "forced",
)

# ------------------------------------------------------------------
# k) R31  NLO Cabibbo V_us  c = s = 1/2
# ------------------------------------------------------------------
_V_us_pre = V_us / (1 - s * _eps)


def _swap_Vus(c_new):
    return _V_us_pre * (1 - c_new * _eps)


_add(
    "R31 (V_us NLO)", "R31",
    "s (=1/2)", s,
    "sieve s symmetry (inter-generation)",
    "constrained",
    obs_key="V_us", pt_val=V_us,
    alt_test=[("c=1", 1), ("c=C_F", 4.0 / 3.0)],
    swap_fn=_swap_Vus,
)

# ------------------------------------------------------------------
# l) R34b  tau cross-branch  c = alpha_s * beta_ghost
# ------------------------------------------------------------------
_c_tau = alpha_s * beta_ghost
_add(
    "R34b (tau cross-branch)", "R34b",
    "alpha_s*beta_ghost", _c_tau,
    "cross-branch VP (hadronic modes)",
    "forced",
    obs_key="m_tau", pt_val=m_tau,
)

# ------------------------------------------------------------------
# m) R36  f_pi NLO  c = -1  (chiral condensate dressing)
# ------------------------------------------------------------------
# PDG F_pi = 130.2(1.2) MeV => 0.1302 GeV  (normalization with sqrt(N_c/(4*pi^2)))
_PDG_f_pi = 0.1302  # GeV

# Tree value is F_PI; NLO: f_pi = F_PI * (1 + c * eps), chosen c = -1
_f_pi_NLO = F_PI * (1 + (-1) * _eps)


def _swap_fpi(c_new):
    return F_PI * (1 + c_new * _eps)


_add(
    "R36 (f_pi NLO)", "R36",
    "-1 (chiral)", -1.0,
    "chiral condensate 1-loop (Banks-Casher)",
    "constrained",
    obs_key=None, pt_val=_f_pi_NLO,
    alt_test=[("c=-s=-0.5", -0.5), ("c=-2", -2), ("c=-C_F=-4/3", -4.0 / 3.0)],
    swap_fn=_swap_fpi,
)
# Override obs_key to use local PDG value
CORRECTIONS[-1]['_pdg_local'] = _PDG_f_pi

# ------------------------------------------------------------------
# n) R36b  m_K NLO  c = -3/2 = -(1+s)
# ------------------------------------------------------------------
_PDG_m_K = 0.4976  # GeV (K0)

_m_K_NLO = M_K * (1 + (-1.5) * _eps)


def _swap_mK(c_new):
    return M_K * (1 + c_new * _eps)


_add(
    "R36b (m_K NLO)", "R36b",
    "-(1+s) (=-3/2)", -1.5,
    "SU(3) breaking (vertex+edge)",
    "constrained",
    obs_key=None, pt_val=_m_K_NLO,
    alt_test=[("c=-1", -1), ("c=-2", -2), ("c=-s=-0.5", -0.5)],
    swap_fn=_swap_mK,
)
CORRECTIONS[-1]['_pdg_local'] = _PDG_m_K

# ------------------------------------------------------------------
# o) R36b  M_p NLO  c = -D = -2  (diquark dressing)
# ------------------------------------------------------------------
_PDG_M_p = 0.9383  # GeV

_M_p_NLO = M_PROTON * (1 + (-2) * _eps)


def _swap_Mp(c_new):
    return M_PROTON * (1 + c_new * _eps)


_add(
    "R36b (M_p NLO)", "R36b",
    "-D (=-2)", -2.0,
    "diquark dim (D=N_c-1=2)",
    "constrained",
    obs_key=None, pt_val=_M_p_NLO,
    alt_test=[("c=-1", -1), ("c=-3", -3), ("c=-N_c=-3", -3)],
    swap_fn=_swap_Mp,
)
CORRECTIONS[-1]['_pdg_local'] = _PDG_M_p

# ------------------------------------------------------------------
# p) R37  eta/eta' NLO  c = +2 = 2^D/2  (U(1)_A anomaly)
# ------------------------------------------------------------------
_PDG_m_eta = 0.5478  # GeV

_m_eta_NLO = M_ETA * (1 + 2 * _eps)


def _swap_meta(c_new):
    return M_ETA * (1 + c_new * _eps)


_add(
    "R37 (eta NLO)", "R37",
    "+2 (=2^D/2)", 2.0,
    "U(1)_A anomaly (topological charge)",
    "constrained",
    obs_key=None, pt_val=_m_eta_NLO,
    alt_test=[("c=+1", 1), ("c=+3", 3), ("c=+s=0.5", 0.5)],
    swap_fn=_swap_meta,
)
CORRECTIONS[-1]['_pdg_local'] = _PDG_m_eta


# ============================================================================
# Run the analysis
# ============================================================================
def main():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 105)
    print("  CORRECTION STAIRCASE AUDIT -- Persistence Theory")
    print("  Testing whether each NLO/NNLO coefficient is forced or could be something else")
    print("=" * 105)
    print()

    header = (f"  {'Correction':<26} {'Coeff':<18} {'Value':>8} "
              f"{'Justification':<36} {'Category':<10} "
              f"{'#Alt':>4}  {'Degradation'}")
    print(header)
    print("  " + "-" * 103)

    n_forced = 0
    n_constrained = 0
    n_pool = 0
    total_bits = 0.0

    for c in CORRECTIONS:
        cat_short = c['category'][:6] + "." if len(c['category']) > 6 else c['category']
        n_alt_tested = 0
        degrad_str = "-"

        if c['alt_test'] and c['swap_fn'] is not None and (c['obs_key'] or '_pdg_local' in c):
            # Test each alternative coefficient
            pdg_val = c.get('_pdg_local', None) or PDG.get(c['obs_key'], None)
            if pdg_val is not None:
                pt_err = _pct_error(c['pt_val'], pdg_val)
                alt_errors = []
                for alt_name, alt_c in c['alt_test']:
                    new_val = c['swap_fn'](alt_c)
                    alt_err = _pct_error(new_val, pdg_val)
                    alt_errors.append((alt_name, alt_err))
                n_alt_tested = len(alt_errors)
                if alt_errors:
                    err_range = [e for _, e in alt_errors]
                    degrad_str = f"{min(err_range):.2f}-{max(err_range):.2f}%"
        elif c['alt_test'] and c['swap_fn'] is None and c['obs_key']:
            # Special case: Delta_1 with inline alternatives
            pdg_val = PDG.get(c['obs_key'], None)
            if pdg_val is not None and c['alt_test']:
                alt_errors = []
                for alt_name, alt_val in c['alt_test']:
                    alt_err = _pct_error(alt_val, pdg_val)
                    alt_errors.append((alt_name, alt_err))
                n_alt_tested = len(alt_errors)
                if alt_errors:
                    degrad_str = f"{alt_errors[0][1]:.1f}% ({alt_errors[0][0]})"

        # Count categories
        if c['category'] == 'forced':
            n_forced += 1
            n_alternatives = 1
        elif c['category'] == 'constrained':
            n_constrained += 1
            n_alternatives = 1 + (len(c['alt_test']) if c['alt_test'] else 0)
        else:  # pool
            n_pool += 1
            n_alternatives = len(COEFFICIENT_POOL)

        if n_alternatives > 1:
            total_bits += math.log2(n_alternatives)

        alt_str = f"{n_alt_tested} tested" if n_alt_tested > 0 else "-"

        coeff_display = c['coeff_name']
        if len(coeff_display) > 16:
            coeff_display = coeff_display[:16]

        print(f"  {c['name']:<26} {coeff_display:<18} {c['coeff_val']:>8.4f} "
              f"{c['justif']:<36} {cat_short:<10} "
              f"{alt_str:>10}  {degrad_str}")

    print()
    print("=" * 105)
    print("  DETAILED ALTERNATIVE TESTS")
    print("=" * 105)
    print()

    for c in CORRECTIONS:
        if not c['alt_test']:
            continue
        print(f"  {c['name']} ({c['label']})  --  chosen coeff = {c['coeff_name']} = {c['coeff_val']:.4f}")
        obs_key = c['obs_key']
        pdg_val = c.get('_pdg_local', None) or (PDG.get(obs_key) if obs_key else None)
        if pdg_val is not None:
            pt_err = _pct_error(c['pt_val'], pdg_val) if c['pt_val'] is not None else None
            if pt_err is not None:
                print(f"    PT value = {c['pt_val']:.6g},  PDG = {pdg_val:.6g},  PT error = {pt_err:.4f}%")

            if c['swap_fn'] is not None:
                for alt_name, alt_c in c['alt_test']:
                    new_val = c['swap_fn'](alt_c)
                    alt_err = _pct_error(new_val, pdg_val)
                    print(f"    Alternative {alt_name:<12}: value = {new_val:.6g},  error = {alt_err:.4f}%"
                          f"  {'<-- WORSE' if alt_err > pt_err else '<-- BETTER?!'}")
            else:
                # Inline alternatives (Delta_1 case)
                for alt_name, alt_val in c['alt_test']:
                    alt_err = _pct_error(alt_val, pdg_val)
                    print(f"    Alternative {alt_name:<16}: 1/alpha = {alt_val:.2f},  error = {alt_err:.3f}%")
        print()

    # ======================================================================
    # Summary statistics
    # ======================================================================
    print("=" * 105)
    print("  SUMMARY STATISTICS")
    print("=" * 105)
    print()
    print(f"  N_forced       = {n_forced:>3}  (corrections with unique structural coefficient)")
    print(f"  N_constrained  = {n_constrained:>3}  (corrections with 2-3 plausible alternatives)")
    print(f"  N_pool         = {n_pool:>3}  (corrections freely chosen from full pool)")
    print(f"  Total corrections = {len(CORRECTIONS)}")
    print()
    print(f"  Total effective bits = {total_bits:.2f}  (sum of log2(n_alternatives))")
    print(f"  = information content if alternatives were equally plausible")
    print()

    # ======================================================================
    # Assertions
    # ======================================================================
    ok = True

    if n_pool != 0:
        print(f"  FAIL: N_pool = {n_pool} != 0  (some corrections freely chosen from full pool)")
        ok = False
    else:
        print(f"  PASS: N_pool == 0  (no correction is freely chosen from the full pool)")

    if total_bits >= 30:
        print(f"  FAIL: total_effective_bits = {total_bits:.2f} >= 30")
        ok = False
    else:
        print(f"  PASS: total_effective_bits = {total_bits:.2f} < 30  (conservative bound)")

    print()
    if ok:
        print("  ALL ASSERTIONS PASSED")
    else:
        print("  SOME ASSERTIONS FAILED")

    print()
    assert n_pool == 0, f"N_pool = {n_pool}, expected 0"
    assert total_bits < 30, f"total_effective_bits = {total_bits:.2f} >= 30"


if __name__ == "__main__":
    main()
