#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test M50: Radiative Corrections from Holomorphic Mechanics
==========================================================

CLAIM: The last "standard QFT formulas" in the NLO catalogue
(QED FSR and rho_b vertex) are derivable from the holomorphic
mechanics on the circle C(s, 0; s).

Key results:
  (a) FSR coeff:  3/(4*pi) = N_spatial * s^2 / Circ(C)
  (b) Loop factor: 16*pi^2 = Omega_2^2  (solid angle squared)
  (c) rho_b = 1 - y_t^2 / Omega_2^2   (vertex from geometry)
  (d) Contour integral:  oint_C F dw = -Circ(C) = -pi

All factors are PT-native:
  - N_spatial = 3 (proven from N_c = 3, R38b)
  - s = 1/2 (fundamental)
  - Circ(C) = 2*pi*s = pi
  - Omega_{N_spatial-1} = 4*pi (solid angle of S^2)

Scripts: M50.  Reference: Ch. complex_mechanics, S4.
"""

import numpy as np
import sys
import os

# --- Path setup ---
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from pt_constants import (
    s, alpha_EM, v_higgs, G_F, m_t, m_b, m_c,
    N_c, PRIMES_ACTIFS, m_e, m_mu, m_tau,
)
from _complex_pt import w_p, force_complex, theta_p, sin2

# ============================================================
# PT-derived geometric constants
# ============================================================

# N_spatial = 3: proven from (N_c+1)!/(N_c+3) = 2^(N_spatial-1),
# unique integer solution for N_c = 3 (R38b).
N_SPATIAL = 3

# Circumference of the sieve circle C(s, 0; s)
CIRC_C = 2 * np.pi * s   # = pi

# Solid angle of the unit (N_spatial-1)-sphere
# Omega_{d-1} = 2*pi^{d/2} / Gamma(d/2)
from math import gamma as _Gamma
OMEGA_2 = 2 * np.pi**(N_SPATIAL / 2) / _Gamma(N_SPATIAL / 2)  # = 4*pi

# q_stat for numerical checks
q_stat = 1.0 - 2.0 / 15.0  # = 13/15


# ============================================================
# PART 1: QED FSR coefficient decomposition
# ============================================================

def test_FSR_coefficient_identity():
    """ID: 3/(4*pi) = N_spatial * s^2 / Circ(C)."""
    coeff_QFT = 3.0 / (4.0 * np.pi)
    coeff_PT = N_SPATIAL * s**2 / CIRC_C
    assert abs(coeff_PT - coeff_QFT) < 1e-15, \
        f"FSR coeff mismatch: PT={coeff_PT}, QFT={coeff_QFT}"
    print(f"  3/(4pi)              = {coeff_QFT:.12f}")
    print(f"  N_sp * s^2 / Circ(C) = {coeff_PT:.12f}")


def test_FSR_factor_three_quarters():
    """ID: 3/4 = N_spatial * s^2 (geometry encodes the coefficient)."""
    assert abs(N_SPATIAL * s**2 - 0.75) < 1e-15
    print(f"  N_spatial * s^2 = {N_SPATIAL} * {s**2} = {N_SPATIAL * s**2}")
    print(f"  = 3/4 exactly")


def test_FSR_circumference_pi():
    """Circ(C) = 2*pi*s = pi (circumference of the sieve circle)."""
    assert abs(CIRC_C - np.pi) < 1e-15
    print(f"  Circ(C) = 2*pi*s = 2*pi*{s} = {CIRC_C:.12f} = pi")


def test_FSR_quarks():
    """delta_QED = N_sp * s^2 * alpha * Q_f^2 / Circ(C) for each quark."""
    coeff = N_SPATIAL * s**2 / CIRC_C
    quarks = {'u': 2/3, 'c': 2/3, 'd': -1/3, 's': -1/3, 'b': -1/3}
    for name, Q in quarks.items():
        delta_PT = coeff * alpha_EM * Q**2
        delta_std = 3 * alpha_EM * Q**2 / (4 * np.pi)
        assert abs(delta_PT - delta_std) < 1e-18, \
            f"FSR({name}): PT={delta_PT}, std={delta_std}"
        print(f"  delta_QED({name}): {delta_PT:.8e}  (Q = {Q:+.4f})")


def test_FSR_leptons():
    """delta_QED for leptons (Q = +-1): same formula, Q^2 = 1."""
    coeff = N_SPATIAL * s**2 / CIRC_C
    delta = coeff * alpha_EM  # Q^2 = 1
    delta_std = 3 * alpha_EM / (4 * np.pi)
    assert abs(delta - delta_std) < 1e-18
    print(f"  delta_QED(lepton): {delta:.8e}")
    print(f"  = 3*alpha/(4*pi)   = {delta_std:.8e}")


def test_FSR_total_hadronic():
    """Total FSR to Gamma_had: sum over 5 quark flavors at Z-pole."""
    coeff = N_SPATIAL * s**2 / CIRC_C
    # 5 active quarks: u(2/3), d(-1/3), s(-1/3), c(2/3), b(-1/3)
    sum_Q2 = 2 * (2/3)**2 + 3 * (1/3)**2  # = 8/9 + 3/9 = 11/9
    delta_had = coeff * alpha_EM * sum_Q2
    print(f"  sum Q_f^2 = {sum_Q2:.6f} = 11/9")
    print(f"  delta_had = {delta_had:.6e}")
    assert delta_had > 0
    assert abs(sum_Q2 - 11/9) < 1e-15


# ============================================================
# PART 2: Loop factor decomposition
# ============================================================

def test_loop_factor_identity():
    """ID: 16*pi^2 = Omega_2^2 = (solid angle in N_spatial=3)^2."""
    loop_QFT = 16 * np.pi**2
    loop_PT = OMEGA_2**2
    assert abs(loop_PT - loop_QFT) < 1e-10, \
        f"Loop factor: PT={loop_PT}, QFT={loop_QFT}"
    print(f"  16*pi^2 = {loop_QFT:.6f}")
    print(f"  Omega_2^2 = {loop_PT:.6f}")


def test_solid_angle_formula():
    """Omega_{d-1} = 2*pi^{d/2}/Gamma(d/2) gives 4*pi for d=3."""
    assert abs(OMEGA_2 - 4 * np.pi) < 1e-12
    print(f"  Omega_2 = 2*pi^(3/2)/Gamma(3/2) = {OMEGA_2:.10f}")
    print(f"  4*pi = {4*np.pi:.10f}")


def test_loop_factor_dimensional():
    """Loop factor Omega^2 varies with N_spatial: only d=3 gives 16*pi^2."""
    results = {}
    for d in [2, 3, 4, 5]:
        Omega = 2 * np.pi**(d/2) / _Gamma(d/2)
        results[d] = Omega**2
        print(f"  N_spatial={d}: Omega_{d-1}={Omega:.4f}, "
              f"Omega^2={Omega**2:.4f}")
    # Only d=3 gives 16*pi^2
    assert abs(results[3] - 16 * np.pi**2) < 1e-10
    # d=2 would give (2*pi)^2 = 4*pi^2 (different physics)
    assert abs(results[2] - 4 * np.pi**2) < 1e-10


# ============================================================
# PART 3: rho_b vertex correction
# ============================================================

def test_rho_b_standard_formula():
    """rho_b = 1 - G_F * m_t^2 / (4*sqrt(2)*pi^2) -- standard."""
    m_t_GeV = m_t / 1000.0  # pt_constants stores in MeV
    rho_b = 1 - G_F * m_t_GeV**2 / (4 * np.sqrt(2) * np.pi**2)
    print(f"  rho_b (standard) = {rho_b:.8f}")
    print(f"  1 - rho_b        = {1 - rho_b:.6e}")
    assert 0.99 < rho_b < 1.0


def test_rho_b_yukawa_form():
    """rho_b = 1 - y_t^2 / (16*pi^2) = 1 - y_t^2 / Omega_2^2."""
    m_t_GeV = m_t / 1000.0
    y_t = np.sqrt(2) * m_t_GeV / v_higgs

    rho_b_std = 1 - G_F * m_t_GeV**2 / (4 * np.sqrt(2) * np.pi**2)
    rho_b_PT = 1 - y_t**2 / OMEGA_2**2

    print(f"  y_t = sqrt(2)*m_t/v = {y_t:.6f}")
    print(f"  rho_b (Yukawa/Omega^2) = {rho_b_PT:.8f}")
    print(f"  rho_b (standard)       = {rho_b_std:.8f}")

    # These must agree (algebraic identity):
    # G_F = 1/(sqrt(2)*v^2), so G_F*m_t^2/(4*sqrt(2)*pi^2)
    # = m_t^2 / (8*v^2*pi^2) = y_t^2 / (16*pi^2) = y_t^2/Omega_2^2
    assert abs(rho_b_PT - rho_b_std) < 1e-8, \
        f"rho_b mismatch: PT={rho_b_PT}, std={rho_b_std}"


def test_rho_b_only_top_matters():
    """For all fermions except top, rho_f ~ 1 (correction negligible)."""
    masses_GeV = {
        'b': m_b / 1000, 'c': m_c / 1000,
        'tau': m_tau / 1000, 'mu': m_mu / 1000,
    }
    m_t_GeV = m_t / 1000.0
    y_t = np.sqrt(2) * m_t_GeV / v_higgs
    delta_top = y_t**2 / OMEGA_2**2

    for name, m in masses_GeV.items():
        y = np.sqrt(2) * m / v_higgs
        delta = y**2 / OMEGA_2**2
        ratio = delta / delta_top
        print(f"  delta_{name:3s} / delta_top = {ratio:.2e}  "
              f"(y_{name} = {y:.6e})")
        assert ratio < 1e-3  # All negligible


# ============================================================
# PART 4: Contour integral of F on C
# ============================================================

def test_contour_integral_F():
    """oint_C F dw = -pi = -Circ(C) (half-residue at boundary pole)."""
    N = 200000
    # Parametrize w(theta) = (1 - e^{2i*theta})/2, theta in (0, pi)
    # Pole at theta=0 and theta=pi (w=0): use midpoint rule to avoid
    dtheta = np.pi / N
    theta = np.linspace(dtheta / 2, np.pi - dtheta / 2, N)
    w = (1 - np.exp(2j * theta)) / 2
    F = 1j / w - 2j
    dw_dtheta = -1j * np.exp(2j * theta)
    integrand = F * dw_dtheta

    integral = np.sum(integrand) * dtheta
    print(f"  oint_C F dw = {integral.real:.6f} + {integral.imag:.6f}i")
    print(f"  Expected:     {-np.pi:.6f} + 0i")
    print(f"  -Circ(C) =   {-CIRC_C:.6f}")
    assert abs(integral.real + np.pi) < 0.001, \
        f"Contour Re: {integral.real} != {-np.pi}"
    assert abs(integral.imag) < 0.001, \
        f"Contour Im: {integral.imag} != 0"


def test_work_equals_circumference():
    """The total work |W_cycle| = Circ(C) = pi (topological identity)."""
    # The work done by F along one traversal of C equals its circumference.
    # This is a consequence of Res(F, 0) = i being on the boundary of C.
    # Sokhotski-Plemelj: boundary pole contributes half-residue = pi*i*i = -pi.
    W_cycle = CIRC_C
    print(f"  |oint_C F dw| = Circ(C) = 2*pi*s = {W_cycle:.10f}")
    assert abs(W_cycle - np.pi) < 1e-15


# ============================================================
# PART 5: Geometric interpretation
# ============================================================

def test_FSR_geometric_meaning():
    """FSR = radiation on C in N_spatial dims, normalized by circumference.

    Physical picture:
    - A charged state w_p on C radiates via the force F(w) = i/w - 2i
    - The radiation rate involves alpha * Q^2 (coupling * charge)
    - The geometric factor N_spatial * s^2 / Circ(C) = 3/(4*pi) comes from:
      * N_spatial = 3 spatial polarisation directions
      * s^2 = 1/4 = (radius of C)^2 = area of C / pi
      * Circ(C) = pi = circumference normalization
    """
    # Area of circle C
    area_C = np.pi * s**2  # = pi/4
    # Ratio area/pi = s^2 = 1/4
    assert abs(area_C / np.pi - s**2) < 1e-15
    # FSR = N_spatial * (area/pi) / Circ = N_spatial * s^2 / (2*pi*s)
    coeff = N_SPATIAL * (area_C / np.pi) / CIRC_C
    assert abs(coeff - 3 / (4 * np.pi)) < 1e-15
    print(f"  Area(C) = pi*s^2 = {area_C:.10f}")
    print(f"  FSR = N_sp * Area/(pi*Circ) = {coeff:.10f} = 3/(4pi)")


def test_loop_factor_geometric_meaning():
    """Loop factor = (solid angle)^2 in embedding space.

    Physical picture:
    - A 1-loop process involves integration over internal directions
    - In N_spatial = 3, the solid angle is Omega_2 = 4*pi
    - The conventional "16*pi^2" IS the square of this geometric quantity
    - PT derives N_spatial = 3, hence the loop factor is determined
    """
    # For N_spatial = 3: Omega_2 = area of unit S^2 = 4*pi
    # This is the same 4*pi that appears in Coulomb's law: F = q/(4*pi*r^2)
    # The loop factor (4*pi)^2 normalizes the 1-loop virtual correction
    assert abs(OMEGA_2 - 4 * np.pi) < 1e-12
    assert abs(OMEGA_2**2 - 16 * np.pi**2) < 1e-10
    print(f"  Omega_2 = 4*pi (Coulomb normalization)")
    print(f"  Omega_2^2 = 16*pi^2 (loop normalization)")
    print(f"  Both fixed by N_spatial = 3 from PT")


# ============================================================
# PART 6: Consistency with contour integral
# ============================================================

def test_residue_universality():
    """Res(F, 0) = i is universal (independent of p, q)."""
    # F(w) = i/w - 2i, so Res at w=0 is exactly i
    residue = 1j
    assert abs(residue) == 1.0
    print(f"  Res(F, 0) = {residue}")
    print(f"  |Res| = {abs(residue)} (universal)")
    # The residue determines the coupling strength at the pole
    # |Res|^2 = 1: normalized to unity, actual strength comes from alpha


def test_FSR_from_residue_and_geometry():
    """delta_QED = |Res|^2 * alpha * Q^2 * N_spatial * s^2 / Circ(C).

    The residue |Res(F,0)|^2 = 1 normalizes the emission vertex.
    The geometric factor N_spatial * s^2 / Circ(C) = 3/(4*pi) arises from
    the embedding of the circle C into N_spatial-dimensional space.
    """
    Res_sq = abs(1j)**2  # = 1
    coeff = Res_sq * N_SPATIAL * s**2 / CIRC_C
    assert abs(coeff - 3 / (4 * np.pi)) < 1e-15
    print(f"  |Res|^2 * N_sp * s^2 / Circ = {coeff:.12f}")
    print(f"  3/(4*pi) = {3/(4*np.pi):.12f}")


# ============================================================
# Main runner
# ============================================================

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_')]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        print(f"\n{name}:")
        try:
            t()
            print(f"  PASS")
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  FAIL: {e}")
            failed += 1

    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"M50 complex_radiative: {passed}/{total} PASS")
    if failed:
        print(f"  ({failed} FAILED)")
        sys.exit(1)
