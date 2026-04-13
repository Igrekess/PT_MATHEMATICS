#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D20 — sin^2(theta_W) Dressing: Closed-form dressing of sin^2(theta_W).
GENUINE DERIVATION: Same C_Koide (Fisher-Koide Identity, Prop. fisher_koide)
mechanism as alpha_EM (D09 -- Bare alpha_EM), zero free parameters.

Theorem chain: D09 (Bare alpha_EM) -> C_Koide (Fisher-Koide Identity)
              -> D20 (sin^2(theta_W) Dressing)

GEOMETRIC RULE (constraint on U(1)^3 spin foam):
    The sieve correction base is UNIVERSAL:
        Correction_base = C_Koide * ln(cost_3D * cost_2D) / (2*pi)

    1. VERTEX (alpha_EM):  1/alpha  += Correction_base * 26/27   [order 0, charged]
    2. METRIC (sin^2_W):   sin^2    -= Correction_base * alpha   [order 1, neutral]

    - 26/27 for alpha: EM couples to 26 charged states out of 27 = N_gen^N_gen
    - alpha for sin^2: Z (neutral current) couples to ALL 27 states => no 26/27
      but acquires one factor of alpha (one level deeper in the hierarchy)
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import sys

# --- Experimental references ---
SIN2_CODATA = 0.23121    # sin^2(theta_W) MS-bar at Z-pole, PDG 2024 (consistent with pt_constants)
ALPHA_CODATA = 1.0 / 137.035999177  # CODATA 2022

# --- Step 1: Tree-level sin^2 from gamma_p ---
mu_star = 15.0
q_stat = 1.0 - 2.0 / mu_star   # = 13/15

print('mu* = {}, q_stat = 1 - 2/{} = {:.10f}'.format(mu_star, int(mu_star), q_stat))

# Compute gamma_p (anomalous dimensions) at mu* = 15
active_primes = [3, 5, 7]
gamma_values = {}

print('\nComputing gamma_p at mu* = {}:'.format(int(mu_star)))
for p in active_primes:
    qp = q_stat ** p
    delta_p = (1.0 - qp) / p
    sin2_p = delta_p * (2.0 - delta_p)
    # gamma_p = -d(ln sin^2)/d(ln mu) = analytic formula
    numerator = 4.0 * p * q_stat**(p - 1) * (1.0 - delta_p)
    denominator = mu_star * (1.0 - qp) * (2.0 - delta_p)
    gamma_p = numerator / denominator
    gamma_values[p] = gamma_p
    print('  p={}: delta_p={:.8f}, sin^2={:.8f}, gamma_p={:.6f}'.format(
        p, delta_p, sin2_p, gamma_p))

# Tree-level Weinberg angle: gamma_7^2 / sum(gamma_p^2)
sum_gamma2 = sum(g**2 for g in gamma_values.values())
sin2_tree = gamma_values[7]**2 / sum_gamma2

print('\nsin^2(theta_W) tree = gamma_7^2 / sum(gamma_p^2)')
print('  = {:.6f}^2 / {:.6f} = {:.6f}'.format(
    gamma_values[7], sum_gamma2, sin2_tree))

# --- Step 2: Universal correction base (same as alpha_EM dressing) ---
# cost_3D and cost_2D from Catalan: 3^2 - 2^3 = 1
cost_3D = np.log(9) / np.log(7)   # ln(3^2) / ln(7) = ln(9)/ln(7)
cost_2D = np.log(8) / np.log(6)   # ln(2^3) / ln(6) = ln(8)/ln(6)

# C_Koide DERIVED from Q = 2/3 (same as pt_constants.py)
def _gamma_p_exact(p, mu):
    if mu <= 2.01: return 0.0
    q = 1.0 - 2.0 / mu
    qp = q**p; d = (1.0 - qp) / p
    if d < 1e-15 or abs(2.0 - d) < 1e-15: return 0.0
    return 4.0*p*q**(p-1)*(1.0-d) / (mu*(1.0-qp)*(2.0-d))

_mu_end = 3.0 * np.pi
_S_int = {p: quad(lambda mu, pp=p: _gamma_p_exact(pp, mu)/mu, p, _mu_end, limit=200)[0]
          for p in active_primes}
def _koide_Q(m1, m2, m3):
    return (m1+m2+m3) / (m1**0.5+m2**0.5+m3**0.5)**2
C_Koide = brentq(lambda C: _koide_Q(np.exp(-C*_S_int[3]), np.exp(-C*_S_int[5]),
                                      np.exp(-C*_S_int[7])) - 2.0/3.0, 5, 50)

correction_base = C_Koide * np.log(cost_3D * cost_2D) / (2.0 * np.pi)

print('\nUniversal correction base:')
print('  C_Koide       = {:.2f} (Q = 2/3, D17b)'.format(C_Koide))
print('  cost_3D       = ln(9)/ln(7) = {:.6f}'.format(cost_3D))
print('  cost_2D       = ln(8)/ln(6) = {:.6f}'.format(cost_2D))
print('  Correction_base = C_Koide * ln(cost_3D * cost_2D) / (2*pi) = {:.6f}'.format(
    correction_base))

# --- Step 3: Verify alpha_EM dressing uses same base ---
# For alpha_EM (D09 — Bare alpha_EM): 1/alpha += Correction_base * 26/27
alpha_bare_inv = 1.0 / np.prod([delta_p * (2.0 - delta_p)
    for p in active_primes
    for delta_p in [(1.0 - q_stat**p) / p]])
dressing_alpha = correction_base * 26.0 / 27.0
alpha_dressed_inv = alpha_bare_inv + dressing_alpha

print('\nVerification alpha_EM (D09):')
print('  1/alpha_bare    = {:.3f}'.format(alpha_bare_inv))
print('  Dressing        = Correction_base * 26/27 = {:.6f}'.format(dressing_alpha))
print('  1/alpha_dressed = {:.3f}'.format(alpha_dressed_inv))
print('  1/alpha_CODATA  = {:.3f}'.format(1.0 / ALPHA_CODATA))

# --- Step 4: sin^2 dressing ---
# For sin^2 (D20 — sin^2(theta_W) Dressing): sin^2 -= Correction_base * alpha_EM
# - No 26/27: Z couples to ALL 27 states (neutral current)
# - Factor alpha: one level deeper in hierarchy (metric, not vertex)
alpha_EM = ALPHA_CODATA  # use physical alpha for self-consistent dressing
delta_sin2 = correction_base * alpha_EM

sin2_dressed = sin2_tree - delta_sin2

print('\nsin^2(theta_W) dressing (D20):')
print('  sin^2_tree    = {:.6f}'.format(sin2_tree))
print('  delta_sin2    = Correction_base * alpha_EM = {:.6f}'.format(delta_sin2))
print('  sin^2_dressed = {:.6f}'.format(sin2_dressed))
print('  sin^2_CODATA  = {:.6f}'.format(SIN2_CODATA))

err_tree = abs(sin2_tree - SIN2_CODATA) / SIN2_CODATA * 100
err_dressed = abs(sin2_dressed - SIN2_CODATA) / SIN2_CODATA * 100
improvement = err_tree / err_dressed

print('\n  Error tree    = {:.4f}%'.format(err_tree))
print('  Error dressed = {:.4f}%'.format(err_dressed))
print('  Improvement   = {:.1f}x'.format(improvement))

# --- Step 5: Geometric rule summary ---
print('\n=== GEOMETRIC RULE ===')
print('  Same Correction_base = {:.6f} for BOTH alpha and sin^2'.format(
    correction_base))
print('  alpha_EM:  x 26/27    (charged sector, order 0, vertex)')
print('  sin^2_W:   x alpha    (neutral sector, order 1, metric)')
print('  26/27 = (3^3 - 1)/3^3 : only charged states couple to photon')
print('  alpha = 1/137 : neutral Z couples to ALL states, one loop deeper')

# --- Assertions ---
assert abs(sin2_tree - 0.2380) < 0.005, \
    'FAIL: tree-level off: {:.4f}'.format(sin2_tree)
assert err_tree > 2.0, \
    'FAIL: tree error suspiciously low: {:.4f}%'.format(err_tree)
assert err_dressed < 0.5, \
    'FAIL: dressed error too large: {:.4f}%'.format(err_dressed)
assert improvement > 5.0, \
    'FAIL: improvement too small: {:.1f}x'.format(improvement)
assert abs(alpha_dressed_inv - 1.0 / ALPHA_CODATA) < 0.5, \
    'FAIL: alpha dressing inconsistent: {:.3f}'.format(alpha_dressed_inv)

print('\nD20 VERIFIED: sin^2_dressed = {:.4f} ({:.3f}% from CODATA, {:.1f}x improvement).'.format(
    sin2_dressed, err_dressed, improvement))

sys.exit(0)
