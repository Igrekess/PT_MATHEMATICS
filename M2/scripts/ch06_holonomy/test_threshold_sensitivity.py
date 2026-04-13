#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: Sensitivity of PT predictions to the activation threshold tau.

Criterion: a prime p is "active" iff gamma_p(mu*) > tau.
We verify that for any tau in a wide band [0.43, 0.60], the active set
remains {3, 5, 7} and therefore ALL predictions are identical.

The gap gamma_7 - gamma_11 >> 0 guarantees structural robustness:
no fine-tuning of the threshold is needed.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pt_constants import gamma_p_exact, sin2_theta, delta_p, PRIMES_ACTIFS, mu_star

# ── Parameters ───────────────────────────────────────────────────────────────
PRIMES_TEST = [3, 5, 7, 11, 13]
THRESHOLDS = [0.43, 0.45, 0.50, 0.55, 0.59]
REFERENCE_SET = {3, 5, 7}

# ── Compute gamma_p at mu* = 15 ─────────────────────────────────────────────
gamma = {p: gamma_p_exact(p, mu_star) for p in PRIMES_TEST}

print("=" * 80)
print("  Threshold sensitivity analysis for gamma_p > tau")
print("=" * 80)
print()

# ── Formatted table ─────────────────────────────────────────────────────────
header = (f"{'tau':>6s}   {'g3':>6s}  {'g5':>6s}  {'g7':>6s}  "
          f"{'g11':>6s}  {'g13':>6s}  {'Active set':<16s}  "
          f"{'Da_EM':>7s}  {'Status':<10s}")
print(header)
print("-" * len(header))

results = []
for tau in THRESHOLDS:
    active = {p for p in PRIMES_TEST if gamma[p] > tau}
    identical = (active == REFERENCE_SET)
    delta_alpha = 0.0 if identical else float('nan')
    status = "IDENTICAL" if identical else "CHANGED"
    results.append((tau, active, identical))
    print(f"{tau:6.2f}   {gamma[3]:.3f}  {gamma[5]:.3f}  {gamma[7]:.3f}  "
          f"{gamma[11]:.3f}  {gamma[13]:.3f}  "
          f"{{{','.join(str(p) for p in sorted(active))}}}{'':>{16 - 2 - len(','.join(str(p) for p in sorted(active)))}}  "
          f"{delta_alpha:6.3f}%  {status}")

print()

# ── Stability margin ────────────────────────────────────────────────────────
margin = gamma[7] - gamma[11]
band_lo = gamma[11]
band_hi = gamma[7]
band_width = band_hi - band_lo

print(f"Stability margin:  gamma_7 - gamma_11 = {gamma[7]:.4f} - {gamma[11]:.4f} = {margin:.4f}")
print(f"Robustness band:   [gamma_11, gamma_7] = [{band_lo:.3f}, {band_hi:.3f}]  (width = {band_width:.3f})")
print(f"Any threshold tau in ({band_lo:.3f}, {band_hi:.3f}) gives the same physics.")
print()

# ── Tests ────────────────────────────────────────────────────────────────────
all_pass = True

# T1: All thresholds give active set {3,5,7}
t1 = all(ident for _, _, ident in results)
tag1 = "PASS" if t1 else "FAIL"
print(f"T1  All 5 thresholds give active set {{3,5,7}}      : {tag1}")
if not t1:
    all_pass = False

# T2: Stability margin > 0.1
t2 = margin > 0.1
tag2 = "PASS" if t2 else "FAIL"
print(f"T2  Stability margin gamma_7 - gamma_11 > 0.1      : {tag2}  ({margin:.4f})")
if not t2:
    all_pass = False

# T3: Robustness band width > 0.15
t3 = band_width > 0.15
tag3 = "PASS" if t3 else "FAIL"
print(f"T3  Robustness band width > 0.15                    : {tag3}  ({band_width:.4f})")
if not t3:
    all_pass = False

print()
if all_pass:
    print("All tests PASSED.")
else:
    print("Some tests FAILED.")

sys.exit(0 if all_pass else 1)
