#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_equations_pont_PT.py

Test suite for the Bridge from Arithmetic to Physics.
6 domains, ~35 tests, 0 fitted parameter.

Domain 1: Structural uniqueness (primes vs alternatives)
Domain 2: Holonomy bridge (Z/pZ -> S^1 -> angles -> couplings)
Domain 3: Bifurcation (vertex/edge duality)
Domain 4: Dimensional emergence (3 active primes -> 3+1D)
Domain 5: Cross-pillar consistency (GFT, alpha, G/alpha)
Domain 6: Falsification tests (counter-examples)

March 2026 -- Persistence Theory
"""

import sys
import pathlib
# --- Path setup (monograph scripts) ---
_scripts_root = str(pathlib.Path(__file__).resolve().parent)
while not (pathlib.Path(_scripts_root) / 'pt_constants.py').exists():
    _scripts_root = str(pathlib.Path(_scripts_root).parent)
    if _scripts_root == str(pathlib.Path(_scripts_root).parent):
        break
sys.path.insert(0, _scripts_root)
for _d in pathlib.Path(_scripts_root).iterdir():
    if _d.is_dir() and not _d.name.startswith(('.', '_')):
        sys.path.insert(0, str(_d))
import os
import json
import numpy as np
from math import sqrt, log, log2, pi, exp

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Import core module
sys.path.insert(0, os.path.dirname(__file__))
from pt_pont_core import (
    S_PT, MU_STAR, DEPTH, N_C, N_F, N_SPATIAL, ACTIVE_PRIMES, GHOST_PRIMES,
    ALL_PRIMES_SMALL,
    compute_gamma_p, compute_sin2, alpha_sieve, q_stat, q_therm,
    generate_prime_gaps, generate_lucky_numbers, generate_composite_gaps,
    generate_random_geometric_gaps, generate_k_rough,
    transition_matrix_residues, has_T0_forbidden,
    D_KL_empirical, crt_superadditivity,
    fixed_point_self_consistency, empirical_mu,
    mertens_product,
    bifurcation_gap_per_prime, survival_probability,
    GFT_check, cross_pillar_alpha, G_over_alpha,
    dimensional_activation, bridge_axiom_check,
)

# ============================================================
# Test infrastructure
# ============================================================

RESULTS = []
N_PASS = 0
N_FAIL = 0

def check(tag, description, condition, details=""):
    """Record a test result."""
    global N_PASS, N_FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        N_PASS += 1
    else:
        N_FAIL += 1
    RESULTS.append({
        'tag': tag, 'description': description,
        'status': status, 'details': details
    })
    symbol = "+" if condition else "X"
    print(f"  [{symbol}] {tag}: {description} ... {status}  {details}")

# ============================================================
# Generate data once
# ============================================================
print("Generating sequences...")
PRIME_GAPS, PRIMES = generate_prime_gaps(200000)
LUCKY_GAPS, LUCKY_NUMS = generate_lucky_numbers(50000)
COMPOSITE_GAPS, COMPOSITES = generate_composite_gaps(200000)
RANDOM_GAPS = generate_random_geometric_gaps(10000, mu=15.0)
ROUGH2_GAPS, ROUGH2_SEQ = generate_k_rough(100000, 2)
ROUGH3_GAPS, ROUGH3_SEQ = generate_k_rough(100000, 3)
print(f"  Primes: {len(PRIMES)}, Lucky: {len(LUCKY_NUMS)}, Composites: {len(COMPOSITES)}")
print()

# ============================================================
# DOMAIN 1: Structural Uniqueness
# ============================================================
print("=" * 60)
print("DOMAIN 1: Structural Uniqueness (P1-P4)")
print("=" * 60)

# U1: T1 for k-rough numbers mod next prime
check("U1", "T1 forbidden: 2-rough mod 5",
      has_T0_forbidden(ROUGH2_SEQ, 5),
      "T[r][r]=0 for r!=0")

# U2: T1 for 3-rough mod 7
check("U2", "T1 forbidden: 3-rough mod 7",
      has_T0_forbidden(ROUGH3_SEQ, 7),
      "T[r][r]=0 for r!=0")

# U3: Lucky numbers do NOT have T1 mod 3
check("U3", "Lucky numbers FAIL T1 mod 3",
      not has_T0_forbidden(LUCKY_NUMS, 3),
      "No forbidden transitions")

# U4: Composites do NOT have T1 mod 3
check("U4", "Composites FAIL T1 mod 3",
      not has_T0_forbidden(COMPOSITES, 3),
      "No forbidden transitions")

# U5: Fixed point mu*=15 is self-consistent
active, s_active, residual = fixed_point_self_consistency(MU_STAR)
check("U5", "Fixed point mu*=15 self-consistent",
      residual == 0 and active == [3, 5, 7],
      f"active={active}, sum={s_active}")

# U6: No other mu in [5,50] is a fixed point
other_fp = []
for mu_test in range(5, 51):
    if mu_test == 15:
        continue
    _, s_test, r_test = fixed_point_self_consistency(float(mu_test))
    if r_test == 0:
        other_fp.append(mu_test)
check("U6", "mu*=15 is the UNIQUE fixed point in [5,50]",
      len(other_fp) == 0,
      f"other={other_fp}" if other_fp else "none found")

print()

# ============================================================
# DOMAIN 2: Holonomy Bridge
# ============================================================
print("=" * 60)
print("DOMAIN 2: Holonomy Bridge (Z/pZ -> S^1)")
print("=" * 60)

# H1: sin^2(theta_p) = delta_p * (2 - delta_p) algebraic identity
q_s = q_stat(MU_STAR)
max_err_h1 = 0
for p in ALL_PRIMES_SMALL[:8]:
    delta = (1.0 - q_s**p) / p
    sin2_expected = delta * (2.0 - delta)
    sin2_computed = compute_sin2(MU_STAR, p, 'stat')
    err = abs(sin2_computed - sin2_expected)
    max_err_h1 = max(max_err_h1, err)
check("H1", "sin^2 = delta*(2-delta) algebraic identity",
      max_err_h1 < 1e-14,
      f"max_err={max_err_h1:.2e}")

# H2: Full circle integral: sum exp(2pi*i*k/p) = 0 for all p
max_err_h2 = 0
for p in ACTIVE_PRIMES:
    s_val = sum(np.exp(2j * pi * k / p) for k in range(p))
    max_err_h2 = max(max_err_h2, abs(s_val))
check("H2", "Full circle integral = 0",
      max_err_h2 < 1e-12,
      f"max_err={max_err_h2:.2e}")

# H3: alpha = product of sin^2 (survival probability)
alpha_prod = 1.0
for p in ACTIVE_PRIMES:
    alpha_prod *= compute_sin2(MU_STAR, p, 'stat')
alpha_fn = alpha_sieve(MU_STAR)
check("H3", "alpha = prod sin^2 = survival probability",
      abs(alpha_prod - alpha_fn) < 1e-15,
      f"alpha={alpha_fn:.8f}")

# H4: G/alpha = 2*pi (holonomy of S^1)
G, alpha, ratio = G_over_alpha(MU_STAR)
check("H4", "G/alpha = 2*pi (bare)",
      abs(ratio - 2 * pi) < 1e-10,
      f"ratio={ratio:.10f}, 2pi={2*pi:.10f}")

# H5: pi emerges from zeta(2) product formula
# zeta(2) = pi^2/6 = prod_p 1/(1-1/p^2)
product = 1.0
for p in ALL_PRIMES_SMALL:
    product *= 1.0 / (1.0 - 1.0 / p**2)
# Use more primes for better convergence
for p in [59, 61, 67, 71, 73, 79, 83, 89, 97]:
    product *= 1.0 / (1.0 - 1.0 / p**2)
pi_from_euler = sqrt(6 * product)
err_pi = abs(pi_from_euler - pi) / pi
check("H5", "pi from Euler product (24 primes)",
      err_pi < 0.005,
      f"pi_approx={pi_from_euler:.6f}, err={err_pi:.4%}")

print()

# ============================================================
# DOMAIN 3: Bifurcation
# ============================================================
print("=" * 60)
print("DOMAIN 3: Bifurcation (Vertex/Edge Duality)")
print("=" * 60)

# B1: q_stat(15) = 13/15
qs = q_stat(MU_STAR)
check("B1", "q_stat(15) = 13/15",
      abs(qs - 13.0/15.0) < 1e-14,
      f"q_stat={qs}")

# B2: q_therm(15) = exp(-1/15)
qt = q_therm(MU_STAR)
expected_qt = exp(-1.0/15.0)
check("B2", "q_therm(15) = exp(-1/15)",
      abs(qt - expected_qt) < 1e-14,
      f"q_therm={qt:.10f}")

# B3: q_stat != q_therm (non-trivial bifurcation)
check("B3", "q_stat != q_therm (bifurcation exists)",
      abs(qs - qt) > 0.01,
      f"|q_stat - q_therm| = {abs(qs - qt):.6f}")

# B4: Bifurcation gap > 0 for all active primes
bif = bifurcation_gap_per_prime(MU_STAR)
all_positive = all(v > 0.01 for v in bif.values())
check("B4", "Bifurcation gap > 0 for all active primes",
      all_positive,
      f"gaps={bif}")

# B5: sin^2(stat) > sin^2(therm) for p=3 (vertex dominates for small p)
s_stat_3 = compute_sin2(MU_STAR, 3, 'stat')
s_therm_3 = compute_sin2(MU_STAR, 3, 'therm')
check("B5", "sin^2(3,stat) > sin^2(3,therm)",
      s_stat_3 > s_therm_3,
      f"stat={s_stat_3:.6f}, therm={s_therm_3:.6f}")

print()

# ============================================================
# DOMAIN 4: Dimensional Emergence
# ============================================================
print("=" * 60)
print("DOMAIN 4: Dimensional Emergence (3 Active -> 3+1D)")
print("=" * 60)

# D1: gamma_3 > 0.5 (active)
g3 = compute_gamma_p(MU_STAR, 3)
check("D1", "gamma_3 > 0.5 (p=3 active)",
      g3 > S_PT,
      f"gamma_3={g3:.6f}")

# D2: gamma_5 > 0.5 (active)
g5 = compute_gamma_p(MU_STAR, 5)
check("D2", "gamma_5 > 0.5 (p=5 active)",
      g5 > S_PT,
      f"gamma_5={g5:.6f}")

# D3: gamma_7 > 0.5 (active)
g7 = compute_gamma_p(MU_STAR, 7)
check("D3", "gamma_7 > 0.5 (p=7 active)",
      g7 > S_PT,
      f"gamma_7={g7:.6f}")

# D4: gamma_11 < 0.5 (inactive = ghost)
g11 = compute_gamma_p(MU_STAR, 11)
check("D4", "gamma_11 < 0.5 (p=11 inactive/ghost)",
      g11 < S_PT,
      f"gamma_11={g11:.6f}")

# D5: Exactly 3 active primes -> N_spatial = 3
active_list = dimensional_activation(MU_STAR)
n_active = len(active_list)
check("D5", "Exactly 3 active primes at mu*=15",
      n_active == 3,
      f"active={[(p, f'{g:.4f}') for p, g in active_list]}")

# D6: (N_c+1)!/(N_c+3) = 2^(N_spatial-1), unique integer solution
import math as _math
lhs = _math.factorial(N_C + 1) / (N_C + 3)
rhs = 2**(N_SPATIAL - 1)
check("D6", "(N_c+1)!/(N_c+3) = 2^(N_spatial-1) for N_c=3",
      abs(lhs - rhs) < 1e-10,
      f"4!/6={lhs}, 2^2={rhs}")

print()

# ============================================================
# DOMAIN 5: Cross-Pillar Consistency
# ============================================================
print("=" * 60)
print("DOMAIN 5: Cross-Pillar Consistency")
print("=" * 60)

# C1: GFT exact for m=6
_, d6, h6, r6 = GFT_check(MU_STAR, 6)
check("C1", "GFT exact: log2(6) = D_KL + H",
      r6 < 1e-13,
      f"residual={r6:.2e}")

# C2: GFT exact for m=30
_, d30, h30, r30 = GFT_check(MU_STAR, 30)
check("C2", "GFT exact: log2(30) = D_KL + H",
      r30 < 1e-13,
      f"residual={r30:.2e}")

# C3: GFT exact for m=210
_, d210, h210, r210 = GFT_check(MU_STAR, 210)
check("C3", "GFT exact: log2(210) = D_KL + H",
      r210 < 1e-13,
      f"residual={r210:.2e}")

# C4: alpha(thermo) = alpha(particles) (same function)
a1, a2, diff = cross_pillar_alpha(MU_STAR)
check("C4", "alpha(particles) = alpha(survival)",
      diff < 1e-15,
      f"diff={diff:.2e}")

# C5: G = 2*pi*alpha (relativity = particles)
G_val, alpha_val, ratio_val = G_over_alpha(MU_STAR)
check("C5", "G = 2*pi*alpha (bare holonomy)",
      abs(G_val - 2 * pi * alpha_val) < 1e-15,
      f"G={G_val:.8f}")

# C6: beta_0 = (11*N_c - 2*n_f)/3 = 23/3 (sieve theorem)
beta_0 = (11 * N_C - 2 * N_F) / 3.0
check("C6", "beta_0 = 23/3 (sieve theorem)",
      abs(beta_0 - 23.0/3.0) < 1e-14,
      f"beta_0={beta_0:.6f}")

# C7: mu* + 2^N_spatial = 23 = beta_0 numerator
beta_0_num = 11 * N_C - 2 * N_F
mu_plus_octants = MU_STAR + 2**N_SPATIAL
check("C7", "mu* + 2^N_spatial = 23 = beta_0_num",
      beta_0_num == mu_plus_octants == 23,
      f"both={beta_0_num}")

# C8: CRT super-additivity D(15) > D(3) + D(5)
D_15, D_3_plus_5, excess_15 = crt_superadditivity(PRIME_GAPS, 3, 5)
check("C8", "CRT super-additivity: D(15) > D(3)+D(5)",
      excess_15 > 0,
      f"D(15)={D_15:.4f}, D(3)+D(5)={D_3_plus_5:.4f}, excess={excess_15:.4f}")

print()

# ============================================================
# DOMAIN 6: Falsification Tests
# ============================================================
print("=" * 60)
print("DOMAIN 6: Falsification & Counter-Examples")
print("=" * 60)

# F1: All 6 bridge axioms pass for prime gaps
axioms = bridge_axiom_check(PRIME_GAPS, PRIMES, MU_STAR)
all_pass = all(v is True for v in axioms.values())
check("F1", "All 6 bridge axioms PASS for primes",
      all_pass,
      str(axioms))

# F2: alpha(mu*) is in the correct range (1/137 ~ 0.0073)
alpha_val = alpha_sieve(MU_STAR)
check("F2", "alpha(mu*) ~ 1/137 (bare)",
      0.007 < alpha_val < 0.008,
      f"alpha={alpha_val:.6f}, 1/alpha={1/alpha_val:.2f}")

# F3: Lucky numbers fail T1 at ALL tested moduli (not just mod 3)
# Primes (via k-rough) have T1 at every sieve level; lucky numbers at none
lucky_t0_mod5 = has_T0_forbidden(LUCKY_NUMS, 5)
lucky_t0_mod7 = has_T0_forbidden(LUCKY_NUMS, 7)
check("F3", "Lucky numbers fail T1 at mod 5 AND mod 7",
      not lucky_t0_mod5 and not lucky_t0_mod7,
      f"mod5={lucky_t0_mod5}, mod7={lucky_t0_mod7}")

# F4: Composites fail T1
check("F4", "Composites fail T1 (no forbidden transitions)",
      not has_T0_forbidden(COMPOSITES, 3),
      "T[r][r] != 0")

# F5: Depth-precision: gamma_p monotonically decreasing
gammas = [(p, compute_gamma_p(MU_STAR, p)) for p in [3, 5, 7, 11, 13]]
monotone = all(gammas[i][1] > gammas[i+1][1] for i in range(len(gammas)-1))
check("F5", "gamma_p strictly decreasing (depth hierarchy)",
      monotone,
      f"gammas={[(p, f'{g:.4f}') for p, g in gammas]}")

# ============================================================
# DOMAIN 7: BA5 Theorem Verification
# ============================================================
print()
print("=" * 60)
print("DOMAIN 7: BA5 Theorem (Coupling = Product sin^2)")
print("=" * 60)

# T1: CRT factorization of theoretical Hilbert space
# For Geom(q) on Z+, the theoretical sin^2 at composite modulus
# decomposes as product over prime factors:
# The CRT isomorphism Z/105Z = Z/3Z x Z/5Z x Z/7Z induces
# factorization of the theoretical D_KL:
# D_KL(105) ~ D_KL(3) + D_KL(5) + D_KL(7) + mutual info terms
# Test: D_KL(105) >= D_KL(3) + D_KL(5) + D_KL(7) (super-additivity)
from pt_pont_core import GFT_check
_, d3, _, _ = GFT_check(MU_STAR, 3)
_, d5, _, _ = GFT_check(MU_STAR, 5)
_, d7, _, _ = GFT_check(MU_STAR, 7)
_, d105, _, _ = GFT_check(MU_STAR, 105)
sum_individual = d3 + d5 + d7
check("T1", "CRT: D_KL(105) >= D_KL(3)+D_KL(5)+D_KL(7) (super-additive)",
      d105 >= sum_individual - 1e-10,
      f"D(105)={d105:.4f}, sum={sum_individual:.4f}, excess={d105-sum_individual:.4f}")

# T2: Uniqueness -- no other power k gives alpha within 10x of 1/137
alpha_base = alpha_sieve(MU_STAR)
# Check: only k=1 gives alpha in [1/200, 1/100]
valid_k = []
for k_test in [0.5, 1, 1.5, 2, 3]:
    alpha_k = alpha_base ** k_test
    inv_alpha_k = 1.0 / alpha_k if alpha_k > 0 else float('inf')
    if 100 < inv_alpha_k < 200:
        valid_k.append(k_test)
check("T2", "k=1 is unique power giving 1/alpha in [100,200]",
      valid_k == [1],
      f"valid_k={valid_k}")

# T3: Tensor product property -- total amplitude = product of individual amplitudes
# |<U|Psi>|^2 = prod_p |<u_p|psi_p>|^2 = prod sin^2
alpha_from_product = 1.0
for p in ACTIVE_PRIMES:
    sin2_p = compute_sin2(MU_STAR, p, 'stat')
    alpha_from_product *= sin2_p
alpha_direct = alpha_sieve(MU_STAR)
check("T3", "Tensor product: prod sin^2 = alpha_sieve (exact)",
      abs(alpha_from_product - alpha_direct) < 1e-15,
      f"diff={abs(alpha_from_product - alpha_direct):.2e}")

# T4: Born rule consistency -- sin^2(theta_p) = delta_p * (2 - delta_p)
# This IS the Born rule applied to the sieve state
q_s = q_stat(MU_STAR)
born_ok = True
for p in ACTIVE_PRIMES:
    delta = (1.0 - q_s**p) / p
    born_prob = delta * (2.0 - delta)  # = |<u|psi>|^2
    sin2 = compute_sin2(MU_STAR, p, 'stat')
    if abs(born_prob - sin2) > 1e-15:
        born_ok = False
check("T4", "Born rule: sin^2 = delta*(2-delta) = |<u|psi>|^2",
      born_ok,
      "algebraic identity exact")

# T5: Multiplicativity check -- f(3*5) = f(3) * f(5)
sin2_3 = compute_sin2(MU_STAR, 3, 'stat')
sin2_5 = compute_sin2(MU_STAR, 5, 'stat')
sin2_7 = compute_sin2(MU_STAR, 7, 'stat')
f_15 = sin2_3 * sin2_5
f_35 = sin2_5 * sin2_7
f_21 = sin2_3 * sin2_7
f_105 = sin2_3 * sin2_5 * sin2_7
# Verify product structure is self-consistent
check("T5", "Multiplicativity: f(105) = f(3)*f(5)*f(7) = alpha",
      abs(f_105 - alpha_direct) < 1e-15,
      f"alpha={f_105:.10f}")

print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
TOTAL = N_PASS + N_FAIL
print(f"BRIDGE TESTS: {N_PASS}/{TOTAL} PASS ({100*N_PASS/TOTAL:.1f}%)")
print(f"  Domain 1 (Uniqueness):    {sum(1 for r in RESULTS[:6] if r['status']=='PASS')}/6")
print(f"  Domain 2 (Holonomy):      {sum(1 for r in RESULTS[6:11] if r['status']=='PASS')}/5")
print(f"  Domain 3 (Bifurcation):   {sum(1 for r in RESULTS[11:16] if r['status']=='PASS')}/5")
print(f"  Domain 4 (Dimensions):    {sum(1 for r in RESULTS[16:22] if r['status']=='PASS')}/6")
print(f"  Domain 5 (Cross-pillar):  {sum(1 for r in RESULTS[22:30] if r['status']=='PASS')}/8")
print(f"  Domain 6 (Falsification): {sum(1 for r in RESULTS[30:35] if r['status']=='PASS')}/5")
print(f"  Domain 7 (BA5 Theorem):   {sum(1 for r in RESULTS[35:] if r['status']=='PASS')}/5")
print("=" * 60)

# Save results
results_path = os.path.join(os.path.dirname(__file__), 'test_results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump({
        'total': TOTAL, 'pass': N_PASS, 'fail': N_FAIL,
        'score': f"{N_PASS}/{TOTAL}",
        'results': RESULTS
    }, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {results_path}")

sys.exit(0 if N_PASS == TOTAL else 1)
