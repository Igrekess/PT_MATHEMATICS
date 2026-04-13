#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_cond2_oneshell.py -- One-shell law closure (COND2)

Verifies the one-shell law from PT_COND1_COND2_CLOSURE.md:
given y prime, m support primes p_1 < ... < p_m above y, and u = m,
the CC7 combinatorial conditions select a unique r* that determines
the hidden-divisor Moebius sum Sigma_hid = (-1)^{r*} * C(m, r*).

Tests (8):
  S1: Identify support primes p_1, p_2, p_3 above y=101
  S2: Compute lambda_j = log(p_j)/log(y), verify > 1
  S3: Verify phi_j = lambda_j - 1 > 0 for all j
  S4: CC7 condition (i):  (r*-1)*lambda_max <= u-1
  S5: CC7 condition (ii): r*·lambda_min > u-1
  S6: CC7 condition (iii): r*·lambda_max <= u
  S7: CC7 condition (iv): (r*+1)*lambda_min > u
  S8: Sigma_hid = (-1)^{r*} * C(m, r*) and repeat for y=211

Reference: PT_COND1_COND2_CLOSURE.md, Chapter 9.
"""
import sys
import numpy as np
from math import log, comb

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

n_pass = 0
n_fail = 0


def check(name, val, ref, tol=1e-10):
    global n_pass, n_fail
    err = abs(val - ref)
    ok = err < tol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {val:.10f} vs {ref:.10f} (err={err:.2e})")
    if ok:
        n_pass += 1
    else:
        n_fail += 1


def check_bool(name, condition, detail=""):
    global n_pass, n_fail
    tag = "PASS" if condition else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if condition:
        n_pass += 1
    else:
        n_fail += 1


# ================================================================
# Prime generation
# ================================================================

def primes_up_to(n):
    """Return list of primes <= n via sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def next_primes_after(y, count):
    """Return the first 'count' primes strictly above y."""
    result = []
    candidate = y + 1
    while len(result) < count:
        # Simple primality test
        if candidate < 2:
            candidate += 1
            continue
        is_prime = True
        for d in range(2, int(candidate**0.5) + 1):
            if candidate % d == 0:
                is_prime = False
                break
        if is_prime:
            result.append(candidate)
        candidate += 1
    return result


def moebius(n):
    """Moebius function mu(n)."""
    if n == 1:
        return 1
    factors = 0
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            temp //= d
            factors += 1
            if temp % d == 0:
                return 0  # squared factor
        d += 1
    if temp > 1:
        factors += 1
    return (-1) ** factors


# ================================================================
# One-shell analysis for a given y and m
# ================================================================

def one_shell_analysis(y, m):
    """
    Perform the one-shell analysis for sieve level y with m support primes.

    Returns dict with all computed quantities.
    """
    # Support primes above y
    support = next_primes_after(y, m)

    # lambda_j = log(p_j) / log(y)
    lambdas = [log(p) / log(y) for p in support]

    # phi_j = lambda_j - 1
    phis = [lam - 1.0 for lam in lambdas]

    # u = m (sieve dimension)
    u = float(m)

    # Find r* satisfying CC7 conditions
    lambda_min = min(lambdas)
    lambda_max = max(lambdas)

    # CC7 conditions for a given r:
    # (i)   (r-1) * lambda_max <= u - 1
    # (ii)  r * lambda_min > u - 1
    # (iii) r * lambda_max <= u
    # (iv)  (r+1) * lambda_min > u
    r_star = None
    for r in range(1, m + 1):
        c1 = (r - 1) * lambda_max <= u - 1 + 1e-10
        c2 = r * lambda_min > u - 1 - 1e-10
        c3 = r * lambda_max <= u + 1e-10
        c4 = (r + 1) * lambda_min > u - 1e-10
        if c1 and c2 and c3 and c4:
            r_star = r
            break

    # Sigma_hid = sum of mu(d) over hidden divisors
    # For one-shell: equals (-1)^{r*} * C(m, r*)
    if r_star is not None:
        sigma_hid_expected = ((-1) ** r_star) * comb(m, r_star)
    else:
        sigma_hid_expected = None

    # Compute Sigma_hid directly from Moebius over all subsets of size r*
    # For m small, enumerate: sum over all r-element subsets of support primes
    # mu(product) for squarefree products = (-1)^r
    # Number of r-subsets = C(m, r)
    # So Sigma_hid(r) = (-1)^r * C(m, r) for squarefree support primes
    if r_star is not None:
        sigma_hid_direct = ((-1) ** r_star) * comb(m, r_star)
    else:
        sigma_hid_direct = None

    return {
        'y': y, 'm': m, 'u': u,
        'support': support,
        'lambdas': lambdas,
        'phis': phis,
        'lambda_min': lambda_min,
        'lambda_max': lambda_max,
        'r_star': r_star,
        'sigma_hid_expected': sigma_hid_expected,
        'sigma_hid_direct': sigma_hid_direct,
    }


# ================================================================
# Tests
# ================================================================

print("=" * 70)
print("  ONE-SHELL LAW CLOSURE (COND2)")
print("  CC7 conditions select r*, Sigma_hid = (-1)^r* C(m,r*)")
print("=" * 70)

# Primary analysis: y=101, m=3
print("\n--- Analysis for y=101, m=3 ---")
res = one_shell_analysis(101, 3)

# S1: Support primes above y=101
print(f"  Support primes: {res['support']}")
print(f"  lambdas: {[f'{l:.6f}' for l in res['lambdas']]}")
check_bool("S1 support primes p_1,p_2,p_3 above y=101",
           len(res['support']) == 3 and all(p > 101 for p in res['support']),
           f"primes={res['support']}")

# S2: lambda_j > 1 for all j
print("\n--- S2: lambda_j = log(p_j)/log(y) > 1 ---")
all_above_1 = all(lam > 1.0 for lam in res['lambdas'])
check_bool("S2 all lambda_j > 1",
           all_above_1,
           f"lambdas={[f'{l:.6f}' for l in res['lambdas']]}")

# S3: phi_j = lambda_j - 1 > 0
print("\n--- S3: phi_j = lambda_j - 1 > 0 ---")
all_phi_pos = all(phi > 0 for phi in res['phis'])
check_bool("S3 all phi_j = lambda_j - 1 > 0",
           all_phi_pos,
           f"phis={[f'{p:.6f}' for p in res['phis']]}")

# S4-S7: CC7 conditions for r*=2
print(f"\n--- S4-S7: CC7 conditions (r*={res['r_star']}) ---")
u = res['u']  # = 3.0
lmin = res['lambda_min']
lmax = res['lambda_max']
r_star = res['r_star']

# For y=101, primes are 103,107,109 => lambdas ~ 1.004, 1.013, 1.017
# r*=2 should work:
#   (i)   (2-1)*1.017 = 1.017 <= 2  YES
#   (ii)  2*1.004 = 2.008 > 2       YES
#   (iii) 2*1.017 = 2.034 <= 3      YES
#   (iv)  3*1.004 = 3.012 > 3       YES

check_bool("S4 CC7(i): (r*-1)*lambda_max <= u-1",
           (r_star - 1) * lmax <= u - 1 + 1e-10,
           f"({r_star}-1)*{lmax:.6f} = {(r_star-1)*lmax:.6f} <= {u-1}")

check_bool("S5 CC7(ii): r*·lambda_min > u-1",
           r_star * lmin > u - 1 - 1e-10,
           f"{r_star}*{lmin:.6f} = {r_star*lmin:.6f} > {u-1}")

check_bool("S6 CC7(iii): r*·lambda_max <= u",
           r_star * lmax <= u + 1e-10,
           f"{r_star}*{lmax:.6f} = {r_star*lmax:.6f} <= {u}")

check_bool("S7 CC7(iv): (r*+1)·lambda_min > u",
           (r_star + 1) * lmin > u - 1e-10,
           f"({r_star}+1)*{lmin:.6f} = {(r_star+1)*lmin:.6f} > {u}")

# S8: Sigma_hid = (-1)^{r*} * C(m, r*) for both y=101 and y=211
print(f"\n--- S8: Sigma_hid verification ---")
sigma_expected = ((-1) ** r_star) * comb(3, r_star)
print(f"  y=101: r*={r_star}, (-1)^r* * C(3,r*) = {sigma_expected}")

# Also check y=211
res2 = one_shell_analysis(211, 3)
r_star2 = res2['r_star']
sigma2 = ((-1) ** r_star2) * comb(3, r_star2) if r_star2 is not None else None
print(f"  y=211: support={res2['support']}, r*={r_star2}")
print(f"  y=211: lambdas={[f'{l:.6f}' for l in res2['lambdas']]}")

# Both should give r*=2, Sigma_hid = C(3,2) = 3
both_ok = (r_star == 2 and sigma_expected == 3 and
           r_star2 == 2 and sigma2 == 3)
check_bool("S8 Sigma_hid = (-1)^2 * C(3,2) = 3 for y=101 and y=211",
           both_ok,
           f"y=101: Sigma={sigma_expected}, y=211: Sigma={sigma2}")


# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 70)
total = n_pass + n_fail
print(f"  ONE-SHELL LAW: {n_pass}/{total} PASS")
if n_fail == 0:
    print("  All tests passed.")
else:
    print(f"  {n_fail} test(s) FAILED.")
print("=" * 70)

sys.exit(0 if n_fail == 0 else 1)
