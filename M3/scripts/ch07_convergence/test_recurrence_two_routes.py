#!/usr/bin/env python3
"""
Two independent derivations of the recurrence D(k+1) = (p-3)*D(k) + Delta.

Route 1: Direct enumeration -- compute D(k) = n12 - n10 from the sieve
          at each level k (no prediction, raw counting).
Route 2: CRT linearity -- from D(k) and 3-gram boundary terms, predict
          D(k+1) = (p_{k+1} - 3)*D(k) + Delta(k), where
          Delta = (A12 - A10) + (B12 - B10) extracted from gram3.

Both routes must give identical EXACT INTEGER values for all k.

Reference: Chapter 7, Remark 'Two independent derivations of the recurrence'.
"""

import numpy as np
import time
import sys

n_pass = 0
n_fail = 0


def check(name, val, ref, tol=0):
    global n_pass, n_fail
    ok = (val == ref) if tol == 0 else abs(val - ref) < tol
    tag = "PASS" if ok else "FAIL"
    print("  [{}] {}: {} vs {}".format(tag, name, val, ref))
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    return ok


# ============================================================
# Sieve infrastructure (from test_T00_induction_proof.py)
# ============================================================

def compute_sieve_stats(prime_list):
    """Compute sieve survivors, gaps, 2-gram and 3-gram counts exactly."""
    P = 1
    for p in prime_list:
        P *= p

    if P > 300_000_000:
        return None

    sieve = np.ones(P + 1, dtype=bool)
    sieve[0] = False
    for p in prime_list:
        sieve[::p] = False
    survivors = np.where(sieve)[0]
    del sieve

    n = len(survivors)

    # Cyclic gaps
    gaps = np.empty(n, dtype=np.int64)
    gaps[:-1] = survivors[1:] - survivors[:-1]
    gaps[-1] = P + survivors[0] - survivors[-1]

    # Gap classes mod 3
    classes = gaps % 3

    # 2-gram transitions
    cls_from = classes
    cls_to = np.roll(classes, -1)
    trans = np.zeros((3, 3), dtype=np.int64)
    for a in range(3):
        ma = (cls_from == a)
        for b in range(3):
            trans[a, b] = int((ma & (cls_to == b)).sum())

    # 3-gram tensor
    cls_to2 = np.roll(classes, -2)
    gram3 = np.zeros((3, 3, 3), dtype=np.int64)
    for a in range(3):
        ma = (cls_from == a)
        for b in range(3):
            mab = ma & (cls_to == b)
            for c in range(3):
                gram3[a, b, c] = int((mab & (cls_to2 == c)).sum())

    return {'trans': trans, 'gram3': gram3, 'n': n, 'P': P}


def compute_delta_from_gram3(gram3):
    """Compute Delta = (A12-A10) + (B12-B10) from boundary 3-grams.

    A_{ab} = sum_{c+d = b mod 3} gram3(a, c, d)
    B_{ab} = sum_{c+d = a mod 3} gram3(c, d, b)
    """
    A12, A10 = 0, 0
    B12, B10 = 0, 0

    for c in range(3):
        for d in range(3):
            cd = (c + d) % 3
            if cd == 2:       # A_{1,2}: a=1, b=2
                A12 += int(gram3[1, c, d])
            if cd == 0:       # A_{1,0}: a=1, b=0
                A10 += int(gram3[1, c, d])
            if cd == 1:       # B_{1,2}: b=2, a=1  =>  c+d = 1 mod 3
                B12 += int(gram3[c, d, 2])
            if cd == 1:       # B_{1,0}: b=0, a=1  =>  c+d = 1 mod 3
                B10 += int(gram3[c, d, 0])

    return (A12 - A10) + (B12 - B10), A12, A10, B12, B10


# ============================================================
# Main computation
# ============================================================

all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

print("=" * 70)
print("TWO INDEPENDENT DERIVATIONS OF THE RECURRENCE")
print("D(k+1) = (p_{k+1} - 3) * D(k) + Delta(k)")
print("=" * 70)

# ---- Route 1: Direct enumeration at each level ----
print("\n" + "=" * 70)
print("ROUTE 1: DIRECT ENUMERATION")
print("=" * 70)

D_direct = {}  # k -> D(k) = n12 - n10

for k in range(2, len(all_primes) + 1):
    plist = all_primes[:k]
    t0 = time.time()
    r = compute_sieve_stats(plist)
    if r is None:
        print("  k={}: P too large, skipping".format(k))
        break
    dt = time.time() - t0
    n12 = int(r['trans'][1, 2])
    n10 = int(r['trans'][1, 0])
    D = n12 - n10
    D_direct[k] = D
    print("  k={}, p_max={:2d}, phi={:>12,}, n12={:>10,}, n10={:>10,}, D={:>10,}  ({:.1f}s)".format(
        k, plist[-1], r['n'], n12, n10, D, dt))

# ---- Route 2: CRT linearity (predict from previous level) ----
print("\n" + "=" * 70)
print("ROUTE 2: CRT LINEARITY (3-gram prediction)")
print("=" * 70)

D_crt = {}
delta_values = {}

# Base case: D at k=2 comes from direct enumeration
k_min = min(D_direct.keys())
k_max = max(D_direct.keys())

# Recompute stats for 3-gram extraction
stats_cache = {}
for k in range(k_min, k_max + 1):
    plist = all_primes[:k]
    r = compute_sieve_stats(plist)
    if r is not None:
        stats_cache[k] = r

# Base case
D_crt[k_min] = D_direct[k_min]
print("  k={}: D = {} (base case from direct enumeration)".format(k_min, D_crt[k_min]))

for k in range(k_min, k_max):
    if k not in stats_cache:
        break
    p_new = all_primes[k]  # prime added at level k+1
    delta, A12, A10, B12, B10 = compute_delta_from_gram3(stats_cache[k]['gram3'])
    delta_values[k] = delta
    D_pred = (p_new - 3) * D_crt[k] + delta
    D_crt[k + 1] = D_pred
    print("  k={}->{}: p={:2d}, (p-3)*D={:>10,}, Delta={:>8,}, D_pred={:>10,}".format(
        k, k + 1, p_new, (p_new - 3) * D_crt[k], delta, D_pred))

# ---- Cross-route verification ----
print("\n" + "=" * 70)
print("CROSS-ROUTE VERIFICATION")
print("=" * 70)

for k in sorted(set(D_direct.keys()) & set(D_crt.keys())):
    check("D(k={}): Route1 == Route2".format(k), D_direct[k], D_crt[k])

# ---- Additional checks ----
print("\n" + "=" * 70)
print("STRUCTURAL CHECKS")
print("=" * 70)

# D(k) > 0 for all computed levels
for k in sorted(D_direct.keys()):
    check("D(k={}) > 0".format(k), D_direct[k] > 0, True)

# Delta(k) > 0 for k >= 3 (key for induction; k=2 is the base case
# where Delta = -1, but D(2) = D(3) = 1 > 0 regardless)
for k in sorted(delta_values.keys()):
    if k == 2:
        check("Delta(k=2) = -1 (base case, harmless)", delta_values[k], -1)
    else:
        check("Delta(k={}) > 0".format(k), delta_values[k] > 0, True)

# Amplification: (p-3)*D(k) / |Delta(k)| grows
print("\n  Amplification factors:")
for k in sorted(delta_values.keys()):
    if delta_values[k] != 0:
        p_new = all_primes[k]
        amp = (p_new - 3) * D_direct[k] / abs(delta_values[k])
        print("    k={}: (p-3)*D/|Delta| = {:.2f}".format(k, amp))

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
total = n_pass + n_fail
print("TWO ROUTES TO RECURRENCE: {}/{} PASS, {} FAIL".format(n_pass, total, n_fail))
if n_fail == 0:
    print("Both routes converge -- recurrence ARMORED.")
else:
    print("WARNING: {} failures detected.".format(n_fail))
print("=" * 70)

sys.exit(0 if n_fail == 0 else 1)
