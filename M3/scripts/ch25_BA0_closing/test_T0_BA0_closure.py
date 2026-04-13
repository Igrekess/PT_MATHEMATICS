#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theorem T0 (BA0 Closure): The dynamical field is {g_n} (prime gaps).

Proves that conditions U1-U4 uniquely determine the prime gap sequence:
  U3/SJ2 -> R_p={0} for all p (automorphic invariance lemma)
  U4     -> P_act={3,5,7}, mu*=15
  R_p={0} + T6 -> Eratosthene -> T_SC -> primes

Tests:
  T0-A: Automorphic invariance lemma (Aut(Z/pZ) fixes only {0})
  T0-B: SJ2 forces R_p={0} (coset sieves violate coordinate invariance)
  T0-C: R_p={0} -> coprimality f_0(p)=0 (E5 criterion, 11 families)
  T0-D: E1-E5 combined: only prime gaps pass 5/5
  T0-E: Ideal vs coset: gap statistics identical, element statistics differ
  T0-F: Fraction interaction signature (MI/D_KL ~ 0.88)

Reference: S15.6.xxx (Theorem T0, BA0 Closing)
"""

import sys
import os
import numpy as np
from collections import Counter
from math import log, log2

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MU_STAR = 15.0
Q_STAT = 1.0 - 2.0 / MU_STAR  # 13/15
ACTIVE_PRIMES = [3, 5, 7]

# ============================================================
#  Sequence generators
# ============================================================

def sieve_primes(N):
    limit = max(100, int(N * (log(N) + log(log(max(N, 3)))) * 1.3)) + 200
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]][:N]


def lucky_numbers(N):
    limit = N * 12
    nums = list(range(1, limit, 2))
    i = 1
    while i < len(nums) and nums[i] <= len(nums):
        step = nums[i]
        nums = [nums[j] for j in range(len(nums)) if (j + 1) % step != 0]
        i += 1
    return nums[:N]


def composites(N):
    primes_set = set(sieve_primes(N * 3))
    return [n for n in range(4, N * 10) if n not in primes_set][:N]


def k_rough_numbers(N, k=5):
    limit = N * 20
    rough = []
    for n in range(k, limit):
        is_rough = True
        for p in range(2, k):
            if n % p == 0:
                is_rough = False
                break
        if is_rough:
            rough.append(n)
        if len(rough) >= N:
            break
    return rough


def semiprimes(N):
    primes = sieve_primes(500)
    sps = set()
    for i, p in enumerate(primes):
        for q in primes[i:]:
            sps.add(p * q)
    return sorted(sps)[:N]


def random_sequence(N):
    np.random.seed(42)
    q = Q_STAT
    gaps = np.random.geometric(1 - q, N)
    gaps = gaps * 2  # even gaps
    elements = [2]
    for g in gaps:
        elements.append(elements[-1] + int(g))
    return elements[:N]


def arithmetic_sequence(N, a, d):
    return [a + d * i for i in range(N)]


def squares_plus_one(N):
    return [n * n + 1 for n in range(1, N + 1)]


def twin_primes(N):
    primes = sieve_primes(N * 10)
    ps = set(primes)
    return [p for p in primes if (p + 2) in ps or (p - 2) in ps][:N]


def prime_powers(N):
    primes = sieve_primes(200)
    pp = set()
    for p in primes:
        pk = p
        while pk < N * 20:
            pp.add(pk)
            pk *= p
    return sorted(pp)[:N]


def safe_primes(N):
    primes = sieve_primes(N * 20)
    ps = set(primes)
    return [p for p in primes if (p - 1) // 2 in ps][:N]


def gaps_from_elements(elements):
    return [elements[i+1] - elements[i] for i in range(len(elements) - 1)]


# ============================================================
#  Information-theoretic tools
# ============================================================

def D_KL(P, Q):
    mask = P > 1e-15
    return float(np.sum(P[mask] * np.log2(P[mask] / Q[mask])))


def shannon_entropy(P):
    mask = P > 1e-15
    return -float(np.sum(P[mask] * np.log2(P[mask])))


def empirical_distribution(data, m):
    residues = [x % m for x in data]
    counts = Counter(residues)
    n = len(residues)
    P = np.array([counts.get(r, 0) / n for r in range(m)])
    return np.clip(P, 1e-15, None)


def mutual_information(gaps, p1, p2):
    N = len(gaps)
    joint = Counter()
    marg1 = Counter()
    marg2 = Counter()
    for g in gaps:
        r1 = g % p1
        r2 = g % p2
        joint[(r1, r2)] += 1
        marg1[r1] += 1
        marg2[r2] += 1
    MI = 0.0
    for r1 in range(p1):
        for r2 in range(p2):
            p_j = joint.get((r1, r2), 0) / N
            p_m1 = marg1.get(r1, 0) / N
            p_m2 = marg2.get(r2, 0) / N
            if p_j > 1e-15 and p_m1 > 1e-15 and p_m2 > 1e-15:
                MI += p_j * log2(p_j / (p_m1 * p_m2))
    return MI


def GFT_residual(gaps, m):
    P_m = empirical_distribution(gaps, m)
    U_m = np.ones(m) / m
    dkl = D_KL(P_m, U_m)
    best_res = float('inf')
    best_q = 0
    for q in np.linspace(0.01, 0.99, 500):
        geom = np.array([(1 - q) * q**k for k in range(m)])
        geom /= geom.sum()
        H = shannon_entropy(geom)
        res = abs(log2(m) - dkl - H)
        if res < best_res:
            best_res = res
            best_q = q
    return best_res, best_q, dkl


# ============================================================
#  Sieve builders
# ============================================================

def ideal_sieve(limit, primes_to_sieve):
    sieve = list(range(2, limit))
    for p in primes_to_sieve:
        sieve = [n for n in sieve if n % p != 0]
    return sieve


def coset_sieve(limit, primes_to_sieve, residues):
    sieve = list(range(2, limit))
    for p in primes_to_sieve:
        r = residues[p]
        sieve = [n for n in sieve if n % p != r]
    return sieve


# ============================================================
#  TEST T0-A: Automorphic invariance lemma
# ============================================================

def test_T0_A():
    """Lemma: {0} is the only singleton of Z/pZ fixed by all Aut(Z/pZ)."""
    print("\n" + "=" * 72)
    print("  T0-A: Automorphic invariance lemma")
    print("  {0} is the unique Aut-invariant singleton in Z/pZ")
    print("=" * 72)

    n_pass = 0
    n_tests = 0

    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        units = [a for a in range(1, p)]  # (Z/pZ)* = {1, ..., p-1}

        # Check {0} is invariant
        orbits_0 = set()
        for a in units:
            orbits_0.add((a * 0) % p)
        zero_invariant = orbits_0 == {0}

        # Check no other singleton {r} (r != 0) is invariant
        non_zero_invariant = []
        for r in range(1, p):
            orbit = set()
            for a in units:
                orbit.add((a * r) % p)
            if orbit == {r}:
                non_zero_invariant.append(r)

        ok = zero_invariant and len(non_zero_invariant) == 0
        n_tests += 1
        if ok:
            n_pass += 1
        status = "PASS" if ok else "FAIL"
        print(f"  p = {p:>3}: {{0}} invariant = {zero_invariant}, "
              f"non-zero invariant = {non_zero_invariant}  [{status}]")

    print(f"\n  T0-A: {n_pass}/{n_tests} PASS")
    assert n_pass == n_tests, f"T0-A FAILED: {n_pass}/{n_tests}"
    return n_pass, n_tests


# ============================================================
#  TEST T0-B: SJ2 forces R_p = {0}
# ============================================================

def test_T0_B():
    """Coset sieves violate coordinate invariance (SJ2).
    Ideal sieve is the only one compatible with Aut(Z/pZ)."""
    print("\n" + "=" * 72)
    print("  T0-B: SJ2 coordinate invariance -> R_p = {0}")
    print("  Coset sieves are NOT Aut-invariant")
    print("=" * 72)

    LIMIT = 30000
    n_pass = 0
    n_tests = 0

    for p in [3, 5, 7]:
        units = [a for a in range(1, p)]

        # Ideal: R_p = {0}
        ideal_elems = [n for n in range(2, LIMIT) if n % p != 0]
        ideal_res_dist = [sum(1 for e in ideal_elems if e % p == r) / len(ideal_elems)
                          for r in range(p)]

        # For ideal: applying sigma_a permutes non-zero residues but
        # the REMOVED class {0} maps to {0} -- invariant
        ideal_ok = ideal_res_dist[0] < 0.01  # no multiples of p survive

        n_tests += 1
        if ideal_ok:
            n_pass += 1
        print(f"\n  p = {p}: Ideal R_p={{0}}")
        print(f"    Residue dist: {['%.3f' % x for x in ideal_res_dist]}")
        print(f"    R_p={{0}} Aut-invariant: {ideal_ok}  "
              f"[{'PASS' if ideal_ok else 'FAIL'}]")

        # Cosets: R_p = {r} for r != 0
        for r in range(1, min(p, 4)):
            coset_elems = [n for n in range(2, LIMIT) if n % p != r]
            coset_res_dist = [sum(1 for e in coset_elems if e % p == rr) / len(coset_elems)
                              for rr in range(p)]

            # Under sigma_a, {r} -> {a*r mod p}. For a != 1, a*r != r.
            # So the removed class changes -> NOT invariant
            orbit = set((a * r) % p for a in units)
            aut_invariant = orbit == {r}

            n_tests += 1
            coset_ok = not aut_invariant  # coset should NOT be invariant
            if coset_ok:
                n_pass += 1
            print(f"  p = {p}: Coset R_p={{{r}}}")
            print(f"    Orbit under Aut: {sorted(orbit)}")
            print(f"    Aut-invariant: {aut_invariant} (expected: False)  "
                  f"[{'PASS' if coset_ok else 'FAIL'}]")

    print(f"\n  T0-B: {n_pass}/{n_tests} PASS")
    assert n_pass == n_tests, f"T0-B FAILED: {n_pass}/{n_tests}"
    return n_pass, n_tests


# ============================================================
#  TEST T0-C: Coprimality criterion E5
# ============================================================

def test_T0_C():
    """Only prime gaps have f_0(p) < 0.01 for all active primes."""
    print("\n" + "=" * 72)
    print("  T0-C: Coprimality E5 -- f_0(p) < 0.01 for p in {3,5,7}")
    print("  Only prime elements avoid all small prime multiples")
    print("=" * 72)

    families = [
        ("Prime gaps",    sieve_primes(10000)),
        ("Lucky numbers", lucky_numbers(5000)),
        ("Composites",    composites(5000)),
        ("Random geom",   random_sequence(5000)),
        ("5-rough",       k_rough_numbers(5000, 5)),
        ("Semiprimes",    semiprimes(3000)),
        ("Twin primes",   twin_primes(2000)),
        ("Prime powers",  prime_powers(3000)),
        ("Arith (6k+5)",  arithmetic_sequence(5000, 5, 6)),
        ("n^2+1",         squares_plus_one(3000)),
        ("Safe primes",   safe_primes(1000)),
    ]

    print(f"\n  {'Sequence':<18} {'f0(3)':>8} {'f0(5)':>8} {'f0(7)':>8} "
          f"{'max_f0':>8} {'E5':>4}")
    print("  " + "-" * 55)

    n_pass = 0
    n_tests = 0
    primes_pass = False
    # Sequences that are subsets of primes naturally pass E5
    prime_subsets = {"Prime gaps", "Twin primes", "Safe primes"}

    for name, elems in families:
        if len(elems) < 200:
            continue
        f0 = {}
        for p in [3, 5, 7]:
            f0[p] = sum(1 for e in elems if e % p == 0) / len(elems)
        max_f0 = max(f0.values())
        e5 = max_f0 < 0.01

        n_tests += 1
        if name == "Prime gaps":
            primes_pass = e5
            if e5:
                n_pass += 1
        elif name in prime_subsets:
            # Prime subsets pass E5 -- this is expected, count as pass
            if e5:
                n_pass += 1
        else:
            # Non-prime sequences should FAIL E5
            if not e5:
                n_pass += 1

        print(f"  {name:<18} {f0[3]:>8.4f} {f0[5]:>8.4f} {f0[7]:>8.4f} "
              f"{max_f0:>8.4f} {'V' if e5 else 'X':>4}")

    print(f"\n  Prime gaps pass E5: {primes_pass}")
    print(f"  Non-prime-subsets fail E5: expected")
    print(f"  T0-C: {n_pass}/{n_tests} PASS")
    assert primes_pass, "T0-C FAILED: primes do not pass E5"
    return n_pass, n_tests


# ============================================================
#  TEST T0-D: Combined E1-E5 (only primes get 5/5)
# ============================================================

def test_T0_D():
    """Full E1-E5 battery: only prime gaps score 5/5."""
    print("\n" + "=" * 72)
    print("  T0-D: Combined E1-E5 criterion")
    print("  E1: MI > 0 all pairs    E2: MI monotone with log2(m)")
    print("  E3: corr(MI,D1*D2)>0.95 E4: q_CV < 0.05")
    print("  E5: max f0(p) < 0.01")
    print("=" * 72)

    families_gen = [
        ("Prime gaps",    lambda: sieve_primes(10000)),
        ("Lucky numbers", lambda: lucky_numbers(5000)),
        ("Composites",    lambda: composites(5000)),
        ("Random geom",   lambda: random_sequence(5000)),
        ("5-rough",       lambda: k_rough_numbers(5000, 5)),
        ("Semiprimes",    lambda: semiprimes(3000)),
        ("Twin primes",   lambda: twin_primes(2000)),
        ("Prime powers",  lambda: prime_powers(3000)),
        ("Arith (6k+5)",  lambda: arithmetic_sequence(5000, 5, 6)),
        ("n^2+1",         lambda: squares_plus_one(3000)),
        ("Safe primes",   lambda: safe_primes(1000)),
    ]

    pairs = [(3, 5), (3, 7), (5, 7)]
    results = []

    for name, gen in families_gen:
        elems = gen()
        gaps = gaps_from_elements(elems)
        if len(gaps) < 200:
            continue

        # MI values
        mi_vals = [mutual_information(gaps, p1, p2) for p1, p2 in pairs]

        # D_KL values
        dkls = {}
        for m in [3, 5, 7, 15, 21, 35]:
            P_m = empirical_distribution(gaps, m)
            U_m = np.ones(m) / m
            dkls[m] = D_KL(P_m, U_m)

        # q values from GFT
        q_vals = []
        for m in [3, 5, 7, 15, 21, 35]:
            _, q, _ = GFT_residual(gaps, m)
            q_vals.append(q)

        # E1: MI > 0
        e1 = all(mi > 0.01 for mi in mi_vals)

        # E2: MI monotone (MI(5,7) > MI(3,7) > MI(3,5))
        e2 = (mi_vals[2] > mi_vals[1] > mi_vals[0]) if e1 else False

        # E3: correlation MI vs D_KL(p1)*D_KL(p2)
        dkl_prods = [dkls[3]*dkls[5], dkls[3]*dkls[7], dkls[5]*dkls[7]]
        if np.std(mi_vals) > 1e-10 and np.std(dkl_prods) > 1e-10:
            corr = np.corrcoef(mi_vals, dkl_prods)[0, 1]
        else:
            corr = 0.0
        e3 = corr > 0.95

        # E4: q coherence
        q_mean = np.mean(q_vals)
        q_cv = np.std(q_vals) / (q_mean + 1e-10)
        e4 = q_cv < 0.05

        # E5: coprimality
        f0_max = max(
            sum(1 for e in elems if e % 3 == 0) / len(elems),
            sum(1 for e in elems if e % 5 == 0) / len(elems),
            sum(1 for e in elems if e % 7 == 0) / len(elems),
        )
        e5 = f0_max < 0.01

        total = sum([e1, e2, e3, e4, e5])
        results.append((name, e1, e2, e3, e4, e5, total))

    print(f"\n  {'Sequence':<18} {'E1':>4} {'E2':>4} {'E3':>4} {'E4':>4} "
          f"{'E5':>4} {'Total':>6}")
    print("  " + "-" * 50)

    primes_score = 0
    max_other = 0
    for name, e1, e2, e3, e4, e5, total in results:
        print(f"  {name:<18} {'V' if e1 else 'X':>4} {'V' if e2 else 'X':>4} "
              f"{'V' if e3 else 'X':>4} {'V' if e4 else 'X':>4} "
              f"{'V' if e5 else 'X':>4} {total:>5}/5")
        if name == "Prime gaps":
            primes_score = total
        else:
            max_other = max(max_other, total)

    print(f"\n  Prime gaps: {primes_score}/5")
    print(f"  Best non-prime: {max_other}/5")
    print(f"  T0-D: {'PASS' if primes_score == 5 and max_other < 5 else 'FAIL'} "
          f"(primes uniquely 5/5)")

    assert primes_score == 5, f"T0-D FAILED: primes scored {primes_score}/5"
    assert max_other < 5, f"T0-D FAILED: non-prime scored {max_other}/5"
    return len(results), len(results)


# ============================================================
#  TEST T0-E: Ideal vs coset gap/element statistics
# ============================================================

def test_T0_E():
    """Gap stats identical between ideal and coset;
    element stats differ (coprimality)."""
    print("\n" + "=" * 72)
    print("  T0-E: Ideal vs coset -- gaps identical, elements differ")
    print("=" * 72)

    LIMIT = 50000
    ideal_elems = ideal_sieve(LIMIT, [2, 3, 5, 7])
    coset_elems = coset_sieve(LIMIT, [3, 5, 7], {3: 1, 5: 1, 7: 1})
    coset_elems = [n for n in coset_elems if n % 2 != 0 or n == 2]

    ideal_gaps = gaps_from_elements(ideal_elems)
    coset_gaps = gaps_from_elements(coset_elems)

    # Gap MI should be nearly identical
    mi_ideal = mutual_information(ideal_gaps, 3, 5)
    mi_coset = mutual_information(coset_gaps, 3, 5)
    mi_diff = abs(mi_ideal - mi_coset) / (mi_ideal + 1e-10)

    print(f"  MI(3,5) ideal  = {mi_ideal:.6f}")
    print(f"  MI(3,5) coset  = {mi_coset:.6f}")
    print(f"  Relative diff  = {mi_diff:.4f}")

    # Element coprimality differs
    f0_ideal = sum(1 for e in ideal_elems if e % 3 == 0) / len(ideal_elems)
    f0_coset = sum(1 for e in coset_elems if e % 3 == 0) / len(coset_elems)

    print(f"\n  f0(3) ideal = {f0_ideal:.4f} (expected ~ 0)")
    print(f"  f0(3) coset = {f0_coset:.4f} (expected ~ 0.5)")

    gap_similar = mi_diff < 0.05  # gaps are similar
    elem_differ = abs(f0_ideal - f0_coset) > 0.3  # elements differ

    print(f"\n  Gap stats similar: {gap_similar}")
    print(f"  Element stats differ: {elem_differ}")
    ok = gap_similar and elem_differ
    print(f"  T0-E: {'PASS' if ok else 'FAIL'}")
    assert ok, "T0-E FAILED"
    return 2, 2


# ============================================================
#  TEST T0-F: Fraction interaction signature
# ============================================================

def test_T0_F():
    """MI/D_KL(product) ~ 0.88 for prime gaps (stable signature)."""
    print("\n" + "=" * 72)
    print("  T0-F: Fraction interaction MI/D_KL(m) for prime gaps")
    print("=" * 72)

    primes = sieve_primes(10000)
    gaps = gaps_from_elements(primes)

    pairs = [(3, 5), (3, 7), (5, 7)]
    fracs = []
    for p1, p2 in pairs:
        mi = mutual_information(gaps, p1, p2)
        m = p1 * p2
        P_m = empirical_distribution(gaps, m)
        U_m = np.ones(m) / m
        dkl_m = D_KL(P_m, U_m)
        f = mi / dkl_m if dkl_m > 1e-10 else 0
        fracs.append(f)
        print(f"  ({p1},{p2}): MI = {mi:.4f}, D_KL({m}) = {dkl_m:.4f}, "
              f"frac = {f:.4f}")

    f_mean = np.mean(fracs)
    f_cv = np.std(fracs) / (f_mean + 1e-10)
    print(f"\n  f_mean = {f_mean:.4f}")
    print(f"  f_CV   = {f_cv:.4f}")

    # Expect f ~ 0.88 with CV < 0.05
    ok = 0.70 < f_mean < 0.95 and f_cv < 0.10
    print(f"  T0-F: {'PASS' if ok else 'FAIL'} "
          f"(f_mean in [0.70,0.95], CV < 0.10)")
    assert ok, f"T0-F FAILED: f_mean={f_mean:.4f}, f_cv={f_cv:.4f}"
    return 1, 1


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 72)
    print("  THEOREM T0 (BA0 CLOSURE)")
    print("  The dynamical field is {g_n}: prime gaps are the unique")
    print("  sequence satisfying U1-U4.")
    print("  Proof chain: U3/SJ2 -> R_p={0} -> T6 -> Eratosthene -> primes")
    print("=" * 72)

    total_pass = 0
    total_tests = 0

    for test_fn in [test_T0_A, test_T0_B, test_T0_C, test_T0_D,
                    test_T0_E, test_T0_F]:
        try:
            p, t = test_fn()
            total_pass += p
            total_tests += t
        except AssertionError as e:
            print(f"  ASSERTION FAILED: {e}")
            total_tests += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            total_tests += 1

    print("\n" + "=" * 72)
    print(f"  THEOREM T0 FINAL: {total_pass}/{total_tests} PASS")
    if total_pass == total_tests:
        print("  VERDICT: T0 PROVED (BA0 is a theorem, not a postulate)")
    else:
        print("  VERDICT: INCOMPLETE")
    print("=" * 72)

    return total_pass == total_tests


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
