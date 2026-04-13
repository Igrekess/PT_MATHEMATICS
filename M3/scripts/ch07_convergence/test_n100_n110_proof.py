#!/usr/bin/env python3
"""
Attempt to prove n100 >= n110 in the binary sieve word.

The binary word z_i = 1_{g_i = 0 mod 3} encodes gap classes.
We need: n100 >= n110 (equivalently, sigma >= T00).

This script explores the structure of the CRT update to find
an inductive argument.

Approach:
1. Exact computation of run statistics at each primorial level
2. Analysis of the CRT update: how does adding prime p transform z?
3. Tracking of how n100, n110 change under CRT update
4. Search for a provable invariant

Key identities from T5_convergence.md:
  n100 - n110 = n1_single - n0_single  (run decomposition)
  sigma >= T00 <=> #{singleton 1-blocks} >= #{singleton 0-blocks}
"""

from fractions import Fraction
from math import prod
from collections import Counter

import numpy as np
import sys


PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]


def sieve_survivors(prime_list):
    """Return sorted array of survivors of sieve by prime_list in [1, prod(prime_list)]."""
    P = prod(prime_list)
    sieve = np.ones(P + 1, dtype=np.bool_)
    sieve[0] = False
    for p in prime_list:
        sieve[::p] = False
    return np.flatnonzero(sieve)


def binary_word(prime_list):
    """Compute the cyclic binary word z and gap sequence."""
    survivors = sieve_survivors(prime_list)
    P = prod(prime_list)
    n = len(survivors)
    gaps = np.empty(n, dtype=np.int64)
    gaps[:-1] = survivors[1:] - survivors[:-1]
    gaps[-1] = P + survivors[0] - survivors[-1]
    z = (gaps % 3 == 0).astype(int)
    return z, gaps, survivors


def run_decomposition(z):
    """Decompose cyclic binary word z into runs. Returns list of (value, length)."""
    n = len(z)
    # Find a transition point to break the cycle
    start = 0
    for j in range(n):
        if z[j] != z[(j - 1) % n]:
            start = j
            break
    zz = np.concatenate([z[start:], z[:start]])

    runs = []
    cur = int(zz[0])
    length = 1
    for x in zz[1:]:
        x = int(x)
        if x == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = x
            length = 1
    runs.append((cur, length))
    # Merge first and last if same value (cyclic)
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0] = (runs[0][0], runs[0][1] + runs[-1][1])
        runs.pop()
    return runs


def analyze_level(k):
    """Complete analysis of level k."""
    prime_list = PRIMES[:k]
    z, gaps, survivors = binary_word(prime_list)
    runs = run_decomposition(z)
    n = len(z)

    # Run statistics
    one_runs = [L for v, L in runs if v == 1]
    zero_runs = [L for v, L in runs if v == 0]

    n1_single = sum(1 for L in one_runs if L == 1)
    n0_single = sum(1 for L in zero_runs if L == 1)
    n1_long = sum(1 for L in one_runs if L >= 2)
    n0_long = sum(1 for L in zero_runs if L >= 2)

    n1 = sum(one_runs)  # total 1s
    n0 = sum(zero_runs)  # total 0s
    alpha = Fraction(n1, n)

    # Run length distributions
    one_dist = Counter(one_runs)
    zero_dist = Counter(zero_runs)

    # 3-gram patterns (gap residues mod 3)
    g3 = gaps % 3
    n3_counts = {}
    for a in range(3):
        for b in range(3):
            for c in range(3):
                count = 0
                for i in range(n):
                    if g3[i] == a and g3[(i+1) % n] == b and g3[(i+2) % n] == c:
                        count += 1
                n3_counts[(a, b, c)] = count

    return {
        'k': k, 'n': n, 'n1': n1, 'n0': n0,
        'alpha': alpha,
        'n1_single': n1_single, 'n0_single': n0_single,
        'n1_long': n1_long, 'n0_long': n0_long,
        'one_runs': len(one_runs), 'zero_runs': len(zero_runs),
        'one_dist': one_dist, 'zero_dist': zero_dist,
        'n3': n3_counts,
        'z': z, 'gaps': gaps,
    }


def main():
    print("=" * 78)
    print("ANALYSIS: n100 >= n110 in the binary sieve word")
    print("=" * 78)

    # =========================================================
    # PART 1: Run statistics and the alpha < 1/2 argument
    # =========================================================
    print("\n--- PART 1: Run statistics ---")
    print(f"{'k':>2} {'alpha':>8} {'R1':>6} {'R0':>6} "
          f"{'n1s':>6} {'n0s':>6} {'n1L':>6} {'n0L':>6} "
          f"{'avgL1':>8} {'avgL0':>8} {'n1s-n0s':>8}")
    print("-" * 88)

    for k in range(3, len(PRIMES)):
        s = analyze_level(k)
        avg1 = s['n1'] / s['one_runs'] if s['one_runs'] else 0
        avg0 = s['n0'] / s['zero_runs'] if s['zero_runs'] else 0
        diff = s['n1_single'] - s['n0_single']
        print(f"{k:>2} {float(s['alpha']):>8.4f} "
              f"{s['one_runs']:>6} {s['zero_runs']:>6} "
              f"{s['n1_single']:>6} {s['n0_single']:>6} "
              f"{s['n1_long']:>6} {s['n0_long']:>6} "
              f"{avg1:>8.3f} {avg0:>8.3f} {diff:>8}")

    # =========================================================
    # PART 2: Run length distributions
    # =========================================================
    print("\n--- PART 2: Run length distributions ---")
    for k in range(3, min(len(PRIMES), 7)):
        s = analyze_level(k)
        print(f"\n  k={k}  alpha={float(s['alpha']):.4f}")
        max_len = max(
            max(s['one_dist'].keys()) if s['one_dist'] else 0,
            max(s['zero_dist'].keys()) if s['zero_dist'] else 0,
        )
        print(f"  {'len':>4} {'1-runs':>8} {'0-runs':>8} {'1-frac':>8} {'0-frac':>8}")
        for L in range(1, max_len + 1):
            c1 = s['one_dist'].get(L, 0)
            c0 = s['zero_dist'].get(L, 0)
            f1 = c1 / s['one_runs'] if s['one_runs'] else 0
            f0 = c0 / s['zero_runs'] if s['zero_runs'] else 0
            if c1 or c0:
                print(f"  {L:>4} {c1:>8} {c0:>8} {f1:>8.4f} {f0:>8.4f}")

    # =========================================================
    # PART 3: 3-gram structure and T1 constraints
    # =========================================================
    print("\n--- PART 3: 3-gram analysis (gap residues mod 3) ---")
    print("Key: n3(a,b,c) = #{consecutive gap triples with g%3 = a,b,c}")
    print("In binary word z: '1' = g%3==0, '0' = g%3 in {1,2}")

    for k in range(3, min(len(PRIMES), 7)):
        s = analyze_level(k)
        print(f"\n  k={k}:")

        # Singleton 1-runs: pattern 010 in z = (nonzero, 0, nonzero) in gaps
        sin1 = s['n3'][(1, 0, 2)] + s['n3'][(2, 0, 1)]
        # Also the forbidden ones: n3(1,0,1) + n3(2,0,2)
        sin1_forb = s['n3'][(1, 0, 1)] + s['n3'][(2, 0, 2)]
        print(f"  Singleton 1-runs (010 in z):")
        print(f"    n3(1,0,2) + n3(2,0,1) = {sin1}  [allowed cross-class]")
        print(f"    n3(1,0,1) + n3(2,0,2) = {sin1_forb}  [T1-3gram FORBIDDEN]")
        print(f"    total = {sin1 + sin1_forb}")

        # Singleton 0-runs: pattern 101 in z = (0, nonzero, 0) in gaps
        sin0 = s['n3'][(0, 1, 0)] + s['n3'][(0, 2, 0)]
        print(f"  Singleton 0-runs (101 in z):")
        print(f"    n3(0,1,0) = {s['n3'][(0, 1, 0)]}")
        print(f"    n3(0,2,0) = {s['n3'][(0, 2, 0)]}")
        print(f"    total = {sin0}")

        print(f"  n1_single = {s['n1_single']}, n0_single = {s['n0_single']}")
        print(f"  Verification: sin1={sin1} == n1_single={s['n1_single']}: "
              f"{'OK' if sin1 == s['n1_single'] else 'FAIL'}")
        print(f"  Verification: sin0={sin0} == n0_single={s['n0_single']}: "
              f"{'OK' if sin0 == s['n0_single'] else 'FAIL'}")

    # =========================================================
    # PART 4: The structural argument
    # =========================================================
    print("\n--- PART 4: Structural argument ---")
    print("Since alpha < 1/2, 1s are the MINORITY in z.")
    print("Runs alternate, so #(1-runs) = #(0-runs) (+/- 1).")
    print("Average 1-run length < average 0-run length.")
    print()
    print("Key ratios:")
    print(f"{'k':>2} {'frac1_sing':>12} {'frac0_sing':>12} {'avg1':>8} {'avg0':>8}")
    print("-" * 50)
    for k in range(3, len(PRIMES)):
        s = analyze_level(k)
        f1s = s['n1_single'] / s['one_runs'] if s['one_runs'] else 0
        f0s = s['n0_single'] / s['zero_runs'] if s['zero_runs'] else 0
        avg1 = s['n1'] / s['one_runs'] if s['one_runs'] else 0
        avg0 = s['n0'] / s['zero_runs'] if s['zero_runs'] else 0
        print(f"{k:>2} {f1s:>12.6f} {f0s:>12.6f} {avg1:>8.3f} {avg0:>8.3f}")

    print()
    print("Observation: fraction of singletons among 1-runs > among 0-runs.")
    print("This is CONSISTENT with shorter average 1-run length.")

    # =========================================================
    # PART 5: CRT update analysis
    # =========================================================
    print("\n--- PART 5: CRT update effect on n100, n110 ---")
    print("How does adding prime p transform the counts?")
    print()

    prev = None
    print(f"{'k->k+1':>8} {'p':>3} {'n100_k':>8} {'n110_k':>8} "
          f"{'n100_k1':>8} {'n110_k1':>8} "
          f"{'r100':>8} {'r110':>8} {'diff_k':>8} {'diff_k1':>8}")
    print("-" * 95)
    for k in range(3, len(PRIMES)):
        s = analyze_level(k)
        z = s['z']
        n = len(z)
        z1 = np.roll(z, -1)
        z2 = np.roll(z, -2)
        n100 = int(np.count_nonzero((z == 1) & (z1 == 0) & (z2 == 0)))
        n110 = int(np.count_nonzero((z == 1) & (z1 == 1) & (z2 == 0)))
        diff = n100 - n110
        if prev is not None:
            p = PRIMES[k - 1]
            r100 = n100 / prev[0] if prev[0] else 0
            r110 = n110 / prev[1] if prev[1] else 0
            print(f"{k-1:>3}->{k:>2} {p:>3} {prev[0]:>8} {prev[1]:>8} "
                  f"{n100:>8} {n110:>8} "
                  f"{r100:>8.2f} {r110:>8.2f} {prev[2]:>8} {diff:>8}")
        prev = (n100, n110, diff)

    # =========================================================
    # PART 6: Markov prediction for n100, n110
    # =========================================================
    print("\n--- PART 6: Markov prediction ---")
    print("Under Markov(z_i -> z_{i+1}) with parameters alpha, T00:")
    print("  P(z=1) = alpha, P(z=0) = 1-alpha")
    print("  T00 = P(z_{i+1}=1 | z_i=1)")
    print("  T01 = 1 - T00")
    print("  T10 = alpha*T01/(1-alpha)  [stationarity]")
    print("  T11 = 1 - T10")
    print()
    print(f"{'k':>2} {'n100_exact':>10} {'n100_markov':>12} "
          f"{'n110_exact':>10} {'n110_markov':>12} "
          f"{'diff_exact':>10} {'diff_markov':>12}")
    print("-" * 80)

    for k in range(3, len(PRIMES)):
        s = analyze_level(k)
        a = float(s['alpha'])
        t00 = float(Fraction(s['n3'][(0, 0, 0)] + s['n3'][(0, 0, 1)] + s['n3'][(0, 0, 2)],
                              s['n1'])) if s['n1'] else 0
        # Hmm, T00 in the z-word is P(z_{i+1}=1 | z_i=1)
        z = s['z']
        n = len(z)
        z1 = np.roll(z, -1)
        z2 = np.roll(z, -2)
        n_z1 = int(np.count_nonzero(z == 1))
        n_z11 = int(np.count_nonzero((z == 1) & (z1 == 1)))
        n_z10 = int(np.count_nonzero((z == 1) & (z1 == 0)))
        T00_z = n_z11 / n_z1 if n_z1 else 0
        T01_z = 1 - T00_z
        T10_z = a * T01_z / (1 - a) if (1 - a) > 0 else 0
        T11_z = 1 - T10_z

        # Markov prediction:
        # n100_M = N * P(z=1)*P(z1=0|z=1)*P(z2=0|z1=0)
        #        = N * a * T01_z * T11_z
        n100_M = n * a * T01_z * T11_z
        # n110_M = N * P(z=1)*P(z1=1|z=1)*P(z2=0|z1=1)
        #        = N * a * T00_z * T01_z
        n110_M = n * a * T00_z * T01_z

        n100_exact = int(np.count_nonzero((z == 1) & (z1 == 0) & (z2 == 0)))
        n110_exact = int(np.count_nonzero((z == 1) & (z1 == 1) & (z2 == 0)))

        diff_exact = n100_exact - n110_exact
        diff_M = n100_M - n110_M

        print(f"{k:>2} {n100_exact:>10} {n100_M:>12.1f} "
              f"{n110_exact:>10} {n110_M:>12.1f} "
              f"{diff_exact:>10} {diff_M:>12.1f}")

    print()
    print("--- PART 6b: Markov algebraic argument ---")
    print("Under Markov:")
    print("  n100_M = N * a * (1-T00) * (1-T10)")
    print("  n110_M = N * a * T00 * (1-T00)")
    print("  diff_M = N * a * (1-T00) * (1-T10 - T00)")
    print("         = N * a * (1-T00) * (1 - T10 - T00)")
    print()
    print("  T10 = a*(1-T00)/(1-a)")
    print("  So 1 - T10 - T00 = 1 - T00 - a(1-T00)/(1-a)")
    print("                   = (1-T00)(1 - a/(1-a))")
    print("                   = (1-T00)(1-2a)/(1-a)")
    print()
    print("  diff_M = N * a * (1-T00)^2 * (1-2a) / (1-a)")
    print()
    print("  Since a < 1/2: (1-2a) > 0")
    print("  Since T00 < 1: (1-T00) > 0")
    print("  Since a > 0:   a > 0")
    print("  Therefore: diff_M > 0  [ALWAYS, under Markov]")
    print()
    print("  This proves n100 > n110 under the Markov approximation!")
    print("  The remaining question: does the non-Markov correction preserve this?")

    # Verification
    print()
    print(f"{'k':>2} {'diff_exact':>10} {'diff_markov':>12} {'non_markov':>12} {'sign':>6}")
    print("-" * 50)
    for k in range(3, len(PRIMES)):
        s = analyze_level(k)
        a = float(s['alpha'])
        z = s['z']
        n = len(z)
        z1 = np.roll(z, -1)
        n_z1 = int(np.count_nonzero(z == 1))
        n_z11 = int(np.count_nonzero((z == 1) & (z1 == 1)))
        T00_z = n_z11 / n_z1 if n_z1 else 0

        diff_M = n * a * (1 - T00_z)**2 * (1 - 2*a) / (1 - a)

        z2 = np.roll(z, -2)
        n100 = int(np.count_nonzero((z == 1) & (z1 == 0) & (z2 == 0)))
        n110 = int(np.count_nonzero((z == 1) & (z1 == 1) & (z2 == 0)))
        diff_exact = n100 - n110

        non_markov = diff_exact - diff_M
        sign = "+" if non_markov >= 0 else "-"
        print(f"{k:>2} {diff_exact:>10} {diff_M:>12.1f} {non_markov:>12.1f} {sign:>6}")

    # =========================================================
    # PART 7: Non-Markov correction analysis
    # =========================================================
    print("\n--- PART 7: Non-Markov correction ---")
    print("diff_exact = diff_Markov + correction")
    print("correction = n100_exact - n100_M - (n110_exact - n110_M)")
    print()
    print("The correction measures 3-point correlations beyond Markov.")
    print("If correction >= 0, then diff_exact >= diff_Markov > 0 and we're done.")
    print("If correction < 0 but |correction| < diff_Markov, we still have diff > 0.")
    print()
    print(f"{'k':>2} {'diff_M':>12} {'correction':>12} {'|corr|/diff_M':>14} {'safe':>6}")
    print("-" * 55)
    for k in range(3, len(PRIMES)):
        s = analyze_level(k)
        a = float(s['alpha'])
        z = s['z']
        n = len(z)
        z1 = np.roll(z, -1)
        z2 = np.roll(z, -2)
        n_z1 = int(np.count_nonzero(z == 1))
        n_z11 = int(np.count_nonzero((z == 1) & (z1 == 1)))
        T00_z = n_z11 / n_z1 if n_z1 else 0

        diff_M = n * a * (1 - T00_z)**2 * (1 - 2*a) / (1 - a)
        n100 = int(np.count_nonzero((z == 1) & (z1 == 0) & (z2 == 0)))
        n110 = int(np.count_nonzero((z == 1) & (z1 == 1) & (z2 == 0)))
        diff_exact = n100 - n110
        correction = diff_exact - diff_M
        ratio = abs(correction) / diff_M if diff_M > 0 else 0
        safe = "YES" if abs(correction) < diff_M else "MARGIN"
        print(f"{k:>2} {diff_M:>12.1f} {correction:>12.1f} {ratio:>14.6f} {safe:>6}")

    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print()
    print("1. Under Markov approximation:")
    print("   diff_M = N * alpha * (1-T00)^2 * (1-2*alpha) / (1-alpha)")
    print("   This is PROVABLY > 0 for alpha < 1/2  [EXACT ALGEBRAIC]")
    print()
    print("2. Non-Markov correction:")
    print("   Measures 3-point correlations beyond pair statistics.")
    print("   Sign and magnitude determine whether the full inequality holds.")
    print()
    print("3. Route to closure:")
    print("   If non-Markov correction is non-negative -> DONE (n100 > n110)")
    print("   If negative but bounded by |corr| < diff_M -> DONE")
    print("   Need: bound on 3-point correlations in the sieve word.")


if __name__ == "__main__":
    main()

sys.exit(0)
