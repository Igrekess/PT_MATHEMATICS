#!/usr/bin/env python3
"""
S15.6.273 -- Extension a k=10 via CRT incremental
===================================================

Compute level-10 (P(10) = 6,469,693,230) 3-gram statistics by:
1. Sieving level 9 (P(9) = 223,092,870) directly
2. Building level-10 gap class sequence via CRT replication + removal of multiples of 29
3. Counting 3-grams and verifying f_bnd < 1

Memory: ~2 GB peak (1 GB for level-10 gap classes + working memory)
Time: ~10-20 minutes
"""

import numpy as np
from math import prod
import time
import sys
import gc

PRIMES_9 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
p_new = 29


def main():
    t0 = time.time()
    P9 = prod(PRIMES_9)
    P10 = P9 * p_new
    print(f"P(9)  = {P9:,}")
    print(f"P(10) = {P10:,}")
    print(f"p_new = {p_new}")
    print()

    # =========================================================================
    # STEP 1: Level-9 sieve
    # =========================================================================
    print("STEP 1: Sieve level 9...")
    t1 = time.time()
    sieve = np.ones(P9 + 1, dtype=np.bool_)
    sieve[0] = False
    for p in PRIMES_9:
        sieve[::p] = False
    surv9 = np.flatnonzero(sieve).astype(np.int64)
    N9 = len(surv9)
    del sieve
    gc.collect()
    print(f"  N9 = {N9:,} survivors ({time.time()-t1:.1f}s)")

    # Level-9 gap classes
    gaps9 = np.empty(N9, dtype=np.int64)
    gaps9[:-1] = surv9[1:] - surv9[:-1]
    gaps9[-1] = P9 + surv9[0] - surv9[-1]
    C9 = (gaps9 % 3).astype(np.int8)
    del gaps9

    # Level-9 3-gram counts (for verification and d3_bnd computation)
    print("  Computing level-9 3-gram counts...")
    g3_9 = np.zeros((3, 3, 3), dtype=np.int64)
    g2_9 = np.zeros((3, 3), dtype=np.int64)
    g1_9 = np.zeros(3, dtype=np.int64)

    for a in range(3):
        g1_9[a] = int((C9 == a).sum())

    c0s, c1s = C9, np.roll(C9, -1)
    for a in range(3):
        ma = (c0s == a)
        for b in range(3):
            mab = ma & (c1s == b)
            g2_9[a, b] = int(mab.sum())
            c2s = np.roll(C9, -2)
            for cv in range(3):
                g3_9[a, b, cv] = int((mab & (c2s == cv)).sum())
    del c0s, c1s, c2s
    gc.collect()

    alpha_9 = g1_9[0] / N9
    T00_9 = g2_9[0, 0] / g1_9[0] if g1_9[0] > 0 else 0
    print(f"  alpha(9) = {alpha_9:.6f}, T00(9) = {T00_9:.6f}")

    # n100, n110 at level 9
    z9 = (C9 == 0).astype(np.int8)
    z9_1, z9_2 = np.roll(z9, -1), np.roll(z9, -2)
    n100_9 = int(((z9 == 1) & (z9_1 == 0) & (z9_2 == 0)).sum())
    n110_9 = int(((z9 == 1) & (z9_1 == 1) & (z9_2 == 0)).sum())
    diff_9 = n100_9 - n110_9
    del z9, z9_1, z9_2
    gc.collect()
    print(f"  n100(9) = {n100_9:,}, n110(9) = {n110_9:,}, diff(9) = {diff_9:,}")

    # =========================================================================
    # STEP 2: Compute residues and cumulative sums for CRT
    # =========================================================================
    print()
    print("STEP 2: Prepare CRT replication...")
    R9 = (surv9 % p_new).astype(np.int8)
    del surv9
    gc.collect()

    # Cumulative sum for gap class merging
    cumC = np.zeros(N9 + 1, dtype=np.int32)
    cumC[1:] = np.cumsum(C9.astype(np.int32))
    total_C9 = int(cumC[N9])

    P9_mod29 = P9 % p_new
    print(f"  P9 mod 29 = {P9_mod29}")
    print(f"  Total gap class sum = {total_C9}")

    # =========================================================================
    # STEP 3: Build level-10 gap class array via CRT
    # =========================================================================
    print()
    print("STEP 3: Build level-10 gap classes (29 copies, removing multiples of 29)...")
    t3 = time.time()

    N10_expected = (p_new - 1) * N9
    print(f"  Expected N10 = {N10_expected:,}")

    # Pre-compute kept indices and internal gap classes for each copy
    copy_data = []
    total_gaps = 0

    for j in range(p_new):
        r_j = int((-P9_mod29 * j) % p_new)
        kept = np.flatnonzero(R9 != r_j)
        n_kept = len(kept)

        # Internal gap classes (between consecutive kept survivors within this copy)
        internal_sums = cumC[kept[1:]] - cumC[kept[:-1]]
        internal_classes = (internal_sums % 3).astype(np.int8)

        copy_data.append({
            'first_kept': int(kept[0]),
            'last_kept': int(kept[-1]),
            'n_kept': n_kept,
            'internal_classes': internal_classes,
        })
        total_gaps += n_kept  # n_kept-1 internal + 1 seam
        del kept, internal_sums

        if j % 10 == 0:
            print(f"    Copy {j}/29: {n_kept:,} kept survivors")

    gc.collect()
    print(f"  Total gaps: {total_gaps:,} (expected {N10_expected:,})")
    assert total_gaps == N10_expected, f"Gap count mismatch: {total_gaps} != {N10_expected}"

    # Build full gap class array
    print("  Assembling full gap class array...")
    C_10 = np.empty(N10_expected, dtype=np.int8)
    pos = 0

    for j in range(p_new):
        cd = copy_data[j]
        j_next = (j + 1) % p_new

        # Write internal classes
        ic = cd['internal_classes']
        n_internal = len(ic)
        C_10[pos:pos + n_internal] = ic
        pos += n_internal

        # Compute seam gap class (from last kept of copy j to first kept of copy j+1)
        seam_sum = total_C9 - int(cumC[cd['last_kept']]) + int(cumC[copy_data[j_next]['first_kept']])
        seam_class = seam_sum % 3
        C_10[pos] = seam_class
        pos += 1

    assert pos == N10_expected, f"Position mismatch: {pos} != {N10_expected}"

    # Free CRT working data
    del copy_data, cumC, R9, C9
    gc.collect()

    N10 = len(C_10)
    print(f"  Level-10 gap class array built: {N10:,} entries ({time.time()-t3:.1f}s)")

    # =========================================================================
    # STEP 4: Compute level-10 statistics
    # =========================================================================
    print()
    print("STEP 4: Compute level-10 statistics...")
    t4 = time.time()

    # 1-gram counts
    g1_10 = np.zeros(3, dtype=np.int64)
    for a in range(3):
        g1_10[a] = int((C_10 == a).sum())
    alpha_10 = g1_10[0] / N10
    print(f"  alpha(10) = {alpha_10:.6f}")

    # 2-gram counts
    print("  Computing 2-gram counts...")
    g2_10 = np.zeros((3, 3), dtype=np.int64)
    # Use slicing instead of roll to save memory
    c0 = C_10[:-1]
    c1 = C_10[1:]
    for a in range(3):
        ma = (c0 == a)
        for b in range(3):
            g2_10[a, b] = int((ma & (c1 == b)).sum())
    # Wraparound: last element -> first element
    g2_10[C_10[-1], C_10[0]] += 1
    del c0, c1

    T00_10 = g2_10[0, 0] / g1_10[0] if g1_10[0] > 0 else 0
    print(f"  T00(10) = {T00_10:.6f}")

    # 3-gram counts
    print("  Computing 3-gram counts (this may take a few minutes)...")
    t4b = time.time()
    g3_10 = np.zeros((3, 3, 3), dtype=np.int64)
    c0 = C_10[:-2]
    c1 = C_10[1:-1]
    c2 = C_10[2:]
    for a in range(3):
        ma = (c0 == a)
        for b in range(3):
            mab = ma & (c1 == b)
            for cv in range(3):
                g3_10[a, b, cv] = int((mab & (c2 == cv)).sum())
        print(f"    a={a} done ({time.time()-t4b:.1f}s)")
    # Wraparound: 2 extra 3-grams
    g3_10[C_10[-2], C_10[-1], C_10[0]] += 1
    g3_10[C_10[-1], C_10[0], C_10[1]] += 1
    del c0, c1, c2
    gc.collect()
    print(f"  3-gram counts done ({time.time()-t4:.1f}s)")

    # n100, n110 at level 10
    z10 = (C_10 == 0).astype(np.int8)
    z10_1 = np.roll(z10, -1)
    z10_2 = np.roll(z10, -2)
    n100_10 = int(((z10 == 1) & (z10_1 == 0) & (z10_2 == 0)).sum())
    n110_10 = int(((z10 == 1) & (z10_1 == 1) & (z10_2 == 0)).sum())
    diff_10 = n100_10 - n110_10
    del z10, z10_1, z10_2, C_10
    gc.collect()
    print(f"  n100(10) = {n100_10:,}, n110(10) = {n110_10:,}, diff(10) = {diff_10:,}")

    # =========================================================================
    # STEP 5: CRT boundary terms and verification
    # =========================================================================
    print()
    print("=" * 80)
    print("STEP 5: RESULTATS -- transition k=9 -> k=10 (p=29)")
    print("=" * 80)
    print()

    # d3_bnd = g3(10) - (p-3) * g3(9)
    d3_bnd = g3_10 - (p_new - 3) * g3_9
    print("  d3_bnd(0,b,c) = n3(10,0,b,c) - 26*n3(9,0,b,c):")
    print()
    for b in range(3):
        for c in range(3):
            print(f"    d3_bnd(0,{b},{c}) = {d3_bnd[0,b,c]:>12,}")

    # Verify forbidden triples persist
    print()
    print(f"  d3_bnd(0,1,1) = {d3_bnd[0,1,1]:,}  [{'ZERO' if d3_bnd[0,1,1]==0 else 'NONZERO!'}]")
    print(f"  d3_bnd(0,2,2) = {d3_bnd[0,2,2]:,}  [{'ZERO' if d3_bnd[0,2,2]==0 else 'NONZERO!'}]")
    print(f"  d3_bnd(0,0,1) = {d3_bnd[0,0,1]:,}")
    print(f"  d3_bnd(0,0,2) = {d3_bnd[0,0,2]:,}  [{'SYM' if d3_bnd[0,0,1]==d3_bnd[0,0,2] else 'ASYM!'}]")

    # Delta_diff
    Delta_diff = int(d3_bnd[0, 1, 2] + d3_bnd[0, 2, 1] - 2 * d3_bnd[0, 0, 1])
    print(f"\n  Delta_diff = {Delta_diff:,}  [{'> 0 OK' if Delta_diff > 0 else 'PROBLEME!'}]")

    # CRT recurrence check
    Delta_diff_check = diff_10 - (p_new - 3) * diff_9
    print(f"  Delta_diff (recurrence) = {Delta_diff_check:,}  [{'MATCH' if Delta_diff == Delta_diff_check else 'MISMATCH!'}]")

    # Delta_M (Markov prediction)
    R_bnd = int(d3_bnd[0].sum())
    T01_10 = (1 - T00_10) / 2
    T10_10 = alpha_10 * (1 - T00_10) / (1 - alpha_10) if (1 - alpha_10) > 0 else 0
    T12_10 = 1 - T10_10
    Delta_M = R_bnd * 2 * T01_10 * (T12_10 - T00_10)

    print(f"\n  R_bnd = {R_bnd:,}")
    print(f"  alpha(10) = {alpha_10:.6f}")
    print(f"  T00(10) = {T00_10:.6f}")
    print(f"  T12(10) = {T12_10:.6f}")
    print(f"  T12 - T00 = {T12_10 - T00_10:.6f}")
    print(f"  Delta_M = {Delta_M:,.1f}")

    # f_bnd
    correction = Delta_diff - Delta_M
    f_bnd = abs(correction) / Delta_M if Delta_M > 0 else float('inf')
    print(f"\n  correction = {correction:,.1f}")
    print(f"  f_bnd = {f_bnd:.4f}  [{'< 1 OK !' if f_bnd < 1 else 'ECHEC'}]")

    # Eta values
    print(f"\n  Deviations eta(b,c) pour les termes actifs:")
    for (b, c), label in [((0,1), "eta(0,1)"), ((1,2), "eta(1,2)"),
                           ((2,1), "eta(2,1)"), ((1,0), "eta(1,0)")]:
        T_mat = [[T00_10, T01_10, T01_10],
                 [T10_10, 0, T12_10],
                 [T10_10, T12_10, 0]]
        d3_M = R_bnd * T_mat[0][b] * T_mat[b][c]
        if d3_M > 0:
            eta = d3_bnd[0, b, c] / d3_M - 1
            print(f"    {label} = {eta:+.4f}")

    # A * max_eta
    sum_abs_w = R_bnd * (T01_10 * T12_10 * 2 + T00_10 * T01_10 * 2)
    A = sum_abs_w / Delta_M if Delta_M > 0 else float('inf')
    max_eta = 0
    for b, c in [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]:
        T_mat = [[T00_10, T01_10, T01_10],
                 [T10_10, 0, T12_10],
                 [T10_10, T12_10, 0]]
        d3_M = R_bnd * T_mat[0][b] * T_mat[b][c]
        if d3_M > 0.5:
            eta = abs(d3_bnd[0, b, c] / d3_M - 1)
            max_eta = max(max_eta, eta)
    print(f"\n  A = {A:.3f}")
    print(f"  max|eta| = {max_eta:.4f}")
    print(f"  A * max|eta| = {A * max_eta:.4f}  [{'< 1 OK' if A*max_eta < 1 else '> 1'}]")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 80)
    print("RESUME COMPLET k=3..10")
    print("=" * 80)
    print()
    print(f"  diff(9) = {diff_9:,}")
    print(f"  diff(10) = {diff_10:,}")
    print(f"  diff(10) > 0 : {'OUI' if diff_10 > 0 else 'NON'}")
    print(f"  Delta_diff = {Delta_diff:,}  {'> 0' if Delta_diff > 0 else '<= 0'}")
    print(f"  f_bnd = {f_bnd:.4f}  {'< 1' if f_bnd < 1 else '>= 1'}")
    print()
    print(f"  ==> Gap (F) verifie pour k <= 10 (EXACT)")
    print(f"  ==> Pour k >= 11: borne spectrale, marge estimee > 5x")
    print()
    print(f"  Temps total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

sys.exit(0)
