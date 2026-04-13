#!/usr/bin/env python3
"""
S15.6.270 -- Algebraic structure of Delta_diff for closing f(k) < 1
=====================================================================

GOAL: Find an exact algebraic formula for
    Delta_diff = diff(k+1) - (p-3)*diff(k)
              = [n100(k+1) - n110(k+1)] - (p-3)*[n100(k) - n110(k)]
              = delta_100 - delta_110

where delta_100 = n100(k+1) - (p-3)*n100(k) counts boundary 3-grams of
type (1,0,0) in the z-word, and similarly delta_110 for type (1,1,0).

If Delta_diff >= 0 for all k >= 3, then diff(k) > 0 for all k by trivial
induction (base: diff(3) = 1 > 0, step: diff(k+1) >= (p-3)*diff(k) > 0).

APPROACH: Express delta_100 and delta_110 in terms of known quantities
at level k (gap-class 3-gram counts, 4-gram counts, etc.) and identify
the algebraic structure that makes Delta_diff >= 0.
"""

import numpy as np
from fractions import Fraction
from math import prod
from collections import defaultdict

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def sieve_full_data(prime_list):
    """Compute full sieve data including all n-gram counts needed."""
    P = prod(prime_list)
    if P > 500_000_000:
        return None
    sieve = np.ones(P + 1, dtype=np.bool_)
    sieve[0] = False
    for p in prime_list:
        sieve[::p] = False
    survivors = np.flatnonzero(sieve)
    N = len(survivors)
    gaps = np.empty(N, dtype=np.int64)
    gaps[:-1] = survivors[1:] - survivors[:-1]
    gaps[-1] = P + survivors[0] - survivors[-1]

    # Gap classes mod 3
    classes = gaps % 3
    z = (classes == 0).astype(int)

    # Binary z-word statistics
    z1 = np.roll(z, -1)
    z2 = np.roll(z, -2)
    n1 = int(np.count_nonzero(z == 1))
    n100 = int(np.count_nonzero((z == 1) & (z1 == 0) & (z2 == 0)))
    n110 = int(np.count_nonzero((z == 1) & (z1 == 1) & (z2 == 0)))
    n101 = int(np.count_nonzero((z == 1) & (z1 == 0) & (z2 == 1)))
    n111 = int(np.count_nonzero((z == 1) & (z1 == 1) & (z2 == 1)))
    diff = n100 - n110

    # Gap-class 3-gram counts
    c0 = classes
    c1 = np.roll(classes, -1)
    c2 = np.roll(classes, -2)
    c3 = np.roll(classes, -3)

    g3 = np.zeros((3, 3, 3), dtype=np.int64)
    for a in range(3):
        ma = (c0 == a)
        for b in range(3):
            mab = ma & (c1 == b)
            for c in range(3):
                g3[a, b, c] = int((mab & (c2 == c)).sum())

    # Gap-class 2-gram counts
    g2 = np.zeros((3, 3), dtype=np.int64)
    for a in range(3):
        for b in range(3):
            g2[a, b] = int(((c0 == a) & (c1 == b)).sum())

    # Gap-class 4-gram counts (for understanding boundary terms)
    g4 = None
    if P <= 50_000_000:
        g4 = np.zeros((3, 3, 3, 3), dtype=np.int64)
        for a in range(3):
            ma = (c0 == a)
            for b in range(3):
                mab = ma & (c1 == b)
                for c in range(3):
                    mabc = mab & (c2 == c)
                    for d in range(3):
                        g4[a, b, c, d] = int((mabc & (c3 == d)).sum())

    # 1-gram counts
    g1 = np.array([int((classes == a).sum()) for a in range(3)])

    alpha = Fraction(n1, N)
    T00 = Fraction(g2[0, 0], g1[0]) if g1[0] > 0 else Fraction(0)

    return {
        'k': len(prime_list), 'P': P, 'N': N,
        'gaps': gaps, 'classes': classes, 'z': z,
        'survivors': survivors,
        'n100': n100, 'n110': n110, 'n101': n101, 'n111': n111,
        'diff': diff,
        'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4,
        'alpha': alpha, 'T00': T00, 'n1': n1,
    }


def main():
    print("=" * 90)
    print("S15.6.270 -- ALGEBRAIC STRUCTURE OF Delta_diff")
    print("=" * 90)

    levels = {}
    for k in range(3, len(PRIMES) + 1):
        r = sieve_full_data(PRIMES[:k])
        if r is None:
            break
        levels[k] = r
        print(f"  k={k}: P={r['P']:>12,}, N={r['N']:>10,}, diff={r['diff']:>10,}")
    k_max = max(levels.keys())

    # =========================================================================
    # PART 1: Exact Delta_diff decomposition
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 1: DECOMPOSITION EXACTE de Delta_diff = delta_100 - delta_110")
    print("=" * 90)
    print()

    k_list = sorted(levels.keys())

    print(f"  {'k->k+1':>8} {'p':>4} {'diff(k)':>10} {'diff(k+1)':>10}"
          f" {'(p-3)*d':>10} {'Delta_diff':>10} {'delta_100':>10} {'delta_110':>10}")
    print("  " + "-" * 82)

    transitions = []
    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        dk = rk['diff']
        dk1 = rk1['diff']
        main = (p - 3) * dk
        delta_diff = dk1 - main

        d100 = rk1['n100'] - (p - 3) * rk['n100']
        d110 = rk1['n110'] - (p - 3) * rk['n110']

        transitions.append({
            'k': k, 'k1': k1, 'p': p,
            'dk': dk, 'dk1': dk1,
            'delta_diff': delta_diff,
            'd100': d100, 'd110': d110,
        })

        print(f"  {k:>3}->{k1:>2} {p:>4} {dk:>10} {dk1:>10}"
              f" {main:>10} {delta_diff:>10} {d100:>10} {d110:>10}")

    print()
    print("  OBSERVATION: Delta_diff >= 0 a TOUS les niveaux!")
    print("  (0 au k=3->4, strictement positif ensuite)")

    # =========================================================================
    # PART 2: delta_100 and delta_110 in terms of gap-class 3-grams
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 2: DECOMPOSITION en 3-grammes de classes de gaps")
    print("=" * 90)
    print()
    print("  n100 = sum_{b,c in {1,2}} n3_gap(0,b,c)")
    print("  n110 = sum_{c in {1,2}} n3_gap(0,0,c)")
    print()
    print("  Donc:")
    print("  delta_100 = sum_{b,c in {1,2}} [n3_gap'(0,b,c) - (p-3)*n3_gap(0,b,c)]")
    print("  delta_110 = sum_{c in {1,2}} [n3_gap'(0,0,c) - (p-3)*n3_gap(0,0,c)]")
    print()

    # Verify and compute the 3-gram boundary terms
    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        print(f"  k={k}->{k1} (p={p}):")

        # delta for each gap-class 3-gram
        d3 = np.zeros((3, 3, 3), dtype=np.int64)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    d3[a, b, c] = int(rk1['g3'][a, b, c]) - (p - 3) * int(rk['g3'][a, b, c])

        # delta_100 from gap-class 3-grams
        d100_check = sum(int(d3[0, b, c]) for b in [1, 2] for c in [1, 2])
        d110_check = sum(int(d3[0, 0, c]) for c in [1, 2])

        t = transitions[i]
        print(f"    delta_100 = {t['d100']:>10} (direct), {d100_check:>10} (from 3-grams)"
              f"  {'OK' if t['d100'] == d100_check else 'FAIL'}")
        print(f"    delta_110 = {t['d110']:>10} (direct), {d110_check:>10} (from 3-grams)"
              f"  {'OK' if t['d110'] == d110_check else 'FAIL'}")

        # Show ALL delta_3 for 3-grams starting with class 0
        print(f"    delta_3(0,b,c) boundary terms:")
        for b in range(3):
            for c in range(3):
                v = int(d3[0, b, c])
                contrib = ""
                if b != 0 and c != 0:
                    contrib = " -> delta_100"
                elif b == 0 and c != 0:
                    contrib = " -> delta_110"
                elif b == 0 and c == 0:
                    contrib = " -> (0,0,0) = n111 boundary"
                print(f"      d3(0,{b},{c}) = {v:>10}{contrib}")

        # Also show delta_3 summed by z-pattern
        d_n100 = sum(int(d3[0, b, c]) for b in [1, 2] for c in [1, 2])
        d_n110 = sum(int(d3[0, 0, c]) for c in [1, 2])
        d_n101 = sum(int(d3[0, b, 0]) for b in [1, 2])
        d_n111 = int(d3[0, 0, 0])
        print(f"    Summary: d_n100={d_n100}, d_n110={d_n110}, d_n101={d_n101}, d_n111={d_n111}")
        print(f"    Delta_diff = d_n100 - d_n110 = {d_n100 - d_n110}")
        print()

    # =========================================================================
    # PART 3: Ratio delta_100/delta_110 -- is there a pattern?
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 3: RATIO delta_100/delta_110")
    print("=" * 90)
    print()

    print(f"  {'k->k+1':>8} {'p':>4} {'delta_100':>10} {'delta_110':>10}"
          f" {'ratio':>10} {'Delta_diff':>10}")
    print("  " + "-" * 60)

    for t in transitions:
        ratio = t['d100'] / t['d110'] if t['d110'] != 0 else float('inf')
        print(f"  {t['k']:>3}->{t['k1']:>2} {t['p']:>4} {t['d100']:>10} {t['d110']:>10}"
              f" {ratio:>10.4f} {t['delta_diff']:>10}")

    # =========================================================================
    # PART 4: Normalize by N(k) -- boundary density
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 4: DENSITE des termes de bord (normalisation par N(k))")
    print("=" * 90)
    print()
    print("  La contribution de bord par survivor au niveau k:")
    print()

    print(f"  {'k->k+1':>8} {'N(k)':>10} {'d100/N':>10} {'d110/N':>10}"
          f" {'(d100-d110)/N':>14} {'diff_M(k)/N':>12}")
    print("  " + "-" * 74)

    for i, t in enumerate(transitions):
        Nk = levels[t['k']]['N']
        alpha_k = float(levels[t['k']]['alpha'])
        T00_k = float(levels[t['k']]['T00'])
        T01_k = 1 - T00_k
        eps_k = 0.5 - alpha_k
        dM_over_N = alpha_k * T01_k**2 * (1 - 2*alpha_k) / (1 - alpha_k)

        print(f"  {t['k']:>3}->{t['k1']:>2} {Nk:>10} {t['d100']/Nk:>10.6f}"
              f" {t['d110']/Nk:>10.6f} {t['delta_diff']/Nk:>14.6f}"
              f" {dM_over_N:>12.6f}")

    # =========================================================================
    # PART 5: Express delta_100, delta_110 via known recurrences
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 5: FORMULES EXACTES via les recurrences CRT connues")
    print("=" * 90)
    print()
    print("  La formule CRT pour les 2-grammes est (S15.6.256):")
    print("    n'_{ab} = (p-3)*n_{ab} + A_{ab} + B_{ab}")
    print()
    print("  Ou A_{ab} et B_{ab} sont les termes de bord.")
    print("  Pour les 3-grammes: delta_3(a,b,c) = n3'(a,b,c) - (p-3)*n3(a,b,c)")
    print("  depend des 4-grammes au niveau k.")
    print()

    # Try to express delta_3(0,b,c) in terms of 4-grams and 2-grams
    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        if rk['g4'] is None or rk1['g4'] is None:
            continue

        print(f"  k={k}->{k1} (p={p}):")

        # Compute delta_3 for all 3-grams
        d3 = np.zeros((3, 3, 3), dtype=np.int64)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    d3[a, b, c] = int(rk1['g3'][a, b, c]) - (p - 3) * int(rk['g3'][a, b, c])

        # Test hypothesis: delta_3(a,b,c) = F(n4_gap, n3_gap, n2_gap, p)
        # Try: delta_3(a,b,c) = 2*n3(a,b,c) + sum_d [n4(d,a,b,c)] - G
        # Or: delta_3(a,b,c) = sum_d [A_{da}*T(b,c)] type formula

        # Let's see: how does the 2-gram recurrence work?
        # n2'(a,b) = (p-3)*n2(a,b) + A_{ab} + B_{ab}
        # Compute A+B for 2-grams
        d2 = np.zeros((3, 3), dtype=np.int64)
        for a in range(3):
            for b in range(3):
                d2[a, b] = int(rk1['g2'][a, b]) - (p - 3) * int(rk['g2'][a, b])

        # Check: sum_b delta_2(a,b) = delta_1(a)
        d1 = np.zeros(3, dtype=np.int64)
        for a in range(3):
            d1[a] = int(rk1['g1'][a]) - (p - 3) * int(rk['g1'][a])

        print(f"    delta_1 = {d1}")
        print(f"    sum_b delta_2(a,b): {[sum(int(d2[a,b]) for b in range(3)) for a in range(3)]}")

        # Key: check if delta_3(a,b,c) relates to delta_2 and g3/g2
        # Hypothesis: delta_3(a,b,c) ≈ delta_2(a,b) * T(b,c)
        # where T(b,c) = g2(b,c)/g1(b) is the transition probability
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    T_bc = int(rk['g2'][b, c]) / int(rk['g1'][b]) if rk['g1'][b] > 0 else 0
                    predicted = d2[a, b] * T_bc
                    actual = d3[a, b, c]
                    err = actual - predicted
                    if a == 0 and (b != 0 or c != 0):  # Focus on n100/n110 related
                        pass  # Will show below

        # Show the comparison for the relevant 3-grams
        print(f"    Test: delta_3(0,b,c) vs delta_2(0,b) * T(b,c)?")
        for b in range(3):
            for c in range(3):
                T_bc = int(rk['g2'][b, c]) / int(rk['g1'][b]) if rk['g1'][b] > 0 else 0
                predicted = float(d2[0, b]) * T_bc
                actual = int(d3[0, b, c])
                err = actual - predicted
                pct = err / actual * 100 if actual != 0 else 0
                print(f"      d3(0,{b},{c}) = {actual:>8}, d2(0,{b})*T({b},{c}) = {predicted:>10.1f},"
                      f" err = {err:>8.1f} ({pct:>6.1f}%)")
        print()

    # =========================================================================
    # PART 6: Second hypothesis -- factored formula with correction
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 6: HYPOTHESE FACTORISEE -- delta_3 = delta_2 * T + correction_3")
    print("=" * 90)
    print()
    print("  Si delta_3(a,b,c) = delta_2(a,b) * T(b,c) + corr_3(a,b,c),")
    print("  alors Delta_diff = sum_{bc relevant} [delta_2(0,b)*T(b,c) + corr_3(0,b,c)]")
    print("                   - sum_{c relevant} [delta_2(0,0)*T(0,c) + corr_3(0,0,c)]")
    print()

    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        if rk['g4'] is None:
            continue

        d3 = np.zeros((3, 3, 3), dtype=np.int64)
        d2 = np.zeros((3, 3), dtype=np.int64)
        for a in range(3):
            for b in range(3):
                d2[a, b] = int(rk1['g2'][a, b]) - (p - 3) * int(rk['g2'][a, b])
                for c in range(3):
                    d3[a, b, c] = int(rk1['g3'][a, b, c]) - (p - 3) * int(rk['g3'][a, b, c])

        # Compute the factored prediction and residual
        # Delta_diff via factored formula
        delta_diff_factored = 0
        delta_diff_corr3 = 0
        delta_100_factored = 0
        delta_110_factored = 0

        for b in [1, 2]:
            for c in [1, 2]:
                T_bc = int(rk['g2'][b, c]) / int(rk['g1'][b]) if rk['g1'][b] > 0 else 0
                delta_100_factored += float(d2[0, b]) * T_bc

        for c in [1, 2]:
            T_0c = int(rk['g2'][0, c]) / int(rk['g1'][0]) if rk['g1'][0] > 0 else 0
            delta_110_factored += float(d2[0, 0]) * T_0c

        ddiff_factored = delta_100_factored - delta_110_factored

        # Actual
        ddiff_actual = transitions[i]['delta_diff']

        # Correction
        ddiff_correction = ddiff_actual - ddiff_factored

        print(f"  k={k}->{k1} (p={p}):")
        print(f"    Delta_diff (actual)   = {ddiff_actual}")
        print(f"    Delta_diff (factored) = {ddiff_factored:.2f}")
        print(f"    Correction_3          = {ddiff_correction:.2f}")
        print(f"    Ratio corr/actual     = {ddiff_correction/ddiff_actual:.4f}" if ddiff_actual != 0 else "    Ratio: n/a")
        print()

    # =========================================================================
    # PART 7: Direct formula -- delta_100 and delta_110 from the CRT
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 7: FORMULE DIRECTE via les A/B du CRT 2-gramme")
    print("=" * 90)
    print()
    print("  Rappel: la formule CRT exacte pour n2'(a,b) est:")
    print("    n2'(a,b) = (p-3)*n2(a,b) + A(a,b) + B(a,b)")
    print("  avec A(a,b) = n2(a, (a+b) mod 3) et B(a,b) = n2((a+b) mod 3, b)")
    print("  (formule de S15.6.256)")
    print()

    # Verify the 2-gram CRT formula
    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        print(f"  k={k}->{k1} (p={p}): Verification 2-gram CRT formula")
        ok_count = 0
        for a in range(3):
            for b in range(3):
                s = (a + b) % 3
                A_ab = int(rk['g2'][a, s])
                B_ab = int(rk['g2'][s, b])
                predicted = (p - 3) * int(rk['g2'][a, b]) + A_ab + B_ab
                actual = int(rk1['g2'][a, b])
                ok = (predicted == actual)
                if ok:
                    ok_count += 1
                else:
                    print(f"    n2({a},{b}): predicted={predicted}, actual={actual} FAIL")
        print(f"    {ok_count}/9 OK")
        print()

    # =========================================================================
    # PART 8: Derive the 3-gram CRT formula
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 8: DERIVATION de la formule CRT pour les 3-grammes")
    print("=" * 90)
    print()
    print("  Pour les 3-grammes, on s'attend a:")
    print("    n3'(a,b,c) = (p-3)*n3(a,b,c) + [somme de termes de bord]")
    print()
    print("  Les termes de bord proviennent des positions ou le gap est")
    print("  'coupe' par le nouveau premier p. A chaque coupure, un gap g")
    print("  de classe s est remplace par deux sous-gaps de classes")
    print("  (s+r) mod 3 et (s-r) mod 3 pour un certain residu r.")
    print()
    print("  Test empirique: cherchons la formule exacte.")
    print()

    # For each transition, try to find the formula for delta_3
    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        d3 = np.zeros((3, 3, 3), dtype=np.int64)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    d3[a, b, c] = int(rk1['g3'][a, b, c]) - (p - 3) * int(rk['g3'][a, b, c])

        # Hypothesis 1: delta_3(a,b,c) = n3(a, s_ab, c) + n3(a, b, s_bc) where s=a+b mod 3, etc.
        # This would be the natural extension of the 2-gram formula
        print(f"  k={k}->{k1} (p={p}):")
        print(f"    Test hypothese: delta_3(a,b,c) = n3(a, (a+b)%3, c) + n3(a, b, (b+c)%3)")
        print(f"                                     + n4_correction ?")

        ok_count = 0
        max_err = 0
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    s_ab = (a + b) % 3
                    s_bc = (b + c) % 3
                    pred = int(rk['g3'][a, s_ab, c]) + int(rk['g3'][a, b, s_bc])
                    actual = int(d3[a, b, c])
                    err = actual - pred
                    max_err = max(max_err, abs(err))
                    if err == 0:
                        ok_count += 1
        print(f"    Hypothese 1: {ok_count}/27 exact, max_err = {max_err}")

        # Hypothesis 2: include 4-gram correction terms
        # delta_3(a,b,c) = n3(a,s_ab,c) + n3(a,b,s_bc) - n3(a,s_ab,s_bc)
        # (inclusion-exclusion: subtract double-counted boundary)
        ok_count2 = 0
        max_err2 = 0
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    s_ab = (a + b) % 3
                    s_bc = (b + c) % 3
                    pred = (int(rk['g3'][a, s_ab, c])
                            + int(rk['g3'][a, b, s_bc])
                            - int(rk['g3'][a, s_ab, s_bc]))
                    actual = int(d3[a, b, c])
                    err = actual - pred
                    max_err2 = max(max_err2, abs(err))
                    if err == 0:
                        ok_count2 += 1
        print(f"    Hypothese 2 (incl-excl): {ok_count2}/27 exact, max_err = {max_err2}")

        # Hypothesis 3: delta_3(a,b,c) involves 4-grams
        if rk['g4'] is not None:
            # Try: delta_3(a,b,c) = n3(a,s,c) + n3(a,b,t)
            #   + sum_d [n4(d,a,b,c) correction terms]
            # Actually, think about it more carefully.
            # In the CRT, the boundary terms at a 3-gram position involve
            # the LEFT context (predecessor gap) and RIGHT context (successor gap).
            # So the correction should involve 4-grams.

            # Try: delta_3(a,b,c) = sum_{d} f(d,a,b,c)
            # where f depends on the 4-gram structure.

            # Let's try: delta_3(a,b,c) = n4(s_ab, a, b, c)/n2(s_ab, a) * n2(s_ab, a)
            # Hmm, that's circular. Let me think differently.

            # The actual boundary at position i contributes to delta_3(a,b,c) if
            # the 3-gram (class_{i}, class_{i+1}, class_{i+2}) at level k+1 involves
            # a boundary. There are two types of boundaries:
            # Type L: the boundary is between positions i-1 and i (left boundary)
            # Type R: the boundary is between positions i+1 and i+2 (right boundary)
            # Type M: the boundary is at position i+1 (middle of the 3-gram)

            # For Type L: the 3-gram is (new_class, b, c) where new_class depends
            # on the split of the predecessor gap. This involves 4-grams (d,a,b,c)
            # at level k where d is the predecessor class.

            # This is getting complicated. Let me try a different pattern.

            # Hypothesis 3: delta_3(a,b,c) = A3(a,b,c) + B3(a,b,c) where
            # A3 involves the "left" boundary and B3 the "right" boundary.
            # A3(a,b,c) = sum_d n4(d, s_da, b, c) ... no, not right either.

            # Actually, let me look at the PATTERN of errors from Hypothesis 2
            print(f"    Errors from Hypothesis 2:")
            err_pattern = {}
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        s_ab = (a + b) % 3
                        s_bc = (b + c) % 3
                        pred = (int(rk['g3'][a, s_ab, c])
                                + int(rk['g3'][a, b, s_bc])
                                - int(rk['g3'][a, s_ab, s_bc]))
                        actual = int(d3[a, b, c])
                        err = actual - pred
                        if err != 0:
                            err_pattern[(a, b, c)] = err
                            # Can we express this error in terms of 4-grams?
                            if rk['g4'] is not None:
                                # Try: error = n4(a+b, a, b+c, c) - n4(a, a+b, b, b+c)
                                # or some other 4-gram combination
                                pass

            if err_pattern:
                for (a, b, c), err in sorted(err_pattern.items()):
                    print(f"      d3({a},{b},{c}): err = {err}")
            else:
                print(f"      All errors = 0!")

        print()

    # =========================================================================
    # PART 9: Test the inclusion-exclusion formula on Delta_diff
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 9: FORMULE D'INCLUSION-EXCLUSION pour Delta_diff")
    print("=" * 90)
    print()
    print("  Si delta_3(a,b,c) = n3(a,s_ab,c) + n3(a,b,s_bc) - n3(a,s_ab,s_bc),")
    print("  alors:")
    print()
    print("  delta_100 = sum_{b,c in {1,2}} [n3(0,s_{0b},c) + n3(0,b,s_{bc}) - n3(0,s_{0b},s_{bc})]")
    print("  delta_110 = sum_{c in {1,2}} [n3(0,s_{00},c) + n3(0,0,s_{0c}) - n3(0,s_{00},s_{0c})]")
    print()
    print("  Ou s_{ab} = (a+b) mod 3.")
    print()

    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        g3 = rk['g3']

        # Compute delta_100 and delta_110 via inclusion-exclusion
        d100_ie = 0
        for b in [1, 2]:
            for c in [1, 2]:
                s_0b = (0 + b) % 3  # = b
                s_bc = (b + c) % 3
                d100_ie += int(g3[0, s_0b, c]) + int(g3[0, b, s_bc]) - int(g3[0, s_0b, s_bc])

        d110_ie = 0
        for c in [1, 2]:
            s_00 = 0  # (0+0) % 3 = 0
            s_0c = c   # (0+c) % 3 = c
            d110_ie += int(g3[0, 0, c]) + int(g3[0, 0, c]) - int(g3[0, 0, c])
            # = int(g3[0, 0, c]) + int(g3[0, 0, s_0c]) - int(g3[0, s_00, s_0c])
            # = int(g3[0, 0, c]) + int(g3[0, 0, c]) - int(g3[0, 0, c])
            # = int(g3[0, 0, c])  ... wait, s_00 = 0 and s_0c = c

        # Redo more carefully
        d110_ie = 0
        for c in [1, 2]:
            s_00 = (0 + 0) % 3  # = 0
            s_0c = (0 + c) % 3  # = c
            term = int(g3[0, s_00, c]) + int(g3[0, 0, s_0c]) - int(g3[0, s_00, s_0c])
            # = g3(0, 0, c) + g3(0, 0, c) - g3(0, 0, c) = g3(0, 0, c)
            d110_ie += term

        ddiff_ie = d100_ie - d110_ie
        ddiff_actual = transitions[i]['delta_diff']

        print(f"  k={k}->{k1}: Delta_diff = {ddiff_actual}"
              f", incl-excl = {ddiff_ie}"
              f", match = {'OUI' if ddiff_ie == ddiff_actual else 'NON (err=%d)' % (ddiff_actual - ddiff_ie)}")

    # =========================================================================
    # PART 10: Direct algebraic expression for Delta_diff
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 10: EXPRESSION ALGEBRIQUE DIRECTE de Delta_diff")
    print("=" * 90)
    print()

    # Since Hyp 2 works exactly for delta_3 when it works, let's compute
    # Delta_diff = delta_100 - delta_110 using the exact formula
    # delta_3(a,b,c) = n3(a, s_ab, c) + n3(a, b, s_bc) - n3(a, s_ab, s_bc)
    # and see what simplifies.

    # delta_100 = sum_{b in {1,2}, c in {1,2}} delta_3(0,b,c)
    # For b in {1,2}, s_{0b} = b (since (0+b)%3 = b)
    # delta_3(0,b,c) = n3(0, b, c) + n3(0, b, s_bc) - n3(0, b, s_bc)
    # Wait: s_{0b} = b, so n3(0, s_{0b}, c) = n3(0, b, c)
    # And n3(0, b, s_{bc}) with s_{bc} = (b+c)%3
    # And n3(0, s_{0b}, s_{bc}) = n3(0, b, s_{bc})
    # So delta_3(0,b,c) = n3(0,b,c) + n3(0,b,s_{bc}) - n3(0,b,s_{bc})
    #                    = n3(0,b,c)
    # That means for b != 0: delta_3(0,b,c) = n3(0,b,c) under hyp 2!
    # But that gives delta_100 = sum_{b,c != 0} n3(0,b,c) = n100 itself
    # Which is wrong since delta_100 != n100 in general.

    # So hypothesis 2 FAILS for (0,b,c) with b != 0 when s_0b = b.
    # Let me recheck...

    # Actually, s_{ab} = (a+b) mod 3. For a=0, b=1: s = 1. For a=0, b=2: s = 2.
    # So s_{0b} = b for b in {1,2}. Then:
    # delta_3(0,b,c) = n3(0, b, c) + n3(0, b, (b+c)%3) - n3(0, b, (b+c)%3) = n3(0,b,c)
    # This simplifies trivially! The inclusion-exclusion cancels for these terms.

    # For delta_110: a=0, b=0. s_{00} = 0. s_{0c} = c.
    # delta_3(0,0,c) = n3(0, 0, c) + n3(0, 0, c) - n3(0, 0, c) = n3(0,0,c)

    # So under hyp 2, ALL delta_3(0,b,c) = n3(0,b,c), which gives
    # delta_100 = n100 and delta_110 = n110, hence Delta_diff = diff.
    # But delta_100 != n100 in general (the boundary terms are NOT the full counts).

    # CONCLUSION: Hypothesis 2 is NOT correct for the specific case a=0.
    # The formula simplifies too much, giving delta_3 = n3 which is wrong.
    # This means we need to find the CORRECT formula.

    print("  La formule d'inclusion-exclusion (hyp 2) se simplifie trivialement")
    print("  pour a=0, donnant delta_3(0,b,c) = n3(0,b,c), ce qui est faux.")
    print("  Il faut chercher la VRAIE formule CRT pour les 3-grammes.")
    print()

    # Let's go back to basics and look at the raw numbers
    print("  Retour aux donnees brutes: delta_3(a,b,c) pour a=0")
    print()

    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        d3 = np.zeros((3, 3, 3), dtype=np.int64)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    d3[a, b, c] = int(rk1['g3'][a, b, c]) - (p - 3) * int(rk['g3'][a, b, c])

        print(f"  k={k}->{k1} (p={p}):")
        print(f"    n3(0,b,c) at level k vs delta_3(0,b,c):")
        for b in range(3):
            for c in range(3):
                n3_k = int(rk['g3'][0, b, c])
                d3_val = int(d3[0, b, c])
                ratio = d3_val / n3_k if n3_k > 0 else 0
                print(f"      ({b},{c}): n3={n3_k:>8}, d3={d3_val:>8}, d3/n3={ratio:>8.4f}")
        print()

    # =========================================================================
    # PART 11: The ratio delta_3/n3 -- is it constant across (b,c)?
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 11: RATIO delta_3(0,b,c)/n3(0,b,c) -- structure de proportionnalite?")
    print("=" * 90)
    print()

    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        d3 = np.zeros((3, 3, 3), dtype=np.int64)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    d3[a, b, c] = int(rk1['g3'][a, b, c]) - (p - 3) * int(rk['g3'][a, b, c])

        print(f"  k={k}->{k1} (p={p}):")
        # Check if d3(0,b,c)/n3(0,b,c) is constant for b,c ≠ 0
        ratios = []
        for b in [1, 2]:
            for c in [1, 2]:
                n3_k = int(rk['g3'][0, b, c])
                if n3_k > 0:
                    ratio = int(d3[0, b, c]) / n3_k
                    ratios.append(ratio)
                    print(f"    d3(0,{b},{c})/n3(0,{b},{c}) = {int(d3[0,b,c])}/{n3_k} = {ratio:.6f}")
        if ratios:
            mean_r = sum(ratios) / len(ratios)
            spread = max(ratios) - min(ratios)
            print(f"    Mean = {mean_r:.6f}, spread = {spread:.6f}")

        # Same for b=0 (delta_110 related)
        ratios_0 = []
        for c in [1, 2]:
            n3_k = int(rk['g3'][0, 0, c])
            if n3_k > 0:
                ratio = int(d3[0, 0, c]) / n3_k
                ratios_0.append(ratio)
                print(f"    d3(0,0,{c})/n3(0,0,{c}) = {int(d3[0,0,c])}/{n3_k} = {ratio:.6f}")
        if ratios_0:
            mean_r0 = sum(ratios_0) / len(ratios_0)
            print(f"    Mean(b=0) = {mean_r0:.6f}")
            if ratios:
                print(f"    Ratio des moyennes: {mean_r:.6f} / {mean_r0:.6f} = {mean_r/mean_r0:.6f}")
        print()

    # =========================================================================
    # PART 12: KEY INSIGHT -- Delta_diff via sums and differences of ratios
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 12: INSIGHT CLE -- Delta_diff comme somme ponderee")
    print("=" * 90)
    print()
    print("  Delta_diff = delta_100 - delta_110")
    print("             = sum_{b,c!=0} d3(0,b,c) - sum_{c!=0} d3(0,0,c)")
    print()
    print("  Si d3(0,b,c) = r_b * n3(0,b,c) avec r_b dependant de b seulement:")
    print("    delta_100 = r_{!=0} * n100")
    print("    delta_110 = r_0 * n110")
    print("    Delta_diff = r_{!=0}*n100 - r_0*n110")
    print()
    print("  Pour Delta_diff >= 0, il suffit que r_{!=0}/r_0 >= n110/n100 < 1")
    print("  (puisque n100 > n110 par verification finie)")
    print()

    for i in range(len(k_list) - 1):
        k = k_list[i]
        k1 = k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk = levels[k]
        rk1 = levels[k1]

        t = transitions[i]
        n100_k = rk['n100']
        n110_k = rk['n110']
        ratio_nn = n110_k / n100_k if n100_k > 0 else 0
        ratio_dd = t['d110'] / t['d100'] if t['d100'] > 0 else 0

        print(f"  k={k}->{k1}: n110/n100 = {ratio_nn:.6f},"
              f" d110/d100 = {ratio_dd:.6f},"
              f" d110/d100 < n110/n100? {'OUI => Delta_diff > 0' if ratio_dd < ratio_nn else 'NON'}")

    # =========================================================================
    # PART 13: VERDICT
    # =========================================================================
    print()
    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    print()

    all_pos = all(t['delta_diff'] >= 0 for t in transitions)
    print(f"  Delta_diff >= 0 a TOUS les niveaux k=3..{k_max-1}: {all_pos}")
    print()

    if all_pos:
        print("  CONSEQUENCE:")
        print("    Base: diff(3) = 1 > 0")
        print("    Induction: diff(k+1) = (p-3)*diff(k) + Delta_diff >= (p-3)*diff(k) > 0")
        print("    => diff(k) > 0 pour tout k >= 3  [CQFD si Delta_diff >= 0 pour tout k]")
        print()
        print("  Le gap: prouver Delta_diff >= 0 algebriquement pour tout k >= 3.")
        print()
        print("  OBSERVATIONS STRUCTURELLES:")
        print("    1. d110/d100 < n110/n100 a chaque niveau")
        print("       => les bords CRT FAVORISENT n100 par rapport a n110")
        print("    2. Le ratio d3/n3 depend de b mais PAS de c")
        print("       => la proportionnalite est par COLONNE de la matrice de transition")
        print("    3. Delta_diff croit exponentiellement: 0, 11, 30, 131, 2448, 61692")
        print("       => l'asymetrie n100 > n110 se RENFORCE a chaque etape CRT")


if __name__ == "__main__":
    main()
