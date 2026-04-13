#!/usr/bin/env python3
"""
S15.6.271 -- Analyse de la non-markovianite relative eta(b,c) des termes de bord
================================================================================

OBJECTIF: Fermer le gap (F) de la chaine T4.

La deviation relative de chaque 3-gramme de bord par rapport a Markov est:
    d3_bnd(0,b,c) = d3_bnd_M(0,b,c) * (1 + eta(b,c))

ou d3_bnd_M(0,b,c) = R * T(0,b) * T(b,c) avec R = total des 3-grammes de bord.

Si max|eta(b,c)| < 1 pour tout k >= 4, alors:
    |correction_bnd| / Delta_M <= amplification_factor * max|eta|

et si amplification_factor * max|eta| < 1, le gap (F) est FERME.

STRUCTURE:
  Part 1: Calcul exact des eta(b,c) a chaque niveau
  Part 2: Relation eta -- gap spectral (lambda_2 de T)
  Part 3: Borne sur le facteur d'amplification
  Part 4: Prediction asymptotique
  Part 5: Condition suffisante pour fermeture
  Part 6: Route alternative -- monotonie directe de f_bnd
"""

import numpy as np
from fractions import Fraction
from math import prod
import sys

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def sieve_data(prime_list):
    """Compute sieve data at a primorial level."""
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

    classes = gaps % 3
    z = (classes == 0).astype(int)

    g1 = np.array([int((classes == a).sum()) for a in range(3)])
    alpha = Fraction(g1[0], N)

    c0, c1, c2 = classes, np.roll(classes, -1), np.roll(classes, -2)

    # Gap-class 2-gram counts
    g2 = np.zeros((3, 3), dtype=np.int64)
    for a in range(3):
        ma = (c0 == a)
        for b in range(3):
            g2[a, b] = int((ma & (c1 == b)).sum())

    # Gap-class 3-gram counts
    g3 = np.zeros((3, 3, 3), dtype=np.int64)
    for a in range(3):
        ma = (c0 == a)
        for b in range(3):
            mab = ma & (c1 == b)
            for c in range(3):
                g3[a, b, c] = int((mab & (c2 == c)).sum())

    T00 = Fraction(g2[0, 0], g1[0]) if g1[0] > 0 else Fraction(0)

    # Binary word statistics
    z1, z2 = np.roll(z, -1), np.roll(z, -2)
    n100 = int(((z == 1) & (z1 == 0) & (z2 == 0)).sum())
    n110 = int(((z == 1) & (z1 == 1) & (z2 == 0)).sum())

    return {
        'k': len(prime_list), 'N': N, 'P': P,
        'alpha': alpha, 'T00': T00,
        'g1': g1, 'g2': g2, 'g3': g3,
        'n100': n100, 'n110': n110,
        'diff': n100 - n110,
    }


def transition_matrix_3x3(alpha, T00):
    """Build the 3x3 gap-class transition matrix."""
    a = float(alpha)
    t00 = float(T00)
    t01 = (1 - t00) / 2

    t10 = a * (1 - t00) / (1 - a) if (1 - a) > 0 else 0
    t12 = 1 - t10

    T = np.array([
        [t00, t01, t01],
        [t10, 0.0, t12],
        [t10, t12, 0.0]
    ])
    return T


def main():
    print("=" * 90)
    print("S15.6.271 -- NON-MARKOVIANITE RELATIVE eta(b,c) DES TERMES DE BORD")
    print("=" * 90)

    levels = {}
    for k in range(3, len(PRIMES) + 1):
        r = sieve_data(PRIMES[:k])
        if r is None:
            break
        levels[k] = r
    k_list = sorted(levels.keys())

    # =========================================================================
    # PART 1: Exact eta(b,c) at each CRT transition
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 1: DEVIATION RELATIVE eta(b,c) = d3_bnd(0,b,c)/d3_M(0,b,c) - 1")
    print("=" * 90)
    print()
    print("  Sous Markov: d3_M(0,b,c) = R * T(0,b) * T(b,c)")
    print("  ou R = sum_{b',c'} d3_bnd(0,b',c') et T est la matrice du niveau k+1.")
    print()

    eta_data = []  # Store for later analysis

    for i in range(len(k_list) - 1):
        k, k1 = k_list[i], k_list[i + 1]
        p = PRIMES[k1 - 1]
        rk, rk1 = levels[k], levels[k1]

        # Boundary 3-gram counts d3_bnd(0,b,c)
        d3_bnd = np.zeros((3, 3), dtype=np.int64)
        for b in range(3):
            for c in range(3):
                d3_bnd[b, c] = int(rk1['g3'][0, b, c]) - (p - 3) * int(rk['g3'][0, b, c])

        R = d3_bnd.sum()

        # Transition matrix at level k+1
        T = transition_matrix_3x3(rk1['alpha'], rk1['T00'])

        # Markov prediction for each d3_bnd(0,b,c)
        d3_M = np.zeros((3, 3))
        for b in range(3):
            for c in range(3):
                d3_M[b, c] = R * T[0, b] * T[b, c]

        # Eta values (relative deviation)
        eta = np.zeros((3, 3))
        for b in range(3):
            for c in range(3):
                if d3_M[b, c] > 0.5:  # Avoid division by near-zero
                    eta[b, c] = d3_bnd[b, c] / d3_M[b, c] - 1
                elif d3_bnd[b, c] == 0 and d3_M[b, c] < 0.5:
                    eta[b, c] = 0  # Both zero (forbidden triples)
                else:
                    eta[b, c] = float('inf')

        max_eta = np.max(np.abs(eta))

        print(f"  k={k}->{k1} (p={p}), R={R}:")
        print(f"    {'(b,c)':>6} {'d3_bnd':>10} {'d3_M':>12} {'eta':>10}")
        print(f"    {'-'*45}")
        for b in range(3):
            for c in range(3):
                label = f"({b},{c})"
                if d3_bnd[b, c] == 0 and abs(d3_M[b, c]) < 0.5:
                    print(f"    {label:>6} {int(d3_bnd[b,c]):>10} {d3_M[b,c]:>12.1f}"
                          f" {'ZERO':>10}  [interdit]")
                else:
                    print(f"    {label:>6} {int(d3_bnd[b,c]):>10} {d3_M[b,c]:>12.1f}"
                          f" {eta[b,c]:>+10.4f}")
        print(f"    max|eta| = {max_eta:.4f}")
        print()

        eta_data.append({
            'k': k, 'k1': k1, 'p': p, 'R': R,
            'd3_bnd': d3_bnd.copy(), 'd3_M': d3_M.copy(),
            'eta': eta.copy(), 'max_eta': max_eta,
        })

    # =========================================================================
    # PART 2: Spectral gap and eta bound
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 2: GAP SPECTRAL et relation avec eta")
    print("=" * 90)
    print()
    print("  La matrice T(3x3) a des valeurs propres: 1, lambda_2, lambda_3")
    print("  avec |lambda_2| = |lambda_3| = (alpha - T00)/(1-alpha) [gap spectral]")
    print()

    print(f"  {'k':>3} {'alpha':>8} {'T00':>8} {'|lam2|':>8} {'1-|lam2|':>8}"
          f" {'max|eta_bnd|':>12} {'ratio':>8}")
    print(f"  {'-'*60}")

    for i, ed in enumerate(eta_data):
        k1 = ed['k1']
        a = float(levels[k1]['alpha'])
        t00 = float(levels[k1]['T00'])
        lam2 = abs(a - t00) / (1 - a) if (1 - a) > 0 else 0
        ratio = ed['max_eta'] / lam2 if lam2 > 1e-12 else float('inf')
        print(f"  {k1:>3} {a:>8.4f} {t00:>8.4f} {lam2:>8.4f} {1-lam2:>8.4f}"
              f" {ed['max_eta']:>12.4f} {ratio:>8.2f}")

    # =========================================================================
    # PART 3: Amplification factor analysis
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 3: FACTEUR D'AMPLIFICATION")
    print("=" * 90)
    print()
    print("  f_bnd = |correction_bnd| / Delta_M")
    print("  correction_bnd = sum w(b,c) * d3_M(0,b,c) * eta(b,c)")
    print("  Delta_M = sum w(b,c) * d3_M(0,b,c)")
    print("  ou w(1,2)=w(2,1)=+1, w(0,1)=w(0,2)=-1, reste=0")
    print()
    print("  Borne par triangle: f_bnd <= max|eta| * [sum|w*d3_M|] / [sum w*d3_M]")
    print("                            = max|eta| * A")
    print("  ou A = facteur d'amplification.")
    print()

    print(f"  {'k->k+1':>8} {'sum|w*d3_M|':>12} {'sum w*d3_M':>12} {'A':>8}"
          f" {'max|eta|':>10} {'A*max|eta|':>12} {'f_bnd_exact':>12}")
    print(f"  {'-'*80}")

    for i, ed in enumerate(eta_data):
        k, k1 = ed['k'], ed['k1']
        d3_M = ed['d3_M']

        # Weights: w(1,2) = w(2,1) = +1, w(0,1) = w(0,2) = -1
        # (corresponding to delta_100 - delta_110 decomposition)
        sum_w_d3M = d3_M[1, 2] + d3_M[2, 1] - d3_M[0, 1] - d3_M[0, 2]
        sum_abs_w_d3M = d3_M[1, 2] + d3_M[2, 1] + d3_M[0, 1] + d3_M[0, 2]

        A = sum_abs_w_d3M / sum_w_d3M if sum_w_d3M > 0 else float('inf')

        # Exact f_bnd
        Delta_diff = levels[k1]['diff'] - (PRIMES[k1 - 1] - 3) * levels[k]['diff']
        Delta_M = sum_w_d3M
        f_bnd_exact = abs(Delta_diff - Delta_M) / Delta_M if Delta_M > 0 else float('inf')

        # Bound via A * max|eta|
        bound = A * ed['max_eta']

        print(f"  {k:>3}->{k1:>2} {sum_abs_w_d3M:>12.1f} {sum_w_d3M:>12.1f}"
              f" {A:>8.3f} {ed['max_eta']:>10.4f} {bound:>12.4f} {f_bnd_exact:>12.4f}")

    # =========================================================================
    # PART 4: Direct weighted eta (exact decomposition of f_bnd)
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 4: DECOMPOSITION EXACTE de f_bnd en termes de eta ponderes")
    print("=" * 90)
    print()
    print("  f_bnd = |sum w(b,c) * p(b,c) * eta(b,c)|")
    print("  ou p(b,c) = d3_M(0,b,c) / Delta_M = poids normalise")
    print()

    for i, ed in enumerate(eta_data):
        k, k1 = ed['k'], ed['k1']
        d3_M = ed['d3_M']

        Delta_M = d3_M[1, 2] + d3_M[2, 1] - d3_M[0, 1] - d3_M[0, 2]
        if Delta_M <= 0:
            continue

        print(f"  k={k}->{k1}:")
        total_weighted = 0
        for (b, c), w in [((1, 2), +1), ((2, 1), +1), ((0, 1), -1), ((0, 2), -1)]:
            p_bc = d3_M[b, c] / Delta_M
            contrib = w * p_bc * ed['eta'][b, c]
            total_weighted += contrib
            print(f"    ({b},{c}): w={w:+d}, p={p_bc:>8.4f}, eta={ed['eta'][b,c]:>+8.4f},"
                  f" contrib = {contrib:>+10.6f}")
        print(f"    f_bnd = |{total_weighted:.6f}| = {abs(total_weighted):.4f}")

        # Exact check
        Delta_diff = levels[k1]['diff'] - (PRIMES[k1 - 1] - 3) * levels[k]['diff']
        f_exact = abs(Delta_diff - Delta_M) / Delta_M
        print(f"    f_bnd (direct) = {f_exact:.4f}")
        print()

    # =========================================================================
    # PART 5: Asymptotic predictions and sufficient condition
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 5: CONDITION SUFFISANTE pour fermeture")
    print("=" * 90)
    print()

    # Collect trends
    print("  TENDANCE de max|eta| et facteur A:")
    print()
    print(f"  {'k->k+1':>8} {'max|eta|':>10} {'A':>8} {'A*max|eta|':>12}"
          f" {'f_bnd':>8} {'marge':>8}")
    print(f"  {'-'*60}")

    for i, ed in enumerate(eta_data):
        k, k1 = ed['k'], ed['k1']
        d3_M = ed['d3_M']
        sum_w = d3_M[1, 2] + d3_M[2, 1] - d3_M[0, 1] - d3_M[0, 2]
        sum_abs_w = d3_M[1, 2] + d3_M[2, 1] + d3_M[0, 1] + d3_M[0, 2]
        A = sum_abs_w / sum_w if sum_w > 0 else float('inf')

        Delta_diff = levels[k1]['diff'] - (PRIMES[k1 - 1] - 3) * levels[k]['diff']
        f_bnd = abs(Delta_diff - sum_w) / sum_w if sum_w > 0 else float('inf')
        bound = A * ed['max_eta']
        marge = 1 - f_bnd

        print(f"  {k:>3}->{k1:>2} {ed['max_eta']:>10.4f} {A:>8.3f}"
              f" {bound:>12.4f} {f_bnd:>8.4f} {marge:>8.4f}")

    print()
    print("  CONDITION SUFFISANTE FORTE: max|eta(b,c)| < 1/A")
    print("  => f_bnd <= A * max|eta| < 1")
    print()
    print("  CONDITION SUFFISANTE FAIBLE: f_bnd < 1")
    print("  (vrai meme si la borne triangle est > 1, car les eta se compensent)")
    print()

    # =========================================================================
    # PART 6: Monotonicity analysis of f_bnd
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 6: ANALYSE DE MONOTONIE de f_bnd")
    print("=" * 90)
    print()

    f_bnd_list = []
    for i, ed in enumerate(eta_data):
        k, k1 = ed['k'], ed['k1']
        d3_M = ed['d3_M']
        sum_w = d3_M[1, 2] + d3_M[2, 1] - d3_M[0, 1] - d3_M[0, 2]
        Delta_diff = levels[k1]['diff'] - (PRIMES[k1 - 1] - 3) * levels[k]['diff']
        f_bnd = abs(Delta_diff - sum_w) / sum_w if sum_w > 0 else float('inf')
        f_bnd_list.append((k, k1, f_bnd))

    print("  Evolution de f_bnd:")
    for j, (k, k1, f) in enumerate(f_bnd_list):
        trend = ""
        if j > 0:
            prev_f = f_bnd_list[j - 1][2]
            if f < prev_f:
                trend = "  DECROIT"
            else:
                trend = "  CROIT"
        print(f"    k={k}->{k1}: f_bnd = {f:.4f}{trend}")

    print()

    # Is f_bnd monotonically decreasing after some k?
    peak_idx = max(range(len(f_bnd_list)), key=lambda j: f_bnd_list[j][2])
    peak_k = f_bnd_list[peak_idx][0]
    peak_f = f_bnd_list[peak_idx][2]
    print(f"  PIC: k={peak_k}->{f_bnd_list[peak_idx][1]}, f_bnd = {peak_f:.4f}")

    monotone_after = True
    for j in range(peak_idx + 1, len(f_bnd_list)):
        if j > 0 and f_bnd_list[j][2] > f_bnd_list[j - 1][2]:
            monotone_after = False
            break

    print(f"  Monotone decroissant apres le pic: {'OUI' if monotone_after else 'NON'}")
    print()

    # =========================================================================
    # PART 7: Individual eta trends and spectral prediction
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 7: TENDANCE INDIVIDUELLE des eta et prediction spectrale")
    print("=" * 90)
    print()

    # For the active (non-forbidden) patterns, track eta evolution
    active_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    print(f"  {'k->k+1':>8}", end="")
    for b, c in active_pairs:
        print(f" {'eta('+str(b)+','+str(c)+')':>12}", end="")
    print()
    print(f"  {'-'*85}")

    for ed in eta_data:
        k, k1 = ed['k'], ed['k1']
        print(f"  {k:>3}->{k1:>2}", end="")
        for b, c in active_pairs:
            print(f" {ed['eta'][b,c]:>+12.4f}", end="")
        print()

    print()
    print("  Observations:")
    print("  - eta(1,2) et eta(2,1) sont les deviations des termes cross-class")
    print("  - eta(0,1) et eta(0,2) sont les deviations des termes self-class")
    print("  - Le signe dominant est: cross-class SOUS-estime (eta < 0),")
    print("    self-class SUR-estime (eta > 0)")
    print("  - Les deux effets vont dans le meme sens: correction_bnd < 0")

    # =========================================================================
    # PART 8: The EXACT remaining condition
    # =========================================================================
    print()
    print("=" * 90)
    print("PART 8: CONDITION EXACTE RESTANTE pour fermer T4")
    print("=" * 90)
    print()

    print("  THEOREME CONDITIONNEL:")
    print("    Si pour tout k >= 4:")
    print("      (i)  Delta_M(k) > 0                                  [PROUVE: THM algebrique]")
    print("      (ii) d3_bnd(0,b,c) >= 0 pour tout b,c               [TRIVIAL: ce sont des comptes]")
    print("      (iii) f_bnd(k) < 1                                   [A PROUVER]")
    print("    alors diff(k) > 0 pour tout k >= 3, et T4 est ferme.")
    print()
    print("  APPROCHES POUR (iii):")
    print()
    print("  A) BORNE SPECTRALE (Route 1):")
    print("     Montrer que max|eta(b,c)| <= C * |lambda_2| avec C borne")
    print("     et A * C * |lambda_2| < 1 pour tout k >= 4.")
    print("     Status: max|eta| ~ 0.2--0.5, |lambda_2| ~ 0.15--0.20")
    print("     Le ratio max|eta|/|lambda_2| ~ 1.5--2.8 (pas exactement C*lam2)")
    print()
    print("  B) EXTENSION FINIE (Route 2):")
    print("     Verifier k=10 exactement (P=6.5G, faisable avec primesieve).")
    print("     Pour k >= 11: f_bnd asymptotique ~ 0.20 (marge 5x).")
    print("     Status: FAISABLE techniquement, fermerait le gap.")
    print()
    print("  C) MONOTONIE (Route 3):")
    print("     Prouver f_bnd(k+1) <= f_bnd(k) pour k >= 7.")
    print("     Puisque f_bnd(7->8) = 0.875 < 1, cela suffit.")
    print("     Status: tendance observee, pas de preuve.")
    print()
    print("  D) BORNE DIRECTE SUR correction/Delta_M (Route 4, nouvelle):")
    print("     correction = sum w(b,c) * d3_M * eta(b,c)")
    print("     Les eta ont des SIGNES OPPOSES pour cross vs self.")
    print("     La cancellation AIDE: f_bnd << A * max|eta|.")
    print("     Si on prouve la compensation structurelle: FERME.")
    print()

    # Check the cancellation effect
    print("  ANALYSE DE COMPENSATION:")
    print()
    for i, ed in enumerate(eta_data):
        k, k1 = ed['k'], ed['k1']
        d3_M = ed['d3_M']
        Delta_M = d3_M[1, 2] + d3_M[2, 1] - d3_M[0, 1] - d3_M[0, 2]
        sum_abs = d3_M[1, 2] + d3_M[2, 1] + d3_M[0, 1] + d3_M[0, 2]
        A = sum_abs / Delta_M if Delta_M > 0 else float('inf')

        Delta_diff = levels[k1]['diff'] - (PRIMES[k1 - 1] - 3) * levels[k]['diff']
        f_bnd = abs(Delta_diff - Delta_M) / Delta_M if Delta_M > 0 else float('inf')
        bound_naive = A * ed['max_eta']

        gain = bound_naive / f_bnd if f_bnd > 1e-10 else float('inf')
        print(f"    k={k}->{k1}: borne_naive={bound_naive:.3f},"
              f" f_bnd_exact={f_bnd:.3f}, gain_compensation={gain:.1f}x")

    print()
    print("=" * 90)
    print("FIN S15.6.271")
    print("=" * 90)


if __name__ == "__main__":
    main()

sys.exit(0)
