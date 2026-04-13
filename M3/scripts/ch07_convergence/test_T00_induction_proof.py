#!/usr/bin/env python3
"""
S15.6.257 -- Preuve inductive: n_{12} > n_{10} a tout niveau
=============================================================

STRATEGIE DE PREUVE:
La formule CRT exacte (S15.6.256) donne:
  n'_{ab} = (p-3)*n_{ab} + A_{ab} + B_{ab}

Donc:
  n'_{12} - n'_{10} = (p-3)*(n_{12} - n_{10}) + Delta

  Delta = (A_{12} - A_{10}) + (B_{12} - B_{10})

Si Delta >= 0 et n_{12} > n_{10} au niveau k,
alors n'_{12} > n'_{10} au niveau k+1 (car p >= 5, p-3 >= 2).

Ce script:
1. Calcule les 3-grammes exacts pour k=2..8
2. Calcule Delta a chaque transition k -> k+1
3. Verifie le signe de Delta
4. Tente de prouver Delta >= 0 structurellement
"""

import numpy as np
import time
from fractions import Fraction

# ============================================================
# PARTIE 1: Calcul exact des 3-grammes
# ============================================================

print("="*70)
print("S15.6.257 -- PREUVE INDUCTIVE VIA DELTA >= 0")
print("="*70)

def compute_sieve_stats(prime_list, verbose=True):
    """Calcule sieve + gaps + classes + 2-gram + 3-gram exactement."""
    P = 1
    for p in prime_list:
        P *= p

    if P > 300_000_000:
        return None

    t0 = time.time()

    # Sieve
    sieve = np.ones(P + 1, dtype=bool)
    sieve[0] = False
    for p in prime_list:
        sieve[::p] = False
    survivors = np.where(sieve)[0]
    del sieve

    n = len(survivors)

    # Gaps cycliques
    gaps = np.empty(n, dtype=np.int64)
    gaps[:-1] = survivors[1:] - survivors[:-1]
    gaps[-1] = P + survivors[0] - survivors[-1]

    # Classes mod 3
    classes = gaps % 3

    # Comptages
    n0 = int(np.count_nonzero(classes == 0))
    n1 = int(np.count_nonzero(classes == 1))
    n2 = int(np.count_nonzero(classes == 2))

    # 2-gram (transitions)
    cls_from = classes
    cls_to = np.roll(classes, -1)
    trans = np.zeros((3, 3), dtype=np.int64)
    for a in range(3):
        ma = (cls_from == a)
        for b in range(3):
            trans[a, b] = int((ma & (cls_to == b)).sum())

    # 3-gram
    cls_to2 = np.roll(classes, -2)
    gram3 = np.zeros((3, 3, 3), dtype=np.int64)
    for a in range(3):
        ma = (cls_from == a)
        for b in range(3):
            mab = ma & (cls_to == b)
            for c in range(3):
                gram3[a, b, c] = int((mab & (cls_to2 == c)).sum())

    # T-matrice
    T = np.zeros((3, 3), dtype=np.float64)
    for a in range(3):
        row = trans[a].sum()
        if row > 0:
            T[a] = trans[a] / row

    alpha = n0 / n
    T00 = T[0, 0]
    T12 = T[1, 2]
    eps = 0.5 - alpha
    F = (1 - alpha) * (2 * T12 - 1)
    Q = F / eps if eps > 0 else 0

    t1 = time.time()
    if verbose:
        print(f"  k={len(prime_list)-1}, P={P:,}, phi={n:,}, time={t1-t0:.1f}s")

    return {
        'primes': list(prime_list), 'P': P, 'n': n,
        'n0': n0, 'n1': n1, 'n2': n2,
        'trans': trans, 'T': T, 'gram3': gram3,
        'alpha': alpha, 'T00': T00, 'T12': T12,
        'eps': eps, 'F': F, 'Q': Q
    }


def compute_delta(gram3, verbose=True):
    """Calcule Delta = (A12-A10) + (B12-B10) depuis les 3-grammes.

    A_{ab} = sum_{c+d=b mod 3} n3(a,c,d)
    B_{ab} = sum_{c+d=a mod 3} n3(c,d,b)
    """
    A12, A10 = 0, 0
    B12, B10 = 0, 0

    for c in range(3):
        for d in range(3):
            cd_mod3 = (c + d) % 3
            # A_{1,2}: a=1, b=2 -> c+d = 2 mod 3
            if cd_mod3 == 2:
                A12 += int(gram3[1, c, d])
            # A_{1,0}: a=1, b=0 -> c+d = 0 mod 3
            if cd_mod3 == 0:
                A10 += int(gram3[1, c, d])
            # B_{1,2}: b=2, a=1 -> c+d = 1 mod 3
            if cd_mod3 == 1:
                B12 += int(gram3[c, d, 2])
            # B_{1,0}: b=0, a=1 -> c+d = 1 mod 3
            if cd_mod3 == 1:
                B10 += int(gram3[c, d, 0])

    delta = (A12 - A10) + (B12 - B10)

    if verbose:
        print(f"    A12={A12}, A10={A10}, A12-A10={A12-A10}")
        print(f"    B12={B12}, B10={B10}, B12-B10={B12-B10}")
        print(f"    Delta = {delta}")

    return delta, A12, A10, B12, B10


# ============================================================
# PARTIE 2: Calculer pour chaque niveau et verifier
# ============================================================

print("\n" + "="*70)
print("PARTIE 2: DELTA A CHAQUE TRANSITION k -> k+1")
print("="*70)

all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
results = {}

# Calculer les stats pour chaque niveau
for k in range(2, len(all_primes) + 1):
    plist = all_primes[:k]
    r = compute_sieve_stats(plist, verbose=True)
    if r is None:
        break
    results[k] = r

# Calculer Delta pour chaque transition
print("\n" + "-"*70)
print(f"{'k->k+1':>8} {'p':>4} {'n12-n10':>12} {'Delta':>10} {'(p-3)(n12-n10)':>16} {'n12_new-n10_new':>16} {'pred':>10} {'OK':>4}")
print("-"*70)

all_deltas = []

for k in range(2, max(results.keys())):
    if k not in results or k+1 not in results:
        continue

    rk = results[k]
    rk1 = results[k + 1]
    p_new = all_primes[k]  # premier ajoute au niveau k+1

    n12 = int(rk['trans'][1, 2])
    n10 = int(rk['trans'][1, 0])
    diff = n12 - n10

    # Delta depuis les 3-grammes
    print(f"\n  Transition k={k} -> k+1={k+1} (ajout p={p_new}):")
    delta, A12, A10, B12, B10 = compute_delta(rk['gram3'])

    # Prediction: n'_{12} - n'_{10}
    pred = (p_new - 3) * diff + delta

    # Valeur reelle
    n12_new = int(rk1['trans'][1, 2])
    n10_new = int(rk1['trans'][1, 0])
    actual = n12_new - n10_new

    ok = pred == actual

    print(f"  {'k=%d->%d'%(k,k+1):>8} {p_new:>4} {diff:>12} {delta:>10} {(p_new-3)*diff:>16} {actual:>16} {pred:>10} {'OK' if ok else 'FAIL':>4}")

    all_deltas.append({
        'k': k, 'p': p_new, 'diff': diff, 'delta': delta,
        'main_term': (p_new - 3) * diff, 'actual': actual,
        'A12': A12, 'A10': A10, 'B12': B12, 'B10': B10,
        'ok': ok
    })

# ============================================================
# PARTIE 3: Analyse du signe de Delta
# ============================================================

print("\n" + "="*70)
print("PARTIE 3: SIGNE DE DELTA")
print("="*70)

print(f"\n{'k':>3} {'p':>4} {'Delta':>10} {'Delta/n':>12} {'Signe':>8}")
print("-"*50)

for d in all_deltas:
    k = d['k']
    n = results[k]['n']
    sign = "+" if d['delta'] > 0 else ("0" if d['delta'] == 0 else "-")
    print(f"{k:>3} {d['p']:>4} {d['delta']:>10} {d['delta']/n:>12.6f} {sign:>8}")

all_positive = all(d['delta'] > 0 for d in all_deltas)
print(f"\nDelta > 0 a toutes les transitions? {'OUI' if all_positive else 'NON'}")

# ============================================================
# PARTIE 4: Decomposition structurelle de Delta
# ============================================================

print("\n" + "="*70)
print("PARTIE 4: DECOMPOSITION STRUCTURELLE")
print("="*70)

print("""
Notation pour les 3-grammes independants (avec T1 et symetrie 1<->2):
  a = n3(0,0,0)
  b = n3(0,0,1) = n3(0,0,2) = n3(1,0,0) = n3(2,0,0)   [PROUVE: stationnarite]
  c = n3(0,1,0) = n3(0,2,0)
  d = n3(0,1,2) = n3(0,2,1) = n3(1,2,0) = n3(2,1,0)   [PROUVE: balance detaillee]
  f = n3(1,0,1) = n3(2,0,2)
  g = n3(1,0,2) = n3(2,0,1)
  i = n3(1,2,1) = n3(2,1,2)

Identites de marginalisation:
  a + 2b = n_{00}           (triples commencant par (0,0,...))
  a + 2b = n_{00}           (triples finissant par (...,0,0)) [meme!]
  c + d = n_{01} = n_{10}   (balance detaillee)
  d + i = n_{12} = n_{21}
  b + f + g = n_{01} = n_{10}

DELTA en termes de a,b,c,d,f,g,i:
  A12 = n3(1,0,2) + n3(1,2,0) = g + d
  A10 = n3(1,0,0) + n3(1,2,1) = b + i
  B12 = n3(0,1,2) + n3(1,0,2) = d + g
  B10 = n3(0,1,0) + n3(1,0,0) = c + b

  Delta = (A12-A10) + (B12-B10)
        = (g + d - b - i) + (d + g - c - b)
        = 2g + 2d - 2b - i - c

Substitutions: c = n01 - d, i = n12 - d, g = n01 - b - f:
  Delta = 2(n01-b-f) + 2d - 2b - (n12-d) - (n01-d)
        = 2*n01 - 2b - 2f + 2d - 2b - n12 + d - n01 + d
        = n01 - 4b - 2f + 4d - n12
""")

# Verification numerique
for k in range(2, max(results.keys())):
    if k not in results:
        continue

    rk = results[k]
    g3 = rk['gram3']

    # Extraire les 3-grammes independants
    a = int(g3[0,0,0])
    b_001 = int(g3[0,0,1])
    b_100 = int(g3[1,0,0])
    c_010 = int(g3[0,1,0])
    c_020 = int(g3[0,2,0])
    d_012 = int(g3[0,1,2])
    d_120 = int(g3[1,2,0])
    d_021 = int(g3[0,2,1])
    d_210 = int(g3[2,1,0])
    f_101 = int(g3[1,0,1])
    f_202 = int(g3[2,0,2])
    g_102 = int(g3[1,0,2])
    g_201 = int(g3[2,0,1])
    i_121 = int(g3[1,2,1])
    i_212 = int(g3[2,1,2])

    n00 = int(rk['trans'][0,0])
    n01 = int(rk['trans'][0,1])
    n10 = int(rk['trans'][1,0])
    n12 = int(rk['trans'][1,2])

    # Verifier les identites
    check_b = (b_001 == b_100)
    check_c = (c_010 == c_020)
    check_d1 = (d_012 == d_120)
    check_d2 = (d_012 == d_021)
    check_d3 = (d_012 == d_210)
    check_f = (f_101 == f_202)
    check_g = (g_102 == g_201)
    check_i = (i_121 == i_212)
    check_bd = (b_001 == int(g3[0,0,2]))  # b = n3(0,0,2) aussi
    check_bal = (n01 == n10)
    check_marg1 = (a + 2*b_001 == n00)
    check_marg2 = (c_010 + d_012 == n01)
    check_marg3 = (d_012 + i_121 == n12)
    check_marg4 = (b_001 + f_101 + g_102 == n01)

    b, c, d, f, g, i_val = b_001, c_010, d_012, f_101, g_102, i_121
    delta_formula = 2*g + 2*d - 2*b - i_val - c
    delta_alt = n01 - 4*b - 2*f + 4*d - n12

    all_ok = all([check_b, check_c, check_d1, check_d2, check_d3,
                  check_f, check_g, check_i, check_bd, check_bal,
                  check_marg1, check_marg2, check_marg3, check_marg4])

    print(f"\n  k={k}: identites {'TOUTES OK' if all_ok else 'ECHEC'}")
    print(f"    a={a}, b={b}, c={c}, d={d}, f={f}, g={g}, i={i_val}")
    print(f"    n00={n00}, n01={n01}, n12={n12}")
    print(f"    Delta(2g+2d-2b-i-c) = {delta_formula}")
    print(f"    Delta(n01-4b-2f+4d-n12) = {delta_alt}")

    if not all_ok:
        print(f"    DETAIL: b={check_b}, c={check_c}, d1={check_d1}, d2={check_d2}")
        print(f"    d3={check_d3}, f={check_f}, g={check_g}, i={check_i}")
        print(f"    bd={check_bd}, bal={check_bal}")
        print(f"    m1={check_marg1}, m2={check_marg2}, m3={check_marg3}, m4={check_marg4}")

# ============================================================
# PARTIE 5: Ratios structurels
# ============================================================

print("\n" + "="*70)
print("PARTIE 5: RATIOS STRUCTURELS")
print("="*70)

print("""
Delta = n01 - 4b - 2f + 4d - n12
      = (c + d) - 4b - 2f + 4d - (d + i)  [substitution]
      = c + 4d - 4b - 2f - i

En ratios normalises par N:
  Delta/N = n01/N - 4b/N - 2f/N + 4d/N - n12/N

Les ratios b/n01, d/n01, f/n01 sont les "poids 3-gram normalises".
""")

print(f"{'k':>3} {'b/n01':>10} {'c/n01':>10} {'d/n01':>10} {'f/n01':>10} {'g/n01':>10} {'i/n12':>10} {'Delta/n01':>12}")
print("-"*90)

for k in sorted(results.keys()):
    rk = results[k]
    g3 = rk['gram3']
    n01 = int(rk['trans'][0,1])
    n12 = int(rk['trans'][1,2])

    if n01 == 0:
        continue

    b = int(g3[0,0,1])
    c = int(g3[0,1,0])
    d = int(g3[0,1,2])
    f = int(g3[1,0,1])
    g_val = int(g3[1,0,2])
    i_val = int(g3[1,2,1])

    delta = 2*g_val + 2*d - 2*b - i_val - c

    print(f"{k:>3} {b/n01:>10.6f} {c/n01:>10.6f} {d/n01:>10.6f} {f/n01:>10.6f} {g_val/n01:>10.6f} {i_val/n12:>10.6f} {delta/n01:>12.6f}")

# ============================================================
# PARTIE 6: Reformulation T12 - T10 en termes de 3-gram
# ============================================================

print("\n" + "="*70)
print("PARTIE 6: ANALYSE DU MECANISME")
print("="*70)

print("""
MECANISME STRUCTUREL:

Delta = 2g + 2d - 2b - i - c

Les termes POSITIFS (g et d) proviennent des triples:
  g = n3(1,0,2): apres (1,0), le 3eme gap est de classe 2
  d = n3(0,1,2) = n3(1,2,0): alternance complete (0->1->2) ou (1->2->0)

Les termes NEGATIFS (b, i, c) proviennent des triples:
  b = n3(0,0,1) = n3(1,0,0): deux class-0 consecutifs
  i = n3(1,2,1): retour immediat (1->2->1)
  c = n3(0,1,0): rebond (0->1->0)

INTUITION: Delta > 0 signifie que les patterns "ouvrerts" (qui traversent
les classes) dominent les patterns "fermes" (qui rebondissent). C'est
force par T1: les rebonds (1,1) et (2,2) sont INTERDITS, ce qui favorise
les traversees.
""")

# Verification: contribution relative de chaque terme
print(f"{'k':>3} {'2g':>10} {'2d':>10} {'-(2b+i+c)':>12} {'Delta':>10} {'%pos':>8}")
print("-"*60)

for k in sorted(results.keys()):
    rk = results[k]
    g3 = rk['gram3']
    n01 = int(rk['trans'][0,1])
    n12 = int(rk['trans'][1,2])

    if n01 == 0:
        continue

    b = int(g3[0,0,1])
    c = int(g3[0,1,0])
    d = int(g3[0,1,2])
    g_val = int(g3[1,0,2])
    i_val = int(g3[1,2,1])

    pos = 2*g_val + 2*d
    neg = 2*b + i_val + c
    delta = pos - neg

    print(f"{k:>3} {2*g_val:>10} {2*d:>10} {-neg:>12} {delta:>10} {pos/(pos+neg)*100:>7.1f}%")

# ============================================================
# PARTIE 7: Borne inferieure algebrique
# ============================================================

print("\n" + "="*70)
print("PARTIE 7: TENTATIVE DE BORNE INFERIEURE")
print("="*70)

print("""
OBJECTIF: Montrer Delta >= 0 pour tout k >= 2.

Delta = n01 - 4b - 2f + 4d - n12

Depuis les contraintes:
  b + f + g = n01     =>  f = n01 - b - g
  c + d = n01         =>  c = n01 - d
  d + i = n12         =>  i = n12 - d

Substituons f:
  Delta = n01 - 4b - 2(n01 - b - g) + 4d - n12
        = n01 - 4b - 2n01 + 2b + 2g + 4d - n12
        = -n01 - 2b + 2g + 4d - n12

Avec n01 = n10 (balance detaillee) et n01 = c + d:
  Delta = -(c+d) - 2b + 2g + 4d - n12
        = -c + 3d - 2b + 2g - n12
        = -c + 3d - 2b + 2g - (d + i)
        = -c + 2d - 2b + 2g - i

Hmm, ca ne simplifie pas facilement. Essayons autrement.

STRATEGIE ALTERNATIVE: Montrer que les ratios convergent vers des limites
qui rendent Delta > 0, et que le signe est stable.

En regime asymptotique (Mertens, grands k):
  - alpha -> 1/2, T00 -> 1/2, T12 -> 1/2
  - La chaine de Markov approche l'equilibre
  - Dans l'approximation Markov: n3(a,b,c) ~ N * pi_a * T_{ab} * T_{bc}

Calculons Delta_Markov:
""")

# Calcul de Delta en approximation Markov
for k in sorted(results.keys()):
    rk = results[k]
    n = rk['n']
    alpha = rk['alpha']
    T00 = rk['T00']
    T = rk['T']

    pi_arr = np.array([alpha, (1-alpha)/2, (1-alpha)/2])

    # 3-grammes Markov
    gm = np.zeros((3, 3, 3))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                gm[a,b,c] = n * pi_arr[a] * T[a,b] * T[b,c]

    # Delta Markov
    b_m = gm[0,0,1]
    c_m = gm[0,1,0]
    d_m = gm[0,1,2]
    g_m = gm[1,0,2]
    i_m = gm[1,2,1]

    delta_m = 2*g_m + 2*d_m - 2*b_m - i_m - c_m

    # Delta exact
    g3 = rk['gram3']
    b_e = int(g3[0,0,1])
    c_e = int(g3[0,1,0])
    d_e = int(g3[0,1,2])
    g_e = int(g3[1,0,2])
    i_e = int(g3[1,2,1])
    delta_e = 2*g_e + 2*d_e - 2*b_e - i_e - c_e

    print(f"  k={k}: Delta_exact={delta_e:>10}, Delta_Markov={delta_m:>12.1f}, ratio={delta_e/delta_m:.4f}" if delta_m != 0 else f"  k={k}: Delta_exact={delta_e}, Delta_Markov={delta_m:.1f}")

# ============================================================
# PARTIE 8: Formule analytique de Delta_Markov
# ============================================================

print("\n" + "="*70)
print("PARTIE 8: DELTA MARKOV ANALYTIQUE")
print("="*70)

print("""
En approximation Markov (3-gram = produit de T-matrice):

  n3_M(a,b,c) = N * pi_a * T_{ab} * T_{bc}

Les entrees necessaires:
  b = N*alpha*T00*T01          = N*alpha*T00*(1-T00)/2
  c = N*(1-alpha)/2*T10*T10    -- FAUX, T10 n'existe...

Wait, recalculons. Avec T1:
  T_{10} = 1 - T_{12} (depuis T11 = 0)

  b = n3_M(0,0,1) = N*pi_0*T_{00}*T_{01} = N*alpha*T00*(1-T00)/2
  c = n3_M(0,1,0) = N*pi_0*T_{01}*T_{10} = N*alpha*(1-T00)/2*(1-T12)
  d = n3_M(0,1,2) = N*pi_0*T_{01}*T_{12} = N*alpha*(1-T00)/2*T12
  g = n3_M(1,0,2) = N*pi_1*T_{10}*T_{02} = N*(1-alpha)/2*(1-T12)*(1-T00)/2
  i = n3_M(1,2,1) = N*pi_1*T_{12}*T_{21} = N*(1-alpha)/2*T12*T12

  (car T_{02} = T_{01} = (1-T00)/2, T_{21} = T_{12} par symetrie)

Substituons T10 = 1-T12 et T01 = (1-T00)/2:

  Delta_M = 2g + 2d - 2b - i - c

Calculons chaque terme (facteur N omis):
  2g = 2*(1-alpha)/2*(1-T12)*(1-T00)/2 = (1-alpha)(1-T12)(1-T00)/2
  2d = 2*alpha*(1-T00)/2*T12 = alpha*(1-T00)*T12
  2b = 2*alpha*T00*(1-T00)/2 = alpha*T00*(1-T00)
  i  = (1-alpha)/2*T12^2
  c  = alpha*(1-T00)/2*(1-T12)
""")

from sympy import symbols, simplify, expand, factor, Rational
import sys

alpha_s, T00_s, T12_s = symbols('alpha T00 T12', positive=True)

# En termes symboliques
term_2g = (1-alpha_s)*(1-T12_s)*(1-T00_s)/2
term_2d = alpha_s*(1-T00_s)*T12_s
term_2b = alpha_s*T00_s*(1-T00_s)
term_i = (1-alpha_s)*T12_s**2/2
term_c = alpha_s*(1-T00_s)*(1-T12_s)/2

Delta_M_sym = term_2g + term_2d - term_2b - term_i - term_c
Delta_M_expanded = expand(Delta_M_sym)
Delta_M_factored = factor(Delta_M_sym)

print(f"  Delta_M (expanded) = {Delta_M_expanded}")
print(f"  Delta_M (factored) = {Delta_M_factored}")

# Utiliser la relation de flow balance: T12 = (1-2alpha+alpha*T00)/(1-alpha)
T12_fb = (1 - 2*alpha_s + alpha_s*T00_s) / (1 - alpha_s)
Delta_M_fb = Delta_M_sym.subs(T12_s, T12_fb)
Delta_M_fb_simplified = simplify(Delta_M_fb)
Delta_M_fb_factored = factor(Delta_M_fb)

print(f"\n  Avec T12 = (1-2alpha+alpha*T00)/(1-alpha):")
print(f"  Delta_M (simplified) = {Delta_M_fb_simplified}")
print(f"  Delta_M (factored) = {Delta_M_fb_factored}")

# Verification numerique de la formule
print("\n  Verification numerique:")
for k in sorted(results.keys()):
    rk = results[k]
    n = rk['n']
    alpha_v = rk['alpha']
    T00_v = rk['T00']
    T12_v = rk['T12']

    val = float(Delta_M_fb_simplified.subs({alpha_s: alpha_v, T00_s: T00_v}))

    g3 = rk['gram3']
    b_e = int(g3[0,0,1])
    c_e = int(g3[0,1,0])
    d_e = int(g3[0,1,2])
    g_e = int(g3[1,0,2])
    i_e = int(g3[1,2,1])
    delta_e = 2*g_e + 2*d_e - 2*b_e - i_e - c_e

    print(f"    k={k}: Delta_exact={delta_e}, N*Delta_M_formula={n*val:.1f}, ratio={delta_e/(n*val):.4f}" if n*val != 0 else f"    k={k}: Delta_exact={delta_e}, N*Delta_M_formula={n*val:.1f}")

# ============================================================
# PARTIE 9: Signe de Delta_M en fonction de alpha et T00
# ============================================================

print("\n" + "="*70)
print("PARTIE 9: SIGNE DE Delta_M(alpha, T00)")
print("="*70)

print(f"\n  Delta_M (en alpha, T00) = {Delta_M_fb_factored}")

# Evaluer pour la region pertinente
print("\n  Evaluation pour alpha in [0.25, 0.49], T00 in [0, alpha]:")
print(f"  {'alpha':>8} {'T00':>8} {'Delta_M':>12} {'Signe':>6}")
print("-"*40)

for alpha_v in [0.25, 0.30, 0.33, 0.35, 0.37, 0.40, 0.45, 0.49]:
    for T00_v in [0, alpha_v/2, alpha_v*0.7, alpha_v]:
        val = float(Delta_M_fb_simplified.subs({alpha_s: alpha_v, T00_s: T00_v}))
        sign = "+" if val > 0 else ("-" if val < 0 else "0")
        print(f"  {alpha_v:>8.3f} {T00_v:>8.4f} {val:>12.6f} {sign:>6}")

# ============================================================
# PARTIE 10: Induction complete
# ============================================================

print("\n" + "="*70)
print("PARTIE 10: RECURRENCE INDUCTIVE")
print("="*70)

print("""
RECURRENCE:
  D(k+1) = (p_{k+1} - 3) * D(k) + Delta(k)

  ou D(k) = n_{12}(k) - n_{10}(k) = n_1(k) * (2*T12(k) - 1)

Si Delta(k) >= 0 pour tout k, et D(base) > 0, alors D(k) > 0 pour tout k.
Equivalence: F(k) > 0 pour tout k. QED.
""")

print(f"{'k':>3} {'p':>4} {'D(k)':>12} {'(p-3)*D':>14} {'Delta':>12} {'D(k+1)':>14} {'D_actual':>14} {'OK':>4}")
print("-"*80)

D_prev = None
for k in sorted(results.keys()):
    rk = results[k]
    n12 = int(rk['trans'][1, 2])
    n10 = int(rk['trans'][1, 0])
    D_k = n12 - n10

    if D_prev is not None:
        p_new = all_primes[k - 1]
        # Delta exact
        g3_prev = results[k-1]['gram3']
        delta_exact = compute_delta(g3_prev, verbose=False)[0]
        D_predicted = (p_new - 3) * D_prev + delta_exact
        ok = (D_predicted == D_k)
        print(f"{k:>3} {p_new:>4} {D_prev:>12} {(p_new-3)*D_prev:>14} {delta_exact:>12} {D_predicted:>14} {D_k:>14} {'OK' if ok else 'FAIL':>4}")
    else:
        print(f"{k:>3} {'--':>4} {'--':>12} {'--':>14} {'--':>12} {'--':>14} {D_k:>14} {'BASE':>4}")

    D_prev = D_k

# ============================================================
# PARTIE 11: Taux de croissance
# ============================================================

print("\n" + "="*70)
print("PARTIE 11: TAUX DE CROISSANCE DE D(k)")
print("="*70)

print(f"\n{'k':>3} {'D(k)':>14} {'D(k)/D(k-1)':>14} {'p-3':>6} {'Delta/D':>12}")
print("-"*55)

D_vals = []
for k in sorted(results.keys()):
    rk = results[k]
    D_k = int(rk['trans'][1, 2]) - int(rk['trans'][1, 0])
    D_vals.append((k, D_k))

for idx in range(len(D_vals)):
    k, D_k = D_vals[idx]
    if idx > 0:
        k_prev, D_prev = D_vals[idx-1]
        p_new = all_primes[k - 1]
        ratio = D_k / D_prev if D_prev > 0 else float('inf')
        delta = D_k - (p_new - 3) * D_prev
        delta_frac = delta / D_prev if D_prev > 0 else 0
        print(f"{k:>3} {D_k:>14} {ratio:>14.4f} {p_new-3:>6} {delta_frac:>12.4f}")
    else:
        print(f"{k:>3} {D_k:>14} {'--':>14} {'--':>6} {'--':>12}")

# ============================================================
# PARTIE 12: Verdict
# ============================================================

print("\n" + "="*70)
print("VERDICT S15.6.257")
print("="*70)

# Verifier toutes les predictions
all_ok_pred = all(d['ok'] for d in all_deltas)
all_delta_pos = all(d['delta'] > 0 for d in all_deltas)

print(f"""
RESULTATS:

1. FORMULE INDUCTIVE EXACTE:
   D(k+1) = (p-3)*D(k) + Delta(k)
   Verifiee pour k=2..{max(results.keys())-1}: {'TOUT OK' if all_ok_pred else 'ECHECS'}

2. SIGNE DE DELTA:
   Delta > 0 pour toutes les transitions k=2..{max(results.keys())-1}: {'OUI' if all_delta_pos else 'NON'}

3. PREUVE INDUCTIVE:
""")

if all_delta_pos:
    print("""   BASE: D(2) = n12 - n10 = 1 > 0  (calcul exact: (1,0,2) alternation)

   HYPOTHESE: D(k) > 0 pour k >= 2.

   ETAPE: D(k+1) = (p_{k+1} - 3) * D(k) + Delta(k)
          - p_{k+1} >= 5, donc p_{k+1} - 3 >= 2 > 0
          - D(k) > 0 par hypothese
          - Delta(k) > 0 (VERIFIE numeriquement pour k=2..8)
          => D(k+1) > 2*D(k) > 0.

   CONCLUSION: D(k) > 0 pour tout k >= 2.
   => n_{12} > n_{10} pour tout k >= 2.
   => T12 > 1/2 pour tout k >= 2.
   => F > 0 pour tout k >= 2.
   => Q > 0 pour tout k >= 2.
   => alpha(k) est STRICTEMENT CROISSANT.
   => eps(k) -> 0 par Mertens.

   GAP RESTANT: Delta > 0 verifie pour k=2..8 mais pas encore PROUVE
   pour tout k. La formule analytique Markov donne le signe, mais
   l'ecart exact/Markov doit etre borne.
""")
else:
    print("   Delta n'est PAS toujours positif. L'approche inductive directe echoue.")
    print("   Analyser les cas ou Delta < 0.")

print(f"""
FORMULE ANALYTIQUE MARKOV:
  Delta_M(alpha, T00) = {Delta_M_fb_factored}

  Domaine pertinent: alpha in (1/4, 1/2), T00 in [0, alpha]
  Signe dans ce domaine: A DETERMINER RIGOUREUSEMENT

STATUT: [THM**] si Delta_M > 0 prouvable analytiquement dans le domaine.
""")

sys.exit(0)
