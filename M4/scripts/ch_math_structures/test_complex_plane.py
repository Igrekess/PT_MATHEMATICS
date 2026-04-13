"""
Tool 42: PT Complexe -- Le plan 2D complet du crible
=====================================================
PT standard utilise sin^2(theta_p) = Re(w_p), la projection reelle.
La variable naturelle w_p = (1 - e^{2i*theta_p})/2 vit dans le plan 2D:
  Re(w_p) = sin^2(theta_p)     = PERTE
  Im(w_p) = -sin(2*theta_p)/2  = COHERENCE (signe -)

PT jette systematiquement Im(w_p). Ce script reconstruit la PT COMPLETE
dans le plan complexe 2D, en restant dans la logique PT (crible, mod 3).

|w_p|^2 = sin^2(theta_p) = Re(w_p) est l'identite fondamentale:
  le MODULE CARRE est la PROJECTION REELLE.
  => sin^2 code deja l'information du plan complet via |w|^2 = Re(w).

Tests:
  T1:  Geometrie du plan w_p -- positions, distances, angles dans le plan 2D
  T2:  Produit complexe W = Pi w_p et decomposition polaire
  T3:  Action complexe S_C = -sum ln(w_p) et ses parties
  T4:  D_KL complexe -- divergence dans le plan 2D
  T5:  T12 complexe via chi_3 -- l'eigenvalue dans le plan
  T6:  Matrice de Gram <w_p|w_{p'}> -- structure metrique du crible
  T7:  Courant de probabilite J = Im(w* dw/dp) entre niveaux
  T8:  Conservation: Noether complexe sum(w_p) et loi de courant
  T9:  Contraction complexe r_K dans le plan 2D
  T10: Alpha complexe -- la constante de couplage dans le plan
  T11: GFT complexe -- H_max = D_KL + H avec partie imaginaire
  T12: Holonomie du plan -- phase accumulee le long du chemin p=2,3,...
  T13: Identites nouvelles propres au plan 2D
  T14: Bilan PT complexe
"""

import numpy as np
import cmath
import math
import sys

# ============================================================
# Parametres PT fondamentaux
# ============================================================
q_stat = 13.0 / 15.0
q_therm = np.exp(-1.0 / 15.0)
MU_STAR = 15
PRIMES_ACTIFS = [3, 5, 7]
PRIMES_GHOST = [11, 13]
PRIMES_ALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

# ============================================================
# Fonctions de base PT
# ============================================================
def delta_p(p, q):
    """Deficit de crible: delta_p = (1 - q^p) / p."""
    return (1.0 - q**p) / p

def sin2(p, q):
    """sin^2(theta_p) = delta*(2-delta)."""
    d = delta_p(p, q)
    return d * (2.0 - d)

def cos2(p, q):
    """cos^2(theta_p) = (1-delta)^2."""
    d = delta_p(p, q)
    return (1.0 - d)**2

def theta_p(p, q):
    """theta_p = arcsin(sqrt(sin^2))."""
    return np.arcsin(np.sqrt(sin2(p, q)))

def w_p(p, q):
    """Variable naturelle PT complexe: w_p = (1 - e^{2i*theta_p}) / 2.
    Re(w) = sin^2, Im(w) = -sin(2*theta)/2, |w|^2 = sin^2.
    """
    th = theta_p(p, q)
    z = np.exp(2j * th)
    return (1.0 - z) / 2.0

def chi3(p):
    """Caractere mod 3: chi_3(p) = 0,+1,-1 selon p mod 3."""
    r = p % 3
    return 0 if r == 0 else (1 if r == 1 else -1)

print("=" * 90)
print("TOOL 42: PT COMPLEXE -- LE PLAN 2D COMPLET DU CRIBLE")
print("=" * 90)

# ====================================================================
# T1: Geometrie du plan w_p
# ====================================================================
print("\n### T1: Geometrie du plan w_p -- positions dans le plan 2D")
print("    w_p = (1 - e^{2i*theta})/2 = sin^2 - i*sin(2*theta)/2")
print("    Chaque premier p a une POSITION dans le plan complexe")
print()

q = q_stat
print(f"  q_stat = {q:.6f}")
print(f"  {'p':>4} {'Re(w)=sin^2':>14} {'Im(w)':>14} {'|w|':>10} {'|w|^2':>10} {'arg(w)/pi':>10}")

w_vals = {}
for p in PRIMES_ALL:
    w = w_p(p, q)
    w_vals[p] = w
    print(f"  {p:4d} {w.real:14.8f} {w.imag:14.8f} {abs(w):10.6f} {abs(w)**2:10.6f} {cmath.phase(w)/math.pi:10.6f}")

# Distances entre premiers actifs
print(f"\n  Distances |w_p - w_{p}'| dans le plan 2D (actifs):")
print(f"  {'p':>4} {'p_':>4} {'|w-w_|':>12} {'Re(w-w_)':>12} {'Im(w-w_)':>12}")
for i, p1 in enumerate(PRIMES_ACTIFS):
    for p2 in PRIMES_ACTIFS[i+1:]:
        dw = w_vals[p1] - w_vals[p2]
        print(f"  {p1:4d} {p2:4d} {abs(dw):12.8f} {dw.real:12.8f} {dw.imag:12.8f}")

# Triangle des actifs dans le plan
w3, w5, w7 = w_vals[3], w_vals[5], w_vals[7]
# Aire du triangle via produit vectoriel
aire = 0.5 * abs((w5 - w3).real * (w7 - w3).imag - (w5 - w3).imag * (w7 - w3).real)
print(f"\n  Aire du triangle (3,5,7) dans le plan w: {aire:.10f}")
print(f"  Centre de masse: {(w3+w5+w7)/3:.8f}")

# ====================================================================
# T2: Produit complexe W = Pi w_p
# ====================================================================
print("\n\n### T2: Produit complexe W = Pi w_p")
print("    En PT reel: alpha = Pi sin^2 = Pi Re(w_p)")
print("    En PT complexe: W = Pi w_p vit dans le plan 2D")
print()

for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    # Produit sur actifs
    W_actifs = 1.0 + 0j
    for p in PRIMES_ACTIFS:
        W_actifs *= w_p(p, q)
    alpha_reel = 1.0
    for p in PRIMES_ACTIFS:
        alpha_reel *= sin2(p, q)

    print(f"  {label} (actifs 3,5,7):")
    print(f"    W = Pi w_p      = {W_actifs.real:+.10f} {W_actifs.imag:+.10f}i")
    print(f"    |W|             = {abs(W_actifs):.10f}")
    print(f"    |W|^2           = {abs(W_actifs)**2:.10f}")
    print(f"    alpha_reel      = {alpha_reel:.10f}")
    print(f"    arg(W)/pi       = {cmath.phase(W_actifs)/math.pi:.8f}")
    # Decomposition: W = |W| * e^{i*phi}
    print(f"    Polaire: |W|={abs(W_actifs):.10f}, phi={cmath.phase(W_actifs):.8f} rad")
    print()

    # Verif: |W| = Pi |w_p| = Pi sin(theta_p) = Pi sqrt(sin^2)
    prod_sin = 1.0
    for p in PRIMES_ACTIFS:
        prod_sin *= np.sqrt(sin2(p, q))
    print(f"    Pi sqrt(sin^2)  = {prod_sin:.10f} (= |W|? {abs(abs(W_actifs) - prod_sin) < 1e-12})")

    # Test cle: |W|^2 vs alpha
    # |W|^2 = Pi |w_p|^2 = Pi sin^2 = alpha
    print(f"    |W|^2 = alpha?  {abs(abs(W_actifs)**2 - alpha_reel) < 1e-12} (ecart {abs(abs(W_actifs)**2 - alpha_reel):.2e})")
    print()

# ====================================================================
# T3: Action complexe S_C = -sum ln(w_p)
# ====================================================================
print("\n### T3: Action complexe S_C = -sum ln(w_p)")
print("    PT reel: S_PT = -sum ln(sin^2) = -sum ln(|w|^2)")
print("    PT complexe: S_C = -sum ln(w) = -sum [ln|w| + i*arg(w)]")
print("    Donc: Re(S_C) = -sum ln|w| = S_PT/2")
print("           Im(S_C) = -sum arg(w_p)")
print()

for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    print(f"  {label}:")
    S_PT = 0.0
    S_C = 0.0 + 0j
    for p in PRIMES_ACTIFS:
        w = w_p(p, q)
        s2 = sin2(p, q)
        S_PT += -np.log(s2)
        S_C += -cmath.log(w)

    print(f"    S_PT (reel)     = {S_PT:.10f}")
    print(f"    S_C             = {S_C.real:.10f} + {S_C.imag:.10f}i")
    print(f"    Re(S_C)         = {S_C.real:.10f}")
    print(f"    S_PT/2          = {S_PT/2:.10f}")
    print(f"    Re(S_C) = S_PT/2? {abs(S_C.real - S_PT/2) < 1e-10} (ecart {abs(S_C.real - S_PT/2):.2e})")
    print(f"    Im(S_C)         = {S_C.imag:.10f}")
    print(f"    Im(S_C)/pi      = {S_C.imag/math.pi:.10f}")
    print()

    # Action sur tous les premiers
    S_PT_all = 0.0
    S_C_all = 0.0 + 0j
    for p in PRIMES_ALL:
        w = w_p(p, q)
        s2 = sin2(p, q)
        S_PT_all += -np.log(s2)
        S_C_all += -cmath.log(w)

    print(f"    Tous (2..47):  S_PT = {S_PT_all:.10f}")
    print(f"                   S_C  = {S_C_all.real:.10f} + {S_C_all.imag:.10f}i")
    print(f"                   Im/Re= {S_C_all.imag/S_C_all.real:.8f}")
    print(f"                   |S_C|= {abs(S_C_all):.10f}")
    print()

# ====================================================================
# T4: D_KL complexe -- distance dans le plan 2D
# ====================================================================
print("\n### T4: D_KL complexe -- divergence dans le plan 2D")
print("    D_KL_reel = sum sin^2 * ln(sin^2 / ref)")
print("    D_KL_C = sum w_p * ln(w_p / ref)")
print("    On prend ref = 1/2 (equipartition loss/conserv)")
print()

q = q_stat
# Distribution reelle: p_loss = sin^2, p_conserv = cos^2
# D_KL reel par premier: sin^2 * ln(sin^2 / 0.5) + cos^2 * ln(cos^2 / 0.5)
print(f"  q_stat, par premier:")
print(f"  {'p':>4} {'D_KL_reel':>12} {'D_KL_complex':>28} {'|D_KL_C|':>10}")
D_KL_tot_re = 0.0
D_KL_tot_C = 0.0 + 0j
for p in PRIMES_ACTIFS:
    s2_val = sin2(p, q)
    c2_val = cos2(p, q)
    # KL reel: distance a l'equipartition
    dkl_re = s2_val * np.log(s2_val / 0.5) + c2_val * np.log(c2_val / 0.5)
    # KL complexe: w * ln(w / 0.5)
    w = w_p(p, q)
    dkl_c = w * cmath.log(w / 0.5)
    D_KL_tot_re += dkl_re
    D_KL_tot_C += dkl_c
    print(f"  {p:4d} {dkl_re:12.8f} {dkl_c.real:+12.8f}{dkl_c.imag:+12.8f}i {abs(dkl_c):10.6f}")

print(f"\n  Total actifs: D_KL_reel = {D_KL_tot_re:.8f}")
print(f"                D_KL_C    = {D_KL_tot_C.real:+.8f}{D_KL_tot_C.imag:+.8f}i")
print(f"                |D_KL_C|  = {abs(D_KL_tot_C):.8f}")
print(f"                Re(D_KL_C)= D_KL_reel? Non, car w*ln(w) != Re(w)*ln(Re(w))")

# Mais une version plus naturelle: D_KL via |w|^2
# Puisque |w|^2 = sin^2 = probabilite de perte:
print(f"\n  Identite: |w|^2 = Re(w) = sin^2")
print(f"  => ln |w|^2 = ln Re(w) = ln sin^2")
print(f"  => S_PT = -sum ln|w|^2 = -2 sum ln|w| = 2 Re(S_C)")

# ====================================================================
# T5: T12 complexe via chi_3
# ====================================================================
print("\n\n### T5: T12 complexe via chi_3")
print("    T12_reel = (1/3)*sum chi_3(p)*sin^2")
print("    T12_C = (1/3)*sum chi_3(p)*w_p -- dans le plan 2D")
print("    La position de T12 dans le plan encode l'asymetrie mod 3")
print()

for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    T12_re = 0.0
    T12_C = 0.0 + 0j
    for p in PRIMES_ACTIFS:
        c = chi3(p)
        if c != 0:
            T12_re += c * sin2(p, q)
            T12_C += c * w_p(p, q)
    T12_re /= 3.0
    T12_C /= 3.0

    print(f"  {label}:")
    print(f"    T12_reel = {T12_re:.10f}")
    print(f"    T12_C    = {T12_C.real:+.10f} {T12_C.imag:+.10f}i")
    print(f"    |T12_C|  = {abs(T12_C):.10f}")
    print(f"    arg(T12_C)/pi = {cmath.phase(T12_C)/math.pi:.8f}")
    print(f"    Re(T12_C) = T12_reel? {abs(T12_C.real - T12_re) < 1e-12}")
    print()

    # T12 complexe inclut le deficit MOD 3 dans les 2 dimensions
    # chi_3(7) = +1 (7 mod 3 = 1), chi_3(5) = -1 (5 mod 3 = 2)
    # T12_C = (w_7 - w_5)/3
    w5_val = w_p(5, q)
    w7_val = w_p(7, q)
    T12_direct = (w7_val - w5_val) / 3.0
    print(f"    (w_7 - w_5)/3 = {T12_direct.real:+.10f} {T12_direct.imag:+.10f}i")
    print(f"    Match T12_C? {abs(T12_C - T12_direct) < 1e-12}")
    print(f"    => T12_C mesure la DISTANCE ORIENTEE entre w_5 et w_7 dans le plan")
    print()

# ====================================================================
# T6: Matrice de Gram <w_p|w_{p'}> -- metrique du crible
# ====================================================================
print("\n### T6: Matrice de Gram G_{pp'} = w_p* . w_{p'}")
print("    Produit hermitien dans le plan 2D")
print("    G_{pp} = |w_p|^2 = sin^2(theta_p)")
print("    G_{pp'} = Re + i*Im encode angle et distance")
print()

q = q_stat
primes_gram = [3, 5, 7, 11, 13]
n_gram = len(primes_gram)
G_mat = np.zeros((n_gram, n_gram), dtype=complex)
for i, p1 in enumerate(primes_gram):
    for j, p2 in enumerate(primes_gram):
        G_mat[i, j] = np.conj(w_p(p1, q)) * w_p(p2, q)

# Print matrix
print("  q_stat, w_p* . w_p' (premiers 3,5,7,11,13):")
header = "     " + "".join(f"{p:>14d}" for p in primes_gram)
print(header)
for i, p1 in enumerate(primes_gram):
    row = f"  {p1:3d}"
    for j in range(n_gram):
        g = G_mat[i, j]
        row += f"  {g.real:+.4f}{g.imag:+.4f}i"
    print(row)

# Diagonale = sin^2
print(f"\n  Diag = sin^2: ", end="")
for i, p in enumerate(primes_gram):
    print(f"{G_mat[i,i].real:.6f} ", end="")
print()

# Partie imaginaire = courant J
print(f"  Im(G) = courant J: antisymetrique?")
for i in range(n_gram):
    for j in range(i+1, n_gram):
        J_ij = G_mat[i, j].imag
        J_ji = G_mat[j, i].imag
        print(f"    J({primes_gram[i]},{primes_gram[j]}) = {J_ij:+.8f}, J({primes_gram[j]},{primes_gram[i]}) = {J_ji:+.8f}, somme = {J_ij+J_ji:.2e}")

# Eigenvalues de la matrice de Gram
evals_G = np.linalg.eigvalsh(G_mat.real)  # partie reelle = metrique
print(f"\n  Eigenvalues de Re(G): {[f'{e:.8f}' for e in sorted(evals_G, reverse=True)]}")
print(f"  Toutes positives? {all(e > -1e-12 for e in evals_G)}")

# Determinant
det_G = np.linalg.det(G_mat)
print(f"  det(G) = {det_G.real:+.2e} {det_G.imag:+.2e}i")

# ====================================================================
# T7: Courant de probabilite J entre niveaux adjacents
# ====================================================================
print("\n\n### T7: Courant J(p, p') = Im(w_p* . w_{p'}) entre niveaux")
print("    J mesure le flux de 'phase' entre deux niveaux du crible")
print("    J > 0: flux de p vers p', J < 0: flux de p' vers p")
print("    En PT, les premiers sont ordonnes: le courant suit l'ordre naturel")
print()

q = q_stat
print(f"  Courant entre niveaux adjacents (q_stat):")
print(f"  {'p->p_':>8} {'J':>12} {'|w_p|^2':>10} {'|w_{p_}|^2':>10} {'J/sqrt(s2*s2_)':>14}")
J_total = 0.0
for i in range(len(PRIMES_ALL) - 1):
    p1, p2 = PRIMES_ALL[i], PRIMES_ALL[i + 1]
    w1 = w_p(p1, q)
    w2 = w_p(p2, q)
    J_val = (np.conj(w1) * w2).imag
    s2_1 = sin2(p1, q)
    s2_2 = sin2(p2, q)
    J_norm = J_val / np.sqrt(s2_1 * s2_2) if s2_1 > 0 and s2_2 > 0 else 0
    J_total += J_val
    print(f"  {p1:3d}->{p2:<3d} {J_val:12.8f} {s2_1:10.6f} {s2_2:10.6f} {J_norm:14.8f}")

print(f"\n  J_total = sum J(p_i, p_{i+1}) = {J_total:.10f}")
print(f"  J_total != 0 => le courant NE SE CONSERVE PAS entre niveaux adjacents")
print(f"  (normal: le systeme n'est pas ferme, les premiers continuent)")

# Courant normalise: J/sqrt(sin^2 * sin^2')
# C'est le SINUS de l'angle entre w_p et w_{p'} dans le plan
# Cos(angle) = Re(w* w') / (|w||w'|), Sin(angle) = Im(w* w') / (|w||w'|)
print("\n  Le courant normalise J/(|w||w'|) = sin(angle entre w_p, w_p' dans le plan)")
print(f"  C'est l'angle entre les POSITIONS des premiers dans le plan complexe")

# ====================================================================
# T8: Conservation de Noether: sum w_p
# ====================================================================
print("\n\n### T8: Somme complexe Sigma = sum w_p (conservation)")
print("    Si PT a une symetrie U(1), Sigma = sum w_p devrait etre 'special'")
print()

for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    Sigma_actifs = sum(w_p(p, q) for p in PRIMES_ACTIFS)
    Sigma_all = sum(w_p(p, q) for p in PRIMES_ALL)
    Sigma_ghost = sum(w_p(p, q) for p in PRIMES_GHOST)

    S_re_actifs = sum(sin2(p, q) for p in PRIMES_ACTIFS)

    print(f"  {label}:")
    print(f"    Sigma_actifs = {Sigma_actifs.real:+.10f} {Sigma_actifs.imag:+.10f}i")
    print(f"    |Sigma_actifs| = {abs(Sigma_actifs):.10f}")
    print(f"    Re(Sigma) = sum sin^2 = {S_re_actifs:.10f} (check: {abs(Sigma_actifs.real - S_re_actifs) < 1e-12})")
    print(f"    Im(Sigma) = -sum sin(2theta)/2 = {Sigma_actifs.imag:.10f}")
    print(f"    arg(Sigma)/pi = {cmath.phase(Sigma_actifs)/math.pi:.8f}")
    print()

    # Rapport Im/Re
    ratio = Sigma_actifs.imag / Sigma_actifs.real if abs(Sigma_actifs.real) > 1e-15 else float('inf')
    print(f"    Im(Sigma)/Re(Sigma) = {ratio:.10f}")

    # Ghost contribution
    print(f"    Sigma_ghost = {Sigma_ghost.real:+.10f} {Sigma_ghost.imag:+.10f}i")
    print(f"    Sigma_total(2..47) = {Sigma_all.real:+.10f} {Sigma_all.imag:+.10f}i")
    print(f"    arg(Sigma_total)/pi = {cmath.phase(Sigma_all)/math.pi:.8f}")
    print()

# ====================================================================
# T9: Contraction complexe r_K dans le plan 2D
# ====================================================================
print("\n### T9: Contraction complexe -- |w_p| decroit avec p")
print("    |w_p| = sqrt(sin^2) = sin(theta_p)")
print("    La contraction PT dans le plan = les points se rapprochent de l'origine")
print("    Vitesse de rapprochement dans la direction Re vs Im")
print()

q = q_stat
print(f"  q_stat:")
print(f"  {'p':>4} {'|w|':>10} {'Re(w)':>12} {'Im(w)':>12} {'|Re/Im|':>10} {'angle/pi':>10}")
for p in PRIMES_ALL:
    w = w_p(p, q)
    ratio = abs(w.real / w.imag) if abs(w.imag) > 1e-15 else float('inf')
    print(f"  {p:4d} {abs(w):10.6f} {w.real:12.8f} {w.imag:12.8f} {ratio:10.4f} {cmath.phase(w)/math.pi:10.6f}")

# La direction dans le plan: angle de w_p
# Si tous les w_p ont le meme angle => la contraction est RADIALE
# Si les angles changent => la contraction a une composante TANGENTIELLE
angles = [cmath.phase(w_p(p, q)) for p in PRIMES_ALL]
angle_var = np.var(angles)
angle_mean = np.mean(angles)
print(f"\n  Angle moyen de w_p: {angle_mean:.8f} rad = {angle_mean/math.pi:.6f} pi")
print(f"  Variance des angles: {angle_var:.10f}")
print(f"  Ecart-type: {np.sqrt(angle_var):.8f} rad = {np.sqrt(angle_var)/math.pi:.6f} pi")
print(f"  Contraction purement radiale? {'OUI' if angle_var < 0.01 else 'NON'} (variance {'<' if angle_var < 0.01 else '>'} 0.01)")

# ====================================================================
# T10: Alpha complexe -- la constante de couplage dans le plan
# ====================================================================
print("\n\n### T10: Alpha complexe -- constante de couplage 2D")
print("    alpha_reel = Pi sin^2 = Pi |w|^2")
print("    W = Pi w_p: le produit COMPLEXE")
print("    |W|^2 = Pi |w_p|^2 = Pi sin^2 = alpha_reel")
print("    => alpha est le MODULE CARRE du produit complexe!")
print()

q = q_stat
W_actifs = 1.0 + 0j
alpha_re = 1.0
for p in PRIMES_ACTIFS:
    W_actifs *= w_p(p, q)
    alpha_re *= sin2(p, q)

print(f"  q_stat, actifs (3,5,7):")
print(f"    W        = {W_actifs.real:+.12f} {W_actifs.imag:+.12f}i")
print(f"    |W|^2    = {abs(W_actifs)**2:.12f}")
print(f"    alpha_re = {alpha_re:.12f}")
print(f"    1/|W|^2  = {1/abs(W_actifs)**2:.6f}")
print(f"    1/alpha  = {1/alpha_re:.6f}")
print()

# L'INFORMATION NOUVELLE: l'argument de W
phi_W = cmath.phase(W_actifs)
print(f"    arg(W) = {phi_W:.10f} rad = {phi_W/math.pi:.8f} pi")
print(f"    => alpha_EM = |W|^2 code le MODULE (force du couplage)")
print(f"    => arg(W) code la PHASE du couplage (nouvelle observable)")
print()

# Decomposition polaire: W = sqrt(alpha) * e^{i*phi}
print(f"    Decomposition polaire: W = sqrt(alpha) * e^{{i*phi}}")
print(f"    sqrt(alpha) = {np.sqrt(alpha_re):.10f}")
print(f"    phi         = {phi_W:.10f}")
print(f"    Verification: sqrt(alpha)*cos(phi) = {np.sqrt(alpha_re)*np.cos(phi_W):.12f} vs Re(W) = {W_actifs.real:.12f}")
print(f"    Verification: sqrt(alpha)*sin(phi) = {np.sqrt(alpha_re)*np.sin(phi_W):.12f} vs Im(W) = {W_actifs.imag:.12f}")
print()

# Avec ghosts
W_ghost = W_actifs
for p in PRIMES_GHOST:
    W_ghost *= w_p(p, q)
phi_ghost = cmath.phase(W_ghost)
print(f"    Avec ghosts (3..13): |W|^2 = {abs(W_ghost)**2:.12f}, arg(W) = {phi_ghost/math.pi:.8f} pi")
print(f"    Changement de phase par ghost: {(phi_ghost - phi_W)/math.pi:.8f} pi")

# ====================================================================
# T11: GFT complexe -- H_max = D_KL + H dans le plan
# ====================================================================
print("\n\n### T11: GFT complexe")
print("    En PT reel: H_max = D_KL + H (identite GFT exacte)")
print("    En PT complexe: les entropies deviennent complexes?")
print("    H = -sum p*ln(p) ou p = sin^2 / (sum sin^2)")
print()

q = q_stat
# Normaliser sin^2 comme distribution de probabilite sur les actifs
S2_actifs = [sin2(p, q) for p in PRIMES_ACTIFS]
S_total = sum(S2_actifs)
prob_re = [s / S_total for s in S2_actifs]

# Entropie reelle
H_re = -sum(p * np.log(p) for p in prob_re if p > 0)
H_max = np.log(len(PRIMES_ACTIFS))
D_KL_re = H_max - H_re

print(f"  Reel (sin^2 normalises sur actifs):")
print(f"    Probabilites: {[f'{p:.6f}' for p in prob_re]}")
print(f"    H    = {H_re:.10f}")
print(f"    H_max = {H_max:.10f} = ln(3)")
print(f"    D_KL = {D_KL_re:.10f}")
print(f"    H + D_KL = {H_re + D_KL_re:.10f} (= H_max? {abs(H_re + D_KL_re - H_max) < 1e-12})")
print()

# Version complexe: normaliser w_p
W_actifs_list = [w_p(p, q) for p in PRIMES_ACTIFS]
W_total = sum(W_actifs_list)
prob_C = [w / W_total for w in W_actifs_list]

# "Entropie complexe" H_C = -sum p_C * ln(p_C)
H_C = -sum(pc * cmath.log(pc) for pc in prob_C)
print(f"  Complexe (w_p normalises sur actifs):")
print(f"    w normalises: {[f'{p.real:.6f}{p.imag:+.6f}i' for p in prob_C]}")
print(f"    H_C = {H_C.real:+.10f} {H_C.imag:+.10f}i")
print(f"    Re(H_C) = {H_C.real:.10f}")
print(f"    Im(H_C) = {H_C.imag:.10f}")
print(f"    H_C = H_reel? Re(H_C) vs H_re: ecart {abs(H_C.real - H_re):.6e}")
print()

# L'identite GFT tient-elle dans le plan complexe?
D_KL_C = H_max - H_C  # Serait la "D_KL complexe"
print(f"    D_KL_C (defini par H_max - H_C) = {D_KL_C.real:+.10f} {D_KL_C.imag:+.10f}i")
print(f"    H_C + D_KL_C = H_max? {abs(H_C + D_KL_C - H_max) < 1e-12} (tautologie par definition)")

# ====================================================================
# T12: Holonomie -- phase accumulee le long de p=2,3,5,...
# ====================================================================
print("\n\n### T12: Holonomie -- phase accumulee le long du crible")
print("    Quand on parcourt les premiers p=2,3,5,...,47, w_p trace un CHEMIN")
print("    dans le plan 2D. La phase accumulee arg(w_{p+1}/w_p) est l'holonomie.")
print()

q = q_stat
print(f"  q_stat:")
print(f"  {'p->p_':>8} {'d(arg)/pi':>12} {'|w_{p+1}/w_p|':>14} {'d(ln|w|)':>12}")

phase_total = 0.0
log_mod_total = 0.0
for i in range(len(PRIMES_ALL) - 1):
    p1, p2 = PRIMES_ALL[i], PRIMES_ALL[i + 1]
    w1 = w_p(p1, q)
    w2 = w_p(p2, q)
    ratio = w2 / w1
    darg = cmath.phase(ratio)
    dmod = np.log(abs(w2)) - np.log(abs(w1))
    phase_total += darg
    log_mod_total += dmod
    print(f"  {p1:3d}->{p2:<3d} {darg/math.pi:12.8f} {abs(ratio):14.8f} {dmod:12.8f}")

print(f"\n  Phase totale accumulee: {phase_total:.8f} rad = {phase_total/math.pi:.6f} pi")
print(f"  Log-module total: {log_mod_total:.8f}")
print(f"  Verification: exp(log_mod_total) * |w_2| = {np.exp(log_mod_total) * abs(w_p(2, q)):.8f}")
print(f"                |w_47| = {abs(w_p(47, q)):.8f}")

# La holonomie est-elle constante par pas?
dargs = []
for i in range(len(PRIMES_ALL) - 1):
    w1 = w_p(PRIMES_ALL[i], q)
    w2 = w_p(PRIMES_ALL[i + 1], q)
    dargs.append(cmath.phase(w2 / w1))
print(f"\n  Variance de d(arg): {np.var(dargs):.10f}")
print(f"  Phase/pas constante? {'OUI' if np.var(dargs) < 0.001 else 'NON'}")

# ====================================================================
# T13: Identites nouvelles propres au plan 2D
# ====================================================================
print("\n\n### T13: Identites fondamentales du plan 2D")
print()

q = q_stat

# ID1: |w|^2 = Re(w)  [fondamentale]
print("  ID1: |w_p|^2 = Re(w_p) pour tout premier p")
all_pass = True
for p in PRIMES_ALL:
    w = w_p(p, q)
    diff = abs(abs(w)**2 - w.real)
    if diff > 1e-12:
        all_pass = False
print(f"  PASS: {all_pass} (|w|^2 = Re(w) pour les 15 premiers)")
print()

# ID2: w_p = sin^2(theta) * (1 - i*cot(theta))   (si sin != 0)
# En fait: w = sin^2 - i*sin(2theta)/2 = sin^2 - i*sin(theta)*cos(theta)
#         = sin(theta) * [sin(theta) - i*cos(theta)]
#         = sin(theta) * (-i) * [cos(theta) + i*sin(theta)]
#         = -i * sin(theta) * e^{i*theta}
print("  ID2: w_p = -i * sin(theta_p) * e^{i*theta_p}")
all_pass = True
for p in PRIMES_ALL:
    w = w_p(p, q)
    th = theta_p(p, q)
    w_test = -1j * np.sin(th) * np.exp(1j * th)
    diff = abs(w - w_test)
    if diff > 1e-12:
        all_pass = False
        print(f"    p={p}: FAIL, diff = {diff:.2e}")
print(f"  PASS: {all_pass}")
print(f"  => w_p est le produit de sin(theta) (AMPLITUDE) et e^{{i*theta}} (PHASE)")
print(f"     avec un facteur -i (rotation de pi/2)")
print()

# ID3: w = -i |w| e^{i*theta} car |w| = sin(theta)
# Donc arg(w) = theta - pi/2
print("  ID3: arg(w_p) = theta_p - pi/2")
all_pass = True
for p in PRIMES_ALL:
    w = w_p(p, q)
    th = theta_p(p, q)
    diff = abs(cmath.phase(w) - (th - math.pi / 2))
    if diff > 1e-12:
        all_pass = False
print(f"  PASS: {all_pass}")
print(f"  => Les w_p vivent dans le 4eme quadrant (Re>0, Im<0)")
print(f"     car theta < pi/2 pour tout premier")
print()

# ID4: w_p * conj(w_p) = |w|^2 = sin^2 = Re(w)
# Donc: w * conj(w) = Re(w)
# Equivalent: Im(w)^2 = Re(w) - Re(w)^2 = Re(w)*(1 - Re(w)) = sin^2 * cos^2
print("  ID4: Im(w)^2 = Re(w) * (1 - Re(w)) = sin^2 * cos^2")
all_pass = True
for p in PRIMES_ALL:
    w = w_p(p, q)
    lhs = w.imag**2
    rhs = w.real * (1 - w.real)
    diff = abs(lhs - rhs)
    if diff > 1e-12:
        all_pass = False
print(f"  PASS: {all_pass}")
print(f"  => w_p est SUR LA PARABOLE Im^2 = Re*(1-Re) dans le plan!")
print()

# ID5: La parabole Im^2 = Re(1-Re) est un DEMI-CERCLE
# En effet: Re = sin^2, Im = -sin*cos, donc Re^2 + Im^2 = sin^4 + sin^2*cos^2 = sin^2(sin^2+cos^2) = sin^2 = Re
# Donc: Re^2 + Im^2 = Re, soit (Re - 1/2)^2 + Im^2 = 1/4
# C'est un CERCLE de centre (1/2, 0) et rayon 1/2 !
print("  ID5: (Re(w) - 1/2)^2 + Im(w)^2 = 1/4")
print("       => Les w_p vivent sur un CERCLE de centre (1/2, 0) et rayon 1/2")
all_pass = True
for p in PRIMES_ALL:
    w = w_p(p, q)
    lhs = (w.real - 0.5)**2 + w.imag**2
    rhs = 0.25
    diff = abs(lhs - rhs)
    if diff > 1e-12:
        all_pass = False
        print(f"    p={p}: FAIL, diff = {diff:.2e}")
print(f"  PASS: {all_pass}")
print()

# This is the KEY discovery: all w_p lie on a CIRCLE
# The circle passes through 0 (when sin^2=0, i.e. delta=0, p->inf)
# and through 1 (when sin^2=1, i.e. delta=1)
# w=0 is the limit p->inf (no sieve effect)
# w=1 is the limit of maximal sieve (total loss)
print("  INTERPRETATION GEOMETRIQUE FONDAMENTALE:")
print("  Le crible PT vit sur le CERCLE C(1/2, 0; r=1/2)")
print("  dans le plan complexe.")
print("  - w = 0 : pas de perte (p -> infini)")
print("  - w = 1 : perte totale")
print("  - w = 1/2 : equipartition (sin^2 = 1/2)")
print("  - Le chemin p=2,3,5,... parcourt le cercle de w~0.23 vers w~0.04")
print("  - La contraction PT = les points convergent vers 0 SUR LE CERCLE")
print()

# ID6: Parametrisation par theta: w = (1/2)(1 - e^{2i*theta}) = (1/2) - (1/2)e^{2i*theta}
# C'est le centre 1/2 MOINS un rayon tournant (1/2)*e^{2i*theta}
print("  ID6: w_p = 1/2 - (1/2)*e^{2i*theta_p}")
print("       Le point w_p est obtenu en partant du centre (1/2,0)")
print("       et en soustrayant un vecteur de longueur 1/2 d'angle 2*theta_p")
all_pass = True
for p in PRIMES_ALL:
    w = w_p(p, q)
    th = theta_p(p, q)
    w_test = 0.5 - 0.5 * np.exp(2j * th)
    diff = abs(w - w_test)
    if diff > 1e-12:
        all_pass = False
print(f"  PASS: {all_pass}")
print(f"  => theta_p parametrise la POSITION sur le cercle")
print(f"  => La PT reside ENTIEREMENT sur ce cercle")
print()

# ID7: Lien w_p et delta_p
# delta = (1-q^p)/p, sin^2 = delta*(2-delta), cos = 1-delta
# w = sin^2 - i*sin*cos = delta*(2-delta) - i*sqrt(delta*(2-delta))*(1-delta)
# w = sin^2 * (1 - i*(1-delta)/sin(theta))
# Mais plus elegant: w = delta*(2-delta) - i*(1-delta)*sqrt(delta*(2-delta))
#                      = sqrt(sin^2) * [sqrt(sin^2) - i*(1-delta)]
print("  ID7: w_p = sqrt(sin^2) * [sqrt(sin^2) - i*(1-delta_p)]")
all_pass = True
for p in PRIMES_ALL:
    w = w_p(p, q)
    d = delta_p(p, q)
    s = np.sqrt(sin2(p, q))
    w_test = s * (s - 1j * (1 - d))
    diff = abs(w - w_test)
    if diff > 1e-12:
        all_pass = False
print(f"  PASS: {all_pass}")
print(f"  => En termes de delta: w est le produit sin(theta) x [sin(theta) - i*cos(theta)]")

# ====================================================================
# T14: Bilan PT complexe
# ====================================================================
print("\n\n" + "=" * 90)
print("### T14: BILAN -- PT COMPLEXE DANS LE PLAN 2D")
print("=" * 90)
print("""
DECOUVERTE FONDAMENTALE:
  La variable naturelle w_p = (1 - e^{2i*theta_p})/2 trace un CERCLE
  dans le plan complexe:

    (Re(w) - 1/2)^2 + Im(w)^2 = 1/4

  Cercle C de centre (1/2, 0) et rayon 1/2.
  Ce cercle passe par 0 (pas de perte) et 1 (perte totale).

STRUCTURE:
  w_p = -i * sin(theta_p) * e^{i*theta_p}

  => w_p est le PRODUIT de:
     - sin(theta_p) = AMPLITUDE (module, decroit avec p)
     - e^{i*theta_p} = PHASE (tourne, croit avec p... NON, decroit aussi)
     - -i = rotation de pi/2 (fixe le quadrant)

  => Le chemin p=2,3,5,...,47 parcourt le cercle C
     depuis w~0.23 (p=2, forte perte) vers w~0.04 (p=47, faible perte)
     La contraction PT = convergence vers 0 SUR LE CERCLE.

IDENTITES CLES:
  |w_p|^2 = Re(w_p)                    [module = projection]
  Im(w_p)^2 = Re(w_p)*(1-Re(w_p))      [parabole = cercle]
  arg(w_p) = theta_p - pi/2            [phase = angle - pi/2]
  (Re-1/2)^2 + Im^2 = 1/4              [CERCLE]

OBSERVABLES:
  alpha_EM = |W|^2 ou W = Pi w_p       [alpha = module carre du produit]
  arg(W) = NOUVELLE observable          [phase du couplage]
  T12_C = (w_7 - w_5)/3                [vecteur dans le plan]
  J(p,p') = Im(w_p* w_{p'})            [courant = flux de phase]
  S_C = -sum ln(w_p) = S_PT/2 + i*Sigma_arg  [action complexe]

CE QUE PT JETTE:
  En projetant sur Re(w), PT perd:
  1. La position sur le cercle (seul |w|^2 = Re(w) est garde)
  2. L'orientation (Im, ie la coherence sin*cos)
  3. Le courant J entre premiers
  4. L'argument de W = Pi w_p
  MAIS: |w|^2 = Re(w) implique que le MODULE est entierement
  determine par la projection reelle. La perte est dans la PHASE seulement.
""")

print("=" * 90)
print("FIN TOOL 42")
print("=" * 90)

sys.exit(0 if passes == total else 1)
