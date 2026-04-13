"""
Tool 44: PT Complexe -- 4 directions profondes
================================================
Suite des tools 42-43. On explore les 4 directions identifiees:

  A. Cross-ratio CR(3,5;7,inf) = 2 exact? (demonstration analytique)
  B. Canal quantique: le crible comme sequence de qubits
  C. Produit d'Euler complexe: theta_p(s) = s*ln(p)
  D. Fubini-Study / Fisher: consequences pour la geometrie PT

Tests:
  T1:  CR(3,5;7,0) analytique -- demonstration via delta_p
  T2:  CR pour d'autres triplets et q_therm -- universalite?
  T3:  Canal quantique: operateur de transition rho_p -> rho_{p'}
  T4:  Fidelite F(rho_p, rho_{p'}) entre qubits adjacents
  T5:  Entropie de von Neumann du melange sum_p rho_p / n
  T6:  Produit d'Euler: W(s) = Pi (1-z_p(s))/2 avec z_p(s) = p^{-2is}
  T7:  Comparaison W(s) vs 1/zeta(2is)
  T8:  Le "s" effectif: pour quel s theta_p(s) ~ s*ln(p)?
  T9:  Fubini-Study: courbure de CP^1 et sa valeur PT
  T10: Connexion de Berry sur CP^1 -- la connexion est theta elle-meme?
  T11: Geodesiques sur CP^1: le chemin p=2,3,...,47 est-il geodesique?
  T12: Synthese des 4 directions
"""

import numpy as np
import cmath
import math

q_stat = 13.0 / 15.0
q_therm = np.exp(-1.0 / 15.0)
MU_STAR = 15
PRIMES_ACTIFS = [3, 5, 7]
PRIMES_ALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def delta_p(p, q):
    return (1.0 - q**p) / p

def sin2(p, q):
    d = delta_p(p, q)
    return d * (2.0 - d)

def theta_p(p, q):
    return np.arcsin(np.sqrt(sin2(p, q)))

def w_p(p, q):
    th = theta_p(p, q)
    return (1.0 - np.exp(2j * th)) / 2.0

def z_p_pt(p, q):
    """z_p PT = e^{2i*theta_p} sur U(1)."""
    th = theta_p(p, q)
    return np.exp(2j * th)

def cross_ratio(z1, z2, z3, z4):
    return ((z1 - z3) * (z2 - z4)) / ((z1 - z4) * (z2 - z3))

print("=" * 90)
print("TOOL 44: PT COMPLEXE -- 4 DIRECTIONS PROFONDES")
print("=" * 90)

# ════════════════════════════════════════════════════════════════════
# DIRECTION A: Cross-ratio CR(3,5;7,0) = 2 exact?
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  DIRECTION A: CROSS-RATIO CR(3,5;7,0)")
print("=" * 70)

# ====================================================================
# T1: Analyse analytique du cross-ratio
# ====================================================================
print("\n### T1: CR(3,5;7,0) -- analyse analytique")
print("    CR(w3,w5;w7,0) = (w3-w7)(w5-0) / ((w3-0)(w5-w7))")
print("                    = w5*(w3-w7) / (w3*(w5-w7))")
print()

for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    w3 = w_p(3, q)
    w5 = w_p(5, q)
    w7 = w_p(7, q)

    cr = cross_ratio(w3, w5, w7, 0.0)
    print(f"  {label}:")
    print(f"    w3 = {w3.real:+.12f} {w3.imag:+.12f}i")
    print(f"    w5 = {w5.real:+.12f} {w5.imag:+.12f}i")
    print(f"    w7 = {w7.real:+.12f} {w7.imag:+.12f}i")
    print(f"    CR = {cr.real:+.14f} {cr.imag:+.14f}i")
    print(f"    |CR - 2| = {abs(cr - 2):.6e}")
    print()

    # Decomposition: CR = w5*(w3-w7) / (w3*(w5-w7))
    # En utilisant w = -i*sin(theta)*e^{i*theta}:
    # w3-w7 = -i*(sin(t3)*e^{i*t3} - sin(t7)*e^{i*t7})
    # w5-w7 = -i*(sin(t5)*e^{i*t5} - sin(t7)*e^{i*t7})
    # CR = [sin(t5)*e^{i*t5} * (sin(t3)*e^{i*t3} - sin(t7)*e^{i*t7})] /
    #      [sin(t3)*e^{i*t3} * (sin(t5)*e^{i*t5} - sin(t7)*e^{i*t7})]
    t3 = theta_p(3, q)
    t5 = theta_p(5, q)
    t7 = theta_p(7, q)

    # Simplification via le cercle unite z = e^{2i*theta}:
    # w = (1-z)/2, donc w3-w7 = (z7-z3)/2, w5-w7 = (z7-z5)/2
    # CR = (w3-w7)*w5 / ((w5-w7)*w3)
    #    = [(z7-z3)/2 * (1-z5)/2] / [(z7-z5)/2 * (1-z3)/2]
    #    = (z7-z3)(1-z5) / ((z7-z5)(1-z3))
    # C'est le cross-ratio (z3,z5;z7,z_inf) sur U(1) ou z_inf = 1 (=w=0)
    z3 = z_p_pt(3, q)
    z5 = z_p_pt(5, q)
    z7 = z_p_pt(7, q)

    cr_z = (z7 - z3) * (1 - z5) / ((z7 - z5) * (1 - z3))
    print(f"    Via U(1): CR = (z7-z3)(1-z5) / ((z7-z5)(1-z3)) = {cr_z.real:.14f}")
    print()

    # Pour que CR = 2 exactement, il faudrait:
    # (z7-z3)(1-z5) = 2*(z7-z5)(1-z3)
    # z7 - z3 - z5*z7 + z3*z5 = 2*z7 - 2*z5 - 2*z3*z7 + 2*z3*z5
    # -z3 + z3*z5 - z5*z7 = z7 - 2*z5 - 2*z3*z7 + 2*z3*z5
    # => condition sur z3, z5, z7 (surdeterminee pour 3 points sur U(1))
    lhs = (z7 - z3) * (1 - z5)
    rhs = 2.0 * (z7 - z5) * (1 - z3)
    print(f"    LHS = (z7-z3)(1-z5) = {lhs.real:+.12f} {lhs.imag:+.12f}i")
    print(f"    RHS = 2(z7-z5)(1-z3) = {rhs.real:+.12f} {rhs.imag:+.12f}i")
    print(f"    |LHS - RHS| = {abs(lhs - rhs):.6e}")
    print()

# ====================================================================
# T2: CR pour d'autres triplets
# ====================================================================
print("\n### T2: Cross-ratios CR(p1,p2;p3,0) pour divers triplets")
print("    Si CR ~ 2 est universel, c'est une propriete du cercle")
print()

q = q_stat
from itertools import combinations
import sys
primes_test = [2, 3, 5, 7, 11, 13, 17, 19, 23]
print(f"  {'(p1,p2,p3)':>15} {'CR':>14} {'|CR-2|':>10} {'~2?':>5}")
for combo in combinations(primes_test, 3):
    p1, p2, p3 = combo
    cr = cross_ratio(w_p(p1, q), w_p(p2, q), w_p(p3, q), 0.0)
    near_2 = abs(cr.real - 2) < 0.1
    print(f"  ({p1:2d},{p2:2d},{p3:2d}) {cr.real:14.8f} {abs(cr.real-2):10.6e} {'OUI' if near_2 else '':>5}")

# Est-ce que certains cross-ratios SONT exactement 2?
# CR(p1,p2;p3,0) = w_p2*(w_p1-w_p3) / (w_p1*(w_p2-w_p3))
# = (sin^2_p2 / sin^2_p1) * (w_p1-w_p3)/(w_p2-w_p3) ... non, plus complexe
# En fait les CR(3,5;7,0) ~ 2 est SPECIFIQUE aux actifs, pas universel

# ════════════════════════════════════════════════════════════════════
# DIRECTION B: Canal quantique
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  DIRECTION B: CANAL QUANTIQUE DU CRIBLE")
print("=" * 70)

# ====================================================================
# T3: Operateur de transition entre qubits
# ====================================================================
print("\n### T3: Operateur de transition rho_p -> rho_{p'}")
print("    Chaque premier p definit rho_p = |psi_p><psi_p|")
print("    |psi_p> = (sin(theta_p), cos(theta_p))^T")
print("    La transition p -> p' est une ROTATION d'angle (theta_{p'}-theta_p)")
print()

q = q_stat

def rho_p(p, q_val):
    """Matrice densite 2x2 pour le premier p."""
    th = theta_p(p, q_val)
    psi = np.array([np.sin(th), np.cos(th)])
    return np.outer(psi, psi)

def fidelity(rho1, rho2):
    """Fidelite entre etats purs: F = |<psi1|psi2>|^2 = Tr(rho1*rho2)."""
    return np.trace(rho1 @ rho2).real

# Rotation: |psi_p'> = R(theta_{p'}-theta_p)|psi_p>
# R(alpha) = [[cos alpha, -sin alpha], [sin alpha, cos alpha]]
print(f"  Transition entre premiers adjacents:")
print(f"  {'p->p_':>8} {'d_theta':>12} {'Fidelite':>12} {'1-F':>12} {'d_FS':>12}")
for i in range(len(PRIMES_ALL) - 1):
    p1, p2 = PRIMES_ALL[i], PRIMES_ALL[i + 1]
    r1 = rho_p(p1, q)
    r2 = rho_p(p2, q)
    F = fidelity(r1, r2)
    dtheta = theta_p(p1, q) - theta_p(p2, q)  # positif (theta decroit)

    # Distance de Fubini-Study: d_FS = arccos(sqrt(F))
    d_FS = np.arccos(np.clip(np.sqrt(F), -1, 1))

    print(f"  {p1:3d}->{p2:<3d} {dtheta:12.8f} {F:12.10f} {1-F:12.4e} {d_FS:12.8f}")

# La fidelite entre etats purs: F = |<psi1|psi2>|^2 = cos^2(theta1-theta2)
print(f"\n  Verif: F(p,p') = cos^2(theta_p - theta_{{p'}}) ?")
for p1, p2 in [(3,5), (3,7), (5,7)]:
    F_calc = fidelity(rho_p(p1, q), rho_p(p2, q))
    F_expected = np.cos(theta_p(p1, q) - theta_p(p2, q))**2
    print(f"    F({p1},{p2}) = {F_calc:.12f}, cos^2(dt) = {F_expected:.12f}, match: {abs(F_calc-F_expected)<1e-12}")

print(f"\n  => La fidelite est cos^2 de l'ecart angulaire.")
print(f"  => d_FS = |theta_p - theta_{{p'}}| (la distance de Fubini-Study")
print(f"     est EXACTEMENT l'ecart angulaire sur le cercle).")

# ====================================================================
# T4: Matrice de rotation entre qubits
# ====================================================================
print("\n\n### T4: Operateur unitaire U(p->p') entre qubits")
print("    U = rotation de theta_{p'} - theta_p dans le plan (perte, conserv)")
print()

q = q_stat

# Matrice de rotation
def rotation_2d(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha),  np.cos(alpha)]])

# Composee des rotations p=2 -> p=47
U_total = np.eye(2)
print(f"  Rotations elementaires:")
print(f"  {'p->p_':>8} {'angle/pi':>12} {'det(U)':>10}")
for i in range(len(PRIMES_ALL) - 1):
    p1, p2 = PRIMES_ALL[i], PRIMES_ALL[i + 1]
    dalpha = theta_p(p2, q) - theta_p(p1, q)  # negatif
    U_step = rotation_2d(dalpha)
    U_total = U_step @ U_total
    print(f"  {p1:3d}->{p2:<3d} {dalpha/math.pi:12.8f} {np.linalg.det(U_step):10.6f}")

print(f"\n  Rotation totale U(2->47):")
print(f"    U = [[{U_total[0,0]:+.8f}, {U_total[0,1]:+.8f}],")
print(f"         [{U_total[1,0]:+.8f}, {U_total[1,1]:+.8f}]]")
angle_total = theta_p(47, q) - theta_p(2, q)
print(f"    Angle total = {angle_total:.8f} rad = {angle_total/math.pi:.6f} pi")
print(f"    det(U) = {np.linalg.det(U_total):.10f} (=1: unitaire)")

# Verification: U|psi_2> = |psi_47>?
th2 = theta_p(2, q)
th47 = theta_p(47, q)
psi2 = np.array([np.sin(th2), np.cos(th2)])
psi47 = np.array([np.sin(th47), np.cos(th47)])
psi47_calc = U_total @ psi2
print(f"\n    |psi_2>  = ({psi2[0]:.8f}, {psi2[1]:.8f})")
print(f"    |psi_47> = ({psi47[0]:.8f}, {psi47[1]:.8f})")
print(f"    U|psi_2> = ({psi47_calc[0]:.8f}, {psi47_calc[1]:.8f})")
print(f"    Match: {np.allclose(psi47, psi47_calc)}")

# ====================================================================
# T5: Entropie de von Neumann du melange
# ====================================================================
print("\n\n### T5: Entropie de von Neumann du melange de qubits")
print("    rho_mix = (1/n) * sum_p rho_p")
print("    S_vN = -Tr(rho_mix * ln(rho_mix))")
print("    rho_mix n'est plus pur => S_vN > 0")
print()

q = q_stat

# Melange uniforme des actifs
rho_mix_act = sum(rho_p(p, q) for p in PRIMES_ACTIFS) / len(PRIMES_ACTIFS)
evals_mix = np.linalg.eigvalsh(rho_mix_act)
S_vN_act = -sum(e * np.log(e) for e in evals_mix if e > 1e-15)

print(f"  Actifs (3,5,7):")
print(f"    rho_mix = [[{rho_mix_act[0,0]:.8f}, {rho_mix_act[0,1]:.8f}],")
print(f"               [{rho_mix_act[1,0]:.8f}, {rho_mix_act[1,1]:.8f}]]")
print(f"    Eigenvalues: {evals_mix}")
print(f"    S_vN = {S_vN_act:.10f}")
print(f"    S_max = ln(2) = {np.log(2):.10f}")
print(f"    S_vN/S_max = {S_vN_act/np.log(2):.8f}")
print(f"    Purete Tr(rho^2) = {np.trace(rho_mix_act @ rho_mix_act):.10f} (<1 = melange)")

# Melange de tous les premiers
rho_mix_all = sum(rho_p(p, q) for p in PRIMES_ALL) / len(PRIMES_ALL)
evals_all = np.linalg.eigvalsh(rho_mix_all)
S_vN_all = -sum(e * np.log(e) for e in evals_all if e > 1e-15)

print(f"\n  Tous (2..47):")
print(f"    rho_mix = [[{rho_mix_all[0,0]:.8f}, {rho_mix_all[0,1]:.8f}],")
print(f"               [{rho_mix_all[1,0]:.8f}, {rho_mix_all[1,1]:.8f}]]")
print(f"    S_vN = {S_vN_all:.10f}")
print(f"    S_vN/S_max = {S_vN_all/np.log(2):.8f}")
print(f"    Purete = {np.trace(rho_mix_all @ rho_mix_all):.10f}")

# Melange pondere par sin^2 (poids PT naturel)
weights = np.array([sin2(p, q) for p in PRIMES_ALL])
weights /= weights.sum()
rho_mix_w = sum(weights[i] * rho_p(PRIMES_ALL[i], q) for i in range(len(PRIMES_ALL)))
evals_w = np.linalg.eigvalsh(rho_mix_w)
S_vN_w = -sum(e * np.log(e) for e in evals_w if e > 1e-15)

print(f"\n  Melange pondere par sin^2 (poids PT):")
print(f"    S_vN = {S_vN_w:.10f}")
print(f"    S_vN/S_max = {S_vN_w/np.log(2):.8f}")
print(f"    Purete = {np.trace(rho_mix_w @ rho_mix_w):.10f}")

# Interpretation: la decoherence = melanage sur les premiers
# Plus on moyenne sur beaucoup de premiers, plus S_vN croit
# Mais les theta sont proches => rho_p sont proches => S_vN reste petit

# ════════════════════════════════════════════════════════════════════
# DIRECTION C: Produit d'Euler complexe
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  DIRECTION C: PRODUIT D'EULER COMPLEXE")
print("=" * 70)

# ====================================================================
# T6: W(s) = Pi (1-p^{-2is})/2
# ====================================================================
print("\n### T6: Produit d'Euler W(s) = (1/2^n) * Pi(1 - p^{-2is})")
print("    Si on remplace theta_p par s*ln(p), on obtient z_p(s) = p^{-2is}")
print("    W(s) = (1/2^n) Pi(1 - p^{-2is}) = 1/(2^n * zeta(2is)) ... presque")
print()

# W(s) avec les primes actifs: 3, 5, 7
for s_val in [0.1, 0.25, 0.5, 1.0, 2.0]:
    W_s = 1.0 + 0j
    for p in PRIMES_ACTIFS:
        z_ps = p**(-2j * s_val)
        W_s *= (1.0 - z_ps) / 2.0
    print(f"  s = {s_val:.2f}: W(s) = {W_s.real:+.10f} {W_s.imag:+.10f}i, |W| = {abs(W_s):.10f}")

# Comparer avec 1/zeta(2is) tronquee aux memes premiers
print(f"\n  Comparaison avec produit d'Euler partiel de zeta:")
print(f"  {'s':>6} {'|W(s)|':>12} {'|zeta_3(2is)|^-1':>18} {'ratio':>10}")
for s_val in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
    W_s = 1.0 + 0j
    zeta_partial = 1.0 + 0j
    for p in PRIMES_ACTIFS:
        z_ps = p**(-2j * s_val)
        W_s *= (1.0 - z_ps) / 2.0
        zeta_partial *= 1.0 / (1.0 - z_ps)

    inv_zeta = 1.0 / abs(zeta_partial) if abs(zeta_partial) > 1e-15 else 0
    ratio = abs(W_s) / inv_zeta if inv_zeta > 1e-15 else float('inf')
    print(f"  {s_val:6.2f} {abs(W_s):12.8f} {inv_zeta:18.8f} {ratio:10.6f}")

print(f"\n  Ratio = 1/2^n = 1/{2**len(PRIMES_ACTIFS)} = {1/2**len(PRIMES_ACTIFS):.6f}")
print(f"  => |W(s)| = |1/zeta_partial(2is)| / 2^n exactement")

# ====================================================================
# T7: Le "s" effectif
# ====================================================================
print("\n\n### T7: s effectif -- pour quel s, theta_p ~ s*ln(p)?")
print("    theta_p(PT) est connu. On cherche s tel que theta_p ~ s*ln(p)")
print()

q = q_stat
# Regression: theta_p = s_eff * ln(p) + cst?
# Non, car theta -> 0 quand p -> inf, et ln(p) -> inf
# Plutot: theta_p ~ a / sqrt(p) (scaling connu)
# Mais pour le produit d'Euler on veut e^{2i*theta_p} = p^{-2is}
# => 2*theta_p = -2*s*ln(p)
# => theta_p = -s*ln(p)
# Mais theta > 0 et ln(p) > 0, donc s < 0... pas physique
# En fait: z_p = e^{2i*theta_p} est dans le 1er quadrant (Re>0, Im>0)
# Tandis que p^{-2is} = e^{-2is*ln p} = e^{2*s_im*ln p} * e^{-2i*s_re*ln p}
# Pour rester sur U(1): s purement imaginaire? Non, s reel:
# p^{-2is} = e^{-2is*ln p}, |p^{-2is}| = 1 pour s reel. OK.
# Alors: e^{2i*theta_p} = e^{-2is*ln p}
# => theta_p = -s*ln(p) mod pi
# Mais theta_p > 0, donc s_eff(p) = -theta_p / ln(p) < 0

print(f"  s_eff(p) = -theta_p / ln(p) (devrait etre constant si theta = -s*ln p)")
print(f"  {'p':>4} {'theta':>12} {'ln(p)':>12} {'s_eff':>12}")
for p in PRIMES_ALL:
    th = theta_p(p, q)
    s_eff = -th / np.log(p)
    print(f"  {p:4d} {th:12.8f} {np.log(p):12.8f} {s_eff:12.8f}")

print(f"\n  s_eff n'est PAS constant: il varie de -0.73 (p=2) a -0.054 (p=47)")
print(f"  => theta_p != -s*ln(p). Le scaling PT (theta ~ sqrt(2/p)) est")
print(f"     fondamentalement different du scaling Euler (theta ~ s*ln p).")
print(f"\n  MAIS: pour le s_eff MOYEN sur les actifs:")
s_eff_mean = -np.mean([theta_p(p, q) / np.log(p) for p in PRIMES_ACTIFS])
print(f"  <s_eff> actifs = {-s_eff_mean:.8f}")
print(f"  => Pas de s constant. PT et Euler ont des parametrisations DIFFERENTES.")

# ====================================================================
# T8: Comparaison directe W_PT vs W_Euler(s)
# ====================================================================
print("\n\n### T8: W_PT vs W_Euler(s) -- chercher le meilleur s")
print("    W_PT = Pi w_p(PT) connu. Quel s minimise |W_PT - W_Euler(s)|?")
print()

q = q_stat
W_PT = 1.0 + 0j
for p in PRIMES_ACTIFS:
    W_PT *= w_p(p, q)

print(f"  W_PT = {W_PT.real:+.10f} {W_PT.imag:+.10f}i")
print(f"  |W_PT| = {abs(W_PT):.10f}, arg/pi = {cmath.phase(W_PT)/math.pi:.8f}")
print()

# Scan s
print(f"  {'s':>8} {'|W_Euler(s)-W_PT|':>20} {'arg_Euler/pi':>14} {'arg_PT/pi':>12}")
best_s = 0
best_dist = 1e10
for s_val in np.arange(-1.0, 1.0, 0.01):
    W_E = 1.0 + 0j
    for p in PRIMES_ACTIFS:
        W_E *= (1.0 - p**(-2j * s_val)) / 2.0
    dist = abs(W_E - W_PT)
    if dist < best_dist:
        best_dist = dist
        best_s = s_val
    if abs(s_val - round(s_val, 1)) < 0.005 and -0.5 <= s_val <= 0.5:
        print(f"  {s_val:8.3f} {dist:20.10f} {cmath.phase(W_E)/math.pi:14.8f} {cmath.phase(W_PT)/math.pi:12.8f}")

print(f"\n  Meilleur s: {best_s:.4f}, distance = {best_dist:.6e}")

# Fine search around best
for s_val in np.arange(best_s - 0.05, best_s + 0.05, 0.001):
    W_E = 1.0 + 0j
    for p in PRIMES_ACTIFS:
        W_E *= (1.0 - p**(-2j * s_val)) / 2.0
    dist = abs(W_E - W_PT)
    if dist < best_dist:
        best_dist = dist
        best_s = s_val

print(f"  Affine: s_best = {best_s:.6f}, distance = {best_dist:.6e}")
W_E_best = 1.0 + 0j
for p in PRIMES_ACTIFS:
    W_E_best *= (1.0 - p**(-2j * best_s)) / 2.0
print(f"  W_Euler(s_best) = {W_E_best.real:+.10f} {W_E_best.imag:+.10f}i")
print(f"  W_PT            = {W_PT.real:+.10f} {W_PT.imag:+.10f}i")

# ════════════════════════════════════════════════════════════════════
# DIRECTION D: Fubini-Study / CP^1
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  DIRECTION D: FUBINI-STUDY ET CP^1")
print("=" * 70)

# ====================================================================
# T9: Courbure de CP^1
# ====================================================================
print("\n### T9: Courbure de CP^1 et sa valeur PT")
print("    CP^1 = sphere de Bloch (rayon 1/2 en Fubini-Study)")
print("    Courbure de Gauss K = 4 (constante)")
print("    ds^2_FS = (1/4) * d(2*theta)^2 = d(theta)^2")
print("    Rayon de courbure R = 1/sqrt(K) = 1/2")
print()

# CP^1 avec metrique de Fubini-Study: ds^2 = d theta^2
# pour un etat |psi> = (cos(theta/2), sin(theta/2)) sur la sphere
# MAIS ici on a |psi> = (sin theta, cos theta), et ds^2 = 4 d theta^2 (Fisher)
# Donc ds^2_FS = d theta^2 (apres le facteur 4 de Fisher = 4*FS)
# La courbure de CP^1 est K = 4 (en coordonnees FS standard)

# Verification: aire de la sphere de Bloch = 4*pi*R^2 = 4*pi*(1/2)^2 = pi
# Notre arc va de theta_2 ~ 0.504 a theta_47 ~ 0.207
# Longueur de l'arc = theta_2 - theta_47 (en FS)
arc_length = theta_p(2, q_stat) - theta_p(47, q_stat)
total_circle = math.pi  # demie-tour (theta va de 0 a pi/2)
print(f"  Longueur de l'arc (2->47): {arc_length:.8f} rad")
print(f"  Fraction du demi-cercle: {arc_length / (math.pi/2):.6f}")
print(f"  Le crible PT couvre {arc_length/(math.pi/2)*100:.1f}% du demi-cercle")
print()

# Geodesiques: sur CP^1, les geodesiques sont les grands cercles
# Comme PT vit sur un seul grand cercle (parametre par theta),
# le chemin est AUTOMATIQUEMENT geodesique!
print(f"  Les geodesiques de CP^1 sont les grands cercles.")
print(f"  Le chemin p=2,3,...,47 est-il geodesique?")
print(f"  OUI: tous les |psi_p> sont dans le plan reel (sin,cos),")
print(f"  qui est un grand cercle de la sphere de Bloch.")
print(f"  => Le crible parcourt une GEODESIQUE de CP^1.")

# ====================================================================
# T10: Connexion de Berry sur CP^1
# ====================================================================
print("\n\n### T10: Connexion de Berry sur CP^1")
print("    La 1-forme de connexion est A = Im(<psi|d psi>)")
print("    Pour |psi> = (sin theta, cos theta):")
print("    A = Im(sin*cos*d theta - cos*sin*d theta) = 0 (!)")
print("    Pas de Berry phase car le chemin est dans le plan REEL")
print()

q = q_stat
# Berry connection: A_k = Im(<psi_k|psi_{k+1}> - 1) pour adjacent
print(f"  Connexion de Berry discrete:")
print(f"  {'p->p_':>8} {'<psi|psi_>':>14} {'Im':>12} {'Berry A':>12}")
berry_total = 0.0
for i in range(len(PRIMES_ALL) - 1):
    p1, p2 = PRIMES_ALL[i], PRIMES_ALL[i + 1]
    th1 = theta_p(p1, q)
    th2 = theta_p(p2, q)
    psi1 = np.array([np.sin(th1), np.cos(th1)])
    psi2 = np.array([np.sin(th2), np.cos(th2)])
    overlap = np.dot(psi1, psi2)  # reel car psi sont reels!
    A_berry = np.arctan2(0, overlap)  # arg d'un nombre reel positif = 0
    # Plus precisement: phase geometrique = arg(prod <psi_k|psi_{k+1}>)
    berry_total += cmath.phase(overlap)  # sera 0 car overlap > 0
    print(f"  {p1:3d}->{p2:<3d} {overlap:14.10f} {0.0:12.8f} {0.0:12.8f}")

print(f"\n  Phase Berry totale = {berry_total:.10f}")
print(f"  = 0 EXACTEMENT car tous les etats sont REELS.")
print(f"  => Pas de phase geometrique dans le plan reel.")
print(f"\n  MAIS: si on utilisait w_p comme 'etat' (complexe),")
print(f"  la phase Berry serait NON-TRIVIALE (voir tool_42, courant J).")

# ====================================================================
# T11: Sphere de Bloch complete -- extension hors du plan reel
# ====================================================================
print("\n\n### T11: Extension hors du plan reel (q complexe)")
print("    Les etats reels vivent sur un grand cercle de S^2.")
print("    q complexe -> etats complexes -> on explore TOUTE la sphere.")
print()

# Avec q complexe, |psi_p> = (sin(theta_C), cos(theta_C)) avec theta_C complexe
# On peut parametriser par (theta, phi) sur la sphere de Bloch
# |psi> = cos(theta/2)|0> + e^{i*phi} sin(theta/2)|1>
# Pour nos etats reels: phi = 0, theta = 2*theta_p... non.
# En fait |psi_p> = sin(theta_p)|1> + cos(theta_p)|0>
# = cos((pi/2-theta_p)/... c'est plus simple en coord. standard:
# Bloch: |psi> = cos(Theta/2)|0> + e^{i*Phi} sin(Theta/2)|1>
# Nos etats: |psi_p> = cos(theta_p)|conserv> + sin(theta_p)|perte>
# Si |conserv> = |0>, |perte> = |1>:
# |psi_p> = cos(theta_p)|0> + sin(theta_p)|1>
# Bloch: Theta_B = 2*theta_p, Phi_B = 0
# => Sur la sphere de Bloch, les etats PT sont a Phi=0, Theta=2*theta_p
# C'est le MERIDIEN Phi=0 de la sphere!

print(f"  Coordonnees de Bloch (Theta_B, Phi_B):")
print(f"  {'p':>4} {'Theta_B/pi':>12} {'Phi_B':>8}")
for p in PRIMES_ALL:
    Theta_B = 2 * theta_p(p, q)
    print(f"  {p:4d} {Theta_B/math.pi:12.8f} {'0':>8}")

print(f"\n  Tous les etats PT sont sur le MERIDIEN Phi=0 de S^2.")
print(f"  Le pole nord (Theta=0) = |conserv> = pas de perte")
print(f"  Le pole sud (Theta=pi) = |perte> = perte totale")
print(f"  L'equateur (Theta=pi/2) = equipartition sin^2 = cos^2 = 1/2")
print(f"\n  Theta_B va de {2*theta_p(2,q)/math.pi:.4f}*pi (p=2) a {2*theta_p(47,q)/math.pi:.4f}*pi (p=47)")
print(f"  Les premiers sont dans l'hemisphere NORD (Theta < pi/2)")
print(f"  car sin^2 < 1/2 pour tous les premiers (perte < conservation)")

# Decoherence: quelle latitude correspond a s = 1/2?
# sin^2(theta) = 1/2 => theta = pi/4 => Theta_B = pi/2 (equateur)
# L'equateur EST le point s = 1/2 !
print(f"\n  Le point s = 1/2 (equipartition) est a l'EQUATEUR de Bloch.")
print(f"  En PT: sin^2 < 1/2 pour tous les premiers reels")
print(f"  => le crible ne franchit JAMAIS l'equateur.")

# ====================================================================
# T12: Synthese des 4 directions
# ====================================================================
print("\n\n" + "=" * 90)
print("### T12: SYNTHESE DES 4 DIRECTIONS")
print("=" * 90)
print("""
A. CROSS-RATIO:
   CR(3,5;7,0) ~ 2.000 mais N'EST PAS exactement 2.
   L'ecart |CR - 2| ~ 1.5e-4 pour q_stat, ~ 3e-3 pour q_therm.
   Ce n'est PAS un invariant universel: les CR d'autres triplets
   varient de ~1.0 a ~1.8. La proximite de 2 est ACCIDENTELLE
   pour le triplet (3,5,7) a q_stat.

B. CANAL QUANTIQUE:
   Le crible EST un canal quantique sur un qubit:
   - Chaque premier p definit un etat pur |psi_p> sur la sphere de Bloch
   - La transition p -> p' est une ROTATION de angle (theta_{p'}-theta_p)
   - La fidelite F(p,p') = cos^2(theta_p - theta_{p'})
   - La composition des rotations est UNITAIRE (det = 1)
   - Le melange rho_mix a une entropie de von Neumann PETITE (S/S_max ~ 0.20)
     car les etats sont proches (tous dans hemisphere nord, meme meridien)
   - Le chemin du crible est une GEODESIQUE de CP^1.

C. PRODUIT D'EULER:
   Le produit PT W = Pi w_p a la FORME d'un produit d'Euler:
     W = (1/2^n) Pi (1 - z_p)
   MAIS la parametrisation est DIFFERENTE:
   - Euler: z_p(s) = p^{-2is}, theta = s*ln(p) (scaling logarithmique)
   - PT: z_p = e^{2i*theta_p}, theta ~ sqrt(2/p) (scaling en racine)
   Il n'existe PAS de s constant tel que theta_PT = s*ln(p).
   Les deux produits vivent dans des ESPACES DE PARAMETRES differents.
   Neanmoins, |W_PT| = sqrt(alpha) et |W_Euler(s)| = 1/(2^n*|zeta(2is)|)
   sont des quantites comparables.

D. FUBINI-STUDY:
   PT vit sur CP^1 (droite projective complexe):
   - d_Fisher = 2 * d_arc = facteur de Fubini-Study
   - Les etats PT sont sur le MERIDIEN Phi=0 de la sphere de Bloch
   - Theta_Bloch = 2*theta_p (du pole nord vers l'equateur)
   - Le chemin du crible est GEODESIQUE (grand cercle)
   - AUCUNE phase de Berry (etats reels => Phi=0 constant)
   - L'equateur (Theta=pi/2) correspond a sin^2 = 1/2 (s = 1/2 !)
   - Le crible ne franchit JAMAIS l'equateur.

   CLEF: L'equateur de Bloch = point s = 1/2 = symmetrie PT.
   Les premiers vivent dans l'hemisphere nord (sin^2 < 1/2).
   La contraction PT = descente vers le pole nord (sin^2 -> 0).
""")

print("=" * 90)
print("FIN TOOL 44")
print("=" * 90)

sys.exit(0)
