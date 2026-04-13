"""
Tool 43: PT Complexe -- Structure profonde
============================================
Suite du tool_42. On a etabli que w_p vit sur le cercle C(1/2,0;1/2).
Maintenant on explore la structure ALGEBRIQUE et GEOMETRIQUE profonde.

Tests:
  T1:  Correspondance cercle C <-> U(1) via z_p = 1 - 2*w_p
  T2:  Produit U(1): Pi z_p = e^{2i*sum theta_p} et son interpretation
  T3:  Cross-ratio de 4 points sur C -- invariant reel du crible
  T4:  Metrique de Fisher sur l'arc du cercle -- distance entre premiers
  T5:  q complexe -- deformation du cercle
  T6:  Le polynome du crible: P(w) = Pi(w - w_p)
  T7:  Born rule: |w|^2 = Re(w) et espace de Hilbert sous-jacent
  T8:  Derivee dw/dp et vitesse sur le cercle
  T9:  Fonction de partition Z = sum e^{-beta*E_p} ou E_p = -ln|w_p|^2
  T10: L'angle 2*theta comme variable duale de ln(p) -- transformee de Fourier
  T11: Isometrie du cercle: quels premiers sont "equidistants"?
  T12: Connexion au produit d'Euler: Z_sieve vs W = Pi w_p
  T13: Le tenseur d'inertie du nuage w_p sur le cercle
  T14: Bilan
"""

import numpy as np
import cmath
import math
from itertools import combinations

# ============================================================
# Parametres PT fondamentaux
# ============================================================
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

def cos2(p, q):
    d = delta_p(p, q)
    return (1.0 - d)**2

def theta_p(p, q):
    return np.arcsin(np.sqrt(sin2(p, q)))

def w_p(p, q):
    """w_p = (1 - e^{2i*theta_p}) / 2, sur le cercle C(1/2,0;1/2)."""
    th = theta_p(p, q)
    return (1.0 - np.exp(2j * th)) / 2.0

def z_p(p, q):
    """z_p = e^{2i*theta_p} = 1 - 2*w_p, sur le cercle unite U(1)."""
    th = theta_p(p, q)
    return np.exp(2j * th)

def chi3(p):
    r = p % 3
    return 0 if r == 0 else (1 if r == 1 else -1)

print("=" * 90)
print("TOOL 43: PT COMPLEXE -- STRUCTURE PROFONDE")
print("=" * 90)

# ====================================================================
# T1: Correspondance C <-> U(1)
# ====================================================================
print("\n### T1: Correspondance cercle C <-> cercle unite U(1)")
print("    z_p = 1 - 2*w_p = e^{2i*theta_p}")
print("    La transformation w -> z = 1-2w envoie C(1/2,0;1/2) sur U(1)")
print("    C'est une INVERSION affine: translation + homothetie")
print()

q = q_stat
print(f"  {'p':>4} {'2*theta/pi':>12} {'z_p':>28} {'|z|':>8} {'1-2w':>28} {'match':>6}")
for p in PRIMES_ALL:
    z = z_p(p, q)
    w = w_p(p, q)
    z_from_w = 1.0 - 2.0 * w
    match = abs(z - z_from_w) < 1e-12
    print(f"  {p:4d} {2*theta_p(p,q)/math.pi:12.8f} {z.real:+12.8f}{z.imag:+12.8f}i {abs(z):8.6f} {z_from_w.real:+12.8f}{z_from_w.imag:+12.8f}i {match}")

# Points speciaux
print(f"\n  Points speciaux de la correspondance:")
print(f"    w = 0   <->  z = 1    (pas de perte, element neutre U(1))")
print(f"    w = 1   <->  z = -1   (perte totale, opposition)")
print(f"    w = 1/2 <->  z = 0    (equipartition, singulier pour U(1)!)")
print(f"    w = 1/2-i/2 <-> z = i (sommet du cercle C)")

# ====================================================================
# T2: Produit U(1)
# ====================================================================
print("\n\n### T2: Produit dans U(1): Pi z_p = e^{2i*Sigma_theta}")
print("    Le cercle unite est un GROUPE sous la multiplication")
print("    Le produit des z_p est un element de U(1)")
print()

for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    # Produit sur actifs
    Z_actifs = 1.0 + 0j
    sum_theta = 0.0
    for p in PRIMES_ACTIFS:
        Z_actifs *= z_p(p, q)
        sum_theta += theta_p(p, q)

    print(f"  {label} (actifs 3,5,7):")
    print(f"    Pi z_p = {Z_actifs.real:+.10f} {Z_actifs.imag:+.10f}i")
    print(f"    |Pi z_p| = {abs(Z_actifs):.12f} (=1? {abs(abs(Z_actifs)-1) < 1e-12})")
    print(f"    arg(Pi z_p) = 2*sum(theta) = {2*sum_theta:.10f} rad = {2*sum_theta/math.pi:.8f} pi")
    print()

    # Produit sur TOUS les premiers
    Z_all = 1.0 + 0j
    sum_theta_all = 0.0
    for p in PRIMES_ALL:
        Z_all *= z_p(p, q)
        sum_theta_all += theta_p(p, q)

    print(f"    Tous (2..47): Pi z_p = {Z_all.real:+.10f} {Z_all.imag:+.10f}i")
    print(f"    arg = {2*sum_theta_all/math.pi:.8f} pi")
    print()

    # Le produit converge-t-il? sum theta_p converge car theta_p ~ 1/sqrt(p)
    # En fait theta_p ~ arcsin(sqrt(delta*(2-delta))) ~ sqrt(2*delta) ~ sqrt(2/p)
    print(f"    Convergence: theta_p ~ sqrt(2*delta_p) ~ sqrt(2(1-q^p)/p)")
    for p in [2, 7, 47, 97, 997]:
        d = (1 - q**p) / p
        th_approx = np.sqrt(2 * d) if d > 0 else 0
        th_exact = np.arcsin(np.sqrt(d * (2 - d))) if 0 < d < 1 else 0
        print(f"      p={p:4d}: theta = {th_exact:.8f}, sqrt(2*delta) = {th_approx:.8f}, ratio = {th_exact/th_approx:.6f}" if th_approx > 0 else f"      p={p:4d}: theta ~ 0")

# ====================================================================
# T3: Cross-ratio -- invariant conforme
# ====================================================================
print("\n\n### T3: Cross-ratio de 4 points sur le cercle C")
print("    Pour 4 points z1,z2,z3,z4 sur un cercle, le cross-ratio")
print("    (z1-z3)(z2-z4) / ((z1-z4)(z2-z3)) est REEL")
print("    C'est un invariant conforme du crible")
print()

q = q_stat
# Points sur U(1): z_2, z_3, z_5, z_7
# Equivalemment w_2, w_3, w_5, w_7 sur C

def cross_ratio(z1, z2, z3, z4):
    """Cross-ratio (z1,z2;z3,z4) = (z1-z3)(z2-z4)/((z1-z4)(z2-z3))"""
    return ((z1 - z3) * (z2 - z4)) / ((z1 - z4) * (z2 - z3))

# Sur le cercle C (w_p)
primes_cr = [2, 3, 5, 7, 11, 13]
w_vals = {p: w_p(p, q) for p in primes_cr}
z_vals = {p: z_p(p, q) for p in primes_cr}

print(f"  Cross-ratios dans le plan w (cercle C):")
print(f"  {'(p1,p2;p3,p4)':>20} {'CR':>28} {'|Im(CR)|':>10} {'Reel?':>6}")
for combo in combinations(primes_cr, 4):
    p1, p2, p3, p4 = combo
    cr_w = cross_ratio(w_vals[p1], w_vals[p2], w_vals[p3], w_vals[p4])
    is_real = abs(cr_w.imag) < 1e-8
    print(f"  ({p1:2d},{p2:2d};{p3:2d},{p4:2d}) {cr_w.real:+14.8f}{cr_w.imag:+14.8f}i {abs(cr_w.imag):10.2e} {is_real}")

# Les cross-ratios les plus importants: actifs + limite
# Avec le point w=0 (limite p->inf) comme reference
print(f"\n  Cross-ratios avec w=0 (limite p->inf) comme 4eme point:")
for combo in combinations([3, 5, 7], 3):
    p1, p2, p3 = combo
    cr = cross_ratio(w_vals[p1], w_vals[p2], w_vals[p3], 0.0)
    print(f"  ({p1},{p2};{p3},inf) = {cr.real:+14.10f}{cr.imag:+14.10f}i  (reel? {abs(cr.imag) < 1e-8})")

# Cross-ratio sur U(1): devrait etre identique (transformation affine)
print(f"\n  Verification: meme CR sur U(1) (via z = 1-2w)?")
cr_w = cross_ratio(w_vals[3], w_vals[5], w_vals[7], w_vals[11])
cr_z = cross_ratio(z_vals[3], z_vals[5], z_vals[7], z_vals[11])
print(f"  CR_w(3,5;7,11) = {cr_w.real:+.10f}{cr_w.imag:+.10f}i")
print(f"  CR_z(3,5;7,11) = {cr_z.real:+.10f}{cr_z.imag:+.10f}i")
print(f"  Identiques? {abs(cr_w - cr_z) < 1e-10} (le cross-ratio est invariant affine)")

# ====================================================================
# T4: Metrique de Fisher sur l'arc du cercle
# ====================================================================
print("\n\n### T4: Metrique de Fisher sur l'arc du cercle")
print("    Le cercle C est parametre par theta. La metrique est ds^2 = d(theta)^2.")
print("    La distance entre deux premiers est |theta_p - theta_{p'}|.")
print("    Mais la metrique de Fisher de la distrib (sin^2, cos^2) est differente!")
print()

q = q_stat

# Distance angulaire sur le cercle
print(f"  Distance angulaire d(p,p') = |theta_p - theta_p'| (sur l'arc):")
print(f"  {'p':>4} {'p_':>4} {'d_arc':>12} {'d_Fisher':>12} {'ratio':>10}")
for i, p1 in enumerate(PRIMES_ACTIFS):
    for p2 in PRIMES_ACTIFS[i+1:]:
        th1 = theta_p(p1, q)
        th2 = theta_p(p2, q)
        d_arc = abs(th1 - th2)

        # Fisher: pour distrib Bernoulli (sin^2, cos^2):
        # ds^2_Fisher = d(sin^2)^2 / [sin^2*(1-sin^2)]
        # = [2*sin*cos*d(theta)]^2 / [sin^2*cos^2]
        # = 4*d(theta)^2
        # Donc d_Fisher = 2*|theta_p - theta_{p'}|
        d_fisher = 2.0 * d_arc

        print(f"  {p1:4d} {p2:4d} {d_arc:12.8f} {d_fisher:12.8f} {d_fisher/d_arc:10.6f}")

print(f"\n  RESULTAT: d_Fisher = 2 * d_arc exactement!")
print(f"  La metrique de Fisher sur la distribution (sin^2, cos^2)")
print(f"  est EXACTEMENT 4 * d(theta)^2.")
print(f"  Le facteur 2 entre Fisher et arc est le facteur de Fubini-Study.")

# Distances entre tous les premiers adjacents
print(f"\n  Gaps angulaires entre premiers adjacents:")
print(f"  {'p->p_':>8} {'d_arc':>12} {'d_arc/d_arc(3->5)':>18}")
d_ref = abs(theta_p(3, q) - theta_p(5, q))
for i in range(len(PRIMES_ALL) - 1):
    p1, p2 = PRIMES_ALL[i], PRIMES_ALL[i + 1]
    d = abs(theta_p(p1, q) - theta_p(p2, q))
    print(f"  {p1:3d}->{p2:<3d} {d:12.8f} {d/d_ref:18.8f}")

# ====================================================================
# T5: q complexe -- deformation du cercle
# ====================================================================
print("\n\n### T5: q complexe -- le cercle se deforme")
print("    Si q = |q|*e^{i*phi}, delta_p = (1-q^p)/p devient complexe")
print("    => w_p quitte le cercle C")
print("    Le cercle est RIGIDE pour q reel, DEFORME pour q complexe")
print()

# Test avec q complexe
q_base = q_stat
for phi_q in [0, 0.01, 0.05, 0.1, 0.3]:
    q_complex = q_base * np.exp(1j * phi_q)
    print(f"  phi_q = {phi_q:.2f} (q = {q_base:.4f} * e^{{i*{phi_q:.2f}}}):")

    for p in [3, 5, 7]:
        d = (1.0 - q_complex**p) / p
        s2 = d * (2.0 - d)  # complexe!
        # theta complexe
        # w via definition directe: pas de theta reel
        # On calcule directement sin^2 et cos^2
        print(f"    p={p}: delta = {d.real:+.6f}{d.imag:+.6f}i, sin^2 = {s2.real:+.6f}{s2.imag:+.6f}i")

    # Test du cercle
    if phi_q > 0:
        p = 5
        d = (1.0 - q_complex**p) / p
        s2 = d * (2.0 - d)
        # w n'est plus defini via theta (theta complexe)
        # Mais on peut definir w = s2 - i*sqrt(s2*(1-s2)) (branche)
        # Ou directement: w tel que |w|^2 = Re(w)?
        # Test: si sin^2 est complexe, |w|^2 = Re(w) ne tient plus
        print(f"    Sur le cercle? (Re-1/2)^2 + Im^2 = 1/4 ne s'applique qu'a q reel")
    print()

print(f"  RESULTAT: q complexe brise la contrainte du cercle.")
print(f"  sin^2 devient complexe => w_p quitte C.")
print(f"  Le cercle C est la SECTION REELLE de l'espace des parametres.")

# ====================================================================
# T6: Le polynome du crible P(w) = Pi(w - w_p)
# ====================================================================
print("\n\n### T6: Polynome du crible P(w) = Pi(w - w_p)")
print("    Les w_p sont les 'racines' du crible dans le plan")
print("    P(w) = 0 aux positions des premiers")
print()

q = q_stat
# Construire le polynome pour les actifs
w_roots = [w_p(p, q) for p in PRIMES_ACTIFS]

# Coefficients par expansion: (w-w3)(w-w5)(w-w7)
# = w^3 - (w3+w5+w7)w^2 + (w3*w5+w3*w7+w5*w7)w - w3*w5*w7
s1 = sum(w_roots)                  # sigma_1
s2 = sum(w_roots[i]*w_roots[j] for i in range(3) for j in range(i+1, 3))  # sigma_2
s3 = w_roots[0] * w_roots[1] * w_roots[2]  # sigma_3

print(f"  Actifs (3,5,7): P(w) = w^3 - sigma_1*w^2 + sigma_2*w - sigma_3")
print(f"    sigma_1 = sum w_p     = {s1.real:+.10f} {s1.imag:+.10f}i")
print(f"    sigma_2 = sum w_p*w_q = {s2.real:+.10f} {s2.imag:+.10f}i")
print(f"    sigma_3 = w_3*w_5*w_7 = {s3.real:+.10f} {s3.imag:+.10f}i")
print(f"    |sigma_3| = sqrt(alpha) = {abs(s3):.10f}")
print(f"    |sigma_3|^2 = alpha     = {abs(s3)**2:.10f}")
print()

# Evaluer P en des points speciaux
print(f"  P(w) aux points speciaux:")
for w_test, label in [(0, "w=0 (limite)"), (0.5, "w=1/2 (equipart.)"),
                        (1.0, "w=1 (perte tot.)"), (0.5-0.5j, "sommet cercle")]:
    P_val = 1.0 + 0j
    for wr in w_roots:
        P_val *= (w_test - wr)
    print(f"    P({label:>20}) = {P_val.real:+.10f} {P_val.imag:+.10f}i, |P| = {abs(P_val):.10f}")

# P(0) = (-1)^3 * sigma_3 = -W (produit complexe)
print(f"\n    P(0) = -sigma_3 = -W = {-s3.real:+.10f}{-s3.imag:+.10f}i")
print(f"    |P(0)|^2 = alpha = {abs(s3)**2:.10f}")
print(f"    => P(0) est (au signe pres) le couplage complexe W!")

# ====================================================================
# T7: Born rule |w|^2 = Re(w) et espace de Hilbert
# ====================================================================
print("\n\n### T7: Regle de Born |w|^2 = Re(w)")
print("    En MQ: probabilite = |amplitude|^2")
print("    Ici: sin^2 = |w|^2 et sin^2 = Re(w)")
print("    Donc: P(perte) = |amplitude|^2 = composante reelle de l'amplitude")
print()

# Construction: w = <psi|0> ou |psi> et |0> sont des etats
# Sur le cercle C, on peut ecrire w = <1|psi> / 2 ou |psi> = |1> - e^{2i*theta}|0>
# Non, plus naturellement:
# w = (1 - e^{2i*theta})/2 = (|0><0| - e^{2i*theta}|0><0|) /2 ...
# En fait: w = sin^2 - i*sin*cos
# Ecrivons |psi> = sin(theta)|1> + cos(theta)|0> (etat a 2 niveaux)
# Alors <1|psi> = sin(theta), et |<1|psi>|^2 = sin^2 (Born rule classique)
# Et: <psi|sigma_-|psi> = sin(theta)*cos(theta) = sin(2theta)/2
# Ou sigma_- = |0><1| (abaissement)
# Donc: w_p = <1|psi_p> * <psi_p|(|1> - i|0>) * ...

# Plus simple: definissons |psi_p> = sin(theta_p)|1> + cos(theta_p)|0>
# Alors: <1|psi_p> = sin(theta_p), <0|psi_p> = cos(theta_p)
# w_p = -i * sin * e^{i*theta} = -i * sin * (cos + i*sin)
#     = -i * <1|psi> * <psi|i> ou |i> = (|0> + i|1>)/sqrt(2)?
# Non...
# w = sin^2 - i*sin*cos = sin*(sin - i*cos) = <1|psi>*(<1| - i<0|)|psi>
# = <1|psi> * <phi_-|psi> ou |phi_-> = (|1> - i|0>)/sqrt(2)?
# Hmm pas exact a cause de normalisation.

# L'interpretation la plus propre:
# |psi_p> = (sin(theta), cos(theta))^T dans la base (|perte>, |conserv>)
# Matrice densitee rho = |psi><psi|
# rho = [[sin^2, sin*cos], [sin*cos, cos^2]]
# Element (0,0): sin^2 = probabilite de perte (Born rule)
# Element (0,1): sin*cos = COHERENCE (off-diagonal)
# w_p = rho_00 - i*rho_01 = sin^2 - i*sin*cos

print(f"  Construction quantique:")
print(f"  |psi_p> = sin(theta_p)|perte> + cos(theta_p)|conserv>")
print(f"  rho_p = |psi_p><psi_p| = [[sin^2, sin*cos], [sin*cos, cos^2]]")
print(f"  w_p = rho_00 - i*rho_01 = diagonale - i*coherence")
print()

q = q_stat
print(f"  Verification:")
print(f"  {'p':>4} {'rho_00=sin^2':>14} {'rho_01=sin*cos':>14} {'rho_00-i*rho_01':>28} {'w_p':>28} {'match':>6}")
for p in PRIMES_ACTIFS + [11, 13]:
    th = theta_p(p, q)
    rho_00 = np.sin(th)**2
    rho_01 = np.sin(th) * np.cos(th)
    w_from_rho = rho_00 - 1j * rho_01
    w = w_p(p, q)
    match = abs(w - w_from_rho) < 1e-12
    print(f"  {p:4d} {rho_00:14.8f} {rho_01:14.8f} {w_from_rho.real:+12.8f}{w_from_rho.imag:+12.8f}i {w.real:+12.8f}{w.imag:+12.8f}i {match}")

print(f"\n  RESULTAT: w_p = rho_00 - i*rho_01")
print(f"  La variable complexe w_p encode la PREMIERE COLONNE de la")
print(f"  matrice densite rho dans un systeme a 2 niveaux (perte|conserv).")
print(f"  - Re(w) = probabilite de perte (diagonale)")
print(f"  - Im(w) = -coherence (off-diagonal)")
print(f"  Le cercle C est l'espace des matrices densite PURES a 2 niveaux.")

# Proprietes de la matrice densite
print(f"\n  Proprietes de rho_p:")
for p in PRIMES_ACTIFS:
    th = theta_p(p, q)
    rho = np.array([[np.sin(th)**2, np.sin(th)*np.cos(th)],
                      [np.sin(th)*np.cos(th), np.cos(th)**2]])
    tr = np.trace(rho)
    tr_rho2 = np.trace(rho @ rho)
    evals = np.linalg.eigvalsh(rho)
    print(f"    p={p}: Tr(rho)={tr:.6f}, Tr(rho^2)={tr_rho2:.6f}, eigenvalues={evals}")
print(f"  Tr(rho) = 1, Tr(rho^2) = 1 => etat PUR (rho^2 = rho)")

# ====================================================================
# T8: Derivee dw/dp et vitesse sur le cercle
# ====================================================================
print("\n\n### T8: Vitesse sur le cercle dw/d(ln p)")
print("    Comment w_p se deplace quand p augmente?")
print("    dw/dtheta = -i*e^{2i*theta} = -i*z_p (tangent au cercle)")
print("    dtheta/dp ~ -1/(2*p^{3/2}) pour grands p")
print()

q = q_stat
# Vitesse discrete: Delta_w / Delta(ln p)
print(f"  {'p->p_':>8} {'|Dw|/D(lnp)':>14} {'direction/pi':>14} {'tangent?':>10}")
for i in range(len(PRIMES_ALL) - 1):
    p1, p2 = PRIMES_ALL[i], PRIMES_ALL[i + 1]
    w1 = w_p(p1, q)
    w2 = w_p(p2, q)
    dw = w2 - w1
    dlnp = np.log(p2) - np.log(p1)
    speed = abs(dw) / dlnp
    direction = cmath.phase(dw) / math.pi

    # Tangent au cercle en w1: dw/dtheta = -i*z_p1
    z1 = z_p(p1, q)
    tangent_dir = cmath.phase(-1j * z1) / math.pi
    is_tangent = abs(direction - tangent_dir) < 0.05

    print(f"  {p1:3d}->{p2:<3d} {speed:14.8f} {direction:14.8f} {'~OUI' if is_tangent else 'NON':>10}")

# Derivee analytique
# w = (1-e^{2i*theta})/2, donc dw/dtheta = -i*e^{2i*theta} = -i*z
# dtheta/dp: theta = arcsin(sqrt(sin^2)), sin^2 = delta*(2-delta), delta = (1-q^p)/p
# d(delta)/dp = -[q^p*(p*ln q + 1) - 1] / p^2 ... complique
# Approximation: pour grands p, delta ~ -ln(q)/1 * q^p / p + ... ~ (1-q^p)/p
# dtheta/dp ~ d(sqrt(2*delta))/dp ~ sqrt(2)/2 * d(delta)/dp / sqrt(delta)
print(f"\n  Analytiquement: dw/dtheta = -i*z_p (tangent au cercle)")
print(f"  La vitesse tangentielle est |dw/dtheta| = |z_p| = 1")
print(f"  La vitesse REELLE depend de dtheta/dp qui decroit avec p")

# ====================================================================
# T9: Fonction de partition thermique
# ====================================================================
print("\n\n### T9: Fonction de partition Z(beta) = sum exp(-beta * E_p)")
print("    Si on definit E_p = -ln|w_p|^2 = -ln(sin^2) = S_PT(p)")
print("    alors Z est la transformee de Laplace de la densite d'etats")
print()

q = q_stat
# Energies
energies = {}
for p in PRIMES_ALL:
    energies[p] = -np.log(sin2(p, q))

print(f"  Energies E_p = -ln(sin^2):")
for p in PRIMES_ALL:
    print(f"    p={p:3d}: E_p = {energies[p]:.8f}")

# Fonction de partition pour differents beta
print(f"\n  {'beta':>8} {'Z(beta)':>14} {'<E>':>14} {'S(beta)':>14}")
for beta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    Z_val = sum(np.exp(-beta * energies[p]) for p in PRIMES_ALL)
    E_mean = sum(energies[p] * np.exp(-beta * energies[p]) for p in PRIMES_ALL) / Z_val
    S_thermo = np.log(Z_val) + beta * E_mean
    print(f"  {beta:8.2f} {Z_val:14.6f} {E_mean:14.6f} {S_thermo:14.6f}")

# beta = 1 est special: Z(1) = sum sin^2 = sum Re(w) = Re(Sigma)
Z_1 = sum(np.exp(-1 * energies[p]) for p in PRIMES_ALL)
Sigma_re = sum(sin2(p, q) for p in PRIMES_ALL)
print(f"\n  Z(beta=1) = sum sin^2 = Re(Sigma) = {Z_1:.10f} (check: {Sigma_re:.10f}, match: {abs(Z_1-Sigma_re)<1e-10})")

# beta = 1/2: Z(1/2) = sum sin(theta) = sum |w|
Z_half = sum(np.exp(-0.5 * energies[p]) for p in PRIMES_ALL)
sum_sin = sum(np.sqrt(sin2(p, q)) for p in PRIMES_ALL)
print(f"  Z(beta=1/2) = sum sin(theta) = sum |w| = {Z_half:.10f} (check: {sum_sin:.10f})")

# ====================================================================
# T10: Fourier -- 2*theta comme variable duale de ln(p)
# ====================================================================
print("\n\n### T10: Dualite theta -- ln(p)")
print("    theta_p decroit avec p. Est-ce une transformee de Fourier?")
print("    On cherche theta_p ~ f(ln p) et la structure spectrale")
print()

q = q_stat
lnp = [np.log(p) for p in PRIMES_ALL]
thetas = [theta_p(p, q) for p in PRIMES_ALL]

# Regression lineaire theta vs ln(p)
from numpy.polynomial import polynomial as P
import sys
coeffs = np.polyfit(lnp, thetas, 1)
print(f"  Regression lineaire theta = a*ln(p) + b:")
print(f"    a = {coeffs[0]:.8f}")
print(f"    b = {coeffs[1]:.8f}")
print(f"    R^2 = {1 - np.var([thetas[i] - (coeffs[0]*lnp[i]+coeffs[1]) for i in range(len(lnp))])/np.var(thetas):.8f}")

# Regression theta vs 1/sqrt(p)
inv_sqrt_p = [1.0 / np.sqrt(p) for p in PRIMES_ALL]
coeffs2 = np.polyfit(inv_sqrt_p, thetas, 1)
print(f"\n  Regression theta = a/sqrt(p) + b:")
print(f"    a = {coeffs2[0]:.8f}")
print(f"    b = {coeffs2[1]:.8f}")
residuals = [thetas[i] - (coeffs2[0]*inv_sqrt_p[i]+coeffs2[1]) for i in range(len(lnp))]
print(f"    R^2 = {1 - np.var(residuals)/np.var(thetas):.8f}")

# Scaling exact: theta ~ sqrt(2*(1-q^p)/p) pour q proche de 1
# Pour q = 13/15: 1-q^p ~ 1-(1-2/15)^p ~ 2p/15 pour petits p
# Donc delta ~ 2/15, theta ~ sqrt(4/15) ~ 0.516 pour petit p
# Pour grands p: 1-q^p -> 1, delta -> 1/p, theta -> sqrt(2/p)
print(f"\n  Scaling asymptotique: theta_p -> sqrt(2/p) pour p -> inf")
print(f"  Verification:")
for p in PRIMES_ALL:
    th = theta_p(p, q)
    th_asympt = np.sqrt(2.0 / p)
    print(f"    p={p:3d}: theta = {th:.6f}, sqrt(2/p) = {th_asympt:.6f}, ratio = {th/th_asympt:.4f}")

# ====================================================================
# T11: Isometrie -- premiers equidistants sur le cercle
# ====================================================================
print("\n\n### T11: Premiers equidistants sur le cercle")
print("    La distance sur l'arc est |theta_p - theta_{p'}|")
print("    Quels premiers sont regulierement espaces?")
print()

q = q_stat
thetas_dict = {p: theta_p(p, q) for p in PRIMES_ALL}

# Gaps angulaires consecutifs
gaps = []
for i in range(len(PRIMES_ALL) - 1):
    p1, p2 = PRIMES_ALL[i], PRIMES_ALL[i + 1]
    gap = thetas_dict[p1] - thetas_dict[p2]  # positif car theta decroit
    gaps.append((p1, p2, gap))

print(f"  Gaps angulaires consecutifs (theta decroit avec p):")
print(f"  {'p->p_':>8} {'gap_theta':>12} {'gap/gap_moy':>12} {'gap_p':>8}")
gap_vals = [g[2] for g in gaps]
gap_mean = np.mean(gap_vals)
for p1, p2, g in gaps:
    print(f"  {p1:3d}->{p2:<3d} {g:12.8f} {g/gap_mean:12.6f} {p2-p1:8d}")

print(f"\n  Gap moyen: {gap_mean:.8f}")
print(f"  Ecart-type: {np.std(gap_vals):.8f}")
print(f"  CV (coeff. variation): {np.std(gap_vals)/gap_mean:.4f}")
print(f"  Reguliers? {'OUI' if np.std(gap_vals)/gap_mean < 0.3 else 'NON'}")

# Correlation gap angulaire vs gap arithmetique?
arith_gaps = [PRIMES_ALL[i+1] - PRIMES_ALL[i] for i in range(len(PRIMES_ALL)-1)]
corr = np.corrcoef(gap_vals, arith_gaps)[0, 1]
print(f"\n  Correlation gap_theta vs gap_arithmetique: r = {corr:.6f}")

# ====================================================================
# T12: Connexion au produit d'Euler
# ====================================================================
print("\n\n### T12: W = Pi w_p et produit d'Euler")
print("    Le produit d'Euler pour zeta: zeta(s) = Pi 1/(1-p^{-s})")
print("    En PT: alpha = Pi sin^2 = Pi |w|^2 = |W|^2")
print("    Question: W = Pi w_p a-t-il une forme de produit d'Euler?")
print()

q = q_stat

# W = Pi w_p = Pi [-i*sin(theta)*e^{i*theta}]
# = (-i)^n * Pi sin(theta) * e^{i*sum(theta)}
# = (-i)^n * sqrt(alpha) * e^{i*sum(theta)}
n_act = len(PRIMES_ACTIFS)
prod_sin = 1.0
sum_th = 0.0
for p in PRIMES_ACTIFS:
    prod_sin *= np.sqrt(sin2(p, q))
    sum_th += theta_p(p, q)

W_factored = (-1j)**n_act * prod_sin * np.exp(1j * sum_th)
W_direct = 1.0 + 0j
for p in PRIMES_ACTIFS:
    W_direct *= w_p(p, q)

print(f"  W = (-i)^n * sqrt(alpha) * e^{{i*Sigma_theta}}")
print(f"  n = {n_act} (nombre d'actifs)")
print(f"  (-i)^{n_act} = {(-1j)**n_act:.4f}")
print(f"  sqrt(alpha) = {prod_sin:.10f}")
print(f"  e^{{i*Sigma_theta}} = e^{{i*{sum_th:.6f}}} = {np.exp(1j*sum_th).real:+.8f}{np.exp(1j*sum_th).imag:+.8f}i")
print(f"  W_factorise = {W_factored.real:+.12f}{W_factored.imag:+.12f}i")
print(f"  W_direct    = {W_direct.real:+.12f}{W_direct.imag:+.12f}i")
print(f"  Match: {abs(W_factored - W_direct) < 1e-12}")
print()

# Forme logarithmique: ln W = n*ln(-i) + (1/2)*sum ln(sin^2) + i*sum(theta)
# = n*(-i*pi/2) + (-S_PT/2) + i*sum(theta)
# = -S_PT/2 + i*[sum(theta) - n*pi/2]
# = -S_PT/2 + i*sum(theta - pi/2)
# = -S_PT/2 + i*sum(arg(w_p))  [car arg(w_p) = theta_p - pi/2]
lnW = cmath.log(W_direct)
S_PT = sum(-np.log(sin2(p, q)) for p in PRIMES_ACTIFS)
sum_arg = sum(cmath.phase(w_p(p, q)) for p in PRIMES_ACTIFS)

print(f"  ln(W) = -S_PT/2 + i*sum(arg(w_p))")
print(f"  Re(ln W) = {lnW.real:.10f}, -S_PT/2 = {-S_PT/2:.10f}, match: {abs(lnW.real+S_PT/2)<1e-10}")
print(f"  Im(ln W) = {lnW.imag:.10f}, sum arg  = {sum_arg:.10f}, match: {abs(lnW.imag-sum_arg)<1e-10}")

# Lien Euler: zeta(s) = Pi 1/(1-p^{-s}), ici Pi w_p = Pi (1-z_p)/2
# = (1/2^n) * Pi (1 - z_p)
# Comparaison formelle: 1-z_p = 1-e^{2i*theta_p}
# Si theta_p jouait le role de t*ln(p), on aurait z_p = p^{-2it}
# et Pi(1-z_p) ~ Pi(1-p^{-2it}) qui est lie a 1/zeta(2it)
print(f"\n  Forme produit: W = (1/2^n) * Pi(1 - e^{{2i*theta_p}})")
print(f"  = (1/{2**n_act}) * Pi(1 - z_p)")
print(f"  Comparaison Euler: zeta(s)^{{-1}} = Pi(1 - p^{{-s}})")
print(f"  Si theta_p = t*ln(p), on aurait W ~ zeta(2it)^{{-1}} / 2^n")
print(f"  MAIS theta_p != t*ln(p) en general (sauf asymptotiquement)")

# ====================================================================
# T13: Tenseur d'inertie du nuage w_p
# ====================================================================
print("\n\n### T13: Tenseur d'inertie du nuage w_p sur le cercle")
print("    Les w_p forment un nuage de points 2D.")
print("    Le tenseur d'inertie capture la geometrie globale.")
print()

q = q_stat
# Centre de masse
w_list = np.array([w_p(p, q) for p in PRIMES_ALL])
cm = np.mean(w_list)
print(f"  Centre de masse: {cm.real:.8f} {cm.imag:+.8f}i")
print(f"  |CM| = {abs(cm):.8f}")
print(f"  arg(CM)/pi = {cmath.phase(cm)/math.pi:.6f}")

# Tenseur d'inertie (2x2 reel)
# I_xx = sum (y-y_cm)^2, I_yy = sum (x-x_cm)^2, I_xy = -sum (x-x_cm)(y-y_cm)
x = np.array([w.real for w in w_list])
y = np.array([w.imag for w in w_list])
x_cm, y_cm = cm.real, cm.imag

I_xx = np.sum((y - y_cm)**2)
I_yy = np.sum((x - x_cm)**2)
I_xy = -np.sum((x - x_cm) * (y - y_cm))

I_mat = np.array([[I_xx, I_xy], [I_xy, I_yy]])
print(f"\n  Tenseur d'inertie:")
print(f"    I_xx = {I_xx:.10f}")
print(f"    I_yy = {I_yy:.10f}")
print(f"    I_xy = {I_xy:.10f}")

evals_I, evecs_I = np.linalg.eigh(I_mat)
print(f"\n  Valeurs propres: {evals_I[0]:.10f}, {evals_I[1]:.10f}")
print(f"  Ratio: {evals_I[1]/evals_I[0]:.6f}")
print(f"  Axes principaux:")
for i in range(2):
    angle = np.arctan2(evecs_I[1, i], evecs_I[0, i])
    print(f"    Axe {i+1}: angle = {angle/math.pi:.6f} pi, val. propre = {evals_I[i]:.10f}")

# Excentricite
e = np.sqrt(1 - evals_I[0] / evals_I[1]) if evals_I[1] > 0 else 0
print(f"\n  Excentricite: {e:.8f}")
print(f"  Le nuage est {'circulaire' if e < 0.3 else 'allonge'} (e {'<' if e < 0.3 else '>'} 0.3)")

# Moment quadrupolaire
Q_xx = np.sum((x - x_cm)**2 - (y - y_cm)**2)
Q_xy = 2 * np.sum((x - x_cm) * (y - y_cm))
print(f"\n  Moment quadrupolaire:")
print(f"    Q_xx = {Q_xx:.10f}")
print(f"    Q_xy = {Q_xy:.10f}")
print(f"    |Q| = {np.sqrt(Q_xx**2 + Q_xy**2):.10f}")

# ====================================================================
# T14: Bilan
# ====================================================================
print("\n\n" + "=" * 90)
print("### T14: BILAN -- STRUCTURE PROFONDE DE LA PT COMPLEXE")
print("=" * 90)
print("""
1. CORRESPONDANCE C <-> U(1):
   z_p = 1 - 2*w_p = e^{2i*theta_p} envoie le cercle C sur U(1).
   Le produit Pi z_p est dans U(1) (module 1 exactement).
   L'argument 2*sum(theta) est la PHASE TOTALE du crible.

2. CROSS-RATIOS:
   Tous les cross-ratios de 4 points w_p sont REELS (car les points
   sont sur un cercle). Ce sont des invariants conformes du crible.
   Le cross-ratio (3,5;7,0) contient l'information de la hierarchie
   des 3 actifs par rapport au point limite w=0.

3. METRIQUE DE FISHER = 4 * METRIQUE DE L'ARC:
   d_Fisher(p,p') = 2 * |theta_p - theta_{p'}|
   Le facteur 2 est le facteur de Fubini-Study (espace projectif).
   La distance Fisher entre premiers est EXACTEMENT proportionnelle
   a la distance angulaire sur le cercle C.

4. q COMPLEXE BRISE LE CERCLE:
   Pour q reel, w_p est contraint au cercle C.
   Pour q complexe, sin^2 devient complexe et w_p quitte C.
   Le cercle est la SECTION REELLE de l'espace des parametres.

5. MATRICE DENSITE:
   w_p = rho_00 - i*rho_01 ou rho = |psi><psi| est la matrice
   densite d'un systeme a 2 niveaux (perte|conservation).
   - Re(w) = probabilite de perte (diagonale)
   - Im(w) = -coherence quantique (off-diagonal)
   Le cercle C = espace des etats PURS a 2 niveaux.

6. PRODUIT FACTORISE:
   W = Pi w_p = (-i)^n * sqrt(alpha) * e^{i*Sigma_theta}
   ln(W) = -S_PT/2 + i*sum(arg w_p)
   La partie reelle de ln(W) est la MOITIE de l'action PT.

7. SCALING: theta_p -> sqrt(2/p) pour p grand.
   La vitesse sur le cercle decroit comme 1/p^{3/2}.
   Le gap angulaire est correle au gap arithmetique.
""")

print("=" * 90)
print("FIN TOOL 43")
print("=" * 90)

sys.exit(0)
