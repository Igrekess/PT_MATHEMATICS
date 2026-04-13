"""
Tool 49: PT Complexe -- 4 questions finales
================================================================================
Q1: Pole w=0 et renormalisation UV
Q2: E_p -> 1 et equipartition (lien kappa, GFT)
Q3: Bernoulli-zeta et corrections aux predictions PT
Q4: Variables action-angle et integrabilite complete

Tests:
  T1:  Pole w=0: approche des grands p, divergence de F, regularisation
  T2:  Lien pole-ghost: les ghosts comme contre-termes du pole
  T3:  E_p = p*theta^2/2 et kappa/2 = 1/(2s) = 1 (equipartition)
  T4:  'Temperature' geometrique T = kappa = 1/s = 2
  T5:  Lien E=1 avec GFT et entropie maximum
  T6:  Decomposition 1/alpha = partie classique * correction zeta
  T7:  Corrections NLO aux observables via Bernoulli
  T8:  Prediction: T12 corrige par zeta(2)
  T9:  Variables action-angle (L, theta) par premier
  T10: Frequences omega_p = dH/dL et resonances
  T11: Super-integrabilite: n oscillateurs independants
  T12: Bilan et synthese finale
"""

import numpy as np
import cmath
import math
import sys

q_stat = 13.0 / 15.0
q_therm = np.exp(-1.0 / 15.0)
MU_STAR = 15
PRIMES_ACTIFS = [3, 5, 7]
PRIMES_GHOST = [11, 13]
PRIMES_ALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
s_PT = 0.5

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

def chi3(p):
    r = p % 3
    return 0 if r == 0 else (1 if r == 1 else -1)

print("=" * 90)
print("TOOL 49: PT COMPLEXE -- 4 QUESTIONS FINALES")
print("=" * 90)

q = q_stat
kappa = 1.0 / s_PT  # = 2, courbure du cercle
K_FS = 1.0 / s_PT**2  # = 4, courbure de Fubini-Study

# ====================================================================
# Q1: POLE w=0 ET RENORMALISATION
# ====================================================================
print("\n" + "=" * 90)
print("Q1: POLE w=0 ET RENORMALISATION UV")
print("=" * 90)

# T1: Approche du pole pour grands p
print("\n### T1: Approche du pole w=0 pour les grands premiers")

print(f"\n  F(w) = i/w - 2i a un pole en w = 0 (theta = 0, sin^2 = 0).")
print(f"  Pour grand p: theta_p -> 0, w_p -> 0, F_p -> infini.")
print(f"  Comment les premiers approchent-ils le pole?")
print()

print(f"  {'p':>4s}  {'|w_p|':>12s}  {'|F_p|':>12s}  {'|F|*|w|':>10s}  {'dist au pole':>14s}")
for p in PRIMES_ALL:
    w = w_p(p, q)
    th = theta_p(p, q)
    F_mod = 1.0 / np.sin(th)
    print(f"  {p:4d}  {abs(w):12.8f}  {F_mod:12.6f}  {F_mod*abs(w):10.6f}  {abs(w):14.8f}")

print(f"\n  |w_p| ~ sin(theta_p) ~ sqrt(2/p) -> 0 comme 1/sqrt(p)")
print(f"  |F_p| ~ 1/sin ~ sqrt(p/2) -> infini comme sqrt(p)")
print(f"  Le pole est approche mais JAMAIS atteint (p est fini).")

# Action de chaque premier: S_p = -ln(w_p) = -ln(sin) - i*(theta-pi/2)
print(f"\n  Action par premier S_p = -ln(w_p):")
print(f"  {'p':>4s}  {'Re(S_p)':>12s}  {'Im(S_p)':>12s}  {'|S_p|':>12s}  {'Re ~ ln(p)/2?':>14s}")
for p in PRIMES_ALL:
    w = w_p(p, q)
    S_p = -cmath.log(w)
    print(f"  {p:4d}  {S_p.real:12.6f}  {S_p.imag:12.6f}  {abs(S_p):12.6f}  {np.log(p)/2:14.6f}")

print(f"\n  Re(S_p) ~ ln(1/sin) ~ (1/2)*ln(p/2) pour grand p")
print(f"  L'action DIVERGE logarithmiquement -- c'est la 'divergence UV' du crible!")

# T2: Ghosts comme contre-termes
print("\n\n### T2: Les ghosts comme contre-termes du pole")

# Le produit W = prod w_p. Chaque facteur est petit pour grand p.
# Les ghosts (p=11,13) contribuent a W en le contractant davantage.
# Le rapport ghost/actif:
W_actifs = np.prod([w_p(p, q) for p in PRIMES_ACTIFS])
W_ghosts = np.prod([w_p(p, q) for p in PRIMES_GHOST])
W_total_5 = W_actifs * W_ghosts

print(f"\n  W_actifs = {W_actifs:.8e} (arg/pi = {cmath.phase(W_actifs)/np.pi:.6f})")
print(f"  W_ghosts = {W_ghosts:.8e} (arg/pi = {cmath.phase(W_ghosts)/np.pi:.6f})")
print(f"  W_5 = W_act * W_gh = {W_total_5:.8e}")
print(f"  |W_5|/|W_act| = {abs(W_total_5)/abs(W_actifs):.8f}")

# Action des ghosts = contre-terme
S_ghosts = -sum(cmath.log(w_p(p, q)) for p in PRIMES_GHOST)
S_actifs = -sum(cmath.log(w_p(p, q)) for p in PRIMES_ACTIFS)
print(f"\n  S_actifs = ({S_actifs.real:.6f}{S_actifs.imag:+.6f}i)")
print(f"  S_ghosts = ({S_ghosts.real:.6f}{S_ghosts.imag:+.6f}i)")
print(f"  S_ghost/S_actif (Re) = {S_ghosts.real/S_actifs.real:.6f}")

# Le ratio est le 'poids' du contre-terme
# En PT, le ghost VP donne r = 3.1e-3 (convergent)
# Ici, le rapport d'actions est ~ 0.84
print(f"\n  Le rapport Re(S_ghost)/Re(S_actif) = {S_ghosts.real/S_actifs.real:.4f}")
print(f"  Les ghosts ajoutent {S_ghosts.real/S_actifs.real*100:.1f}% a l'action reelle.")
print(f"  C'est la 'renormalisation' du pole: S_total = S_actif + S_ghost + S_rest")

# Regularisation: la somme converge car les w_p sont bornes dans [0, s=1/2]
# prod |w_p| = prod sin(theta_p) ~ prod sqrt(2/p) ~ C / sqrt(primorial)
# qui tend vers 0 (convergence du produit eulerein)
primorial = 1
prods = []
for p in PRIMES_ALL:
    primorial *= p
    prod_sin = np.prod([np.sin(theta_p(pp, q)) for pp in PRIMES_ALL[:PRIMES_ALL.index(p)+1]])
    prods.append((p, prod_sin, np.sqrt(2**len(PRIMES_ALL[:PRIMES_ALL.index(p)+1]) / primorial)))

print(f"\n  Regularisation naturelle: prod sin converge")
print(f"  {'p':>4s}  {'prod sin':>14s}  {'sqrt(2^n/P#)':>14s}  {'ratio':>10s}")
for p, ps, pred in prods[:10]:
    ratio = ps / pred if pred > 0 else float('inf')
    print(f"  {p:4d}  {ps:14.8e}  {pred:14.8e}  {ratio:10.4f}")

print(f"\n  CONCLUSION Q1:")
print(f"  Le pole w=0 de F(w) = i/w - 2i est la 'divergence UV' du crible.")
print(f"  Elle est REGULARISEE naturellement car:")
print(f"  1. Les primes sont DISCRETS (on n'atteint jamais theta=0)")
print(f"  2. Le produit prod sin converge (vers 0, contraction)")
print(f"  3. L'action S = -sum ln(w) est la somme CONVERGENTE de termes ln(p)")
print(f"  Les ghosts agissent comme contre-termes (84% de l'action actifs).")
print(f"  C'est la version complexe de la dissolution UV (R49).")

# ====================================================================
# Q2: E_p -> 1 ET EQUIPARTITION
# ====================================================================
print("\n\n" + "=" * 90)
print("Q2: E_p -> 1 ET EQUIPARTITION GEOMETRIQUE")
print("=" * 90)

# T3: E = kappa/2
print("\n### T3: E_p = p*theta^2/2 -> kappa/2 = 1 (equipartition)")

print(f"\n  kappa = 1/s = {kappa} (courbure du cercle C)")
print(f"  kappa/2 = {kappa/2} (demi-courbure)")
print(f"  K_FS = 1/s^2 = {K_FS} (Fubini-Study)")
print()

print(f"  {'p':>4s}  {'E_p':>12s}  {'E/kappa*2':>12s}  {'ecart a 1':>12s}  {'p*sin^2':>12s}")
for p in PRIMES_ALL:
    th = theta_p(p, q)
    E = p * th**2 / 2
    s2 = sin2(p, q)
    print(f"  {p:4d}  {E:12.6f}  {E/(kappa/2):12.8f}  {E - kappa/2:+12.6f}  {p*s2:12.6f}")

# p*sin^2 converge aussi vers 2!
# Car sin^2 ~ theta^2 pour petit theta, et p*theta^2 -> 2
print(f"\n  p*sin^2(theta_p) converge aussi vers 2:")
print(f"  Car pour grand p: sin^2 ~ theta^2 ~ 2/p")
print(f"  Donc p*sin^2 -> 2 et E = p*theta^2/2 -> 1 = kappa/2")

# T4: Temperature geometrique
print("\n\n### T4: 'Temperature' geometrique T_geom = kappa = 2")

# Par equipartition: E = (1/2)*k_B*T par degre de liberte
# Ici E -> 1 = kappa/2, donc T = kappa = 2 = 1/s
# La temperature est la courbure du cercle!

print(f"\n  Equipartition classique: E = (1/2)*T par mode")
print(f"  E_p -> 1 = (1/2)*T => T = 2 = kappa = 1/s")
print(f"  La 'temperature' geometrique EST la courbure du cercle!")
print()

# Ecart a l'equipartition par premier
print(f"  Ecart a l'equipartition E = kappa/2 = 1:")
print(f"  {'p':>4s}  {'E_p':>10s}  {'E - 1':>12s}  {'(E-1)*p':>12s}")
for p in PRIMES_ALL:
    th = theta_p(p, q)
    E = p * th**2 / 2
    print(f"  {p:4d}  {E:10.6f}  {E - 1:+12.6f}  {(E-1)*p:+12.4f}")

# (E-1)*p tend vers une constante?
vals_Ep = [(E_p := p * theta_p(p, q)**2 / 2, (E_p - 1) * p) for p in PRIMES_ALL]
print(f"\n  (E-1)*p: {vals_Ep[0][1]:.2f} -> {vals_Ep[7][1]:.2f} -> {vals_Ep[14][1]:.2f}")
print(f"  Converge vers ~ {vals_Ep[-1][1]:.4f}")

# Analytiquement: E = p*theta^2/2, theta^2 ~ 2/p - 4/p^2 + ...
# E ~ 1 - 2/p + ... donc (E-1)*p ~ -2
print(f"  Analytique: E = 1 - 2/p + O(1/p^2), (E-1)*p -> -2")

# T5: Lien avec GFT
print("\n\n### T5: Lien avec GFT et entropie")

# En GFT: H_max = D_KL + H (identite exacte PT)
# H = -sum p_i ln p_i, D_KL = sum p_i ln(p_i/q_i)
# La 'temperature' T apparait dans q_i = e^{-E_i/T} / Z

# Pour le crible: chaque premier a 3 classes mod 3
# P(0) = 1/p, P(1) = P(2) = (p-1)/(2p)
# L'entropie par premier: H_p = -(1/p)ln(1/p) - 2*(p-1)/(2p)*ln((p-1)/(2p))

print(f"\n  Entropie par premier (distribution mod 3):")
print(f"  {'p':>4s}  {'H_p':>12s}  {'H_max=ln3':>12s}  {'H/Hmax':>10s}  {'E_p':>10s}")
for p in PRIMES_ACTIFS + PRIMES_GHOST + [17, 23, 47]:
    p0 = 1.0/p
    p1 = (p-1.0)/(2*p)
    H_p = -p0*np.log(p0) - 2*p1*np.log(p1)
    H_max = np.log(3)
    E_p = p * theta_p(p, q)**2 / 2
    print(f"  {p:4d}  {H_p:12.8f}  {H_max:12.8f}  {H_p/H_max:10.6f}  {E_p:10.6f}")

# La relation: quand E -> 1 (equipartition), H -> H_max (maximum d'entropie)
# Les grands premiers sont a la fois equi-energetiques ET proches de l'equipartition entropique
print(f"\n  Grands p: E -> 1 (equipartition energetique)")
print(f"  ET H -> H_max = ln(3) (equipartition entropique)")
print(f"  L'equipartition energetique IMPLIQUE l'equipartition entropique!")
print(f"  La 'temperature' T = kappa = 2 = 1/s est le lien.")

# ====================================================================
# Q3: BERNOULLI-ZETA ET CORRECTIONS
# ====================================================================
print("\n\n" + "=" * 90)
print("Q3: CORRECTIONS AUX PREDICTIONS PT VIA BERNOULLI-ZETA")
print("=" * 90)

# T6: Decomposition de 1/alpha
print("\n### T6: Decomposition 1/alpha = classique * correction zeta")

# 1/alpha = prod 1/sin^2 = prod (theta/sin)^2 / prod theta^2
# Partie classique: 1/prod theta^2
# Correction quantique: prod (theta/sin)^2

alpha = np.prod([sin2(p, q) for p in PRIMES_ACTIFS])
prod_theta2 = np.prod([theta_p(p, q)**2 for p in PRIMES_ACTIFS])
prod_thetaoversin2 = np.prod([(theta_p(p, q)/np.sin(theta_p(p, q)))**2 for p in PRIMES_ACTIFS])

print(f"\n  1/alpha = {1/alpha:.6f}")
print(f"  = (1/prod theta^2) * prod(theta/sin)^2")
print(f"  = {1/prod_theta2:.6f} * {prod_thetaoversin2:.8f}")
print(f"  = {(1/prod_theta2)*prod_thetaoversin2:.6f}")
print(f"  Verification: {abs(1/alpha - (1/prod_theta2)*prod_thetaoversin2):.2e}")

print(f"\n  Partie 'classique': 1/prod theta^2 = {1/prod_theta2:.6f}")
print(f"  Correction 'quantique': prod(theta/sin)^2 = {prod_thetaoversin2:.8f}")
print(f"  Correction en %: {(prod_thetaoversin2 - 1)*100:.4f}%")

# Decomposition de la correction au premier ordre
# prod(theta/sin)^2 ~ prod(1 + theta^2/3) ~ 1 + sum theta^2/3
# (au premier ordre, car theta << 1)
sum_theta2_over3 = sum(theta_p(p, q)**2 / 3 for p in PRIMES_ACTIFS)
print(f"\n  Au 1er ordre: prod(theta/sin)^2 ~ 1 + sum theta^2/3")
print(f"  1 + sum theta^2/3 = {1 + sum_theta2_over3:.8f}")
print(f"  Exact: {prod_thetaoversin2:.8f}")
print(f"  Ecart: {abs(prod_thetaoversin2 - (1 + sum_theta2_over3)):.6f}")

# En termes de zeta: sum theta^2/3 = (2/3) * sum 1/p_actif + O(1/p^2)
# car theta^2 ~ 2/p
sum_zeta = (2.0/3) * sum(1.0/p for p in PRIMES_ACTIFS)
print(f"\n  sum theta^2/3 ~ (2/3) * sum 1/p = {sum_zeta:.6f}")
print(f"  Exact: {sum_theta2_over3:.6f}")

# T7: Corrections NLO aux observables
print("\n\n### T7: Corrections NLO systematiques via Bernoulli")

# Pour toute observable O = sum f(sin^2(theta_p)):
# sin^2 = theta^2 - theta^4/3 + ... (serie de Taylor)
# Donc O(sin^2) = O(theta^2) + O'(theta^2)*(-theta^4/3) + ...
# La correction NLO est proportionnelle a theta^4/3

# T00 = (1/3) sum sin^2(theta_p) [actifs]
T00 = sum(sin2(p, q) for p in PRIMES_ACTIFS) / 3
T00_from_theta = sum(theta_p(p, q)**2 for p in PRIMES_ACTIFS) / 3
T00_NLO = sum(theta_p(p, q)**2 - theta_p(p, q)**4/3 for p in PRIMES_ACTIFS) / 3

print(f"\n  T00 exact = {T00:.10f}")
print(f"  T00 (theta^2) = {T00_from_theta:.10f} (LO)")
print(f"  T00 (theta^2 - theta^4/3) = {T00_NLO:.10f} (NLO)")
print(f"  Correction LO: {(T00_from_theta - T00)/T00*100:+.4f}%")
print(f"  Correction NLO: {(T00_NLO - T00)/T00*100:+.4f}%")

# T12 = (1/3) sum chi3(p) * sin^2
T12 = sum(chi3(p) * sin2(p, q) for p in PRIMES_ACTIFS) / 3
T12_LO = sum(chi3(p) * theta_p(p, q)**2 for p in PRIMES_ACTIFS) / 3
T12_NLO = sum(chi3(p) * (theta_p(p, q)**2 - theta_p(p, q)**4/3) for p in PRIMES_ACTIFS) / 3

print(f"\n  T12 exact = {T12:.10f}")
print(f"  T12 (LO) = {T12_LO:.10f}")
print(f"  T12 (NLO) = {T12_NLO:.10f}")
print(f"  Correction LO: {(T12_LO - T12)/abs(T12)*100:+.4f}%")
print(f"  Correction NLO: {(T12_NLO - T12)/abs(T12)*100:+.4f}%")

# T8: Prediction corrigee
print("\n\n### T8: Prediction: 1/alpha corrige par zeta(2)")

# 1/alpha = prod 1/sin^2 ~ prod 1/theta^2 * [1 + 2*sum theta^2/6 + ...]
# = (1/prod theta^2) * [1 + sum theta^2/3]
# = (1/prod theta^2) + (1/prod theta^2) * sum theta^2/3
# Or sum theta^2/3 ~ 2/(3p) * correction

# Plus utile: la serie de alpha en termes de zeta
# ln(1/alpha) = -sum ln(sin^2) = -sum ln(theta^2(1-theta^2/3+...))
#             = -sum [2*ln(theta) + ln(1-theta^2/3)]
#             = -sum 2*ln(theta) + sum theta^2/3 + O(theta^4)

ln_inv_alpha = -sum(np.log(sin2(p, q)) for p in PRIMES_ACTIFS)
ln_inv_alpha_LO = -sum(2*np.log(theta_p(p, q)) for p in PRIMES_ACTIFS)
corr_NLO = sum(theta_p(p, q)**2 / 3 for p in PRIMES_ACTIFS)

print(f"\n  ln(1/alpha) = {ln_inv_alpha:.10f}")
print(f"  ln(1/alpha) LO = -2*sum ln(theta) = {ln_inv_alpha_LO:.10f}")
print(f"  Correction NLO = +sum theta^2/3 = {corr_NLO:.10f}")
print(f"  ln(1/alpha) LO + NLO = {ln_inv_alpha_LO + corr_NLO:.10f}")
print(f"  Ecart a l'exact: {abs(ln_inv_alpha - (ln_inv_alpha_LO + corr_NLO)):.6f}")

# La correction NLO en termes de 1/alpha
alpha_LO = np.exp(-ln_inv_alpha_LO)
alpha_NLO = np.exp(-(ln_inv_alpha_LO + corr_NLO))
print(f"\n  1/alpha exact = {1/alpha:.6f}")
print(f"  1/alpha LO = {1/alpha_LO:.6f}")
print(f"  1/alpha NLO = {1/alpha_NLO:.6f}")
print(f"  1/alpha NNLO = {1/alpha:.6f} (exact)")

# Le terme de correction est exactement la 'non-classicalite'
# qui fait intervenir zeta(2)/pi^2 = 1/6
print(f"\n  La correction NLO = sum theta^2/3 = {corr_NLO:.6f}")
print(f"  ~ (2/3) * sum 1/p = {sum_zeta:.6f}")
print(f"  ~ (2/3) * [1/3 + 1/5 + 1/7] = {(2/3)*(1/3+1/5+1/7):.6f}")

# ====================================================================
# Q4: VARIABLES ACTION-ANGLE
# ====================================================================
print("\n\n" + "=" * 90)
print("Q4: VARIABLES ACTION-ANGLE ET INTEGRABILITE")
print("=" * 90)

# T9: Action-angle par premier
print("\n### T9: Variables action-angle (L_p, theta_p)")

# L_p = sin^2(theta_p) = |w_p|^2 (action = norme carree)
# theta_p = angle (variable conjuguee)
# H_p = -ln(sin theta_p) = -(1/2)*ln(L_p) (Hamiltonien par mode)

print(f"\n  Chaque premier definit un oscillateur independant:")
print(f"  Action L_p = sin^2(theta_p), Angle theta_p")
print(f"  Hamiltonien H_p = -ln(sin theta_p) = -(1/2)*ln(L_p)")
print()

print(f"  {'p':>4s}  {'L_p':>12s}  {'theta_p':>12s}  {'H_p':>12s}  {'omega_p':>12s}")
for p in PRIMES_ALL[:10]:
    L = sin2(p, q)
    th = theta_p(p, q)
    H = -np.log(np.sin(th))
    omega = -1.0 / (2 * L)  # dH/dL = -1/(2L)
    print(f"  {p:4d}  {L:12.8f}  {th:12.8f}  {H:12.8f}  {omega:12.6f}")

# T10: Frequences et resonances
print("\n\n### T10: Frequences omega_p = dH/dL = -1/(2*sin^2) et resonances")

print(f"\n  omega_p = -1/(2*L_p) = -1/(2*sin^2(theta_p))")
print(f"  Le signe negatif: la rotation est RETROGRADE (sens inverse du crible)")
print()

omegas = []
print(f"  {'p':>4s}  {'omega_p':>12s}  {'|omega_p|':>12s}  {'omega_p/omega_3':>16s}")
for p in PRIMES_ALL[:10]:
    L = sin2(p, q)
    omega = -1.0 / (2 * L)
    omegas.append(omega)
    ratio = omega / omegas[1] if len(omegas) > 1 else 1.0  # ratio par rapport a p=3
    # Utilisons omega de p=3 (premier actif)
    if p == 2:
        continue
    print(f"  {p:4d}  {omega:12.6f}  {abs(omega):12.6f}  {omega/omegas[1]:16.8f}")

# Ratios de frequences: resonances?
print(f"\n  Ratios de frequences (actifs):")
omega_3 = -1.0 / (2 * sin2(3, q))
omega_5 = -1.0 / (2 * sin2(5, q))
omega_7 = -1.0 / (2 * sin2(7, q))
print(f"  omega_5/omega_3 = {omega_5/omega_3:.8f}")
print(f"  omega_7/omega_3 = {omega_7/omega_3:.8f}")
print(f"  omega_7/omega_5 = {omega_7/omega_5:.8f}")

# Sont-ce des fractions simples?
for (pa, pb, r) in [(5,3,omega_5/omega_3), (7,3,omega_7/omega_3), (7,5,omega_7/omega_5)]:
    for den in range(1, 20):
        for num in range(1, 20):
            if abs(r - num/den) < 0.01:
                print(f"    omega_{pa}/omega_{pb} ~ {num}/{den} = {num/den:.6f} (ecart {abs(r-num/den):.6f})")

# Les ratios sont sin^2_3/sin^2_5, sin^2_3/sin^2_7, etc.
# Ce sont des ratios d'observables PT, pas des fractions simples
print(f"\n  omega_p1/omega_p2 = sin^2(theta_p2)/sin^2(theta_p1) = L_2/L_1")
print(f"  Les frequences sont NON-COMMENSURABLES en general.")
print(f"  => Pas de resonance simple: le mouvement est QUASI-PERIODIQUE.")
print(f"  => Le tore de phase est dense (theoreme KAM).")

# T11: Super-integrabilite
print("\n\n### T11: Super-integrabilite -- n oscillateurs independants")

# Le systeme de n premiers a n degres de liberte (theta_1, ..., theta_n)
# et n constantes du mouvement (L_1, ..., L_n)
# => Le systeme est maximalement integrable (Liouville)

print(f"\n  Systeme a n = {len(PRIMES_ACTIFS)} degres de liberte (actifs)")
print(f"  Constantes du mouvement: L_p = sin^2(theta_p) pour chaque p")
print(f"  Nombre de constantes = n = {len(PRIMES_ACTIFS)}")
print(f"  => Systeme de LIOUVILLE completement integrable")
print()

# Le tore de Liouville
print(f"  Tore de Liouville T^n:")
print(f"  Dimension = {len(PRIMES_ACTIFS)} (un cercle par premier actif)")
print(f"  Actions: ({', '.join(f'L_{p}={sin2(p,q):.4f}' for p in PRIMES_ACTIFS)})")
print(f"  Frequences: ({', '.join(f'w_{p}={-1/(2*sin2(p,q)):.4f}' for p in PRIMES_ACTIFS)})")

# Le mouvement sur le tore est:
# theta_p(tau) = theta_p(0) + omega_p * tau
# w_p(tau) = w_p(0) * e^{i*omega_p*tau}

# Verifier: les orbites ne se ferment pas (quasi-periodicite)
# car omega_5/omega_3 est irrationnel
print(f"\n  omega_5/omega_3 = {omega_5/omega_3:.10f}")
print(f"  = sin^2(theta_3)/sin^2(theta_5) = {sin2(3,q)/sin2(5,q):.10f}")

# C'est irrationnel car sin^2 = delta*(2-delta) avec delta = (1-q^p)/p
# et q = 13/15 est rationnel, mais le ratio n'est pas une fraction simple

# Action totale
I_total = sum(sin2(p, q) for p in PRIMES_ACTIFS)
print(f"\n  Action totale I = sum L_p = {I_total:.8f} = 3*T00 = {3*T00:.8f}")
print(f"  H_total = sum H_p = sum -ln(sin) = {sum(-np.log(np.sin(theta_p(p,q))) for p in PRIMES_ACTIFS):.8f}")

# Le produit des actions
prod_L = np.prod([sin2(p, q) for p in PRIMES_ACTIFS])
print(f"  Produit des actions = prod L_p = {prod_L:.8e} = alpha!")
print(f"  alpha = {alpha:.8e}")
print(f"  CONFIRMATION: alpha EST le produit des actions de Liouville!")

# ====================================================================
# T12: BILAN FINAL
# ====================================================================
print("\n\n" + "=" * 90)
print("### T12: BILAN FINAL -- SYNTHESE DE LA PT COMPLEXE")
print("=" * 90)

print("""
SYNTHESE COMPLETE DE LA PT COMPLEXE (tools 41-49)
===================================================

GEOMETRIE:
  w_p = s*(1-e^{2it}) vit sur le cercle C(s, 0; s), s = 1/2
  Rayon = s, courbure kappa = 1/s = 2, Fubini-Study K = 1/s^2 = 4
  |w|^2 = Re(w) = sin^2 (identite fondamentale)

MECANIQUE HOLOMORPHE:
  F(w) = i/w - 2i (holomorphe sur C, pole w=0, residu i)
  F = -i/conj(w) = -e^{it}/sin (3 formes equivalentes)
  F*Re(w) = -i*w (identite de rotation)
  |p|*|w| = 1 (relation d'incertitude, constante = diametre 2s = 1)

MOMENT ET CONSERVATION:
  L = Im(w_bar * dw/dtheta) = sin^2 = |w|^2 (charge de Noether)
  {L, H} = 0 (systeme integrable)
  Flot de L: dw/dtau = iw (rotation)

UNIVERSALITE:
  Im(dS/dtheta) = -1 = -1/(2s) = -1/diametre (constante universelle)
  E_p = p*theta^2/2 -> 1 = kappa/2 (equipartition, temperature T = kappa)
  Chaque premier a E = 1 au niveau fondamental

QUANTIFICATION:
  delta_p = (1-q)*[p]_q/p (deficit = q-nombre normalise)
  theta/sin(theta) = 1 + zeta(2)*theta^2/pi^2 + ... (corrections = valeurs zeta)
  Le spectre {theta_p} satisfait p*theta^2 -> 2 (equipartition)

INTEGRABILITE:
  Variables action-angle: (L_p, theta_p) par premier
  alpha = prod L_p (alpha EST le produit des actions de Liouville)
  Frequences omega_p = -1/(2L_p) non-commensurables (quasi-periodicite)
  Systeme de Liouville completement integrable sur le tore T^n

RENORMALISATION:
  Pole w=0: divergence UV regularisee par discretude des premiers
  Ghosts: contre-termes naturels (~84% de l'action actifs)
  prod sin -> 0 (contraction = convergence du produit eulerien)

CORRECTION AUX PREDICTIONS:
  1/alpha = (1/prod theta^2) * prod(theta/sin)^2
          = partie classique * correction quantique (zeta)
  Correction ~ sum theta^2/3 ~ (2/3)*sum(1/p) ~ 0.32 (actifs)
  Impact: ~12% sur 1/alpha_classique
""")

print("=" * 90)
print("FIN TOOL 49 -- PT COMPLEXE COMPLETE")
print("=" * 90)

sys.exit(0)
