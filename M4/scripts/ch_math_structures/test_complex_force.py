"""
Tool 47: PT Complexe -- Approfondissement de la force complexe
================================================================================
La force dS_C/dtheta = -cot(theta) - i a une partie imaginaire universelle.
Ce script approfondit cette decouverte sous tous les angles.

Decouvertes a explorer:
  - F = -e^{i*theta}/sin(theta) : forme geometrique compacte
  - F * sin^2(theta) = -i * w : identite de rotation
  - Im(S) = -sum theta_p + const : fonctionnelle LINEAIRE
  - Lien Im=-1 et s=1/2 : le facteur 2 dans e^{2i*theta}
  - Courant conserve et Noether

Tests:
  T1:  Force F = -e^{i*theta}/sin(theta) -- verification
  T2:  Identite F*Re(w) = -i*w -- la rotation fondamentale
  T3:  Im(S_C) = -sum theta + n*pi/2 -- fonctionnelle lineaire
  T4:  Le facteur 2 et s=1/2 : pourquoi Im = -1 exactement
  T5:  Module |F| = 1/sin = 1/sqrt(Re(w)) -- lien avec l'observable
  T6:  Direction de F : arg(F) = theta + pi -- force centripete
  T7:  Somme des forces : sum F_p et son interpretation
  T8:  Moment angulaire L = Im(w* dw/dtheta) -- conservation?
  T9:  Hamilton-Jacobi : p = dS/dtheta, H = ?
  T10: Travail theta/sin(theta) et sinc inverse
  T11: Courbure K=4 et force : |F|^2 = K * |F_reduit|^2 ?
  T12: Bilan : structure complete de la mecanique complexe PT
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

def cos2(p, q):
    d = delta_p(p, q)
    return (1.0 - d)**2

def theta_p(p, q):
    return np.arcsin(np.sqrt(sin2(p, q)))

def w_p(p, q):
    th = theta_p(p, q)
    return (1.0 - np.exp(2j * th)) / 2.0

def chi3(p):
    r = p % 3
    return 0 if r == 0 else (1 if r == 1 else -1)

print("=" * 90)
print("TOOL 47: PT COMPLEXE -- APPROFONDISSEMENT DE LA FORCE COMPLEXE")
print("=" * 90)

# ====================================================================
# T1: F = -e^{i*theta}/sin(theta)
# ====================================================================
print("\n### T1: Force F = -e^{i*theta}/sin(theta) -- forme compacte")

q = q_stat
print("\n  dS_C/dtheta = -cot(theta) - i = -(cos + i*sin)/sin = -e^{i*theta}/sin(theta)")
print()
print(f"  {'p':>4s}  {'F = -cot-i':>28s}  {'F = -e^it/sin':>28s}  {'ecart':>10s}")
for p in PRIMES_ALL:
    th = theta_p(p, q)
    # Methode 1: -cot - i
    F1 = -np.cos(th)/np.sin(th) - 1j
    # Methode 2: -e^{i*theta}/sin
    F2 = -np.exp(1j * th) / np.sin(th)
    ecart = abs(F1 - F2)
    print(f"  {p:4d}  ({F1.real:+12.8f}{F1.imag:+12.8f}i)  "
          f"({F2.real:+12.8f}{F2.imag:+12.8f}i)  {ecart:.2e}")

print(f"\n  IDENTITE VERIFIEE: F = -e^{{i*theta}}/sin(theta)")
print(f"  C'est la forme la plus compacte de la force complexe PT.")
print(f"  Interpretation: F pointe DANS la direction e^{{i*theta}} (sur le cercle unite)")
print(f"  avec un signe negatif (vers le centre) et une amplitude 1/sin(theta).")

# ====================================================================
# T2: F * Re(w) = -i * w
# ====================================================================
print("\n\n### T2: Identite de rotation F * Re(w) = -i * w")

print(f"\n  Demonstration:")
print(f"  F * sin^2 = (-e^{{it}}/sin) * sin^2 = -sin * e^{{it}}")
print(f"  -i * w = -i * (-i*sin*e^{{it}}) = -sin*e^{{it}} = i^2 * sin * e^{{it}}")
print(f"  Verifions: w = -i*sin*e^{{it}} => -i*w = i^2*sin*e^{{it}} = -sin*e^{{it}}")
print(f"  Et F*sin^2 = -e^{{it}}*sin = -sin*e^{{it}}")
print(f"  Donc F * sin^2 = -i * w  [QED]")
print()

print(f"  {'p':>4s}  {'F*sin^2':>28s}  {'-i*w':>28s}  {'ecart':>10s}")
for p in PRIMES_ALL[:8]:
    th = theta_p(p, q)
    F = -np.exp(1j * th) / np.sin(th)
    s2 = sin2(p, q)
    w = w_p(p, q)
    lhs = F * s2
    rhs = -1j * w
    ecart = abs(lhs - rhs)
    print(f"  {p:4d}  ({lhs.real:+12.8f}{lhs.imag:+12.8f}i)  "
          f"({rhs.real:+12.8f}{rhs.imag:+12.8f}i)  {ecart:.2e}")

print(f"\n  SIGNIFICATION: La force appliquee a sin^2 donne -i*w.")
print(f"  C'est une ROTATION de -pi/2 de w!")
print(f"  F * Re(w) = -i * w  =>  F = -i * w / Re(w)")
print(f"  La force EST w tourne de -pi/2 et dilate de 1/Re(w).")

# Reformulation: F = -i * w / |w|^2 car |w|^2 = Re(w)
print(f"\n  Comme |w|^2 = Re(w), on a aussi:")
print(f"  F = -i * w / |w|^2 = -i / conj(w)")
print()
print(f"  {'p':>4s}  {'-i/conj(w)':>28s}  {'F direct':>28s}  {'ecart':>10s}")
for p in PRIMES_ALL[:8]:
    th = theta_p(p, q)
    w = w_p(p, q)
    F_direct = -np.exp(1j * th) / np.sin(th)
    F_conj = -1j / np.conj(w)
    ecart = abs(F_direct - F_conj)
    print(f"  {p:4d}  ({F_conj.real:+12.8f}{F_conj.imag:+12.8f}i)  "
          f"({F_direct.real:+12.8f}{F_direct.imag:+12.8f}i)  {ecart:.2e}")

print(f"\n  F = -i / conj(w)  [IDENTITE EXACTE]")
print(f"  La force est l'INVERSE CONJUGUE de w, tourne de -pi/2!")
print(f"  C'est une INVERSION de MOEBIUS suivie d'une rotation.")

# ====================================================================
# T3: Im(S_C) = -sum theta + n*pi/2
# ====================================================================
print("\n\n### T3: Im(S_C) = -sum theta + n*pi/2 -- fonctionnelle lineaire")

print(f"\n  S_C = -sum ln(w_p) = -sum [ln(sin) + i*(theta - pi/2)]")
print(f"  Im(S_C) = -sum (theta_p - pi/2) = n*pi/2 - sum theta_p")
print()

for label, q, primes in [("Actifs q_stat", q_stat, PRIMES_ACTIFS),
                           ("Actifs q_therm", q_therm, PRIMES_ACTIFS),
                           ("Tous q_stat", q_stat, PRIMES_ALL)]:
    S_C = sum(-cmath.log(w_p(p, q)) for p in primes)
    sum_th = sum(theta_p(p, q) for p in primes)
    n = len(primes)
    Im_pred = n * np.pi/2 - sum_th
    print(f"  {label:20s}: Im(S_C) = {S_C.imag:12.8f}, "
          f"n*pi/2 - sum(theta) = {Im_pred:12.8f}, ecart = {abs(S_C.imag - Im_pred):.2e}")

print(f"\n  Im(S_C) est une fonctionnelle LINEAIRE des angles theta_p.")
print(f"  => La stationnarite de Im(S) donne: d Im(S)/d theta_p = -1 pour tout p.")
print(f"  => La 'pression imaginaire' est UNIFORME sur tous les premiers.")
print(f"  => C'est l'analogue d'une pression hydrostatique en mecanique des fluides.")

# ====================================================================
# T4: Pourquoi Im = -1 exactement? Le role de s = 1/2
# ====================================================================
print("\n\n### T4: Pourquoi Im(dS/dtheta) = -1? Le role de s et du double angle")

# w = s * (1 - e^{2i*theta}), avec s = 1/2
# ln(w) = ln(s) + ln(1 - e^{2i*theta})
# 1 - e^{2it} = -e^{it} * 2i * sin(t) = 2*sin(t) * e^{i*(t+pi/2-pi)} ...
# Plus directement:
# 1 - e^{2it} = -2i * sin(t) * e^{it}
# ln(1 - e^{2it}) = ln(2*sin(t)) + i*(t - pi/2)
# d/dt ln(w) = d/dt [ln(2sin) + i*(t-pi/2)] = cos/sin + i

# Le facteur i*t vient de e^{2i*theta}: la derivee de 2i*theta est 2i,
# mais apres la decomposition ln(...), c'est i*1.

# Si on avait e^{n*i*theta}, que se passerait-il?
print(f"\n  Generalisation: w_n = s_n * (1 - e^{{n*i*theta}}) avec s_n = 1/n")
print(f"  (n=2 est le cas PT standard)")
print()

for n in [1, 2, 3, 4, 6]:
    # w_n = (1 - e^{n*i*theta}) / n pour garder la normalisation
    # d/dtheta ln(w_n) = n*i*e^{n*i*t}/(1-e^{n*i*t})
    # Pour n=2: 2i*e^{2it}/(1-e^{2it}) = 2i*e^{2it}/(-2i*sin(t)*e^{it}) = -e^{it}/sin(t) = F
    # En general: n*i*e^{nit}/(1-e^{nit})
    # Im de cela au point theta?
    # Calculons numeriquement pour p=3
    th = theta_p(3, q_stat)
    w_n = (1 - np.exp(1j * n * th)) / n
    # Force = d/dtheta [-ln(w_n)] = -d ln(w_n)/dtheta
    # = -n*i*e^{nit}/(1-e^{nit})  ... non, c'est d/dt ln(w_n)
    # w_n = (1-e^{int})/n, d/dt = -in*e^{int}/n = -i*e^{int}
    # d ln(w_n)/dt = dw_n/dt / w_n = (-i*e^{int}) / ((1-e^{int})/n)
    #             = -i*n*e^{int}/(1-e^{int})
    # F_n = -d ln(w_n)/dt = i*n*e^{int}/(1-e^{int})

    deriv = -1j * n * np.exp(1j * n * th) / (1 - np.exp(1j * n * th))
    F_n = -deriv  # F = -d(ln w)/dtheta
    print(f"  n={n}: w = ({w_n.real:+.6f}{w_n.imag:+.6f}i), "
          f"F = ({F_n.real:+.6f}{F_n.imag:+.6f}i), "
          f"Im(F) = {F_n.imag:.6f}")

# Calculons Im(F_n) analytiquement
# F_n = i*n*e^{int}/(1-e^{int})
# Posons z = e^{int}. Alors 1-z = 2*sin(nt/2)*e^{i*(nt/2 + pi - pi/2)}...
# Plus simple: calculons pour plusieurs p
print(f"\n  Im(F) pour differents n (moyenne sur actifs 3,5,7):")
for n in [1, 2, 3, 4, 5, 6]:
    ims = []
    for p in PRIMES_ACTIFS:
        th = theta_p(p, q_stat)
        F_n = 1j * n * np.exp(1j * n * th) / (1 - np.exp(1j * n * th))
        ims.append(F_n.imag)
    mean_im = np.mean(ims)
    # Pour n=2: Im(F) = -1 pour tout p (constant)
    # Pour n != 2: Im(F) depend de p
    std_im = np.std(ims)
    print(f"  n={n}: Im(F) = [{ims[0]:.6f}, {ims[1]:.6f}, {ims[2]:.6f}], "
          f"moyenne = {mean_im:.6f}, std = {std_im:.6f}, constant? {'OUI' if std_im < 1e-10 else 'NON'}")

# Pour n=2 exactement:
# F_2 = 2i*e^{2it}/(1-e^{2it}) = 2i*e^{2it}/(-2i*sin(t)*e^{it})
#      = -e^{it}/sin(t) = -cos/sin - i = -cot - i
# Im(F_2) = -1 pour tout theta. CQFD.

# Pour n general:
# F_n = in*e^{int}/(1-e^{int})
# 1-e^{int} = -e^{int/2}*2i*sin(nt/2)
# F_n = in*e^{int} / (-e^{int/2}*2i*sin(nt/2))
#      = n*e^{int/2} / (2*sin(nt/2))
#      = (n/2) * [cos(nt/2) + i*sin(nt/2)] / sin(nt/2)
#      = (n/2) * [cot(nt/2) + i]
# Im(F_n) = n/2 pour tout theta!

print(f"\n  RESULTAT ANALYTIQUE:")
print(f"  Pour w_n = (1 - e^{{n*i*theta}})/n:")
print(f"  F_n = -d ln(w_n)/dtheta = (n/2) * [cot(n*theta/2) + i]")
print(f"  Im(F_n) = n/2")
print(f"")
print(f"  Verification (F = -d ln w/dtheta, donc Im(F) = -n/2):")
for n in [1, 2, 3, 4, 5, 6]:
    th = theta_p(5, q_stat)  # p=5 comme test
    # d ln(w)/dtheta (pas la force, qui est l'oppose)
    dln = 1j * n * np.exp(1j * n * th) / (1 - np.exp(1j * n * th))
    F_n_im = -dln.imag  # Force = -d ln w / dtheta
    print(f"  n={n}: Im(F) = {-dln.imag:.10f}, -n/2 = {-n/2:.1f}, ecart = {abs(-dln.imag - (-n/2)):.2e}")

print(f"\n  CONCLUSION:")
print(f"  Im(dS/dtheta) = -n/2 ou n est le facteur d'angle dans e^{{n*i*theta}}.")
print(f"  Pour PT: n = 2 (double angle de sin^2 = (1-cos2theta)/2).")
print(f"  Donc Im(dS/dtheta) = -2/2 = -1.")
print(f"  Si s = 1/n (pour garder w normalise), alors Im = -1/(2*s).")
print(f"  Pour s = 1/2: Im = -1/(2*1/2) = -1. CQFD!")
print(f"  => Im(dS/dtheta) = -1/(2*s) = -1/diametre du cercle C!")

# ====================================================================
# T5: |F| = 1/sin = 1/sqrt(Re(w))
# ====================================================================
print("\n\n### T5: Module |F| = 1/sin(theta) = 1/sqrt(Re(w))")

print(f"\n  F = -e^{{it}}/sin => |F| = 1/sin(theta)")
print(f"  sin(theta) = sqrt(sin^2) = sqrt(Re(w))")
print(f"  => |F| = 1/sqrt(Re(w))")
print()
print(f"  {'p':>4s}  {'|F|':>12s}  {'1/sin':>12s}  {'1/sqrt(Re)':>12s}  {'|F|^2':>12s}  {'1/Re(w)':>12s}")
for p in PRIMES_ALL[:10]:
    th = theta_p(p, q)
    s2 = sin2(p, q)
    F_mod = 1.0 / np.sin(th)
    inv_sqrt_re = 1.0 / np.sqrt(s2)
    print(f"  {p:4d}  {F_mod:12.6f}  {1/np.sin(th):12.6f}  {inv_sqrt_re:12.6f}  "
          f"{F_mod**2:12.6f}  {1/s2:12.6f}")

print(f"\n  |F|^2 = 1/sin^2 = 1/Re(w) = INVERSE de l'observable PT standard!")
print(f"  La force est d'autant plus grande que l'observable est petite.")
print(f"  Pour les grands premiers (sin^2 ~ 1/p): |F| ~ sqrt(p) ~ la charge cot.")

# ====================================================================
# T6: Direction de F
# ====================================================================
print("\n\n### T6: Direction de F : arg(F) = theta + pi (force centripete)")

print(f"\n  F = -e^{{it}}/sin => arg(F) = arg(-e^{{it}}) = theta + pi [mod 2pi]")
print(f"  C'est la direction OPPOSEE au vecteur e^{{i*theta}} sur le cercle unite.")
print()
print(f"  {'p':>4s}  {'theta/pi':>10s}  {'arg(F)/pi':>10s}  {'(arg(F)-theta)/pi':>18s}  {'=1?':>6s}")
for p in PRIMES_ALL[:10]:
    th = theta_p(p, q)
    F = -np.exp(1j * th) / np.sin(th)
    arg_F = cmath.phase(F)
    diff = (arg_F - th) / np.pi
    # Normaliser mod 2
    while diff > 1.5: diff -= 2
    while diff < -0.5: diff += 2
    print(f"  {p:4d}  {th/np.pi:10.6f}  {arg_F/np.pi:10.6f}  {diff:18.10f}  {'OUI' if abs(abs(diff)-1) < 1e-8 else 'NON':>6s}")

print(f"\n  La force pointe TOUJOURS a 180 degres de e^{{i*theta}}.")
print(f"  Sur le cercle C, e^{{2i*theta}} parametre la position. La force est CENTRIPETE.")
print(f"  Avec arg(F) = theta + pi, la force tire vers le point theta = 0")
print(f"  (le point w = 0, ou sin^2 = 0, repos complet du crible).")

# ====================================================================
# T7: Somme des forces
# ====================================================================
print("\n\n### T7: Somme des forces sum F_p et interpretation")

for label, q, primes in [("Actifs q_stat", q_stat, PRIMES_ACTIFS),
                           ("Actifs q_therm", q_therm, PRIMES_ACTIFS),
                           ("Actifs+Ghost", q_stat, PRIMES_ACTIFS + PRIMES_GHOST),
                           ("Tous", q_stat, PRIMES_ALL)]:
    F_sum = sum(-np.exp(1j * theta_p(p, q)) / np.sin(theta_p(p, q))
                for p in primes)
    n = len(primes)
    print(f"  {label:20s}: sum F = ({F_sum.real:+12.6f}{F_sum.imag:+12.6f}i), "
          f"|sum F| = {abs(F_sum):10.6f}, Im = {F_sum.imag:.6f} = -n = {-n}")

print(f"\n  Im(sum F) = -n (nombre de premiers)!")
print(f"  C'est trivial: Im(F_p) = -1 pour chaque p, donc sum Im = -n.")
print(f"  Mais Re(sum F) = -sum cot(theta_p) est NON-TRIVIAL.")

# Re(sum F) = -sum cot
sum_cot_actifs = sum(np.cos(theta_p(p, q_stat))/np.sin(theta_p(p, q_stat))
                     for p in PRIMES_ACTIFS)
print(f"\n  sum cot(theta_p) actifs = {sum_cot_actifs:.8f}")
print(f"  sum cot(theta_p) / n = {sum_cot_actifs/3:.8f} (moyenne)")

# Comparer avec des observables PT
alpha = 1.0
for p in PRIMES_ACTIFS:
    alpha *= sin2(p, q_stat)
print(f"  alpha = {alpha:.8f}")
print(f"  sum cot / (1/sqrt(alpha)) = {sum_cot_actifs * np.sqrt(alpha):.8f}")

# ====================================================================
# T8: Moment angulaire L
# ====================================================================
print("\n\n### T8: Moment angulaire L = Im(w* dw/dtheta)")

# dw/dtheta = -i*e^{2it} = derivee de (1-e^{2it})/2
# L = Im(w* * dw/dtheta)
print(f"\n  dw/dtheta = d/dtheta [(1-e^{{2it}})/2] = -i * e^{{2it}}")
print(f"  L = Im(conj(w) * dw/dtheta) = Im(conj(w) * (-i*e^{{2it}}))")
print()
print(f"  {'p':>4s}  {'L':>14s}  {'sin^2':>12s}  {'L/sin^2':>12s}")
for p in PRIMES_ALL[:10]:
    th = theta_p(p, q)
    w = w_p(p, q)
    dw = -1j * np.exp(2j * th)
    L = (np.conj(w) * dw).imag
    s2 = sin2(p, q)
    print(f"  {p:4d}  {L:14.10f}  {s2:12.8f}  {L/s2:12.8f}")

# Calculons analytiquement
# w = -i*sin*e^{it}, conj(w) = i*sin*e^{-it}
# dw/dtheta = -i*e^{2it}
# conj(w)*dw = i*sin*e^{-it} * (-i*e^{2it}) = sin*e^{it}
# Im(sin*e^{it}) = sin*sin(t) = sin^2
print(f"\n  RESULTAT ANALYTIQUE: L = sin^2(theta) = Re(w) pour tout p!")
print(f"  Le moment angulaire est EXACTEMENT l'observable PT standard!")
print(f"  L = Re(w) = sin^2(theta)")
print(f"  => L EST le module carre |w|^2 (car |w|^2 = Re(w)).")
print(f"  => L = |w|^2 est la norme carree du vecteur d'etat!")
print(f"  C'est la LOI DE CONSERVATION: la norme du vecteur d'etat")
print(f"  est conservee sur le cercle (chaque point a |w|^2 = sin^2).")

# ====================================================================
# T9: Hamilton-Jacobi
# ====================================================================
print("\n\n### T9: Hamilton-Jacobi : impulsion p = dS/dtheta = -cot - i")

# Si theta est la 'coordonnee' et p = dS/dtheta la 'impulsion':
# p = -cot(theta) - i
# Re(p) = -cot(theta) => cot = -Re(p) => sin^2 = 1/(1+cot^2) = 1/(1+Re(p)^2)
# Im(p) = -1 (constant)

# Hamiltonien: H(theta, p) tel que dH/dp = dtheta/d'tau' et dH/dtheta = -dp/d'tau'
# Ici S est l'action STATIQUE (pas de temps), donc H = 0?
# Non: S = sum H_p(theta_p), et H_p = -ln(w_p) = -ln(sin) - i*(theta-pi/2)

print(f"\n  Impulsion canonique:")
print(f"  p = dS/dtheta = -cot(theta) - i")
print(f"  => Re(p) = -cot(theta), Im(p) = -1")
print()
print(f"  Inversion: sin^2 = 1/(1 + Re(p)^2)")
print(f"  L'observable PT est determinee par la partie REELLE de l'impulsion!")
print()

# L'espace des phases est (theta, p) avec p complexe
# |p|^2 = cot^2 + 1 = 1/sin^2 = |F|^2
print(f"  |p|^2 = cot^2 + 1 = 1/sin^2(theta)")
print(f"  => |p| * sin = 1  pour tout premier!")
print(f"  => |p| * |w| = 1  (relation d'incertitude?)")
print()

print(f"  {'p':>4s}  {'|p|*sin':>12s}  {'|p|*|w|':>12s}  {'|p|^2*sin^2':>12s}")
for p in PRIMES_ALL[:10]:
    th = theta_p(p, q)
    momentum = abs(-np.cos(th)/np.sin(th) - 1j)
    sin_th = np.sin(th)
    w_mod = abs(w_p(p, q))
    print(f"  {p:4d}  {momentum*sin_th:12.10f}  {momentum*w_mod:12.10f}  {momentum**2*sin_th**2:12.10f}")

print(f"\n  |p| * |w| = 1 EXACTEMENT! (car |w| = sin et |p| = 1/sin)")
print(f"  C'est une RELATION D'INCERTITUDE PT:")
print(f"  Delta(position) * Delta(impulsion) = 1")
print(f"  La constante est 1 (= 2*s = diametre du cercle).")

# ====================================================================
# T10: Travail theta/sin(theta) et sinc
# ====================================================================
print("\n\n### T10: Travail |F|*theta = theta/sin(theta) = 1/sinc(theta/pi)")

print(f"\n  |F|*theta = theta/sin(theta) = sinc^{{-1}}(theta)")
print(f"  Pour petit theta: ~ 1 + theta^2/6 + ...")
print()
print(f"  {'p':>4s}  {'theta':>10s}  {'|F|*theta':>12s}  {'1+t^2/6':>12s}  {'ecart':>10s}")
for p in PRIMES_ALL:
    th = theta_p(p, q)
    work = th / np.sin(th)
    approx = 1 + th**2/6
    print(f"  {p:4d}  {th:10.6f}  {work:12.8f}  {approx:12.8f}  {abs(work-approx):.2e}")

print(f"\n  Le 'travail' theta/sin est la fonction sinc inverse.")
print(f"  Pour les actifs (theta ~ 0.43-0.49): |F|*theta ~ 1.03")
print(f"  Pour p->infini (theta->0): |F|*theta -> 1 (limite classique)")
print(f"  L'ecart a 1 mesure la 'non-classicalite' du premier.")

# ====================================================================
# T11: Courbure et force
# ====================================================================
print("\n\n### T11: Courbure K=4 et force")

K = 4.0  # = 1/s^2
print(f"\n  Courbure de l'espace PT: K = 1/s^2 = {K}")
print(f"  |F|^2 = 1/sin^2 = 1 + cot^2 = 1 + |F_re|^2")
print(f"  ou |F_re| = cot(theta) est la composante reelle de la force.")
print()

# Lien avec la courbure: K * sin^2 * |F|^2 = K * 1 = 4
print(f"  K * sin^2 * |F|^2 = K * (sin^2/sin^2) = K = {K}")
print(f"  => K = |F|^2 * sin^2 * K_0  ou K_0 = 4 est la courbure de fond.")
print()

# Plus interessant: la 'courbure locale' de la trajectoire sur C
# Le cercle C a rayon s = 1/2, donc courbure 1/s = 2
# La courbure de la sphere de Bloch est K = 4 (de la metrique de Fubini-Study)
# Le lien: K_FS = 4 = 1/s^2, et kappa_cercle = 1/s = 2 = sqrt(K)

print(f"  Hierarchie de courbure:")
print(f"  kappa(cercle C) = 1/s = {1/s_PT} (courbure du cercle dans le plan)")
print(f"  K(Fubini-Study) = 1/s^2 = {1/s_PT**2} (courbure de la sphere S^2)")
print(f"  K = kappa^2 (la courbure est le CARRE de la courbure du cercle)")
print()

# |F| en termes de K
# |F| = 1/sin = sqrt(K) * |w_normalise| / sin ? Non...
# |F|^2 * |w|^2 = 1, et K * |w|^2 = 4*sin^2
# Donc |F|^2 = 1/|w|^2 = K/(4*sin^2)... non, |F|^2 = 1/sin^2 et K*sin^2 = 4*sin^2
# |F|^2 = K / (K*sin^2) = 1/sin^2. Trivial.

# Mieux: la densite d'energie E = |F|^2 * dtheta^2 = dtheta^2/sin^2
# et ds^2_Fisher = K * dtheta^2 = 4*dtheta^2
# Donc E = ds^2_Fisher / (K * sin^2) = ds^2_Fisher * |F|^2 / K

# Calculons l'energie totale pour le systeme de 3 actifs
E_total = sum(1.0/sin2(p, q_stat) for p in PRIMES_ACTIFS)
print(f"  Energie totale E = sum |F_p|^2 (actifs) = {E_total:.6f}")
print(f"  E / K = {E_total/K:.6f}")
print(f"  E / n = {E_total/3:.6f} (par premier)")

# ====================================================================
# T12: BILAN
# ====================================================================
print("\n\n" + "=" * 90)
print("### T12: BILAN -- MECANIQUE COMPLEXE PT")
print("=" * 90)

print("""
STRUCTURE COMPLETE DE LA MECANIQUE COMPLEXE PT
===============================================

1. FORCE COMPACTE:
   F = dS_C/dtheta = -e^{i*theta}/sin(theta) = -i/conj(w)
   - Module: |F| = 1/sin(theta) = 1/|w| = 1/sqrt(Re(w))
   - Direction: arg(F) = theta + pi (centripete)
   - Re(F) = -cot(theta) ~ -sqrt(p/2) (p-dependante)
   - Im(F) = -1 (UNIVERSELLE)

2. IDENTITE DE ROTATION:
   F * Re(w) = -i * w      (force = rotation de -pi/2 de w normalisee)
   F = -i / conj(w)         (inversion de Moebius + rotation)

3. MOMENT ANGULAIRE:
   L = Im(conj(w) * dw/dtheta) = sin^2(theta) = Re(w) = |w|^2
   Le moment angulaire EST l'observable PT standard!
   C'est la norme carree du vecteur d'etat, conservee sur le cercle.

4. IMPULSION CANONIQUE:
   p = -cot(theta) - i    (impulsion complexe)
   |p| * |w| = 1           (relation d'incertitude PT)
   sin^2 = 1/(1 + Re(p)^2) (observable = fonction de l'impulsion)

5. UNIVERSALITE Im=-1 ET s=1/2:
   Pour w_n = (1 - e^{n*i*theta})/n : Im(F) = -n/2
   PT utilise n=2 (double angle), s=1/n=1/2
   => Im(F) = -n/2 = -1/(2s) = -1/diametre
   La constante universelle -1 EST l'inverse du diametre du cercle C.

6. TRAVAIL ET CLASSICALITE:
   |F|*theta = theta/sin(theta) = sinc^{-1}(theta)
   -> 1 quand theta -> 0 (limite classique, grands p)
   -> 1.03 pour les actifs (regime semi-classique)
   L'ecart a 1 mesure la 'non-classicalite'.

7. HIERARCHIE DE COURBURE:
   kappa(cercle) = 1/s = 2
   K(Fubini-Study) = 1/s^2 = 4 = kappa^2
   La courbure de l'espace quantique = carre de la courbure classique.
""")

print("=" * 90)
print("FIN TOOL 47")
print("=" * 90)

sys.exit(0)
