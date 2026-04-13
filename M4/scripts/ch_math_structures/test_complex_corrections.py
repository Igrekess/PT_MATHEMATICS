"""
Tool 45: PT Complexe -- Consequences pour la PT et corrections dimensionnelles
================================================================================
Que nous apporte concretement la PT complexe? Ce script explore les
connexions entre la structure du cercle C et les observables PT etablis.

Tests:
  T1:  Le rayon r=1/2 et la borne s^2=1/4 -- le cercle ENCODE s
  T2:  |T12_C| vs T12_re -- renforcement de la borne spectrale
  T3:  sin(2theta) comme correction NLO de sin^2
  T4:  alpha complexe W et corrections physiques
  T5:  La metrique ds^2=4*dtheta^2 et les bornes PT (Q<=1, alpha(1-T00)>=1/4)
  T6:  Corrections dimensionnelles: Im(w) par couche de crible
  T7:  Le ratio Im/Re comme fonction de p -- la "charge" complexe
  T8:  D_KL corrige par la coherence
  T9:  La contraction r_K revisitee dans le plan complexe
  T10: Corrections aux observables physiques (alpha_EM, masses)
  T11: L'argument de W comme angle de Weinberg/mixing?
  T12: Bilan: ce que la PT complexe ajoute a la PT
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
s_PT = 0.5  # parametre fondamental PT

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

def sin_2theta(p, q):
    """sin(2*theta) = 2*sin*cos = coherence."""
    return 2.0 * np.sqrt(sin2(p, q) * cos2(p, q))

def w_p(p, q):
    th = theta_p(p, q)
    return (1.0 - np.exp(2j * th)) / 2.0

def chi3(p):
    r = p % 3
    return 0 if r == 0 else (1 if r == 1 else -1)

print("=" * 90)
print("TOOL 45: PT COMPLEXE -- CONSEQUENCES ET CORRECTIONS DIMENSIONNELLES")
print("=" * 90)

# ====================================================================
# T1: Le rayon r=1/2 et la borne s^2=1/4
# ====================================================================
print("\n### T1: Le cercle C encode le parametre fondamental s = 1/2")
print("    Cercle C: centre (1/2, 0), rayon r = 1/2")
print("    r = s = 1/2 (parametre PT fondamental)")
print("    r^2 = s^2 = 1/4 (borne fondamentale PT)")
print()

# La borne PT: alpha*(1-T00) >= s^2 = 1/4
# C'est AUSSI: distance maximale au centre^2 = 1/4
# Car |w - 1/2|^2 = 1/4 pour tout w sur C
q = q_stat
print(f"  Pour tout premier p, |w_p - 1/2|^2 = 1/4 = s^2:")
for p in PRIMES_ALL:
    w = w_p(p, q)
    dist2 = abs(w - 0.5)**2
    print(f"    p={p:3d}: |w-1/2|^2 = {dist2:.12f} (ecart a 1/4: {abs(dist2-0.25):.2e})")

print(f"\n  Le rayon du cercle EST le parametre s:")
print(f"    r_cercle = 1/2 = s = {s_PT}")
print(f"    r^2 = 1/4 = s^2 = {s_PT**2}")
print(f"\n  Le centre du cercle est a (s, 0) = (1/2, 0)")
print(f"  Le cercle est C(s, 0; s) -- TOUT est parametrise par s!")

# Decomposition: w = s - s*e^{2i*theta} = s*(1 - e^{2i*theta})
# Ou de maniere equivalente: w = s + s*vecteur_unitaire
print(f"\n  w_p = s * (1 - e^{{2i*theta_p}}) = s - s*z_p")
print(f"  Le prefacteur global est s = 1/2!")
print(f"  Si s etait different, le cercle aurait rayon s.")
print(f"  => La TAILLE de l'espace complexe PT est fixee par s.")

# ====================================================================
# T2: |T12_C| vs T12_re -- renforcement spectral
# ====================================================================
print("\n\n### T2: T12 complexe -- renforcement de la borne spectrale")
print("    T12_re = Re(T12_C), mais |T12_C| > |T12_re|")
print("    Le module complexe est la 'vraie' valeur de T12")
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

    ratio = abs(T12_C) / abs(T12_re)
    print(f"  {label}:")
    print(f"    T12_re   = {T12_re:.10f}")
    print(f"    |T12_C|  = {abs(T12_C):.10f}")
    print(f"    Ratio |T12_C|/|T12_re| = {ratio:.6f}")
    print(f"    Gain: +{(ratio-1)*100:.1f}% sur la borne spectrale")
    print()

    # Decomposition: T12_C = T12_re + i*T12_im
    T12_im = T12_C.imag
    print(f"    T12_im   = {T12_im:.10f}")
    print(f"    |T12_C|^2 = T12_re^2 + T12_im^2 = {T12_re**2 + T12_im**2:.12f}")
    print(f"    = {abs(T12_C)**2:.12f} (check)")
    print(f"    T12_im/T12_re = {T12_im/T12_re:.6f}")
    print()

# ====================================================================
# T3: sin(2theta) comme correction NLO
# ====================================================================
print("\n### T3: sin(2theta) comme correction NLO de sin^2")
print("    sin^2 est l'observable LO (leading order)")
print("    sin(2theta)/2 = sqrt(sin^2 * cos^2) = NLO?")
print("    Ratio correction: |Im(w)|/Re(w) = sin(2theta)/(2*sin^2) = cos/sin = cot(theta)")
print()

q = q_stat
print(f"  {'p':>4} {'sin^2 (LO)':>12} {'sin2t/2 (NLO)':>14} {'cot(theta)':>12} {'NLO/LO':>10}")
for p in PRIMES_ALL:
    lo = sin2(p, q)
    nlo = sin_2theta(p, q) / 2.0
    th = theta_p(p, q)
    cot = np.cos(th) / np.sin(th)
    ratio = nlo / lo
    print(f"  {p:4d} {lo:12.8f} {nlo:14.8f} {cot:12.6f} {ratio:10.6f}")

print(f"\n  Le ratio NLO/LO = cot(theta_p) CROIT avec p!")
print(f"  Pour p grand: theta ~ sqrt(2/p) -> 0, cot ~ 1/theta ~ sqrt(p/2)")
print(f"  => La correction NLO DOMINE pour les grands premiers!")
print(f"  => C'est pourquoi Im(Sigma) >> Re(Sigma) dans la somme totale.")

# La correction cumulee
sum_LO = sum(sin2(p, q) for p in PRIMES_ACTIFS)
sum_NLO = sum(sin_2theta(p, q) / 2.0 for p in PRIMES_ACTIFS)
print(f"\n  Actifs: sum LO = {sum_LO:.8f}, sum NLO = {sum_NLO:.8f}")
print(f"  NLO/LO global = {sum_NLO/sum_LO:.6f}")

# ====================================================================
# T4: Alpha complexe et corrections
# ====================================================================
print("\n\n### T4: Alpha complexe W = sqrt(alpha) * e^{i*phi}")
print("    alpha_EM = |W|^2 (module carre)")
print("    phi_W = arg(W) (phase du couplage)")
print("    La correction complexe: alpha_corrige = alpha * f(phi)")
print()

q = q_stat
W = 1.0 + 0j
alpha_re = 1.0
for p in PRIMES_ACTIFS:
    W *= w_p(p, q)
    alpha_re *= sin2(p, q)

phi_W = cmath.phase(W)
print(f"  alpha = |W|^2 = {alpha_re:.12f}")
print(f"  1/alpha = {1/alpha_re:.6f}")
print(f"  phi_W = {phi_W:.10f} rad = {phi_W/math.pi:.8f} pi")
print()

# Le couplage "vrai" incluant la phase:
# Re(W) = sqrt(alpha)*cos(phi) -- la projection sur l'axe reel
# Mais alpha = Pi sin^2 est deja le bon couplage (module carre).
# La phase ajoute une DIRECTION, pas une correction de magnitude.

# Cependant, pour les PRODUITS de constantes PT, la phase intervient:
# Exemple: W1 * W2 => alpha12 = |W1*W2|^2 = alpha1*alpha2 (meme module)
# MAIS arg(W1*W2) = arg(W1) + arg(W2) (les phases s'additionnent)

# La phase comme angle de mixing?
# En physique: l'angle de Weinberg theta_W ~ 0.23 (sin^2 theta_W = 0.231)
# En PT: phi_W/pi = 0.937 => phi_W = 2.943 rad
# Pas de lien direct evident.

# Mais: arg(W) = sum(theta_p - pi/2) pour actifs
# = sum(theta_p) - 3*pi/2
# = (theta_3 + theta_5 + theta_7) - 3*pi/2
sum_th = sum(theta_p(p, q) for p in PRIMES_ACTIFS)
print(f"  arg(W) = sum(theta) - n*pi/2 = {sum_th:.6f} - {len(PRIMES_ACTIFS)}*pi/2 = {sum_th - len(PRIMES_ACTIFS)*math.pi/2:.6f}")
print(f"  Verif: {phi_W:.6f} (match: {abs(phi_W - (sum_th - len(PRIMES_ACTIFS)*math.pi/2)) < 1e-6})")
print()

# Correction a alpha via la phase: alpha_eff = alpha * (1 + epsilon)
# ou epsilon capture l'effet de la coherence
# epsilon ~ Im(W)/Re(W) * (facteur geometrique)
# En fait: W = Re(W) + i*Im(W), |W|^2 = Re^2 + Im^2 = alpha
# Si on ne gardait que Re(W): alpha_re_only = Re(W)^2
alpha_re_only = W.real**2
alpha_im_only = W.imag**2
print(f"  Decomposition de alpha = Re(W)^2 + Im(W)^2:")
print(f"    Re(W)^2  = {alpha_re_only:.12f} ({alpha_re_only/alpha_re*100:.2f}%)")
print(f"    Im(W)^2  = {alpha_im_only:.12f} ({alpha_im_only/alpha_re*100:.2f}%)")
print(f"    Total    = {alpha_re_only+alpha_im_only:.12f} = alpha")
print(f"  => {alpha_im_only/alpha_re*100:.2f}% de alpha vient de la partie imaginaire!")

# ====================================================================
# T5: Metrique ds^2 = 4*dtheta^2 et bornes PT
# ====================================================================
print("\n\n### T5: La metrique et les bornes PT")
print("    ds^2_Fisher = 4*dtheta^2 sur le cercle C")
print("    Borne: alpha*(1-T00) >= s^2 = 1/4 = r_cercle^2")
print("    Question: le 4 de Fisher = le 4 de la borne?")
print()

q = q_stat
# La borne alpha*(1-T00) >= 1/4:
# alpha = Pi sin^2, T00 est la fraction de gaps = 0 mod 3
# En PT: T00 ~ 1/3 asymptotiquement

# Calculons alpha*(1-T00) pour les actifs
alpha_val = 1.0
for p in PRIMES_ACTIFS:
    alpha_val *= sin2(p, q)

# T00 n'est pas directement dans les actifs, mais on peut estimer
# T00 ~ 1/3 (equipartition mod 3)
T00_est = 1.0/3.0
bound_val = alpha_val * (1 - T00_est)

print(f"  alpha = {alpha_val:.10f}")
print(f"  T00 ~ {T00_est:.6f}")
print(f"  alpha*(1-T00) = {bound_val:.10f}")
print(f"  s^2 = {s_PT**2:.6f}")
print(f"  Borne satisfaite? {bound_val >= s_PT**2 - 1e-10}")
print()

# Le lien geometrique:
# Sur le cercle C(s, 0; s):
# |w - s|^2 = s^2 pour tout w sur C
# => s^2 est la distance MAXIMALE au centre (carree)
# C'est aussi la VARIANCE de w sur le cercle (si w uniformement distribue)

# Le facteur 4 de Fisher:
# ds^2_F = 4*dtheta^2
# La longueur Fisher d'une transition p -> p+1:
# L_F = 2 * |theta_p - theta_{p+1}|
# La longueur angulaire: L_arc = |theta_p - theta_{p+1}|
# Ratio: L_F / L_arc = 2 = 2*s (ou s = 1/2)
print(f"  Le facteur 2 = 2*s = 2*(1/2) = 1")
print(f"  NON: le facteur 2 vient de la metrique de Fubini-Study,")
print(f"  pas directement de s. Mais le cercle a rayon s = 1/2,")
print(f"  et la courbure K = 1/s^2 = 4.")
print(f"  => La courbure de l'espace PT est K = 4 = 1/s^2")
print(f"  => s fixe la courbure de l'espace des etats!")

# ====================================================================
# T6: Corrections dimensionnelles par couche
# ====================================================================
print("\n\n### T6: Corrections dimensionnelles par couche de crible")
print("    En PT, les corrections NLO viennent des ordres superieurs.")
print("    La coherence Im(w) = -sin*cos fournit une correction NATURELLE.")
print("    Par couche: correction(p) = Im(w_p) / Re(w_p) = -cot(theta_p)")
print()

q = q_stat

# Pour chaque couche (actifs, ghosts, au-dela):
print(f"  Couche par couche:")
print(f"  {'Couche':>20} {'sum Re(w)':>12} {'sum |Im(w)|':>12} {'|Im|/Re':>10} {'correction%':>12}")

layers = [
    ("Actifs (3,5,7)", PRIMES_ACTIFS),
    ("Ghosts (11,13)", PRIMES_GHOST),
    ("p=17..47", [p for p in PRIMES_ALL if p >= 17]),
    ("Tous (2..47)", PRIMES_ALL),
]

for name, primes in layers:
    sum_re = sum(sin2(p, q) for p in primes)
    sum_im = sum(abs(w_p(p, q).imag) for p in primes)
    ratio = sum_im / sum_re if sum_re > 0 else 0
    print(f"  {name:>20} {sum_re:12.8f} {sum_im:12.8f} {ratio:10.6f} {ratio*100:11.2f}%")

# La correction dimensionnelle:
# En PT, alpha_EM = Pi sin^2 (actifs).
# La version "corrigee" inclurait l'angle:
# alpha_C = |Pi w_p|^2 = Pi sin^2 = alpha (meme module!)
# MAIS: si on calcule une observable qui depend de la PHASE,
# comme la masse effective m_eff = Re(W) au lieu de |W|,
# alors la correction est:
# m_eff = sqrt(alpha) * cos(phi_W)
# Correction: cos(phi_W) par rapport a 1

W_actifs = 1.0 + 0j
for p in PRIMES_ACTIFS:
    W_actifs *= w_p(p, q)

phi = cmath.phase(W_actifs)
cos_correction = np.cos(phi)
print(f"\n  Correction de phase sur sqrt(alpha):")
print(f"    sqrt(alpha) = {abs(W_actifs):.10f}")
print(f"    cos(phi_W) = {cos_correction:.10f}")
print(f"    sqrt(alpha)*cos(phi) = Re(W) = {W_actifs.real:.10f}")
print(f"    Correction: {(cos_correction - 1)*100:.4f}% (reduction de {(1-cos_correction)*100:.2f}%)")

# ====================================================================
# T7: Le ratio Im/Re comme "charge complexe"
# ====================================================================
print("\n\n### T7: La charge complexe Im/Re = -cot(theta)")
print("    Pour chaque premier, cot(theta) = cos/sin mesure la 'charge'")
print("    = rapport coherence/perte. C'est aussi |Im(w)/Re(w)|.")
print()

q = q_stat
print(f"  {'p':>4} {'cot(theta)':>12} {'1/cot=tan':>12} {'sin^2':>10} {'delta':>10}")
for p in PRIMES_ALL:
    th = theta_p(p, q)
    cot = np.cos(th) / np.sin(th)
    d = delta_p(p, q)
    print(f"  {p:4d} {cot:12.6f} {1/cot:12.6f} {sin2(p,q):10.6f} {d:10.6f}")

# La charge cot(theta) ~ 1/theta ~ sqrt(p/2) pour grands p
# Ceci est la TAILLE de la correction complexe
# Pour p petit (p=3): cot ~ 1.89, correction ~ 189%
# Pour p grand (p=47): cot ~ 4.77, correction ~ 477%
print(f"\n  cot(theta_p) CROIT comme sqrt(p/2):")
print(f"  {'p':>4} {'cot':>10} {'sqrt(p/2)':>10} {'ratio':>10}")
for p in PRIMES_ALL:
    th = theta_p(p, q)
    cot = np.cos(th) / np.sin(th)
    approx = np.sqrt(p / 2.0)
    print(f"  {p:4d} {cot:10.6f} {approx:10.6f} {cot/approx:10.6f}")

print(f"\n  La 'charge complexe' est asymptotiquement sqrt(p/2).")
print(f"  Ceci est DUAL au deficit delta ~ 1/p.")
print(f"  Produit: cot * delta ~ sqrt(2/p) * 1/p ~ sqrt(2)/p^{{3/2}}")

# ====================================================================
# T8: D_KL corrige par la coherence
# ====================================================================
print("\n\n### T8: D_KL et la coherence")
print("    D_KL = sum p_i ln(p_i/q_i) mesure la distance a l'equipartition")
print("    La coherence ajoute un terme croise")
print()

q = q_stat
# D_KL standard: pour distribution (sin^2, cos^2) vs (1/2, 1/2)
# D_KL = sin^2*ln(2*sin^2) + cos^2*ln(2*cos^2)
# = ln(2) + sin^2*ln(sin^2) + cos^2*ln(cos^2)
# = ln(2) - H(sin^2, cos^2)

print(f"  D_KL par premier (distance a l'equipartition):")
print(f"  {'p':>4} {'D_KL':>12} {'sin2theta/2':>14} {'D_KL_corr':>12} {'correction%':>12}")

D_KL_total = 0.0
D_KL_corr_total = 0.0
for p in PRIMES_ACTIFS:
    s2 = sin2(p, q)
    c2 = cos2(p, q)
    dkl = s2 * np.log(2 * s2) + c2 * np.log(2 * c2)
    # Correction: terme de coherence sin*cos
    coherence = sin_2theta(p, q) / 2.0
    # La "D_KL complexe" naturelle:
    # D_KL_C = |w*ln(2w)| = module du KL complexe
    w = w_p(p, q)
    dkl_c = abs(w * cmath.log(2 * w))  # module du KL complexe
    correction = (dkl_c - dkl) / dkl * 100
    D_KL_total += dkl
    D_KL_corr_total += dkl_c
    print(f"  {p:4d} {dkl:12.8f} {coherence:14.8f} {dkl_c:12.8f} {correction:11.2f}%")

print(f"\n  Total actifs: D_KL = {D_KL_total:.8f}, |D_KL_C| = {D_KL_corr_total:.8f}")
print(f"  Correction globale: {(D_KL_corr_total/D_KL_total - 1)*100:.2f}%")

# ====================================================================
# T9: Contraction r_K revisitee
# ====================================================================
print("\n\n### T9: Contraction dans le plan complexe")
print("    En PT reel: r_K = max|S_K| / sqrt(N_K)")
print("    En PT complexe: le chemin sur le cercle C a une vitesse")
print("    v_p = |dw/d(ln p)| qui donne une 'contraction angulaire'")
print()

q = q_stat
# Contraction: |w_{p+1}| / |w_p| = sin(theta_{p+1}) / sin(theta_p) < 1
print(f"  Contraction radiale |w_{p+1}|/|w_p| et angulaire:")
print(f"  {'p->p_':>8} {'|w2|/|w1|':>12} {'d_theta':>12} {'d_arg/pi':>12}")
for i in range(len(PRIMES_ALL) - 1):
    p1, p2 = PRIMES_ALL[i], PRIMES_ALL[i + 1]
    w1 = w_p(p1, q)
    w2 = w_p(p2, q)
    r_ratio = abs(w2) / abs(w1)
    dtheta = theta_p(p1, q) - theta_p(p2, q)
    darg = cmath.phase(w2 / w1)
    print(f"  {p1:3d}->{p2:<3d} {r_ratio:12.8f} {dtheta:12.8f} {darg/math.pi:12.8f}")

# Taux de contraction moyen
ratios = [abs(w_p(PRIMES_ALL[i+1], q)) / abs(w_p(PRIMES_ALL[i], q))
          for i in range(len(PRIMES_ALL) - 1)]
r_mean = np.exp(np.mean(np.log(ratios)))
print(f"\n  Contraction geometrique moyenne: {r_mean:.8f}")
print(f"  = exp(mean ln(r)) par pas de premier")

# La contraction totale: |w_47| / |w_2|
total_contraction = abs(w_p(47, q)) / abs(w_p(2, q))
print(f"  Contraction totale (2->47): {total_contraction:.8f}")
print(f"  = sin(theta_47)/sin(theta_2) = {np.sin(theta_p(47,q))/np.sin(theta_p(2,q)):.8f}")

# ====================================================================
# T10: Corrections aux observables physiques
# ====================================================================
print("\n\n### T10: Corrections complexes aux observables physiques")
print("    En PT: alpha_EM ~ 1/137 vient de Pi sin^2(actifs)")
print("    La correction complexe la plus naturelle: inclure la coherence")
print()

q = q_stat

# alpha standard
alpha = 1.0
for p in PRIMES_ACTIFS:
    alpha *= sin2(p, q)
print(f"  alpha_EM (standard) = {alpha:.12f}, 1/alpha = {1/alpha:.6f}")

# Corrections possibles:
# 1. alpha_C = |Pi w_p|^2 = Pi sin^2 = alpha (pas de correction!)
#    Le module carre est invariant.

# 2. Correction par Re(W) au lieu de |W|:
#    alpha_Re = Re(W)^2 (projection sur le reel)
W = 1.0 + 0j
for p in PRIMES_ACTIFS:
    W *= w_p(p, q)
alpha_Re = W.real**2
print(f"  alpha_Re = Re(W)^2 = {alpha_Re:.12f}, 1/alpha_Re = {1/alpha_Re:.6f}")

# 3. Correction NLO via sin(2theta):
# alpha_NLO = Pi [sin^2 + epsilon*sin(2theta)/2]
# avec epsilon petit (couplage coherence)
for eps in [0.01, 0.05, 0.1]:
    alpha_nlo = 1.0
    for p in PRIMES_ACTIFS:
        s2 = sin2(p, q)
        s2t = sin_2theta(p, q) / 2.0
        alpha_nlo *= (s2 + eps * s2t)
    print(f"  alpha_NLO(eps={eps:.2f}) = {alpha_nlo:.12f}, 1/alpha = {1/alpha_nlo:.6f}, delta = {(alpha_nlo-alpha)/alpha*100:.4f}%")

# 4. Correction ghost via Im:
# L'effet des ghosts (11, 13) sur la phase
W_ghost = W
for p in PRIMES_GHOST:
    W_ghost *= w_p(p, q)
print(f"\n  Avec ghosts: |W_ghost|^2 = {abs(W_ghost)**2:.12f}")
print(f"  Phase shift ghost: {(cmath.phase(W_ghost)-cmath.phase(W))/math.pi:.6f} pi")

# ====================================================================
# T11: arg(W) comme angle de mixing?
# ====================================================================
print("\n\n### T11: arg(W) et angles de mixing PT")
print("    arg(W) = sum(theta_p - pi/2) pour les actifs")
print("    Valeur: arg(W) = 0.937*pi = 2.943 rad")
print("    Y a-t-il un lien avec les angles de mixing connus?")
print()

q = q_stat
W_act = 1.0 + 0j
for p in PRIMES_ACTIFS:
    W_act *= w_p(p, q)
phi_act = cmath.phase(W_act)

# Angles de mixing PT connus:
# Weinberg: sin^2(theta_W) = 0.2312 (~ sin^2(theta_2) = 0.2334!)
# Cabibbo: sin(theta_C) ~ 0.225, theta_C ~ 0.227 rad
# CKM: V_us ~ 0.225

print(f"  arg(W_actifs) = {phi_act:.8f} rad = {phi_act/math.pi:.6f} pi")
print(f"  arg(W_actifs) mod pi = {phi_act % math.pi:.8f} rad")
print(f"  arg(W_actifs) mod pi/2 = {phi_act % (math.pi/2):.8f} rad")
print()

# Test: arg(W) mod pi ~ theta_2 ?
# arg(W) = 2.943, mod pi = 2.943 - pi = -0.199... non
# Plutot: pi - arg(W) = pi - 2.943 = 0.199 rad
# Et theta_2 = 0.504 rad. Pas de match.

# Test: les theta_p individuels comme angles de mixing
sin2_W = 0.2312  # Weinberg
print(f"  Angle de Weinberg: sin^2(theta_W) = {sin2_W:.4f}")
print(f"  Comparaison avec sin^2(theta_p):")
for p in [2, 3, 5]:
    s2 = sin2(p, q)
    print(f"    sin^2(theta_{p}) = {s2:.6f} (ecart: {abs(s2-sin2_W)/sin2_W*100:.2f}%)")

# sin^2(theta_2) = 0.2334 vs sin^2(theta_W) = 0.2312 => ecart 0.95%!
print(f"\n  sin^2(theta_2) = {sin2(2, q):.6f} est a {abs(sin2(2,q)-sin2_W)/sin2_W*100:.2f}% de sin^2(theta_W)!")
print(f"  MAIS theta_2 utilise p=2 qui n'est PAS un actif de T_3.")

# Les phases des produits partiels
print(f"\n  Phases des produits partiels (q_stat):")
W_partial = 1.0 + 0j
for p in PRIMES_ALL:
    W_partial *= w_p(p, q)
    if p in [3, 5, 7, 11, 13, 17, 47]:
        print(f"    Pi_{p<=p} w_p: arg/pi = {cmath.phase(W_partial)/math.pi:.8f}, |W|^2 = {abs(W_partial)**2:.4e}")

# ====================================================================
# T12: Bilan
# ====================================================================
print("\n\n" + "=" * 90)
print("### T12: BILAN -- CE QUE LA PT COMPLEXE APPORTE")
print("=" * 90)
print(f"""
1. LE CERCLE ENCODE s:
   C(s, 0; s) avec s = 1/2. Le rayon EST le parametre fondamental.
   La courbure K = 1/s^2 = 4 = facteur de Fisher.
   La borne s^2 = 1/4 est le rayon carre.
   => s est une constante GEOMETRIQUE de l'espace complexe PT.

2. RENFORCEMENT SPECTRAL:
   |T12_C| > |T12_re| de ~29% (q_stat) a ~55% (q_therm).
   Le module complexe donne une borne spectrale PLUS FORTE.
   L'ecart |T12_C|/|T12_re| = sqrt(1 + (Im/Re)^2) = sqrt(1 + cot^2)... = 1/sin.
   En fait |T12_C| = |w_7 - w_5|/3, la DISTANCE dans le plan.

3. CORRECTIONS NLO VIA LA COHERENCE:
   sin(2theta)/2 = coherence = NLO naturel.
   Le ratio NLO/LO = cot(theta) ~ sqrt(p/2), CROISSANT.
   Pour les grands premiers, la correction DOMINE la valeur LO.
   C'est le signe que la PT reelle sous-estime la structure.

4. LA CHARGE COMPLEXE cot(theta):
   cot(theta_p) ~ sqrt(p/2) est DUAL au deficit delta ~ 1/p.
   Produit cot*delta ~ 1/p^(3/2).
   La charge mesure la TAILLE relative de la coherence vs la perte.

5. ALPHA ET LA PHASE:
   alpha = |W|^2 est INVARIANT sous complexification (pas de correction).
   Mais Re(W)^2 = {alpha_Re/alpha*100:.1f}% de alpha.
   Les {alpha_im_only/alpha*100:.1f}% restants viennent de Im(W).
   La phase arg(W) = {phi_act/math.pi:.4f}*pi est une observable independante.

6. CORRECTIONS DIMENSIONNELLES:
   La correction la plus directe vient du rapport |Im(w)|/Re(w) = cot.
   Par couche: actifs cot ~ 2, ghosts cot ~ 3, grands p: cot ~ sqrt(p/2).
   L'espace complexe est PLUS GRAND que le reel d'un facteur cot.
   Pour la physique: cela signifie que les constantes de couplage
   effectives devraient inclure un facteur geometrique lie a cot.

7. DIMENSION 3+1 ET LE CERCLE:
   Le cercle C est un espace 1D (parametre par theta).
   La sphere de Bloch S^2 est un espace 2D.
   L'extension a q complexe donne un espace 3D (theta, phi_Bloch, |q|).
   Avec le "temps" de crible (K croissant): 3+1 D.
   La reduction au plan reel (Phi=0) donne la PT standard 1D.
""")

print("=" * 90)
print("FIN TOOL 45")
print("=" * 90)

sys.exit(0)
