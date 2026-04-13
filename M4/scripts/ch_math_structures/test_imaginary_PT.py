"""
Tool 41: La partie imaginaire de PT
=====================================
PT utilise sin²(θ_p) = δ(2-δ) pour TOUT construire.
Mais sin² = (1-cos 2θ)/2, donc PT utilise Re(1-e^{2iθ})/2.
La partie imaginaire sin(2θ) = Im(e^{2iθ}) est SYSTEMATIQUEMENT jetee.

Qu'est-ce que sin(2θ) represente en PT ?

sin(2θ) = 2 sin(θ) cos(θ) = 2 √(perte) × √(conservation)
         = terme d'INTERFERENCE entre les deux canaux

Ce script explore cette observable SANS lien avec RH.

Tests:
  T1: Decomposition sin²/cos²/sin(2θ) -- les 3 projections
  T2: L'identite fondamentale sin² + cos² = 1 en PT
  T3: sin(2θ) comme observable geometrique (aire du parallelogramme)
  T4: Action complexe S_C = -sum ln(sin²) + 2i*sum(θ)
  T5: Observables PT en version complexe (T12, Q, D_KL)
  T6: Le "courant" J_p = Im(z_p * conj(z_{p'})) entre deux premiers
  T7: Sommes partielles et convergence
  T8: Identites nouvelles
  T9: Bilan
"""

import numpy as np
import math
import sys

# ── Parametres PT fondamentaux ──
q_stat = 13.0 / 15.0
q_therm = np.exp(-1.0 / 15.0)
mu_star = 15
PRIMES_ACTIFS = [3, 5, 7]
PRIMES_ALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def delta_p(p, q):
    return (1.0 - q**p) / p

def sin2_theta(p, q):
    d = delta_p(p, q)
    return d * (2.0 - d)

def cos2_theta(p, q):
    """cos²(θ) = (1 - δ)² -- observable DUAL (conservation)."""
    d = delta_p(p, q)
    return (1.0 - d)**2

def sin_2theta(p, q):
    """sin(2θ) = 2√(sin²)·(1-δ) = 2√(sin²·cos²) -- INTERFERENCE."""
    s2 = sin2_theta(p, q)
    c2 = cos2_theta(p, q)
    return 2.0 * np.sqrt(s2 * c2)

def theta_p(p, q):
    """θ_p = arcsin(√sin²)."""
    return np.arcsin(np.sqrt(sin2_theta(p, q)))

def z_p(p, q):
    """z_p = e^{2iθ_p} = cos(2θ) + i sin(2θ)."""
    th = theta_p(p, q)
    return np.exp(2j * th)

# ── chi3 character ──
def chi3(p):
    r = p % 3
    return 0 if r == 0 else (1 if r == 1 else -1)

print("=" * 90)
print("TOOL 41: LA PARTIE IMAGINAIRE DE PT")
print("=" * 90)

# ══════════════════════════════════════════════════════════════════
# T1: Les trois projections
# ══════════════════════════════════════════════════════════════════
print("\n### T1: Les trois projections de z_p = e^{2iθ_p}")
print("    Re(z_p) = cos(2θ) = 1 - 2sin² = 2cos² - 1")
print("    Im(z_p) = sin(2θ) = 2√(sin²·cos²) = INTERFERENCE")
print("    |z_p| = 1 toujours (cercle unite)")
print()

for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    print(f"  {label} = {q:.6f}:")
    print(f"  {'p':>4} {'sin²':>10} {'cos²':>10} {'sin(2θ)':>10} {'cos(2θ)':>10} {'θ/π':>8} {'check':>8}")
    for p in PRIMES_ALL:
        s2 = sin2_theta(p, q)
        c2 = cos2_theta(p, q)
        s2t = sin_2theta(p, q)
        c2t = 1.0 - 2 * s2  # cos(2θ) = 1 - 2sin²
        th = theta_p(p, q) / np.pi
        check = s2 + c2  # Should be 1
        print(f"  {p:4d} {s2:10.6f} {c2:10.6f} {s2t:10.6f} {c2t:10.6f} {th:8.4f} {check:8.6f}")
    print()

# ══════════════════════════════════════════════════════════════════
# T2: Identite fondamentale et ses consequences PT
# ══════════════════════════════════════════════════════════════════
print("\n### T2: L'identite sin² + cos² = 1 en PT")
print("    sin² = perte (PT standard)")
print("    cos² = (1-δ)² = conservation")
print("    sin² + cos² = 1 est une LOI DE CONSERVATION de l'information!")
print()

# Sum over active primes
for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    S_loss = sum(sin2_theta(p, q) for p in PRIMES_ACTIFS)
    S_cons = sum(cos2_theta(p, q) for p in PRIMES_ACTIFS)
    S_interf = sum(sin_2theta(p, q) for p in PRIMES_ACTIFS)

    print(f"  {label} (p=3,5,7):")
    print(f"    Σ sin²    = {S_loss:.8f}  (perte totale)")
    print(f"    Σ cos²    = {S_cons:.8f}  (conservation totale)")
    print(f"    Σ sin(2θ) = {S_interf:.8f}  (interference totale)")
    print(f"    Σ(sin²+cos²) = {S_loss + S_cons:.8f}  (=3, nombre de premiers actifs)")
    print(f"    Ratio Im/Re: sin(2θ)/sin² = {S_interf / S_loss:.6f}")
    print()

# ══════════════════════════════════════════════════════════════════
# T3: sin(2θ) comme observable geometrique
# ══════════════════════════════════════════════════════════════════
print("\n### T3: Interpretation geometrique de sin(2θ)")
print("    sin(2θ) = 2 sin(θ) cos(θ)")
print("    = 2 × √(perte) × √(conservation)")
print("    = MOYENNE GEOMETRIQUE des deux canaux, fois 2")
print("    C'est l'AIRE du parallelogramme forme par (sin,cos)")
print()

for label, q in [("q_stat", q_stat)]:
    print(f"  {label}:")
    print(f"  {'p':>4} {'√sin²':>10} {'√cos²':>10} {'prod':>10} {'sin(2θ)/2':>10} {'match':>6}")
    for p in PRIMES_ALL[:8]:
        sq_s = np.sqrt(sin2_theta(p, q))
        sq_c = np.sqrt(cos2_theta(p, q))
        prod = sq_s * sq_c
        half_s2t = sin_2theta(p, q) / 2
        print(f"  {p:4d} {sq_s:10.6f} {sq_c:10.6f} {prod:10.6f} {half_s2t:10.6f} {abs(prod - half_s2t) < 1e-12:>6}")

# Maximum de sin(2θ) : quand sin² = cos² = 1/2, soit θ = π/4
# En PT: δ(2-δ) = 1/2 => δ = 1 - 1/√2 ≈ 0.2929
print(f"\n  Maximum de sin(2θ) = 1 atteint quand sin² = cos² = 1/2")
print(f"  En PT: δ_max = 1 - 1/√2 = {1 - 1/np.sqrt(2):.6f}")
print(f"  Aucun δ_p reel n'atteint ce maximum (δ_3 = {delta_p(3, q_stat):.6f})")

# ══════════════════════════════════════════════════════════════════
# T4: Action complexe S_C
# ══════════════════════════════════════════════════════════════════
print("\n\n### T4: Action complexifiee S_C = S_re + i·S_im")
print("    S_re = -Σ ln(sin²) = S_PT standard (Polyakov)")
print("    S_im = 2Σ θ_p = phase totale")
print("    S_C = -Σ ln(z_p) = -Σ [ln|sin θ|² + 2iθ_p]... NON")
print("    En fait: S_C = -Σ ln(sin² · e^{2iθ}) = S_re - 2i·Σθ")
print()

for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    S_re_act = sum(-np.log(sin2_theta(p, q)) for p in PRIMES_ACTIFS)
    S_im_act = 2 * sum(theta_p(p, q) for p in PRIMES_ACTIFS)
    S_re_all = sum(-np.log(sin2_theta(p, q)) for p in PRIMES_ALL)
    S_im_all = 2 * sum(theta_p(p, q) for p in PRIMES_ALL)

    print(f"  {label}:")
    print(f"    Actifs (3,5,7): S_re = {S_re_act:.8f}, S_im = {S_im_act:.8f}")
    print(f"      |S_C| = {np.sqrt(S_re_act**2 + S_im_act**2):.8f}")
    print(f"      arg(S_C) = {np.arctan2(S_im_act, S_re_act) / np.pi:.6f} π")
    print(f"    Tous (2..47): S_re = {S_re_all:.8f}, S_im = {S_im_all:.8f}")
    print(f"      S_im / π = {S_im_all / np.pi:.6f}")
    print(f"      S_im / S_re = {S_im_all / S_re_all:.6f}")
    print()

# ══════════════════════════════════════════════════════════════════
# T5: Observables PT en version complexe
# ══════════════════════════════════════════════════════════════════
print("\n### T5: Observables PT avec la partie imaginaire")
print("    PT standard: produit α_EM = Π sin²(θ_p)")
print("    Complexifie: α_C = Π z_p = Π e^{2iθ_p} = e^{2i·Σθ}")
print()

for label, q in [("q_stat", q_stat), ("q_therm", q_therm)]:
    # Standard alpha
    alpha_real = np.prod([sin2_theta(p, q) for p in PRIMES_ACTIFS])

    # Complex alpha
    alpha_C = np.prod([z_p(p, q) for p in PRIMES_ACTIFS])
    sum_theta = sum(theta_p(p, q) for p in PRIMES_ACTIFS)

    print(f"  {label}:")
    print(f"    α_EM (standard) = Π sin² = {alpha_real:.8f}")
    print(f"    α_C = Π e^{'{2iθ}'} = e^{'{2i·Σθ}'}")
    print(f"    |α_C| = {abs(alpha_C):.8f} (=1 car |z_p|=1)")
    print(f"    arg(α_C) = 2·Σθ = {2 * sum_theta:.8f} rad = {2 * sum_theta / np.pi:.6f} π")
    print(f"    cos(arg α_C) = {np.cos(2 * sum_theta):.8f}")
    print(f"    sin(arg α_C) = {np.sin(2 * sum_theta):.8f}")
    print()

    # T12 complex: χ₃·z_p produit
    T12_re = sum(chi3(p) * sin2_theta(p, q) for p in PRIMES_ACTIFS) / len(PRIMES_ACTIFS)
    T12_im = sum(chi3(p) * sin_2theta(p, q) for p in PRIMES_ACTIFS) / len(PRIMES_ACTIFS)
    print(f"    T12_re = (1/3)·Σ χ₃·sin² = {T12_re:.8f}")
    print(f"    T12_im = (1/3)·Σ χ₃·sin(2θ) = {T12_im:.8f}")
    print(f"    T12_C = {T12_re:.8f} + {T12_im:.8f}i")
    print(f"    |T12_C| = {np.sqrt(T12_re**2 + T12_im**2):.8f}")
    print()

# ══════════════════════════════════════════════════════════════════
# T6: "Courant" entre premiers
# ══════════════════════════════════════════════════════════════════
print("\n### T6: Courant J(p,p') = Im(z_p · conj(z_{p'}))")
print("    J mesure le TRANSFERT DE PHASE entre deux premiers")
print("    J(p,p') = sin(2θ_p - 2θ_{p'}) (difference de phase)")
print()

q = q_stat
print("  q_stat, matrice J(p,p') pour p,p' in {3,5,7,11,13}:")
ps = [3, 5, 7, 11, 13]
print(f"  {'':>4}", end="")
for p2 in ps:
    print(f" {p2:>8}", end="")
print()

for p1 in ps:
    z1 = z_p(p1, q)
    print(f"  {p1:4d}", end="")
    for p2 in ps:
        z2 = z_p(p2, q)
        J = (z1 * z2.conjugate()).imag  # = sin(2θ₁ - 2θ₂)
        print(f" {J:8.5f}", end="")
    print()

# Total current
J_total = 0
for i, p1 in enumerate(PRIMES_ACTIFS):
    for j, p2 in enumerate(PRIMES_ACTIFS):
        if i < j:
            z1, z2 = z_p(p1, q), z_p(p2, q)
            J = (z1 * z2.conjugate()).imag
            J_total += J

print(f"\n  J_total (actifs, paires ordonnees) = {J_total:.8f}")
print(f"  = sin(2θ₃-2θ₅) + sin(2θ₃-2θ₇) + sin(2θ₅-2θ₇)")

# Verify: J = sin(2(θ₁-θ₂))
for p1, p2 in [(3,5), (3,7), (5,7)]:
    th1, th2 = theta_p(p1, q), theta_p(p2, q)
    J_direct = np.sin(2*(th1 - th2))
    z1, z2 = z_p(p1, q), z_p(p2, q)
    J_zprod = (z1 * z2.conjugate()).imag
    print(f"    J({p1},{p2}): sin(2(θ-θ')) = {J_direct:.8f}, Im(z·z'*) = {J_zprod:.8f}, match = {abs(J_direct - J_zprod) < 1e-12}")

# ══════════════════════════════════════════════════════════════════
# T7: Convergence des sommes imaginaires
# ══════════════════════════════════════════════════════════════════
print("\n\n### T7: Sommes partielles cumulees")
print("    Σ_K sin² (PT standard), Σ_K sin(2θ), Σ_K cos(2θ)")
print("    Comment ces sommes croissent-elles avec K?")
print()

q = q_stat
S_sin2, S_cos2, S_sin2t, S_cos2t = 0, 0, 0, 0

print(f"  {'K':>3} {'p':>4} {'Σsin²':>10} {'Σcos²':>10} {'Σsin(2θ)':>10} {'Σcos(2θ)':>10} {'ratio Im/Re':>12}")
for i, p in enumerate(PRIMES_ALL):
    S_sin2 += sin2_theta(p, q)
    S_cos2 += cos2_theta(p, q)
    S_sin2t += sin_2theta(p, q)
    S_cos2t += 1 - 2 * sin2_theta(p, q)

    ratio = S_sin2t / S_sin2 if S_sin2 > 0 else 0
    print(f"  {i+1:3d} {p:4d} {S_sin2:10.6f} {S_cos2:10.6f} {S_sin2t:10.6f} {S_cos2t:10.6f} {ratio:12.6f}")

# ══════════════════════════════════════════════════════════════════
# T8: Identites nouvelles
# ══════════════════════════════════════════════════════════════════
print("\n\n### T8: Identites PT impliquant la partie imaginaire")

q = q_stat

# Identite 1: sin(2θ)² + cos(2θ)² = 1 (triviale mais utile)
print("\n  ID1: sin²(2θ) + cos²(2θ) = 1")
for p in PRIMES_ALL[:5]:
    s2t = sin_2theta(p, q)
    c2t = 1 - 2 * sin2_theta(p, q)
    print(f"    p={p}: {s2t**2:.8f} + {c2t**2:.8f} = {s2t**2 + c2t**2:.8f}")

# Identite 2: sin(2θ)·cos(2θ) = sin(4θ)/2
print("\n  ID2: sin(2θ)·cos(2θ) = sin(4θ)/2")
for p in PRIMES_ALL[:5]:
    th = theta_p(p, q)
    lhs = sin_2theta(p, q) * (1 - 2 * sin2_theta(p, q))
    rhs = np.sin(4 * th) / 2
    print(f"    p={p}: LHS = {lhs:.8f}, RHS = {rhs:.8f}, match = {abs(lhs - rhs) < 1e-12}")

# Identite 3: sin²(θ) = [1 - cos(2θ)]/2 => sin² est la projection RE
print("\n  ID3: sin² = (1 - cos 2θ)/2 -- PT utilise la projection REELLE de (1-z_p)/2")
for p in PRIMES_ALL[:5]:
    s2 = sin2_theta(p, q)
    c2t = np.cos(2 * theta_p(p, q))
    proj = (1 - c2t) / 2
    print(f"    p={p}: sin² = {s2:.8f}, (1-cos2θ)/2 = {proj:.8f}, match = {abs(s2 - proj) < 1e-12}")

# Identite 4: sin(2θ)/2 = Im[(1-z_p)/2] ... NON
# (1-z_p)/2 = (1 - cos2θ - i·sin2θ)/2 = sin² - i·sin(2θ)/2
# Donc Im[(1-z_p)/2] = -sin(2θ)/2
print("\n  ID4: Im[(1 - z_p)/2] = -sin(2θ)/2")
print("    => L'observable PT sin² = Re[(1-z_p)/2]")
print("    => L'observable CACHEE = -Im[(1-z_p)/2] = sin(2θ)/2")
for p in PRIMES_ALL[:5]:
    zp = z_p(p, q)
    obs_re = ((1 - zp) / 2).real
    obs_im = -((1 - zp) / 2).imag
    s2 = sin2_theta(p, q)
    s2t_half = sin_2theta(p, q) / 2
    print(f"    p={p}: Re = {obs_re:.8f} = sin² = {s2:.8f}, "
          f"-Im = {obs_im:.8f} = sin(2θ)/2 = {s2t_half:.8f}")

# Identite 5: La variable NATURELLE est w_p = (1 - z_p)/2 = sin² - i·sin(2θ)/2
print("\n  ID5: Variable naturelle w_p = (1-z_p)/2 = sin² - i·sin(2θ)/2")
print("    |w_p|² = sin⁴ + sin²(2θ)/4 = sin²·(sin² + cos²) = sin²")
print("    => |w_p| = sin(θ)  (le MODULE est √sin² !)")
for p in PRIMES_ALL[:5]:
    zp = z_p(p, q)
    w = (1 - zp) / 2
    s2 = sin2_theta(p, q)
    print(f"    p={p}: |w|² = {abs(w)**2:.8f}, sin² = {s2:.8f}, "
          f"match = {abs(abs(w)**2 - s2) < 1e-12}")

# ══════════════════════════════════════════════════════════════════
# T9: Bilan
# ══════════════════════════════════════════════════════════════════
print("\n\n### T9: BILAN -- La partie imaginaire de PT")
print("""
═══════════════════════════════════════════════════════════════
STRUCTURE FONDAMENTALE:

  PT opere avec sin²(θ_p) = Re[(1 - e^{2iθ_p})/2]

  La variable NATURELLE est w_p = (1 - e^{2iθ_p})/2, qui a:
    Re(w_p) = sin²(θ_p)     [OBSERVABLE PT STANDARD = perte]
    Im(w_p) = -sin(2θ_p)/2  [OBSERVABLE CACHEE = interference]
    |w_p|   = sin(θ_p)      [MODULE = √sin²]

  Les 3 observables sont LIEES par:
    |w_p|² = Re(w_p)        [identite fondamentale]
    => sin²(θ) = sin⁴(θ) + sin²(2θ)/4

═══════════════════════════════════════════════════════════════
INTERPRETATION PT:

  1. sin²(θ_p) = PERTE d'information au premier p
  2. cos²(θ_p) = CONSERVATION d'information
  3. sin(2θ_p) = 2√(perte × conservation) = COHERENCE
     = moyenne geometrique des deux canaux
     = aire du parallelogramme unitaire

  La coherence est MAXIMALE quand perte = conservation = 1/2
  (soit δ = 1 - 1/√2 ≈ 0.293)

  La coherence DECROIT avec p car δ_p ~ 1/p → 0
  => Grands premiers: presque toute conservation, peu de perte,
     et PEU D'INTERFERENCE

═══════════════════════════════════════════════════════════════
CE QUE PT JETTE:

  PT jette sin(2θ)/2 = Im(w_p) a CHAQUE etape.
  C'est le terme qui couplerait les canaux sin/cos.
  En le jetant, PT travaille dans un espace PROJECTIF REEL
  (seule la ligne Re compte), pas dans le plan complexe complet.

  Le "courant" J(p,p') = sin(2θ_p - 2θ_{p'}) mesure le
  TRANSFERT DE PHASE entre premiers. PT l'ignore totalement.

═══════════════════════════════════════════════════════════════
""")

print("=" * 90)
print("FIN TOOL 41")
print("=" * 90)

sys.exit(0)
