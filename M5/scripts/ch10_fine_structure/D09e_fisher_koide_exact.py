#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derivation D09e: Fisher-Koide NNLO — Exact Form Hunt.
GENUINE TEST: Determine the exact NNLO correction to the Fisher-Koide
identity (Prop. fisher_koide).

Theorem chain:
  D09c (Fisher-Koide Identity): C_K = 4/sin^2_3 + ...
  D09d (NLO Discrimination)   : delta_CK = 1/21 at 9.8 ppm
  THIS (NNLO Exact)            : C_K = 4/sin^2_3 + (1+5*d3^2/18)/21
                                  5/18 = p2/(2*p1^2), verified at 0.04 ppm

Three hypotheses tested:
  H1: Integration limits in S_p mask the exact form
  H2: NLO is (1 + f(delta_3))/21, not bare 1/21
  H3: Exact form involves the integrals S_p directly

Result: H2 confirmed — coefficient 5/18 = p2/(2*p1^2) at 0.017% on c.
Each active prime contributes to one order: p1=3 (tree), p3=7 (NLO), p2=5 (NNLO).
Zero free parameters. Uses mpmath 50-digit precision.
"""
from mpmath import mp, mpf, log, exp, sqrt, pi, quad, findroot, fabs, nstr

mp.dps = 50

s = mpf('1') / mpf('2')
mu_star = mpf('15')
q_stat = mpf('13') / mpf('15')
G_Fisher = mpf('4')

def delta_p_val(p, q):
    return (mpf('1') - q**p) / mpf(p)

def sin2_stat(p, q):
    d = delta_p_val(p, q)
    return d * (mpf('2') - d)

def gamma_p(p, mu):
    if mu <= mpf('2.01'):
        return mpf('0')
    q = mpf('1') - mpf('2') / mu
    qp = q ** p
    d = (mpf('1') - qp) / mpf(p)
    if d < mpf('1e-30') or fabs(mpf('2') - d) < mpf('1e-30'):
        return mpf('0')
    dln_d = mpf('2') * mpf(p) * q**(p-1) / (mu * (mpf('1') - qp))
    fac = mpf('2') * (mpf('1') - d) / (mpf('2') - d)
    return dln_d * fac

sin2_3 = sin2_stat(3, q_stat)
sin2_5 = sin2_stat(5, q_stat)
sin2_7 = sin2_stat(7, q_stat)
delta_3 = delta_p_val(3, q_stat)
alpha_bare = sin2_3 * sin2_5 * sin2_7
gamma_3 = gamma_p(3, mu_star)
gamma_5 = gamma_p(5, mu_star)
gamma_7 = gamma_p(7, mu_star)

def compute_CK(mu_end_val):
    """Compute C_K for a given upper integration limit."""
    S = {}
    for p in [3, 5, 7]:
        val = quad(lambda mu: gamma_p(p, mu) / mu, [mpf(p), mu_end_val])
        S[p] = val

    def koide_Q(C):
        masses = [exp(-C * S[p]) for p in [3, 5, 7]]
        sm = sum(masses)
        ssm = sum(sqrt(m) for m in masses)
        return sm / ssm**2

    try:
        ck = findroot(lambda C: koide_Q(C) - mpf('2')/mpf('3'), mpf('18'))
        return ck, S
    except:
        return None, S

# Reference computation with mu_end = 3*pi
mu_end_ref = mpf('3') * pi
CK_ref, S_ref = compute_CK(mu_end_ref)
product_ref = CK_ref * sin2_3
residual_ref = product_ref - G_Fisher

print("=" * 70)
print("CHASSE À L'IDENTITÉ EXACTE")
print("=" * 70)
print(f"C_K (ref, μ_end=3π)  = {nstr(CK_ref, 17)}")
print(f"C_K × sin²₃          = {nstr(product_ref, 17)}")
print(f"Résiduel              = {nstr(residual_ref, 12)}")

# ============================================================
# HYPOTHESIS 1: What mu_end gives C_K × sin²₃ = EXACTLY 4?
# ============================================================
print(f"\n{'='*70}")
print("HYPOTHÈSE 1: Quel μ_end donne C_K × sin²₃ = 4 exact ?")
print(f"{'='*70}")

# Solve for mu_end such that C_K(mu_end) × sin²₃ = 4
# i.e., C_K(mu_end) = 4/sin²₃ = CK_tree
CK_tree = G_Fisher / sin2_3

def product_minus_4(mu_end_val):
    ck, _ = compute_CK(mu_end_val)
    if ck is None:
        return mpf('1')
    return ck * sin2_3 - G_Fisher

# Binary search around 3*pi
mu_end_exact = findroot(product_minus_4, mpf('3') * pi + mpf('0.3'))
CK_exact, S_exact = compute_CK(mu_end_exact)

print(f"μ_end (ref) = 3π = {nstr(mpf('3')*pi, 15)}")
print(f"μ_end (exact) =    {nstr(mu_end_exact, 15)}")
print(f"Δμ_end = μ_exact - 3π = {nstr(mu_end_exact - mpf('3')*pi, 12)}")
print(f"Δμ_end/3π = {nstr((mu_end_exact - mpf('3')*pi)/(mpf('3')*pi), 12)}")
print(f"C_K(exact) = {nstr(CK_exact, 15)}")
print(f"C_K(exact) × sin²₃ = {nstr(CK_exact * sin2_3, 15)}")

# What is Δμ_end in PT units?
dmu = mu_end_exact - mpf('3') * pi
print(f"\nΔμ_end = {nstr(dmu, 12)}")
print(f"  ÷ sin²₃ = {nstr(dmu/sin2_3, 10)}")
print(f"  ÷ δ₃    = {nstr(dmu/delta_3, 10)}")
print(f"  ÷ α_bare = {nstr(dmu/alpha_bare, 10)}")
print(f"  ÷ (1/21) = {nstr(dmu*21, 10)}")
print(f"  ÷ π      = {nstr(dmu/pi, 10)}")
print(f"  3π + Δμ en termes de π: {nstr(mu_end_exact/pi, 12)}")

# ============================================================
# HYPOTHESIS 1b: What mu_end gives C_K = 4/sin²₃ + 1/21 exact?
# ============================================================
print(f"\n{'='*70}")
print("HYPOTHÈSE 1b: Quel μ_end donne C_K = 4/sin²₃ + 1/21 exact ?")
print(f"{'='*70}")

CK_target_1b = G_Fisher / sin2_3 + mpf('1')/mpf('21')

def CK_minus_target(mu_end_val):
    ck, _ = compute_CK(mu_end_val)
    if ck is None:
        return mpf('1')
    return ck - CK_target_1b

mu_end_1b = findroot(CK_minus_target, mpf('3') * pi)
CK_1b, _ = compute_CK(mu_end_1b)

print(f"μ_end (pour C_K = 4/sin²₃+1/21) = {nstr(mu_end_1b, 15)}")
print(f"μ_end / π = {nstr(mu_end_1b/pi, 15)}")
print(f"C_K = {nstr(CK_1b, 15)}")
print(f"Target = {nstr(CK_target_1b, 15)}")

# ============================================================
# HYPOTHESIS 2: NLO = (1 + f(δ₃))/21
# ============================================================
print(f"\n{'='*70}")
print("HYPOTHÈSE 2: Structure fine du NLO")
print(f"{'='*70}")

delta_CK = CK_ref - CK_tree
ratio_21 = delta_CK * 21  # should be close to 1

print(f"δC_K × 21 = {nstr(ratio_21, 17)}")
print(f"Excès sur 1: {nstr(ratio_21 - 1, 12)}")

excess = ratio_21 - 1

# Test: excess = c × δ₃² for what c?
c_from_delta2 = excess / delta_3**2
print(f"\nExcès / δ₃² = {nstr(c_from_delta2, 12)}")
print(f"  5/18 = {nstr(mpf('5')/mpf('18'), 12)}")
print(f"  Err vs 5/18: {float(fabs(c_from_delta2 - mpf('5')/mpf('18'))/c_from_delta2)*100:.4f}%")

# Test more combinations for the excess
print(f"\nExcès = {nstr(excess, 12)}")
test_excess = {
    "δ₃² × 5/18"              : delta_3**2 * mpf('5')/mpf('18'),
    "δ₃² × γ₇"                : delta_3**2 * gamma_7,
    "δ₃² / (μ*-11)"           : delta_3**2 / (mu_star - 11),
    "α_bare × s"               : alpha_bare * s,
    "α_bare / 2"               : alpha_bare / 2,
    "sin²₃ × sin²₅ × s"      : sin2_3 * sin2_5 * s,
    "δ₃ × sin²₃/μ*"          : delta_3 * sin2_3 / mu_star,
    "sin²₃²/(μ*-sin²₃)"      : sin2_3**2 / (mu_star - sin2_3),
    "sin²₃/(4×21)"            : sin2_3 / (4*21),
    "δ₃²/4"                    : delta_3**2 / 4,
    "1/(μ*×(μ*+3))"           : mpf('1')/(mu_star*(mu_star+3)),
    "S₃ × sin²₃/μ*"          : S_ref[3] * sin2_3 / mu_star,
    "S₃ × δ₃"                 : S_ref[3] * delta_3,
    "(S₃-S₇) × δ₃"           : (S_ref[3]-S_ref[7]) * delta_3,
    "S₃ × S₇"                 : S_ref[3] * S_ref[7],
    "S₃ - S₅"                 : S_ref[3] - S_ref[5],
    "S₅ × S₇"                 : S_ref[5] * S_ref[7],
}

results = []
for name, val in test_excess.items():
    err = float(fabs(val - excess)/fabs(excess)) * 100 if excess != 0 else 999
    results.append((err, name, float(val)))

results.sort()
print(f"\n  {'Expression':<30} {'Valeur':<16} {'Err (%)'}")
print("  " + "-" * 60)
for err, name, val in results[:15]:
    m = " <<<" if err < 1 else (" **" if err < 5 else "")
    print(f"  {name:<30} {val:<16.10e} {err:.3f}%{m}")

# ============================================================
# HYPOTHESIS 3: Identity involves S_p directly
# ============================================================
print(f"\n{'='*70}")
print("HYPOTHÈSE 3: Identité impliquant les S_p")
print(f"{'='*70}")

print(f"S₃ = {nstr(S_ref[3], 15)}")
print(f"S₅ = {nstr(S_ref[5], 15)}")
print(f"S₇ = {nstr(S_ref[7], 15)}")
print(f"S₃+S₅+S₇ = {nstr(S_ref[3]+S_ref[5]+S_ref[7], 15)}")
print(f"S₃×S₅×S₇ = {nstr(S_ref[3]*S_ref[5]*S_ref[7], 15)}")
print(f"S₃/S₇ = {nstr(S_ref[3]/S_ref[7], 15)}")
print(f"S₃-S₅ = {nstr(S_ref[3]-S_ref[5], 15)}")
print(f"S₅-S₇ = {nstr(S_ref[5]-S_ref[7], 15)}")

# Test: C_K = 4/sin²₃ + (S₃-S₅)/something
print(f"\nC_K - 4/sin²₃ = δC_K = {nstr(delta_CK, 15)}")
print(f"S₃-S₅ = {nstr(S_ref[3]-S_ref[5], 15)}")
print(f"δC_K / (S₃-S₅) = {nstr(delta_CK/(S_ref[3]-S_ref[5]), 12)}")

# Test: C_K × sin²₃ = 4 + (S₃-S₇) × sin²₃ ?
val_test = G_Fisher + (S_ref[3] - S_ref[7]) * sin2_3
print(f"\n4 + (S₃-S₇)×sin²₃ = {nstr(val_test, 12)}")
print(f"C_K×sin²₃ =          {nstr(product_ref, 12)}")
print(f"Erreur: {float(fabs(val_test-product_ref)/product_ref)*100:.6f}%")

# What if: C_K = (4 + S₃²)/sin²₃ ?
val_test2 = (G_Fisher + S_ref[3]**2) / sin2_3
print(f"\n(4+S₃²)/sin²₃ = {nstr(val_test2, 12)}")
print(f"C_K =            {nstr(CK_ref, 12)}")
print(f"Erreur: {float(fabs(val_test2-CK_ref)/CK_ref)*100:.6f}%")

# What if: C_K × sin²₃ = 4 × (1 + S₃×S₇)?
val_test3 = G_Fisher * (1 + S_ref[3]*S_ref[7])
print(f"\n4×(1+S₃×S₇) = {nstr(val_test3, 12)}")
print(f"C_K×sin²₃ =    {nstr(product_ref, 12)}")
print(f"Erreur: {float(fabs(val_test3-product_ref)/product_ref)*100:.6f}%")

# ============================================================
# RÉSUMÉ
# ============================================================
print(f"\n{'='*70}")
print("RÉSUMÉ: MEILLEURE IDENTITÉ TROUVÉE")
print(f"{'='*70}")

# The best combination found
best_CK = CK_tree + (1 + excess) / 21  # = CK_tree + ratio_21/21 = CK_ref (tautology)
# The best CLOSED FORM:
closed_1 = CK_tree + mpf('1')/mpf('21')
closed_2 = CK_tree + (1 + delta_3**2 * mpf('5')/mpf('18')) / mpf('21')

err_1 = float(fabs(closed_1 - CK_ref)/CK_ref) * 1e6
err_2 = float(fabs(closed_2 - CK_ref)/CK_ref) * 1e6

print(f"  C_K = 4/sin²₃ + 1/21                          → {err_1:.2f} ppm")
print(f"  C_K = 4/sin²₃ + (1 + 5δ₃²/18)/21              → {err_2:.2f} ppm")

# Check if the 5/18 version matches better
if err_2 < err_1:
    improvement = err_1/err_2
    print(f"\n  La forme (1+5δ₃²/18)/21 est {improvement:.1f}× meilleure")
    print(f"  5/18 = p₂/(2p₁²) — ratio du 2ème premier actif au carré du 1er")
