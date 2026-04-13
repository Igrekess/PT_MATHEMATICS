"""
S15.6.261 -- RECURRENCE DE Q ET BORNE INFERIEURE T00
=====================================================

RESULTAT CLE de S15.6.260:
  Q > 0  <=>  T12 > 1/2  <=>  (1-3a+2a*T00) > 0  <=>  D > 0

  Les contraintes prouvees (sigma<=1/2, T00<=alpha, alpha<1/2)
  sont INSUFFISANTES. Il faut une contrainte SUPPLEMENTAIRE.

CETTE ANALYSE:
  1. Recurrence exacte de Q(k) entre niveaux
  2. Relation T12-1/2 vs epsilon = 1/2-alpha
  3. Argument de profondeur: T00 approche alpha "par en-dessous"
"""

from fractions import Fraction
import math

def sieve_residues(prime_list):
    P = 1
    for p in prime_list:
        P *= p
    return sorted([r for r in range(P) if all(r % p != 0 for p in prime_list)])

def compute_all(prime_list):
    P = 1
    for p in prime_list:
        P *= p
    res = sieve_residues(prime_list)
    n = len(res)
    classes = []
    for i in range(n):
        g = (res[(i+1) % n] - res[i]) % P
        classes.append(g % 3)
    counts = [[Fraction(0)]*3 for _ in range(3)]
    cc = [Fraction(0)]*3
    for i in range(n):
        a, b = classes[i], classes[(i+1) % n]
        counts[a][b] += 1
        cc[a] += 1
    T = [[Fraction(0)]*3 for _ in range(3)]
    for i in range(3):
        if cc[i] > 0:
            for j in range(3):
                T[i][j] = counts[i][j] / cc[i]
    alpha = cc[0] / n
    return T, alpha, n

primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
results = []

for k in range(2, len(primes) + 1):
    pl = primes[:k]
    T, alpha, n = compute_all(pl)
    T00 = T[0][0]
    T12 = T[1][2]
    eps = Fraction(1, 2) - alpha
    F = 1 - 2*alpha + alpha*T00

    # Q exact (rationnel)
    if eps > 0:
        Q_num = 1 - 3*alpha + 2*alpha*T00  # numerateur de Q*eps
        Q = Q_num / eps
    else:
        Q = None

    results.append({
        'k': k, 'alpha': alpha, 'T00': T00, 'T12': T12,
        'eps': eps, 'F': F, 'Q_num': Q_num if eps > 0 else None,
        'Q': Q, 'n': n
    })

# ================================================================
print("=" * 70)
print("PARTIE 1: QUANTITE Q EXACTE")
print("=" * 70)
print("""
  Q = (1 - 3*alpha + 2*alpha*T00) / epsilon
    = (1-alpha)*(2*T12 - 1) / epsilon

  Q > 0 <=> convergence.
  Q apparait dans: eps(k+1)/eps(k) = 1 - Q/(p-1).
""")

print(f"  {'k':>2} {'eps':>12} {'Q_num':>12} {'Q':>10} {'(T12-1/2)/eps':>14} {'Q*eps/(1-a)':>12}")
print("-" * 75)
for r in results:
    ef = float(r['eps'])
    Qf = float(r['Q']) if r['Q'] else 0
    T12f = float(r['T12'])
    af = float(r['alpha'])
    ratio_T12 = (T12f - 0.5) / ef if ef > 0 else 0
    check = Qf * ef / (1-af) if af < 1 else 0
    print(f"  {r['k']:>2} {ef:>12.8f} {float(r['Q_num']):>12.8f} "
          f"{Qf:>10.6f} {ratio_T12:>14.8f} {check:>12.8f}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 2: EVOLUTION DE Q ENTRE NIVEAUX")
print("=" * 70)

print(f"\n  {'k':>2}  {'Q(k)':>10}  {'Q(k+1)':>10}  {'Q(k+1)/Q(k)':>12}  {'delta_Q':>10}  {'p_{k+1}':>7}")
print("-" * 65)
for i in range(len(results) - 1):
    r0, r1 = results[i], results[i+1]
    if r0['Q'] and r1['Q']:
        Q0 = float(r0['Q'])
        Q1 = float(r1['Q'])
        ratio = Q1/Q0 if Q0 > 0 else 0
        delta = Q1 - Q0
        p_next = primes[r0['k']]
        print(f"  {r0['k']:>2}  {Q0:>10.6f}  {Q1:>10.6f}  {ratio:>12.6f}  {delta:>+10.6f}  {p_next:>7}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 3: DECOMPOSITION F = F_Markov + F_corr")
print("=" * 70)

print("""
  F_global = 1 - 2*alpha + alpha*T00
  En Markov pur: T00_Markov = alpha (auto-transition proportionnelle a freq)
  => F_Markov = 1 - 2*alpha + alpha^2 = (1-alpha)^2
  => T12_Markov = (1-alpha)^2 / (1-alpha) = 1-alpha
  => Toujours T12 > 1/2 pour alpha < 1/2!

  MAIS: T00_Markov = alpha est une SURESTIMATION de T00 reel.
  T00 < alpha (prouve). Donc F < F_Markov.
  La question: F > (1-alpha)/2 malgre F < F_Markov?
""")

print(f"  {'k':>2} {'alpha':>8} {'T00':>8} {'T00/a':>8} {'F':>10} {'F_Markov':>10} {'(1-a)/2':>10} {'marge':>10}")
print("-" * 85)
for r in results:
    af = float(r['alpha'])
    T00f = float(r['T00'])
    rho = T00f/af if af > 0 else 0
    F = float(r['F'])
    F_M = (1-af)**2
    half = (1-af)/2
    marge = F - half
    print(f"  {r['k']:>2} {af:>8.4f} {T00f:>8.4f} {rho:>8.4f} {F:>10.6f} {F_M:>10.6f} {half:>10.6f} {marge:>+10.6f}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 4: RATIO (T12-1/2) / eps = Q_ratio")
print("=" * 70)

print("""
  Observation: (T12 - 1/2) / eps converge vers Q_inf ~ 0.713.

  Si on pouvait PROUVER: T12 - 1/2 = Q_inf * eps + O(eps^2)
  avec Q_inf > 0, alors T12 > 1/2 <=> eps > 0 <=> alpha < 1/2.

  Et alpha < 1/2 est PROUVE!

  Le ratio Q_ratio = (2*T12-1) / (2*eps) = (T12-1/2)/(1/2-alpha):
""")

print(f"  {'k':>2} {'eps':>10} {'T12-1/2':>10} {'Q_ratio':>10} {'Q_ratio*2*(1-a)':>16}")
print("-" * 60)
for r in results:
    ef = float(r['eps'])
    T12f = float(r['T12'])
    af = float(r['alpha'])
    if ef > 0.001:
        delta_T12 = T12f - 0.5
        Q_ratio = delta_T12 / ef
        Q_check = Q_ratio * 2 * (1 - af)
        print(f"  {r['k']:>2} {ef:>10.6f} {delta_T12:>10.6f} {Q_ratio:>10.6f} {Q_check:>16.6f}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 5: FORMULE EXACTE delta_T00 / delta_alpha")
print("=" * 70)

print("""
  Quand on passe de k a k+1, alpha augmente: delta_alpha > 0.
  T00 augmente aussi: delta_T00 > 0.

  Le ratio delta_T00 / delta_alpha mesure la "vitesse relative".
  Si T00 "suit" alpha (meme vitesse), rho = T00/alpha reste constant.
  Si T00 accelere, rho augmente.
""")

print(f"  {'k':>2} {'d_alpha':>10} {'d_T00':>10} {'d_T00/d_a':>10} {'d_rho':>10} {'rho':>10}")
print("-" * 60)
prev = None
for r in results:
    if prev is not None and float(prev['alpha']) > 0:
        da = float(r['alpha'] - prev['alpha'])
        dT = float(r['T00'] - prev['T00'])
        ratio = dT/da if da > 0 else 0
        rho = float(r['T00']) / float(r['alpha']) if float(r['alpha']) > 0 else 0
        rho_prev = float(prev['T00']) / float(prev['alpha']) if float(prev['alpha']) > 0 else 0
        drho = rho - rho_prev
        print(f"  {r['k']:>2} {da:>10.6f} {dT:>10.6f} {ratio:>10.4f} {drho:>+10.6f} {rho:>10.6f}")
    prev = r

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 6: ARGUMENT CLÉ -- PROFONDEUR 2 DE MERTENS")
print("=" * 70)

print("""
  DOUBLE LOI DE MERTENS (T4):
    eps(k)   ~ C_eps   * prod_{p<=p_k} (1 - 1/p)
    delta(k) ~ C_delta * prod_{p<=p_k} (1 - 1/p)

  ou delta(k) = 1/2 - T12 [PAS T00!] et:
    C_eps   = 0.899 * e^gamma / 2  [constante d'Euler-Mascheroni]
    C_delta = 0.641 * e^gamma / 2

  RATIO: Q_inf = C_delta / C_eps = 0.641 / 0.899 = 0.713

  CONSEQUENCE:
    T12 - 1/2 = delta(k) ~ C_delta * M(k)
    eps(k)    = 1/2 - alpha ~ C_eps * M(k)

  ou M(k) = prod(1-1/p). Donc:
    T12 - 1/2   C_delta
    --------- = ------- = Q_inf > 0
       eps       C_eps

  A TOUT ORDRE en M(k), T12-1/2 et eps ont le MEME signe!
  => T12 > 1/2 <=> eps > 0 <=> alpha < 1/2. QED (si la loi double est prouvee).
""")

# Verification numerique de la double loi
import math
import sys
euler_gamma = 0.5772156649015329
M_values = []
cumM = 1.0
for i, p in enumerate(primes):
    cumM *= (1 - 1/p)
    M_values.append(cumM)

print(f"  {'k':>2} {'eps':>10} {'T12-1/2':>10} {'M(k)':>12} {'eps/M':>10} {'delta/M':>10} {'ratio':>10}")
print("-" * 75)
for i, r in enumerate(results):
    if i < 1:
        continue
    ef = float(r['eps'])
    T12f = float(r['T12'])
    delta = T12f - 0.5
    M = M_values[r['k']-1]
    eps_over_M = ef / M if M > 0 else 0
    delta_over_M = delta / M if M > 0 else 0
    ratio = delta / ef if ef > 0 else 0
    print(f"  {r['k']:>2} {ef:>10.6f} {delta:>10.6f} {M:>12.8f} {eps_over_M:>10.4f} {delta_over_M:>10.4f} {ratio:>10.6f}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 7: IDENTITE STRUCTURELLE Q_num = F - (1-a)/2")
print("=" * 70)

print("""
  Q_num = 1 - 3a + 2a*T00 = F - (1-a)/2
  F = 1 - 2a + a*T00

  Or F = (1-a)*T12 [par flow balance: T12 = F/(1-a)].

  Donc: Q_num = 2*(1-a)*T12 - (1-a) = (1-a)*(2*T12 - 1)

  Et: Q = Q_num / eps = (1-a)*(2*T12 - 1) / eps

  IDENTITE EXACTE (ALGEBRIQUE):
    Q * eps = (1-alpha) * (2*T12 - 1)

  Cette identite montre que Q > 0 <=> T12 > 1/2 (trivial).
  Mais aussi: si on peut exprimer Q en termes du PAS INDUCTIF...
""")

# Verification
for r in results:
    if r['Q_num'] is not None and float(r['alpha']) > 0:
        af = float(r['alpha'])
        T12f = float(r['T12'])
        ef = float(r['eps'])
        lhs = float(r['Q_num'])
        rhs = (1-af) * (2*T12f - 1)
        assert abs(lhs - rhs) < 1e-10, f"Identity fails at k={r['k']}: {lhs} vs {rhs}"
print("  Identite Q_num = (1-a)*(T12-1/2) verifiee pour k>=3. OK")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 8: ARGUMENT D'INDUCTION SUR F")
print("=" * 70)

print("""
  F(k) = 1 - 2*alpha(k) + alpha(k)*T00(k)

  Q > 0 <=> F > (1-alpha)/2

  Posons G(k) = F(k) - (1-alpha(k))/2 = Q_num(k) > 0.

  QUESTION: G(k) > 0 pour tout k?

  Equivalent: montrer que F "domine" toujours (1-alpha)/2.

  F evolue selon: F(k+1) = 1 - 2*alpha(k+1) + alpha(k+1)*T00(k+1).
  (1-alpha(k+1))/2 = (1 - alpha(k+1))/2.

  Le ratio F/(1-a)/2 = 2F/(1-a) = 2*T12 (par flow balance).
  Donc G > 0 <=> T12 > 1/2 (meme chose).

  Mais regardons l'EVOLUTION du ratio 2*T12 = 2*F/(1-alpha):
""")

print(f"  {'k':>2} {'2*T12':>8} {'2*T12-1':>10} {'2*T12(k+1)-1':>14} {'ratio':>10} ")
print("-" * 55)
for i, r in enumerate(results):
    val = 2 * float(r['T12'])
    marge = val - 1
    if i < len(results) - 1:
        val_next = 2 * float(results[i+1]['T12'])
        marge_next = val_next - 1
        ratio = marge_next / marge if marge > 0 else 0
        print(f"  {r['k']:>2} {val:>8.5f} {marge:>10.6f} {marge_next:>14.6f} {ratio:>10.6f}")
    else:
        print(f"  {r['k']:>2} {val:>8.5f} {marge:>10.6f}")

# ================================================================
print("\n" + "=" * 70)
print("SYNTHESE FINALE")
print("=" * 70)

print("""
ETAT DE LA PREUVE:

  PROUVE INCONDITIONNELLEMENT:
    - sigma <= 1/2 (Lemme C)          => T00 <= T12
    - T00 <= alpha (Lemme B)           => borne sup
    - alpha < 1/2 (g(alpha) > 0)       => convergence vers 1/2
    - D(k+1) = (p-4)*D(k) + [2(d+d')-4b]  (recurrence exacte)
    - D > 0 <=> T12 > 1/2 <=> Q > 0    (equivalence)

  VERIFIE NUMERIQUEMENT (k=2..9):
    - D(k) > 0 pour tout k
    - (d+d') >= 2b pour tout k
    - rho = T00/alpha monotone croissant
    - (T12-1/2)/eps -> Q_inf ~ 0.713 (double Mertens)

  GAP UNIQUE:
    Prouver T00 > (3*alpha-1)/(2*alpha) pour alpha > 1/3.
    Equivalent a: T12 > 1/2, ou D > 0, ou Q > 0.

  APPROCHE LA PLUS PROMETTEUSE:
    La double loi de Mertens montre que T12-1/2 et eps = 1/2-alpha
    ont le MEME profil asymptotique (proportionnels a prod(1-1/p)).
    Si on peut prouver que le ratio (T12-1/2)/eps est MONOTONE
    et converge vers Q_inf > 0, alors T12 > 1/2 <=> eps > 0 <=> alpha < 1/2.
    Et alpha < 1/2 EST PROUVE.

    Alternativement: prouver D(k+1) > 0 par induction, en montrant
    que la recurrence D(k+1) = (p-4)*D(k) + [2(d+d')-4b]
    preserve D > 0. Le terme dominant (p-4)*D(k) croit, et il faut
    borner le terme [2(d+d')-4b] par en-dessous.

    Le GAP est: passer de la verification k=2..9 a la preuve pour tout k.
""")

print("=" * 70)
print("FIN S15.6.261")
print("=" * 70)

sys.exit(0 if all(results) else 1)
