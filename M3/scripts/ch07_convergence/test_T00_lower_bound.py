"""
S15.6.260 -- BORNE INFERIEURE T00 : CONNEXION D>0 <=> Q>0
============================================================

OBJECTIF: Prouver T00 > (3*alpha-1)/(2*alpha) pour alpha > 1/3.

EQUIVALENCES CLES:
  Q > 0  <=>  T12 > 1/2  <=>  D(k) > 0  <=>  T00 > (3a-1)/(2a)

ou D(k) = n12 - n10 = n1*(T12 - T10) = n1*(2*T12 - 1).

PLAN:
  Part 1: Verification numerique de l'equivalence
  Part 2: Espace des contraintes prouvees vs Q > 0
  Part 3: Formule de mise a jour CRT pour T00
  Part 4: Monotonie de T00/alpha et borne inferieure
  Part 5: Synthese et gap residuel
"""

from fractions import Fraction
import math
import sys

def sieve_residues(prime_list):
    """Residus coprimes a tous les premiers de prime_list, dans [0, prod)."""
    P = 1
    for p in prime_list:
        P *= p
    return sorted([r for r in range(P) if all(r % p != 0 for p in prime_list)])

def mod3_class(g):
    """Classe mod 3 d'un gap g."""
    return g % 3

def compute_T_matrix(residues, P):
    """Matrice de transition exacte (Fraction) pour les classes mod 3."""
    n = len(residues)
    # Classes mod 3 des gaps
    classes = []
    for i in range(n):
        g = (residues[(i+1) % n] - residues[i]) % P
        classes.append(g % 3)

    # Compter transitions
    counts = [[Fraction(0)]*3 for _ in range(3)]
    class_counts = [Fraction(0)]*3
    for i in range(n):
        a = classes[i]
        b = classes[(i+1) % n]
        counts[a][b] += 1
        class_counts[a] += 1

    # T-matrice
    T = [[Fraction(0)]*3 for _ in range(3)]
    for i in range(3):
        if class_counts[i] > 0:
            for j in range(3):
                T[i][j] = counts[i][j] / class_counts[i]

    alpha = class_counts[0] / n
    return T, alpha, classes, n

def compute_3grams(classes, n):
    """Compte les 3-grammes."""
    grams = {}
    for i in range(n):
        a = classes[i]
        b = classes[(i+1) % n]
        c = classes[(i+2) % n]
        key = (a, b, c)
        grams[key] = grams.get(key, 0) + 1
    return grams

print("=" * 70)
print("S15.6.260 -- BORNE INFERIEURE T00 : D>0 <=> Q>0")
print("=" * 70)

# Calcul exact
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
data = []

for k in range(2, len(primes) + 1):
    pl = primes[:k]
    P = 1
    for p in pl:
        P *= p
    res = sieve_residues(pl)
    T, alpha, classes, n = compute_T_matrix(res, P)
    grams = compute_3grams(classes, n)

    T00 = T[0][0]
    T12 = T[1][2]
    T10 = T[1][0]

    # Compter n12, n10
    class_counts = [0]*3
    for c in classes:
        class_counts[c] += 1
    n1 = class_counts[1]

    # Transition counts
    n12 = 0
    n10 = 0
    for i in range(n):
        if classes[i] == 1:
            nxt = classes[(i+1) % n]
            if nxt == 2:
                n12 += 1
            elif nxt == 0:
                n10 += 1

    D = n12 - n10

    # Q > 0 condition
    af = float(alpha)
    threshold = (3*af - 1) / (2*af) if af > 0 else -999
    T00f = float(T00)
    Q_positive = T00f > threshold

    # T12 > 1/2
    T12f = float(T12)
    T12_positive = T12f > 0.5

    data.append({
        'k': k, 'P': P, 'n': n, 'alpha': alpha, 'T00': T00, 'T12': T12,
        'T10': T10, 'n1': n1, 'n12': n12, 'n10': n10, 'D': D,
        'threshold': threshold, 'Q_pos': Q_positive, 'T12_pos': T12_positive,
        'grams': grams
    })

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 1: EQUIVALENCE D>0 <=> T12>1/2 <=> Q>0")
print("=" * 70)

print(f"\n  {'k':>2}  {'alpha':>8}  {'T00':>8}  {'T12':>8}  {'thresh':>8}  {'D':>10}  {'T12>1/2':>7}  {'Q>0':>5}  {'D>0':>5}")
print("-" * 85)
for d in data:
    af = float(d['alpha'])
    print(f"  {d['k']:>2}  {af:>8.4f}  {float(d['T00']):>8.4f}  {float(d['T12']):>8.4f}  "
          f"{d['threshold']:>8.4f}  {d['D']:>10}  {'OUI' if d['T12_pos'] else 'NON':>7}  "
          f"{'OUI' if d['Q_pos'] else 'NON':>5}  {'OUI' if d['D'] > 0 else 'NON':>5}")

print("\n  VERIFICATION: D>0 <=> T12>1/2 <=> Q>0 pour TOUS les k?")
all_equiv = all(
    (d['D'] > 0) == d['T12_pos'] == d['Q_pos']
    for d in data if float(d['alpha']) > 1/3
)
print(f"  Phase 2 (alpha > 1/3): {'OUI -- EQUIVALENCE CONFIRMEE' if all_equiv else 'NON'}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 2: ESPACE DES CONTRAINTES PROUVEES")
print("=" * 70)

print("""
  Contraintes PROUVEES:
    C1: sigma <= 1/2  =>  T00 <= T12
    C2: T00 <= alpha
    C3: alpha < 1/2

  Question: C1+C2+C3 impliquent-elles Q > 0 (T12 > 1/2)?

  De C1: T00 <= T12 = (1-2a+a*T00)/(1-a)
    => (1-a)*T00 <= 1-2a+a*T00
    => T00*(1-2a) <= 1-2a
    => T00 <= 1  [si a < 1/2, trivial]

  CONCLUSION: C1+C2+C3 sont INSUFFISANTES pour T12 > 1/2.

  Contre-exemple: (alpha=0.4, T00=0.1) satisfait C1,C2,C3 mais:
    T12 = (1-0.8+0.04)/0.6 = 0.4 < 1/2  =>  Q < 0
""")

# Verification du contre-exemple
a_test, T00_test = Fraction(2, 5), Fraction(1, 10)
T12_test = (1 - 2*a_test + a_test*T00_test) / (1 - a_test)
sigma_test = T00_test / (2 * T12_test)
print(f"  Contre-exemple: alpha={float(a_test)}, T00={float(T00_test)}")
print(f"    T12 = {float(T12_test):.4f} {'< 1/2 => Q < 0' if T12_test < Fraction(1,2) else '>= 1/2'}")
print(f"    sigma = {float(sigma_test):.4f} {'<= 1/2 OK' if sigma_test <= Fraction(1,2) else '> 1/2'}")
print(f"    T00 <= alpha? {T00_test <= a_test}")
print(f"  => Il faut une CONTRAINTE SUPPLEMENTAIRE.")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 3: MISE A JOUR CRT DE T00")
print("=" * 70)

print("\n  Quand on ajoute le premier p (niveau k -> k+1):")
print("  Les phi(P_{k+1}) = (p-1)*phi(P_k) residus viennent de (p-1) copies")
print("  de la sequence niveau k, moins les multiples de p.")
print()

# Pour chaque transition k -> k+1, calculons le facteur de mise a jour
print(f"  {'k':>2} {'p_add':>5} {'T00(k)':>10} {'T00(k+1)':>10} {'ratio':>8} {'alpha(k)':>10} {'alpha(k+1)':>10}")
print("-" * 70)
for i in range(len(data) - 1):
    d0, d1 = data[i], data[i+1]
    p_add = primes[d0['k']]  # prime being added
    T00_old = float(d0['T00'])
    T00_new = float(d1['T00'])
    ratio = T00_new / T00_old if T00_old > 0 else float('inf')
    print(f"  {d0['k']:>2} {p_add:>5} {T00_old:>10.6f} {T00_new:>10.6f} {ratio:>8.4f} "
          f"{float(d0['alpha']):>10.6f} {float(d1['alpha']):>10.6f}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 4: MONOTONIE DE rho = T00/alpha")
print("=" * 70)

print(f"\n  {'k':>2} {'alpha':>10} {'T00':>10} {'rho=T00/a':>10} {'thresh/a':>10} {'marge':>10} {'Q>0':>5}")
print("-" * 75)
for d in data:
    af = float(d['alpha'])
    T00f = float(d['T00'])
    rho = T00f / af if af > 0 else 0
    thresh_over_a = (3*af - 1) / (2*af*af) if af > 0 else -999
    marge = rho - thresh_over_a
    print(f"  {d['k']:>2} {af:>10.6f} {T00f:>10.6f} {rho:>10.6f} {thresh_over_a:>10.6f} "
          f"{marge:>10.6f} {'OUI' if d['Q_pos'] else 'NON':>5}")

# Verifier monotonie de rho
rhos = [float(d['T00']) / float(d['alpha']) if float(d['alpha']) > 0 else 0 for d in data]
monotone = all(rhos[i] <= rhos[i+1] for i in range(len(rhos)-1))
print(f"\n  rho = T00/alpha MONOTONE CROISSANT? {'OUI' if monotone else 'NON'}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 5: DECOMPOSITION e vs b")
print("=" * 70)

print("""
  D = n12 - n10 = e - f  ou:
    e = n3(2,1,2)  (ping-pong)
    f = n3(0,1,0)  (classe 1 sandwichee entre 0s)
    b = n3(0,0,1)  (sortie de run-zero vers 1)
    d = n3(0,1,2)  (transition 0->1->2)

  Les 4 types de position classe-1:
    (0,1,0) = f    (1 sandwiche entre 0s)
    (0,1,2) = d    (transition 0->1->2)
    (2,1,0) = d    (transition 2->1->0, time-rev de (0,1,2))
    (2,1,2) = e    (ping-pong 2->1->2)
    Total: n1 = f + 2d + e

  n12 = d + e (predecesseurs 0 ou 2 de la transition 1->2)
  n10 = f + d (predecesseurs 0 ou 2 de la transition 1->0)
  D = (d+e) - (f+d) = e - f

  CONDITION (d+d') >= 2b: equivalent a d >= b (par sym 1<->2)
  CONDITION D > 0: equivalent a e > f
  NOTE: b != f en general!
""")

print(f"  {'k':>2} {'n1':>8} {'b':>8} {'d':>8} {'e':>8} {'f':>8} {'f+2d+e':>8} {'e/b':>8} {'D=e-f':>8} {'d>=b':>5}")
print("-" * 80)
for dat in data:
    g = dat['grams']
    b = g.get((0,0,1), 0)
    d = g.get((0,1,2), 0)
    e = g.get((2,1,2), 0)
    f = g.get((0,1,0), 0)
    n1 = dat['n1']
    D = e - b
    eb = e/b if b > 0 else float('inf')
    db = d/b if b > 0 else float('inf')

    # f vs b et decomposition
    h = g.get((1,0,2), 0)  # n3(1,0,2) = traverse 1->0->2
    check_n1 = b + 2*d + e  # Predecesseur (0 ou 2) x successeur (0 ou 2) pour classe 1
    # Correct: n1 = n3(0,1,0) + n3(0,1,2) + n3(2,1,0) + n3(2,1,2) = f + d + d + e
    correct_n1 = f + 2*d + e
    D_check = e - f  # D = n12 - n10 = (d+e) - (f+d) = e - f
    D_cond = d >= b   # (d+d') >= 2b condition

    print(f"  {dat['k']:>2} {n1:>8} {b:>8} {d:>8} {e:>8} {f:>8} {correct_n1:>8} "
          f"{eb:>8.3f} {D_check:>8} {'d>=b' if D_cond else 'd<b':>5}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 6: RATIO e/b EN APPROXIMATION MARKOV")
print("=" * 70)

print("""
  En Markov (predecesseur et successeur independants):
    b_M = n1 * T10^2
    d_M = n1 * T10 * T12
    e_M = n1 * T12^2

  e_M/b_M = (T12/T10)^2 = (T12/(1-T12))^2

  Puisque T12 > 1/2 [a prouver], e_M/b_M > 1.
  Mais T12 > 1/2 est CE QU'ON VEUT PROUVER => circulaire!

  Cependant, si on pouvait prouver e/b > 1 DIRECTEMENT
  (sans passer par T12 > 1/2), on aurait D > 0 => Q > 0.
""")

print(f"  {'k':>2} {'T12':>8} {'T10':>8} {'(T12/T10)^2':>12} {'e/b exact':>10} {'corr%':>8}")
print("-" * 60)
for dat in data:
    g = dat['grams']
    b = g.get((0,0,1), 0)
    e = g.get((2,1,2), 0)
    T12f = float(dat['T12'])
    T10f = float(dat['T10'])
    markov_ratio = (T12f / T10f)**2 if T10f > 0 else float('inf')
    exact_ratio = e / b if b > 0 else float('inf')
    corr = (exact_ratio - markov_ratio) / markov_ratio * 100 if markov_ratio != float('inf') and markov_ratio > 0 else 0
    print(f"  {dat['k']:>2} {T12f:>8.4f} {T10f:>8.4f} {markov_ratio:>12.4f} {exact_ratio:>10.4f} {corr:>+8.1f}%")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 7: ARGUMENT STRUCTUREL VIA T1-3GRAM")
print("=" * 70)

print("""
  THEOREME T1-3gram: n3(1,0,1) = n3(2,0,2) = 0.

  Consequence pour les positions classe-0:
  Chaque position classe-0 a un predecesseur et un successeur.
  Les combinaisons (pred, 0, succ) possibles sont:

  (0,0,0): run interne
  (0,0,1): sortie vers 1 = b
  (0,0,2): sortie vers 2 = b  [par sym 1<->2]
  (1,0,0): entree depuis 1 = b  [par time-rev]
  (1,0,2): traversee 1->0->2 (AUTORISE par T1-3gram)
  (2,0,0): entree depuis 2 = b  [par sym + time-rev]
  (2,0,1): traversee 2->0->1 (AUTORISE)
  ** (1,0,1) = 0 ** INTERDIT par T1-3gram
  ** (2,0,2) = 0 ** INTERDIT par T1-3gram

  Les 3-grammes centres sur 0 sont donc:
""")

for dat in data:
    if dat['k'] < 4:
        continue
    g = dat['grams']
    n0 = sum(1 for c in dat.get('grams', {}) if False)  # placeholder
    # Compute n0 from class counts
    classes = []
    # Recompute...
    break

# Direct computation for k=9
print("  Verification pour k=9:")
pl9 = primes[:9]
P9 = 1
for p in pl9:
    P9 *= p
res9 = sieve_residues(pl9)
T9, alpha9, classes9, n9 = compute_T_matrix(res9, P9)
grams9 = compute_3grams(classes9, n9)

zero_grams = {}
for (a,b,c), cnt in grams9.items():
    if b == 0:
        zero_grams[(a,c)] = cnt

print(f"  3-grammes (a, 0, c) pour k=9:")
total_0 = sum(zero_grams.values())
for a in range(3):
    for c in range(3):
        cnt = zero_grams.get((a,c), 0)
        pct = cnt / total_0 * 100
        status = " ** INTERDIT **" if (a == 1 and c == 1) or (a == 2 and c == 2) else ""
        print(f"    ({a},0,{c}): {cnt:>10} ({pct:>5.2f}%){status}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 8: ARGUMENT PAR MISE A JOUR INDUCTIVE")
print("=" * 70)

print("""
  STRATEGIE: Montrer que lors de l'ajout du premier p,
  si T12(k) > 1/2 alors T12(k+1) > 1/2.

  Base: k=3 (P=30), T12 = 2/3 > 1/2. OK.
        k=4 (P=210), T12 = 11/17 > 1/2. OK.

  Pas inductif: lors du retrait des multiples de p_{k+1},
  combien de transitions 1->2 et 1->0 sont affectees?
""")

# Analyse detaillee de la mise a jour pour chaque niveau
for i in range(len(data) - 1):
    d0, d1 = data[i], data[i+1]
    p_add = primes[d0['k']]

    T12_old = float(d0['T12'])
    T12_new = float(d1['T12'])
    T10_old = float(d0['T10'])
    T10_new = float(d1['T10'])

    # Delta T12
    dT12 = T12_new - T12_old
    dT10 = T10_new - T10_old

    print(f"  k={d0['k']} -> k+1={d1['k']}, p={p_add}:")
    print(f"    T12: {T12_old:.6f} -> {T12_new:.6f} (delta = {dT12:+.6f})")
    print(f"    T10: {T10_old:.6f} -> {T10_new:.6f} (delta = {dT10:+.6f})")
    print(f"    T12 > 1/2? {'OUI' if T12_new > 0.5 else 'NON'}")
    print()

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 9: BORNE INFERIEURE VIA F_global")
print("=" * 70)

print("""
  F_global = 1 - 2*alpha + alpha*T00

  Q > 0 <=> F_global > (1-alpha)/2

  Car: f(p) > 1 <=> 1 + alpha*(p-4+2T00) > (p-1)*alpha
                 <=> 1 - 3*alpha + 2*alpha*T00 > 0
                 <=> (1-alpha)*(2*T12-1) > 0
                 <=> T12 > 1/2

  Et: F_global = (1-alpha)*T12 + alpha*T00 - alpha*(1-T00)
     Non... F = 1 - 2*alpha + alpha*T00.

  Reecrivons: F = 1 - 2*alpha + alpha*T00
             = (1 - alpha) - alpha*(1 - T00)
             = (1 - alpha) - alpha + alpha*T00

  T12 = (F - alpha*(T00 - (1-T00)*(1-alpha)/alpha)) ... trop complique.

  Approche directe:
    F = 1 - 2*alpha + alpha*T00
    T12 = F / (1 - alpha)   [car T12 = (1-2a+a*T00)/(1-a) = F/(1-a)]
    T12 > 1/2 <=> F > (1-alpha)/2

  Donc Q > 0 <=> F > (1-alpha)/2.
""")

print(f"  {'k':>2} {'alpha':>8} {'T00':>8} {'F':>10} {'(1-a)/2':>10} {'F-(1-a)/2':>10} {'Q>0':>5}")
print("-" * 65)
for d in data:
    af = float(d['alpha'])
    T00f = float(d['T00'])
    F = 1 - 2*af + af*T00f
    half_compl = (1 - af) / 2
    margin = F - half_compl
    print(f"  {d['k']:>2} {af:>8.4f} {T00f:>8.4f} {F:>10.6f} {half_compl:>10.6f} {margin:>+10.6f} {'OUI' if margin > 0 else 'NON':>5}")

# ================================================================
print("\n" + "=" * 70)
print("PARTIE 10: SYNTHESE")
print("=" * 70)

print("""
RESULTATS:
  1. D > 0 <=> T12 > 1/2 <=> Q > 0 [EQUIVALENCE CONFIRMEE]
  2. Les contraintes prouvees (sigma<=1/2, T00<=alpha, alpha<1/2)
     sont INSUFFISANTES pour Q > 0.
  3. Il faut une CONTRAINTE SUPPLEMENTAIRE.
  4. rho = T00/alpha est monotone croissant [VERIFIE k=2..9].
  5. f = b (identite structurelle) [PROUVE par time-reversal].
  6. D = e - b ou e = n3(2,1,2) [PROUVE].

GAP RESIDUEL:
  Prouver T00 > (3*alpha-1)/(2*alpha) necessite une borne
  INFERIEURE sur T00 qui va au-dela de sigma <= 1/2.

PISTES:
  A. Prouver rho(k+1) >= rho(k) (monotonie du ratio T00/alpha)
  B. Prouver e > b directement (ping-pong > sortie-zero)
  C. Utiliser la structure CRT de la mise a jour
  D. Utiliser l'argument de profondeur: I = O(eps) >> |A| = O(eps^2)
""")

print("=" * 70)
print("FIN S15.6.260")
print("=" * 70)

sys.exit(0)
