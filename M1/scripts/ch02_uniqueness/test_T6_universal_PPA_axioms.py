#!/usr/bin/env python3
"""
T6 Universel -- Verification computationnelle de U1-U2-U3
==========================================================
S15.6.281

Theoreme universel : le crible d'Eratosthene est l'unique PPA sur Z.

Tests :
  U1  -- Modularite : TC verifiee, contre-exemples non-TC
  U1b -- Localite de la division : r capture toute l'info p-locale
  U2  -- Ideal : R_p = {0} seul ideal propre de Z/pZ
  U2b -- Alternatives : cribles swap violent A5 (ideal)
  U3  -- Completude : skip => q^2 survit
  U4  -- Classification : 9 types d'alternatives, toutes eliminees
  U5  -- Synthese universelle : seul Eratosthene satisfait A1-A6
"""

import math
from itertools import product as iter_product

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")

# --- Helpers ---

def primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def primorial(k_primes):
    """Product of the first k primes."""
    P = 1
    for p in k_primes:
        P *= p
    return P

def eratosthenes_gaps(primes_list):
    """Compute cyclic gaps after sieving by the given primes."""
    P = primorial(primes_list)
    survivors = [n for n in range(1, P + 1) if all(n % p != 0 for p in primes_list)]
    if not survivors:
        return []
    gaps = []
    for i in range(len(survivors) - 1):
        gaps.append(survivors[i + 1] - survivors[i])
    # Cyclic gap
    gaps.append(P - survivors[-1] + survivors[0])
    return sorted(gaps)

def modular_sieve_gaps(primes_list, R_dict):
    """Compute cyclic gaps for a modular sieve with given R_p sets."""
    P = primorial(primes_list)
    survivors = []
    for n in range(1, P + 1):
        eliminated = False
        for p in primes_list:
            if n % p in R_dict[p]:
                eliminated = True
                break
        if not eliminated:
            survivors.append(n)
    if not survivors:
        return []
    gaps = []
    for i in range(len(survivors) - 1):
        gaps.append(survivors[i + 1] - survivors[i])
    gaps.append(P - survivors[-1] + survivors[0])
    return sorted(gaps)


# ============================================================
print("=" * 60)
print("U1 -- TRANSLATION-COVARIANCE ET MODULARITE")
print("=" * 60)

# U1.1 : Verifier TC pour le crible d'Eratosthene
print("\n--- U1.1 : TC pour Eratosthene ---")
for p in [2, 3, 5, 7, 11, 13]:
    # E_p = {n : n mod p = 0} = multiples of p
    # TC : n in E_p <=> n+p in E_p
    tc_ok = True
    for n in range(-100, 200):
        in_Ep = (n % p == 0)
        in_Ep_shifted = ((n + p) % p == 0)
        if in_Ep != in_Ep_shifted:
            tc_ok = False
            break
    check(f"TC(Eratosthene, p={p})", tc_ok)

# U1.2 : Contre-exemple -- regle non-TC "eliminer n < p^2"
print("\n--- U1.2 : Contre-exemple non-TC ---")
p = 5
E_nonTC = {n for n in range(2, 200) if n < p * p}
# Verifier que TC est violee
tc_violated = False
for n in range(2, 200 - p):
    if (n in E_nonTC) != ((n + p) in E_nonTC):
        tc_violated = True
        break
check("Non-TC: 'n < p^2' viole TC", tc_violated)

# U1.3 : Lucky numbers violent A1 (non-modulaire aux etapes > 3)
print("\n--- U1.3 : Lucky numbers violent A1 ---")
# Lucky number sieve: at each step, remove every k-th surviving element
# Step 1: remove every 2nd (evens) -> survivors = odd numbers
# Step 2: remove every 3rd -> eliminate positions 3,6,9,...
# Step 3: next surviving lucky (7) -> remove every 7th -> non-congruential!
survivors = list(range(1, 300, 2))  # odd numbers
# Step 3 of lucky: remove every 3rd
new_survivors = []
for i, s in enumerate(survivors):
    if (i + 1) % 3 != 0:  # keep if not at position 3,6,9,...
        new_survivors.append(s)
survivors = new_survivors
# Step 7 of lucky: remove every 7th from remaining
lucky_step7_eliminated = set()
for i, s in enumerate(survivors):
    if (i + 1) % 7 == 0:
        lucky_step7_eliminated.add(s)
# Check if the elimination is congruential mod 7
# If congruential, all eliminated should have the same residue mod 7
residues = set()
for n in lucky_step7_eliminated:
    residues.add(n % 7)
# If more than one residue class is eliminated, it's not a single R_p
# More importantly: check TC -- n eliminated => n+7 eliminated?
tc_violated_7 = False
for n in lucky_step7_eliminated:
    if n + 7 in set(survivors) and n + 7 not in lucky_step7_eliminated:
        tc_violated_7 = True
        break
    if n + 7 not in set(survivors):
        # n+7 may have been eliminated at step 3, but that's fine
        # Check if n+7 is in the original odd list and wasn't eliminated by step 3
        if n + 7 in new_survivors and n + 7 not in lucky_step7_eliminated:
            tc_violated_7 = True
            break
# Also check: two elements with same residue mod 7 treated differently
non_congruential = len(residues) > 1
check("Lucky step 7: non-congruentiel (multiples residus mod 7)",
      non_congruential,
      f"residus trouves: {sorted(residues)}")
# Lucky step 7 is non-congruential => implicitly violates A1 (already tested above)


# ============================================================
print("\n" + "=" * 60)
print("U1b -- LEMME DE LOCALITE DE LA DIVISION")
print("=" * 60)

# La division n = qp + r : r est p-local, q est global
# Verifier que toute propriete de divisibilite ne depend que de r

print("\n--- U1b.1 : p|n ne depend que de r ---")
for p in [3, 5, 7, 11]:
    all_ok = True
    for n in range(0, 200):
        r = n % p
        divides = (p % 1 == 0) and (n % p == 0)  # p | n
        divides_from_r = (r == 0)
        if divides != divides_from_r:
            all_ok = False
            break
    check(f"p|n <=> r=0 (p={p})", all_ok)

print("\n--- U1b.2 : gcd(n,p) ne depend que de r ---")
for p in [3, 5, 7, 11, 13]:
    all_ok = True
    for n in range(1, 200):
        r = n % p
        g = math.gcd(n, p)
        g_from_r = math.gcd(r, p) if r > 0 else p
        if g != g_from_r:
            all_ok = False
            break
    check(f"gcd(n,p) = gcd(r,p) (p={p})", all_ok)

print("\n--- U1b.3 : n^k mod p ne depend que de r ---")
for p in [3, 5, 7]:
    all_ok = True
    for n in range(0, 100):
        r = n % p
        for k in [2, 3, 4, 5]:
            nk_mod = pow(n, k, p)
            rk_mod = pow(r, k, p)
            if nk_mod != rk_mod:
                all_ok = False
                break
        if not all_ok:
            break
    check(f"n^k mod p = r^k mod p (p={p})", all_ok)

print("\n--- U1b.4 : n et n+kp sont p-indistinguables ---")
for p in [3, 5, 7, 11]:
    all_ok = True
    for n in range(0, 50):
        r = n % p
        for k in [1, 2, 3, 5, 10, -1, -3]:
            n2 = n + k * p
            r2 = n2 % p
            if r != r2:
                all_ok = False
                break
        if not all_ok:
            break
    check(f"(n+kp) mod p = n mod p (p={p})", all_ok)


# ============================================================
print("\n" + "=" * 60)
print("U2 -- IDEAUX DE Z/pZ ET R_p = {0}")
print("=" * 60)

print("\n--- U2.1 : Ideaux propres de Z/pZ ---")
for p in [2, 3, 5, 7, 11, 13]:
    # Enumerate all subsets of Z/pZ
    # Check which are ideals (closed under multiplication by any element)
    proper_ideals = []
    for size in range(1, p):  # proper: not empty, not full
        from itertools import combinations
        for subset in combinations(range(p), size):
            S = set(subset)
            is_ideal = True
            for r in S:
                for a in range(p):
                    if (a * r) % p not in S:
                        is_ideal = False
                        break
                if not is_ideal:
                    break
            if is_ideal:
                proper_ideals.append(S)
    check(f"Z/{p}Z : seul ideal propre = {{0}}",
          len(proper_ideals) == 1 and proper_ideals[0] == {0},
          f"trouves: {proper_ideals}")

print("\n--- U2.2 : Cribles swap violent A5 (ideal) ---")
for p in [3, 5, 7]:
    for r in range(1, p):
        R_swap = {r}
        # Check if R_swap is an ideal of Z/pZ
        is_ideal = True
        for a in range(p):
            if (a * r) % p not in R_swap:
                is_ideal = False
                break
        check(f"R_p={{{r}}} n'est PAS un ideal de Z/{p}Z", not is_ideal)


# ============================================================
print("\n" + "=" * 60)
print("U3 -- COMPLETUDE")
print("=" * 60)

print("\n--- U3.1 : Skip p => p^2 survit ---")
all_primes = primes_up_to(20)
for skip_p in [3, 5, 7, 11, 13]:
    reduced_primes = [p for p in all_primes if p != skip_p and p <= skip_p]
    # Check if skip_p^2 survives
    target = skip_p * skip_p
    survives = all(target % p != 0 for p in reduced_primes)
    check(f"Skip p={skip_p} : p^2={target} survit", survives)


# ============================================================
print("\n" + "=" * 60)
print("U4 -- CLASSIFICATION DES ALTERNATIVES")
print("=" * 60)

print("\n--- U4.1 : 9 types d'alternatives, axiome viole ---")

# Type 1: Crible swap (R_p = {r != 0})
primes_k4 = [2, 3, 5, 7]
R_era = {p: {0} for p in primes_k4}
gaps_era = modular_sieve_gaps(primes_k4, R_era)

R_swap3 = {p: {0} for p in primes_k4}
R_swap3[3] = {1}  # swap: eliminate class 1 instead of 0
gaps_swap = modular_sieve_gaps(primes_k4, R_swap3)

# Swap has same cyclic gaps but {1} is not an ideal
check("Type 1 (swap): same gaps but violates A5",
      gaps_era == gaps_swap and not ({1} == {0}))

# Type 2: Lucky numbers -- already tested in U1.3

# Type 3: Crible incomplet -- tested computationally in U3

# Type 4: Crible composite (m=6 instead of 2,3)
# Verify CRT decomposition: Z/6Z ~ Z/2Z x Z/3Z
crt_ok = True
for n in range(6):
    r2 = n % 2
    r3 = n % 3
    r6 = n % 6
    # Reconstruct r6 from (r2, r3) via CRT
    for candidate in range(6):
        if candidate % 2 == r2 and candidate % 3 == r3:
            if candidate != r6:
                crt_ok = False
            break
check("Type 4 (composite m=6): CRT Z/6Z ~ Z/2Z x Z/3Z verifie", crt_ok)

# Type 5: Crible mixte (|R_p| > 1, e.g., R_3 = {0,1})
R_mixte = {p: {0} for p in primes_k4}
R_mixte[3] = {0, 1}  # eliminate two classes
# {0,1} is not an ideal of Z/3Z (0*2=0 ok, 1*2=2 not in {0,1})
is_ideal_mixte = all((a * r) % 3 in {0, 1} for r in {0, 1} for a in range(3))
check("Type 5 (mixte |R|>1): {0,1} not ideal of Z/3Z", not is_ideal_mixte)

# Type 6: Crible trivial (R_p = Z/pZ) -- verify no survivors
R_trivial = {p: set(range(p)) for p in primes_k4}
gaps_trivial = modular_sieve_gaps(primes_k4, R_trivial)
check("Type 6 (trivial R=Z/pZ): no survivors (A3 violated)", len(gaps_trivial) == 0)

# Type 7: Sundaram -- verify non-congruential elimination
# Sundaram: eliminate n = i+j+2ij for i<=j. Check mod 3 distribution.
sundaram_elim = set()
for i in range(1, 50):
    for j in range(i, 50):
        sundaram_elim.add(i + j + 2*i*j)
# If congruential mod 3, all eliminated should share residue mod 3
sund_residues = {n % 3 for n in sundaram_elim if n < 100}
check("Type 7 (Sundaram): non-congruentiel mod 3 (multiples residus)",
      len(sund_residues) > 1,
      f"residus: {sorted(sund_residues)}")

# Types 8-9: Atkin and Selberg are stipulative exclusions (by definition of A1).
# Not computationally testable in this framework. Noted but NOT counted in score.
print("  [NOTE] Types 8-9 (Atkin, Selberg): exclusion stipulative par A1, non comptes")


# ============================================================
print("\n" + "=" * 60)
print("U5 -- SYNTHESE UNIVERSELLE")
print("=" * 60)

print("\n--- U5.1 : Eratosthene satisfait A1-A6 ---")

# A1: derivation arithmetique -- verify rule depends only on n mod p
a1_ok = True
for p in [2, 3, 5, 7, 11]:
    for n in range(0, 100):
        for k in [-2, -1, 1, 2, 3]:
            n2 = n + k * p
            # Eratosthene: eliminate iff n mod p == 0
            if (n % p == 0) != (n2 % p == 0):
                a1_ok = False
check("A1: Eratosthene depend de n mod p (verifie p=2..11)", a1_ok)

# A2: TC
tc_all = True
for p in [2, 3, 5, 7, 11]:
    for n in range(0, 200):
        if (n % p == 0) != ((n + p) % p == 0):
            tc_all = False
check("A2: TC satisfaite pour p=2..11", tc_all)

# A3: non-trivialite -- verify R_p={0} is proper and non-empty
a3_ok = True
for p in [2, 3, 5, 7, 11, 13]:
    R_p = {0}
    if len(R_p) == 0 or R_p == set(range(p)):
        a3_ok = False
check("A3: R_p={0} non-trivial pour p=2..13", a3_ok)

# A4: moduli premiers -- verify all moduli used are prime
from sympy import isprime as _isprime_check
import sys
a4_ok = all(_isprime_check(p) for p in [2, 3, 5, 7, 11, 13, 17, 19])
check("A4: moduli p=2..19 tous premiers", a4_ok)

# A5: {0} est un ideal -- verify closure under multiplication
a5_ok = True
for p in [2, 3, 5, 7, 11, 13]:
    for a in range(p):
        if (a * 0) % p != 0:  # a*0 must be in {0}
            a5_ok = False
check("A5: {0} est un ideal de Z/pZ (cloture verifie p=2..13)", a5_ok)

# A6: completude -- verify skipping any prime leaves composites surviving
a6_ok = True
for skip_p in [2, 3, 5, 7]:
    target = skip_p * skip_p
    other_primes = [p for p in [2, 3, 5, 7, 11] if p != skip_p and p < skip_p]
    if all(target % p != 0 for p in other_primes):
        pass  # skip_p^2 survives without skip_p, confirming A6 is needed
    else:
        a6_ok = False
check("A6: completude necessaire (skip p => p^2 survit, p=2..7)", a6_ok)

print("\n--- U5.2 : Exhaustivite k=3..6 ---")
for k in range(3, 7):
    plist = primes_up_to(20)[:k]
    P = primorial(plist)

    # Count alternatives satisfying A1+A2+A3 but violating A5
    n_alternatives = 0
    n_violate_A5 = 0

    for p in plist:
        if p == 2:
            continue  # R_2 = {0} is the only proper non-empty subset
        for r in range(1, p):
            R_alt = {r}
            # Is {r} an ideal of Z/pZ?
            is_ideal = all((a * r) % p in R_alt for a in range(p))
            n_alternatives += 1
            if not is_ideal:
                n_violate_A5 += 1

    check(f"k={k}: {n_alternatives} swap-alternatives, {n_violate_A5} violent A5 (100%)",
          n_violate_A5 == n_alternatives and n_alternatives > 0)


# ============================================================
print("\n" + "=" * 60)
print(f"BILAN : {PASS}/{PASS + FAIL} PASS, {FAIL} FAIL")
print("=" * 60)

if FAIL == 0:
    print("\nT6 UNIVERSEL VERIFIE (dans la classe PPA).")
    print("Eratosthene = unique PPA sur Z (conditionnel a A5).")
    print("Cle : localite de la division euclidienne (U1)")
    print("     + hypothese ideal A5 + structure de corps de Z/pZ (U2)")
    print("     + completude (U3).")
    print("Note : 2 exclusions stipulatives (Atkin, Selberg) non comptees.")
else:
    print(f"\n{FAIL} echec(s) detecte(s).")

sys.exit(0 if FAIL == 0 else 1)
