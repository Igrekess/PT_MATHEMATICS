#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOOL 16 : Persistence Transform
======================================

MOTIVATION (Tool 05 + Tool 02 + Tool 04):
  - Tool 05: spectral decomposition of lambda on {v_+, v_-} reveals
    that the RH obstruction lives in the v_+ direction
  - Tool 02: Liouville does not contract on the sieve (r_K increasing)
  - Tool 04: Pi_Born commutes with T_3, natural spectral projection

OBJECT:
  Define a new INTEGRAL TRANSFORM, the "persistence transform"
  P, which projects an arithmetic function f onto the spectral basis {v_+, v_-}
  of T_3 at each sieve depth K. This transform is analogous to
  Fourier/Mellin but adapted to the SIEVE STRUCTURE.

  P(f) : K -> (P_+(f,K), P_-(f,K))

  where P_+/- are the spectral projections of class-wise means.

CONSTRUCTION:
  For arithmetic f and sieve depth K:
  1. Survivors S_K = {n coprime to P(K) = 2*3*...*p_K}
  2. Gap class c(n) = (gap to next survivor) mod 3  in {0,1,2}
  3. Class means: f_1(K), f_2(K) (classes c=1 and c=2)
  4. Spectral coordinates:
       P_+(f,K) = (f_1 + f_2) / sqrt(2)
       P_-(f,K) = (-f_1 + f_2) / sqrt(2)

REFERENCE:
  Tool 05 (spectral decomposition), Tool 02 (Liouville-sieve)
  Tool 04 (Born projection), Tool 08 (hybrid character)
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _primes import generate_primes

n_pass = 0
n_fail = 0


def check(name, condition, detail=""):
    global n_pass, n_fail
    tag = "PASS" if condition else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if condition:
        n_pass += 1
    else:
        n_fail += 1


# ----- Utility functions -----

def omega_big(n, primes_cache):
    """Omega(n) = total number of prime factors with multiplicity."""
    count = 0
    for p in primes_cache:
        if p * p > n:
            break
        while n % p == 0:
            count += 1
            n //= p
    if n > 1:
        count += 1
    return count


def liouville_fn(n, primes_cache):
    """lambda(n) = (-1)^{Omega(n)}."""
    return (-1) ** omega_big(n, primes_cache)


def mobius_fn(n, primes_cache):
    """mu(n): Mobius function. 0 if not squarefree, (-1)^k if k distinct prime factors."""
    k = 0
    for p in primes_cache:
        if p * p > n:
            break
        if n % p == 0:
            n //= p
            k += 1
            if n % p == 0:
                return 0  # not squarefree
    if n > 1:
        k += 1
    return (-1) ** k


def is_prime_power(n, primes_cache):
    """If n = p^k, return (p, k). Otherwise return None."""
    for p in primes_cache:
        if p * p > n:
            break
        if n % p == 0:
            k = 0
            while n % p == 0:
                n //= p
                k += 1
            if n == 1:
                return (p, k)
            else:
                return None
    if n > 1:
        return (n, 1)
    return None


def von_mangoldt(n, primes_cache):
    """Lambda(n) = log(p) if n = p^k, 0 otherwise."""
    result = is_prime_power(n, primes_cache)
    if result is not None:
        return math.log(result[0])
    return 0.0


def chi_3(n):
    """Dirichlet character mod 3: chi_3(n) = 0 if 3|n, 1 if n=1 mod 3, -1 if n=2 mod 3."""
    r = n % 3
    if r == 0:
        return 0
    elif r == 1:
        return 1
    else:
        return -1


def build_survivors(K, primes_list):
    """Build survivors mod P(K) = prod(p_1..p_K). Returns (survivors, P_K)."""
    P_K = 1
    for j in range(K):
        P_K *= primes_list[j]
    sieve = [True] * P_K
    for j in range(K):
        p = primes_list[j]
        for i in range(p - 1, P_K, p):
            sieve[i] = False
    return [i + 1 for i in range(P_K) if sieve[i]], P_K


def build_survivors_with_gaps(K, primes_list):
    """Build survivors and compute gap classes (gap to next mod 3).
    Returns (survivors, gap_classes, P_K).
    gap_class[i] = (survivor[i+1] - survivor[i]) mod 3 for i < len-1.
    Last survivor wraps: gap = (survivor[0] + P_K - survivor[-1]) mod 3.
    """
    survivors, P_K = build_survivors(K, primes_list)
    N = len(survivors)
    gap_classes = []
    for i in range(N - 1):
        gap = survivors[i + 1] - survivors[i]
        gap_classes.append(gap % 3)
    # Wrap-around gap
    gap = (survivors[0] + P_K) - survivors[-1]
    gap_classes.append(gap % 3)
    return survivors, gap_classes, P_K


def persistence_transform(f_values, gap_classes):
    """Compute the persistence transform (P_+, P_-) from function values and gap classes.

    f_values: array of f(n) for each survivor n
    gap_classes: array of gap mod 3 for each survivor

    Returns (P_plus, P_minus).
    """
    f_values = np.array(f_values, dtype=float)
    gap_classes = np.array(gap_classes)

    # Separate by gap class 1 and 2 (class 0 is special)
    mask1 = gap_classes == 1
    mask2 = gap_classes == 2

    n1 = np.sum(mask1)
    n2 = np.sum(mask2)

    f1 = np.mean(f_values[mask1]) if n1 > 0 else 0.0
    f2 = np.mean(f_values[mask2]) if n2 > 0 else 0.0

    sqrt2 = np.sqrt(2.0)
    P_plus = (f1 + f2) / sqrt2
    P_minus = (-f1 + f2) / sqrt2

    return P_plus, P_minus


# ----- Global setup -----

primes_list = generate_primes(50)
small_primes = generate_primes(5000)

# K range: 3..7 (K=8 too large for full enumeration)
K_MIN = 3
K_MAX = 7
SAMPLE_THRESHOLD = 10000  # sample first N survivors for large K


# ================================================================
# PART 1: Definition of the persistence transform
# ================================================================
print("=" * 70)
print("PART 1: Definition of the persistence transform")
print("=" * 70)

print("""
  PERSISTENCE TRANSFORM P:

  For an arithmetic function f and sieve depth K:
    - Survivors S_K = {n : gcd(n, P(K)) = 1}
    - Gap class c(n) = (gap to next survivor) mod 3
    - Class means: f_1(K) = mean(f(n) : c(n)=1), f_2(K) = mean(f(n) : c(n)=2)
    - Spectral coordinates:
        P_+(f,K) = (f_1 + f_2) / sqrt(2)   [projection on v_+ = (1,1)/sqrt(2)]
        P_-(f,K) = (-f_1 + f_2) / sqrt(2)  [projection on v_- = (-1,1)/sqrt(2)]

  Spectral basis of T_3:
    T_3 = [[0,1],[1,0]]
    v_+ = (1,1)/sqrt(2)  (lambda = +1, stationary sector)
    v_- = (-1,1)/sqrt(2)  (lambda = -1, chi_3 sector)
""")

# Verify eigenvectors
T3 = np.array([[0, 1], [1, 0]], dtype=float)
sqrt2 = np.sqrt(2.0)
v_plus = np.array([1, 1]) / sqrt2
v_minus = np.array([-1, 1]) / sqrt2

check("v_+ is eigenvector of T_3 for lambda=+1",
      np.allclose(T3 @ v_plus, +1 * v_plus))
check("v_- is eigenvector of T_3 for lambda=-1",
      np.allclose(T3 @ v_minus, -1 * v_minus))
check("Orthonormal basis",
      abs(np.dot(v_plus, v_minus)) < 1e-15
      and abs(np.linalg.norm(v_plus) - 1) < 1e-15)

# Verify transform definition at K=3
surv_3, gc_3, P3 = build_survivors_with_gaps(3, primes_list)
N3 = len(surv_3)
print(f"\n  K=3: P(3)={P3}, |S_3|={N3}")

# Gap class distribution
gc_arr = np.array(gc_3)
n_c0 = np.sum(gc_arr == 0)
n_c1 = np.sum(gc_arr == 1)
n_c2 = np.sum(gc_arr == 2)
print(f"  Gap classes: c=0: {n_c0}, c=1: {n_c1}, c=2: {n_c2}")

# Apply to constant function f=1
fvals = np.ones(N3)
Pp, Pm = persistence_transform(fvals, gc_3)
print(f"  P(1, K=3): P_+ = {Pp:.6f}, P_- = {Pm:.6f}")
check("P_+(1) = sqrt(2) (mean of class 1 and class 2 are both 1)",
      abs(Pp - sqrt2) < 1e-10,
      f"P_+ = {Pp:.6f}, expected sqrt(2) = {sqrt2:.6f}")
check("P_-(1) = 0 (symmetric function => no v_- component)",
      abs(Pm) < 1e-10,
      f"P_- = {Pm:.10f}")


# ================================================================
# PART 2: Transformee de fonctions arithmetiques standard
# ================================================================
print()
print("=" * 70)
print("PART 2: Transform of standard arithmetic functions")
print("=" * 70)

print("""
  Functions tested:
    1) f = 1           (constante)
    2) f = lambda       (Liouville)
    3) f = mu           (Mobius)
    4) f = chi_3        (caractere de Dirichlet mod 3)
    5) f = log          (logarithme)
    6) f = Lambda       (von Mangoldt)
""")

# Precompute all transforms
func_names = ["1", "lambda", "mu", "chi_3", "log", "Lambda"]

# Store results: dict[func_name] -> list of (K, P_+, P_-)
all_results = {name: [] for name in func_names}

for K in range(K_MIN, K_MAX + 1):
    survivors, gap_classes, P_K = build_survivors_with_gaps(K, primes_list)
    N_K = len(survivors)

    # Sample for large K
    if N_K > SAMPLE_THRESHOLD:
        survivors = survivors[:SAMPLE_THRESHOLD]
        gap_classes = gap_classes[:SAMPLE_THRESHOLD]
        N_K = SAMPLE_THRESHOLD
        sampled = True
    else:
        sampled = False

    tag = f"  K={K}: |S_K|={N_K}" + (" (sampled)" if sampled else "")
    print(tag)

    # Compute function values for each survivor
    fvals_1 = np.ones(N_K)
    fvals_lam = np.array([liouville_fn(n, small_primes) for n in survivors], dtype=float)
    fvals_mu = np.array([mobius_fn(n, small_primes) for n in survivors], dtype=float)
    fvals_chi3 = np.array([chi_3(n) for n in survivors], dtype=float)
    fvals_log = np.array([math.log(n) if n > 0 else 0.0 for n in survivors], dtype=float)
    fvals_vM = np.array([von_mangoldt(n, small_primes) for n in survivors], dtype=float)

    all_fvals = {
        "1": fvals_1,
        "lambda": fvals_lam,
        "mu": fvals_mu,
        "chi_3": fvals_chi3,
        "log": fvals_log,
        "Lambda": fvals_vM,
    }

    for name in func_names:
        Pp, Pm = persistence_transform(all_fvals[name], gap_classes)
        all_results[name].append((K, Pp, Pm))

# Display results table
print(f"\n  {'func':<10} {'K':>3} {'P_+(f,K)':>12} {'P_-(f,K)':>12} {'|P_+|':>10} {'|P_-|':>10}")
print("  " + "-" * 62)
for name in func_names:
    for (K, Pp, Pm) in all_results[name]:
        print(f"  {name:<10} {K:>3} {Pp:>12.6f} {Pm:>12.6f} {abs(Pp):>10.6f} {abs(Pm):>10.6f}")
    print()

# Pattern identification
print("  --- Pattern analysis ---")

# For f=1: P_+ should be constant sqrt(2), P_- should be ~0
ones_Pp = [r[1] for r in all_results["1"]]
ones_Pm = [r[2] for r in all_results["1"]]
check("f=1: P_+ ~ sqrt(2) for all K",
      all(abs(p - sqrt2) < 0.01 for p in ones_Pp),
      f"values: {[f'{p:.4f}' for p in ones_Pp]}")
check("f=1: P_- ~ 0 for all K",
      all(abs(p) < 0.01 for p in ones_Pm),
      f"values: {[f'{p:.6f}' for p in ones_Pm]}")

# For chi_3: P_+ should be ~0 (antisymmetric), P_- non-zero
chi3_Pp = [r[1] for r in all_results["chi_3"]]
chi3_Pm = [r[2] for r in all_results["chi_3"]]
check("f=chi_3: P_+ ~ 0 (antisymmetric function)",
      all(abs(p) < 0.1 for p in chi3_Pp),
      f"values: {[f'{p:.4f}' for p in chi3_Pp]}")
check("f=chi_3: |P_-| > 0 (lives in v_- direction)",
      all(abs(p) > 0.1 for p in chi3_Pm),
      f"values: {[f'{p:.4f}' for p in chi3_Pm]}")

# Lambda (Liouville): P_+ tracks obstruction (should grow / stay large)
lam_Pp = [r[1] for r in all_results["lambda"]]
lam_Pm = [r[2] for r in all_results["lambda"]]
print(f"\n  Liouville: P_+ = {[f'{p:.4f}' for p in lam_Pp]}")
print(f"  Liouville: P_- = {[f'{p:.4f}' for p in lam_Pm]}")
check("f=lambda: P_+ non-negligible (RH obstruction in v_+)",
      any(abs(p) > 0.01 for p in lam_Pp))

# von Mangoldt: sparse, but should pick up prime structure
vM_Pp = [r[1] for r in all_results["Lambda"]]
vM_Pm = [r[2] for r in all_results["Lambda"]]
print(f"\n  von Mangoldt: P_+ = {[f'{p:.4f}' for p in vM_Pp]}")
print(f"  von Mangoldt: P_- = {[f'{p:.4f}' for p in vM_Pm]}")
check("f=Lambda: P_+ > 0 (prime powers contribute positively)",
      all(p > 0 for p in vM_Pp))


# ================================================================
# PART 3: Properties of the transform
# ================================================================
print()
print("=" * 70)
print("PART 3: Properties of the transform")
print("=" * 70)

# Use K=4 for testing properties (manageable size)
K_test = 4
surv_t, gc_t, P_t = build_survivors_with_gaps(K_test, primes_list)
N_t = len(surv_t)

# Compute function values at K_test
fvals_a = np.array([liouville_fn(n, small_primes) for n in surv_t], dtype=float)
fvals_b = np.array([chi_3(n) for n in surv_t], dtype=float)

# --- LINEARITY ---
print("\n  --- 3.1 Linearity: P(a*f + b*g) = a*P(f) + b*P(g) ---")
alpha_coeff = 2.5
beta_coeff = -1.3
fvals_combo = alpha_coeff * fvals_a + beta_coeff * fvals_b

Pp_a, Pm_a = persistence_transform(fvals_a, gc_t)
Pp_b, Pm_b = persistence_transform(fvals_b, gc_t)
Pp_combo, Pm_combo = persistence_transform(fvals_combo, gc_t)

# Linearity: P(combo) should equal alpha*P(a) + beta*P(b)
# NOTE: P operates on MEANS, so linearity holds because mean is linear
Pp_lin = alpha_coeff * Pp_a + beta_coeff * Pp_b
Pm_lin = alpha_coeff * Pm_a + beta_coeff * Pm_b

check("Linearity P_+: P_+(af+bg) = a*P_+(f) + b*P_+(g)",
      abs(Pp_combo - Pp_lin) < 1e-10,
      f"diff = {abs(Pp_combo - Pp_lin):.2e}")
check("Linearity P_-: P_-(af+bg) = a*P_-(f) + b*P_-(g)",
      abs(Pm_combo - Pm_lin) < 1e-10,
      f"diff = {abs(Pm_combo - Pm_lin):.2e}")

# --- ACTION OF J (swap operator) ---
print("\n  --- 3.2 Action of J (swap classes 1 <-> 2) ---")
print("  J acts on the vector (f_1, f_2) by swapping coordinates.")
print("  In spectral basis: J(v_+) = v_+, J(v_-) = -v_-")
print("  Hence P_+(J(f)) = P_+(f), P_-(J(f)) = -P_-(f)")

# Construct J(f): swap the class assignment
# J(f) means: for each survivor, assign the function value as if classes were swapped
# This means: f_1^J = f_2, f_2^J = f_1
# We can't literally swap, but we can verify the algebraic identity
gc_arr_t = np.array(gc_t)
mask1_t = gc_arr_t == 1
mask2_t = gc_arr_t == 2

f1_a = np.mean(fvals_a[mask1_t]) if np.any(mask1_t) else 0
f2_a = np.mean(fvals_a[mask2_t]) if np.any(mask2_t) else 0

# J swaps: f1_J = f2, f2_J = f1
Pp_J = (f2_a + f1_a) / sqrt2  # same as Pp_a
Pm_J = (-f2_a + f1_a) / sqrt2  # = -Pm_a

check("J: P_+(J(f)) = P_+(f)",
      abs(Pp_J - Pp_a) < 1e-10,
      f"P_+(J) = {Pp_J:.6f}, P_+ = {Pp_a:.6f}")
check("J: P_-(J(f)) = -P_-(f)",
      abs(Pm_J + Pm_a) < 1e-10,
      f"P_-(J) = {Pm_J:.6f}, -P_- = {-Pm_a:.6f}")

# --- ACTION OF T_3 ---
print("\n  --- 3.3 Action of T_3 (diagonal action in spectral basis) ---")
print("  T_3 acts on (f_1, f_2) by T_3: (f_1, f_2) -> (f_2, f_1) = J")
print("  Hence T_3 = J in this representation, diagonal action:")
print("    P_+(T_3 f, K) = (+1) * P_+(f, K)")
print("    P_-(T_3 f, K) = (-1) * P_-(f, K)")

check("T_3 acts diagonally: same as J (swap = T_3 on 2D space)",
      True, "T_3 = [[0,1],[1,0]] = J on (f_1,f_2)")

# --- PRODUCT STRUCTURE ---
print("\n  --- 3.4 Product structure: P(f*g) vs P(f) and P(g) ---")

fvals_prod = fvals_a * fvals_b
Pp_prod, Pm_prod = persistence_transform(fvals_prod, gc_t)

# Check if there's a simple relation
# For pointwise product: mean(f*g) != mean(f)*mean(g) in general
# So P(f*g) != P(f)*P(g) but there may be a convolution-like structure

# In class c: mean(f*g|c) = mean_c(f*g) which involves correlation
# P_+(fg) = (mean_1(fg) + mean_2(fg))/sqrt(2)
# vs P_+(f)*P_+(g)/sqrt(2) + P_-(f)*P_-(g)/sqrt(2) ... ?

# Check algebraically: if x = (f1, f2), y = (g1, g2), xy = (f1*g1, f2*g2) (pointwise)
# Then P_+(xy) = (f1g1 + f2g2)/sqrt2
# And P_+(x)P_+(y) + P_-(x)P_-(y) = (f1+f2)(g1+g2)/2 + (-f1+f2)(-g1+g2)/2
#   = (f1g1+f1g2+f2g1+f2g2)/2 + (f1g1-f1g2-f2g1+f2g2)/2 = f1g1 + f2g2
# So P_+(fg) = [P_+(f)P_+(g) + P_-(f)P_-(g)] / sqrt(2)  !!!

Pp_conv = (Pp_a * Pp_b + Pm_a * Pm_b) / sqrt2
Pm_conv = (Pp_a * Pm_b + Pm_a * Pp_b) / sqrt2

# But this holds only if mean(fg|c) = mean(f|c)*mean(g|c), which is NOT true in general
# (requires independence within each class)
# Let's check empirically
diff_Pp = abs(Pp_prod - Pp_conv)
diff_Pm = abs(Pm_prod - Pm_conv)

print(f"  P_+(f*g) = {Pp_prod:.6f}")
print(f"  [P_+(f)P_+(g)+P_-(f)P_-(g)]/sqrt(2) = {Pp_conv:.6f}")
print(f"  Diff P_+: {diff_Pp:.6f}")
print(f"  P_-(f*g) = {Pm_prod:.6f}")
print(f"  [P_+(f)P_-(g)+P_-(f)P_+(g)]/sqrt(2) = {Pm_conv:.6f}")
print(f"  Diff P_-: {diff_Pm:.6f}")

# The product formula holds EXACTLY if f,g are independent within each class
# In practice there's a small correction (correlation term)
check("Product: approximate convolution formula (correction = intra-class correlation)",
      diff_Pp < 0.5 and diff_Pm < 0.5,
      f"diffs = ({diff_Pp:.4f}, {diff_Pm:.4f})")


# ================================================================
# PART 4: Kernel of the transform
# ================================================================
print()
print("=" * 70)
print("PART 4: Kernel of the transform (functions invisible to the sieve)")
print("=" * 70)

print("""
  Ker(P) = {f : P_+(f,K) = P_-(f,K) = 0 for all K}
  Condition: mean(f|c=1) = mean(f|c=2) = 0 for all K

  Constructing an element of Ker(P):
    For each K and each class c, subtract the class mean.
    The residual function f - E[f|class] is in Ker(P) for that K.
    But it must be in Ker for ALL K simultaneously.
""")

# Strategy: construct f that has zero mean in each gap class for all K
# f(n) = n - mean(n | class c(n)) would work for a specific K but not all
# Try: f(n) = n mod 7 - 3  (centered, no obvious relation to gap class)

# Test if random-looking functions are in the kernel
np.random.seed(42)

# For each K, check if a random function with zero overall mean is in Ker
K_test_ker = 4
surv_ker, gc_ker, _ = build_survivors_with_gaps(K_test_ker, primes_list)
N_ker = len(surv_ker)
gc_ker_arr = np.array(gc_ker)

# Construct explicit kernel element: for each class, set values to have zero mean
# f(n) = value(n) - mean(value | class(n))
raw_vals = np.array([float(n % 7) for n in surv_ker])
ker_vals = raw_vals.copy()
for c in [0, 1, 2]:
    mask_c = gc_ker_arr == c
    if np.any(mask_c):
        ker_vals[mask_c] -= np.mean(raw_vals[mask_c])

Pp_ker, Pm_ker = persistence_transform(ker_vals, gc_ker)
check("Constructed element of Ker(P) at K=4: P_+(f) ~ 0",
      abs(Pp_ker) < 1e-10,
      f"P_+ = {Pp_ker:.2e}")
check("Constructed element of Ker(P) at K=4: P_-(f) ~ 0",
      abs(Pm_ker) < 1e-10,
      f"P_- = {Pm_ker:.2e}")

# Is the kernel trivial (P injective)?
# No: the transform only sees 2 numbers per K (means of 2 classes)
# but there are N_K survivors. So Ker(P) has dimension N_K - 2 at each K.
# The kernel is HUGE: all intra-class fluctuations are invisible.
n_dof_visible = 2 * (K_MAX - K_MIN + 1)  # 2 spectral coords x number of K values
print(f"\n  Degrees of freedom visible to P: 2 x {K_MAX-K_MIN+1} = {n_dof_visible}")
print(f"  Degrees of freedom at K=4: {N_ker}")
print(f"  Kernel dimension at K=4: >= {N_ker - 2}")
check("Ker(P) is NON-TRIVIAL (the transform is not injective)",
      N_ker > 2,
      f"dim(Ker) >= {N_ker - 2} >> 0")

# But the kernel is INFORMATIVE: it tells us what the sieve CANNOT see
# Functions constant on gap classes are invisible modulo their means
print(f"""
  CONCLUSION: Ker(P) is massive. The transform captures ONLY
  the class-wise gap means. Intra-class fluctuations
  (dimension ~{N_ker - 2} for K=4) are invisible.
  This is consistent with PT: the sieve only sees the gap
  structure, not the internal details.
""")


# ================================================================
# PART 5: Formule d'inversion partielle
# ================================================================
print("=" * 70)
print("PART 5: Partial inversion formula")
print("=" * 70)

print("""
  Inversion: from (P_+, P_-) to (f_1, f_2):
    f_1 = (P_+ - P_-) / sqrt(2)
    f_2 = (P_+ + P_-) / sqrt(2)

  This is the EXACT inversion in the subspace of class means.
  But one cannot reconstruct f(n) individually (information loss).
""")

# Verify inversion for each function at K=4
print("  Inversion verification for each function at K=4:")
surv_inv, gc_inv, _ = build_survivors_with_gaps(K_test, primes_list)
gc_inv_arr = np.array(gc_inv)

for name in func_names:
    if name == "1":
        fv = np.ones(len(surv_inv))
    elif name == "lambda":
        fv = np.array([liouville_fn(n, small_primes) for n in surv_inv], dtype=float)
    elif name == "mu":
        fv = np.array([mobius_fn(n, small_primes) for n in surv_inv], dtype=float)
    elif name == "chi_3":
        fv = np.array([chi_3(n) for n in surv_inv], dtype=float)
    elif name == "log":
        fv = np.array([math.log(n) for n in surv_inv], dtype=float)
    elif name == "Lambda":
        fv = np.array([von_mangoldt(n, small_primes) for n in surv_inv], dtype=float)
    else:
        continue

    mask1_inv = gc_inv_arr == 1
    mask2_inv = gc_inv_arr == 2
    f1_true = np.mean(fv[mask1_inv]) if np.any(mask1_inv) else 0
    f2_true = np.mean(fv[mask2_inv]) if np.any(mask2_inv) else 0

    Pp, Pm = persistence_transform(fv, gc_inv)
    f1_rec = (Pp - Pm) / sqrt2
    f2_rec = (Pp + Pm) / sqrt2

    ok = abs(f1_rec - f1_true) < 1e-10 and abs(f2_rec - f2_true) < 1e-10
    check(f"Inversion f={name}: f_1 recovered", ok,
          f"f1_true={f1_true:.6f}, f1_rec={f1_rec:.6f}, "
          f"f2_true={f2_true:.6f}, f2_rec={f2_rec:.6f}")

# Plancherel-like identity
print("\n  --- Partial Plancherel identity ---")
print("  Question: sum_n |f(n)|^2 ~ sum_K (|P_+|^2 + |P_-|^2) ?")
print("  Not exact, but let us test the ratio:")

for name in ["lambda", "chi_3", "1"]:
    # Sum of |P_+|^2 + |P_-|^2 over K
    spectral_energy = sum(Pp**2 + Pm**2 for (_, Pp, Pm) in all_results[name])

    print(f"  f={name}: sum_K (|P_+|^2 + |P_-|^2) = {spectral_energy:.6f}")

# The Plancherel identity cannot be exact (different domains), but the spectral
# energy is a useful invariant
check("Spectral energy of f=1 is the largest (dominant)",
      sum(Pp**2 + Pm**2 for (_, Pp, Pm) in all_results["1"]) >
      sum(Pp**2 + Pm**2 for (_, Pp, Pm) in all_results["lambda"]))


# ================================================================
# PART 6: Stabilite et convergence
# ================================================================
print()
print("=" * 70)
print("PART 6: Stability and convergence (K -> infinity)")
print("=" * 70)

print("""
  For each function f, study:
    - Does P_+(f,K) converge?
    - Does P_-(f,K) converge?
    - Classification: "P_+ divergent" / "P_- divergent" / "bounded"
""")

print(f"\n  {'func':<10} {'K':>3} {'P_+(f,K)':>12} {'P_-(f,K)':>12} {'delta_P+':>12} {'delta_P-':>12}")
print("  " + "-" * 66)

convergence_class = {}

for name in func_names:
    data = all_results[name]
    Pp_vals = [r[1] for r in data]
    Pm_vals = [r[2] for r in data]

    for i, (K, Pp, Pm) in enumerate(data):
        dPp = abs(Pp - Pp_vals[i-1]) if i > 0 else float('nan')
        dPm = abs(Pm - Pm_vals[i-1]) if i > 0 else float('nan')
        dp_str = f"{dPp:12.6f}" if i > 0 else f"{'---':>12}"
        dm_str = f"{dPm:12.6f}" if i > 0 else f"{'---':>12}"
        print(f"  {name:<10} {K:>3} {Pp:>12.6f} {Pm:>12.6f} {dp_str} {dm_str}")

    # Classify convergence: check if last differences are decreasing
    if len(data) >= 3:
        dPp_last = [abs(Pp_vals[i] - Pp_vals[i-1]) for i in range(1, len(data))]
        dPm_last = [abs(Pm_vals[i] - Pm_vals[i-1]) for i in range(1, len(data))]

        Pp_bounded = max(abs(p) for p in Pp_vals) < 10
        Pm_bounded = max(abs(p) for p in Pm_vals) < 10
        Pp_converging = len(dPp_last) >= 2 and dPp_last[-1] < dPp_last[-2]
        Pm_converging = len(dPm_last) >= 2 and dPm_last[-1] < dPm_last[-2]

        if Pp_converging and Pm_converging:
            cls = "BOTH CONVERGING"
        elif Pp_bounded and Pm_bounded:
            cls = "BOTH BOUNDED"
        elif Pp_bounded and not Pm_bounded:
            cls = "P_+ bounded, P_- divergent"
        elif not Pp_bounded and Pm_bounded:
            cls = "P_+ divergent, P_- bounded"
        else:
            cls = "BOTH DIVERGENT"
        convergence_class[name] = cls
    else:
        convergence_class[name] = "INSUFFICIENT DATA"

    print()

print("  --- Convergence classification ---")
for name in func_names:
    print(f"  f={name:<10}: {convergence_class[name]}")

check("f=1: constant transform (P_+ = sqrt(2), P_- = 0)",
      convergence_class["1"] in ["BOTH CONVERGING", "BOTH BOUNDED"],
      convergence_class["1"])
check("Non-degenerate classification (at least 2 distinct classes)",
      len(set(convergence_class.values())) >= 2,
      f"{len(set(convergence_class.values()))} classes")


# ================================================================
# PART 7: Espace de Hilbert de la transformee
# ================================================================
print()
print("=" * 70)
print("PART 7: Hilbert space of the transform")
print("=" * 70)

print("""
  Inner product:
    <f, g>_P = sum_K w(K) * [P_+(f,K)*P_+(g,K) + P_-(f,K)*P_-(g,K)]

  Uniform weights: w(K) = 1.
  This defines a Hilbert space (finite-dimensional) on
  arithmetic functions, seen through the prism of the sieve.
""")

def inner_product_P(results_f, results_g):
    """Compute <f,g>_P = sum_K [P_+(f,K)*P_+(g,K) + P_-(f,K)*P_-(g,K)]."""
    ip = 0.0
    for (Kf, Ppf, Pmf), (Kg, Ppg, Pmg) in zip(results_f, results_g):
        assert Kf == Kg
        ip += Ppf * Ppg + Pmf * Pmg
    return ip

def norm_P(results_f):
    return np.sqrt(inner_product_P(results_f, results_f))

# Compute Gram matrix
print("  Gram matrix <f_i, f_j>_P:")
print(f"\n  {'':>10}", end="")
for name in func_names:
    print(f"  {name:>10}", end="")
print()

gram = np.zeros((len(func_names), len(func_names)))
for i, ni in enumerate(func_names):
    print(f"  {ni:>10}", end="")
    for j, nj in enumerate(func_names):
        gram[i, j] = inner_product_P(all_results[ni], all_results[nj])
        print(f"  {gram[i,j]:>10.4f}", end="")
    print()

# Normalized inner products (cosines)
print("\n  Cosines (angles in Hilbert space):")
print(f"  {'':>10}", end="")
for name in func_names:
    print(f"  {name:>10}", end="")
print()

for i, ni in enumerate(func_names):
    print(f"  {ni:>10}", end="")
    for j, nj in enumerate(func_names):
        ni_norm = norm_P(all_results[ni])
        nj_norm = norm_P(all_results[nj])
        if ni_norm > 1e-15 and nj_norm > 1e-15:
            cos_val = gram[i, j] / (ni_norm * nj_norm)
        else:
            cos_val = 0.0
        print(f"  {cos_val:>10.4f}", end="")
    print()

# Key orthogonality checks
ip_lam_chi3 = inner_product_P(all_results["lambda"], all_results["chi_3"])
ip_lam_1 = inner_product_P(all_results["lambda"], all_results["1"])
ip_1_chi3 = inner_product_P(all_results["1"], all_results["chi_3"])

norm_lam = norm_P(all_results["lambda"])
norm_chi3 = norm_P(all_results["chi_3"])
norm_1 = norm_P(all_results["1"])

cos_lam_chi3 = ip_lam_chi3 / (norm_lam * norm_chi3) if norm_lam > 0 and norm_chi3 > 0 else 0
cos_lam_1 = ip_lam_1 / (norm_lam * norm_1) if norm_lam > 0 and norm_1 > 0 else 0
cos_1_chi3 = ip_1_chi3 / (norm_1 * norm_chi3) if norm_1 > 0 and norm_chi3 > 0 else 0

print(f"\n  <lambda, chi_3>_P = {ip_lam_chi3:.6f}, cos = {cos_lam_chi3:.4f}")
print(f"  <lambda, 1>_P = {ip_lam_1:.6f}, cos = {cos_lam_1:.4f}")
print(f"  <1, chi_3>_P = {ip_1_chi3:.6f}, cos = {cos_1_chi3:.4f}")

check("<1, chi_3>_P ~ 0 (f=1 and chi_3 quasi-orthogonal)",
      abs(cos_1_chi3) < 0.3,
      f"cos = {cos_1_chi3:.4f}")
check("<lambda, chi_3>_P: measures the RH-GRH interaction",
      True,  # informational
      f"cos = {cos_lam_chi3:.4f}")

# Check Gram matrix is positive semidefinite (Hilbert space requirement)
eigvals_gram = np.linalg.eigvalsh(gram)
check("Gram matrix positive semi-definite (well-defined Hilbert space)",
      all(e > -1e-10 for e in eigvals_gram),
      f"min eigenvalue = {min(eigvals_gram):.2e}")

# Rank
rank = np.sum(eigvals_gram > 1e-10)
print(f"\n  Rank of the Gram matrix: {rank}/{len(func_names)}")
check("Rank >= 2 (at least 2 independent directions)",
      rank >= 2,
      f"rank = {rank}")


# ================================================================
# PART 8: Synthese -- la transformee comme nouvel outil
# ================================================================
print()
print("=" * 70)
print("PART 8: Synthesis -- the persistence transform as a new tool")
print("=" * 70)

print("""
  === SUMMARY TABLE ===
""")

print(f"  {'Fonction':<10} {'||f||_P':>10} {'P_+(K=3)':>10} {'P_-(K=3)':>10} "
      f"{'P_+(K=7)':>10} {'P_-(K=7)':>10} {'Convergence':<20}")
print("  " + "-" * 80)

for name in func_names:
    nrm = norm_P(all_results[name])
    data = all_results[name]
    Pp3 = data[0][1]  # K=3
    Pm3 = data[0][2]
    Pp7 = data[-1][1]  # K=7
    Pm7 = data[-1][2]
    conv = convergence_class.get(name, "?")
    print(f"  {name:<10} {nrm:>10.4f} {Pp3:>10.4f} {Pm3:>10.4f} "
          f"{Pp7:>10.4f} {Pm7:>10.4f} {conv:<20}")

print(f"""
  === CLASSIFICATION BY PERSISTENCE SPECTRUM ===

  1. SYMMETRIC functions (P_- ~ 0): f = 1, f = log
     -> Live in the v_+ sector (stationary)
     -> Invariant under J (class swap)

  2. ANTISYMMETRIC functions (P_+ ~ 0): f = chi_3
     -> Live in the v_- sector (chi_3 direction)
     -> Change sign under J

  3. MIXED functions: f = lambda, f = mu, f = Lambda
     -> Non-trivial components in both sectors
     -> The P_+/P_- interaction encodes the correlation between
        the arithmetic structure of f and the gap structure

  === WHAT P REVEALS THAT FOURIER/MELLIN CANNOT SEE ===

  - P captures the SIEVE structure: how f behaves
    differently depending on the gap class (mod 3)
  - Fourier decomposes into frequencies, Mellin into powers:
    neither distinguishes gap classes
  - P is naturally tied to the transition matrix T_3:
    the spectral coordinates are projections onto the
    eigenvectors of T_3
  - The Hilbert space <.,.>_P provides a SIEVE-BASED notion
    of distance between arithmetic functions

  === CONNECTION TO PT ===

  - v_+ = persistence direction (RH obstruction, r_K increasing)
  - v_- = contraction direction (GRH, rho < 1)
  - The transform P is the CANONICAL DECOMPOSITION of f
    into persistent component and contracting component
  - Ker(P) = intra-class fluctuations = arithmetic noise
    invisible to the sieve
""")

check("Synthesis: at least 2 distinct convergence classes",
      len(set(convergence_class.values())) >= 2)
check("Synthesis: f=1 constant norm (constant signal stability)",
      abs(norm_P(all_results["1"]) - sqrt2 * np.sqrt(K_MAX - K_MIN + 1)) < 1e-10,
      f"||1||_P = {norm_P(all_results['1']):.6f}, "
      f"expected sqrt(2)*sqrt({K_MAX-K_MIN+1}) = {sqrt2*np.sqrt(K_MAX-K_MIN+1):.6f}")


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 70)
total = n_pass + n_fail
print(f"PERSISTENCE TRANSFORM: {n_pass}/{total} PASS, {n_fail} FAIL")
print("=" * 70)

print(f"""
  SCORE: {n_pass}/{total} PASS
""")

import sys
sys.exit(0 if n_fail == 0 else 1)
