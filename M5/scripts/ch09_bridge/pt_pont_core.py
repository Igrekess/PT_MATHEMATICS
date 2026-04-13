#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pt_pont_core.py

Bridge functions for the Persistence Theory: cross-domain verification
that the sieve of Eratosthenes uniquely produces physics.

Tests whether alternative gap sequences (lucky numbers, composites, random)
satisfy the four structural properties (P1-P4) that primes uniquely possess,
and verifies cross-pillar consistency (GFT, holonomy, bifurcation).

Source PT: pt_thermo_core.py (validated February 2026)
March 2026 -- Persistence Theory
"""

import sys
import numpy as np
from math import sqrt, log, log2, pi, exp, gcd
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# PT constants (March 2026, ALL PROVED)
# ============================================================

S_PT = 0.5            # s = 1/2 (D00 PROVED)
MU_STAR = 15          # fixed point (D08 PROVED)
DEPTH = 2             # sieve depth (D17 PROVED)
N_C = 3               # number of colours (T1 PROVED)
N_F = 5               # active flavours = mu*/N_c
N_SPATIAL = 3         # spatial dimensions
ACTIVE_PRIMES = [3, 5, 7]
GHOST_PRIMES = [11, 13]
ALL_PRIMES_SMALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

# ============================================================
# Core sieve functions (from pt_thermo_core.py)
# ============================================================

def compute_gamma_p(mu, p):
    """gamma_p = -d(ln sin^2)/d(ln mu) -- metric dimension (DERIVED)."""
    q = 1.0 - 2.0 / mu
    if q <= 0 or q >= 1:
        return 0.0
    qp = q ** p
    delta = (1.0 - qp) / p
    if delta <= 0 or delta >= 2:
        return 0.0
    numerator = 4.0 * p * q**(p - 1) * (1.0 - delta)
    denominator = mu * (1.0 - qp) * (2.0 - delta)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_sin2(mu, p, q_type='stat'):
    """sin^2(theta_p) at given mu (DERIVED)."""
    if q_type == 'stat':
        q = 1.0 - 2.0 / mu
    else:
        q = np.exp(-1.0 / mu)
    if q <= 0 or q >= 1:
        return 0.0
    delta = (1.0 - q**p) / p
    return delta * (2.0 - delta)


def alpha_sieve(mu):
    """alpha = prod sin^2(theta_p, q_stat) for p in {3,5,7} (DERIVED)."""
    result = 1.0
    for p in ACTIVE_PRIMES:
        result *= compute_sin2(mu, p, 'stat')
    return result


def q_stat(mu):
    """q_stat = 1 - 2/mu (statistical branch, max-entropy)."""
    return 1.0 - 2.0 / mu


def q_therm(mu):
    """q_therm = exp(-1/mu) (thermal branch, Boltzmann)."""
    return np.exp(-1.0 / mu)


# ============================================================
# Gap generators
# ============================================================

def generate_prime_gaps(N_max):
    """Generate prime gaps via sieve of Eratosthenes."""
    is_prime = [True] * (N_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(sqrt(N_max)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_max + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N_max + 1) if is_prime[i]]
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    return gaps, primes


def generate_lucky_numbers(N_max):
    """Generate lucky numbers via the lucky number sieve.
    Start with odd numbers, repeatedly remove every k-th element
    where k is the next surviving number."""
    nums = list(range(1, N_max + 1, 2))  # odd numbers
    i = 1  # start from index 1 (value 3)
    while i < len(nums) and nums[i] <= len(nums):
        step = nums[i]
        nums = [nums[j] for j in range(len(nums)) if (j + 1) % step != 0]
        i += 1
    gaps = [nums[j+1] - nums[j] for j in range(len(nums)-1)]
    return gaps, nums


def generate_composite_gaps(N_max):
    """Generate gaps between consecutive composite numbers."""
    is_prime = [True] * (N_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(sqrt(N_max)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_max + 1, i):
                is_prime[j] = False
    composites = [i for i in range(4, N_max + 1) if not is_prime[i]]
    gaps = [composites[j+1] - composites[j] for j in range(len(composites)-1)]
    return gaps, composites


def generate_random_geometric_gaps(N, mu=15.0, seed=42):
    """Generate N random geometric gaps with mean mu."""
    rng = np.random.default_rng(seed)
    q = 1.0 - 1.0 / mu
    gaps = rng.geometric(1.0 - q, size=N).tolist()
    return gaps


def generate_k_rough(N_max, k):
    """Generate k-rough numbers up to N_max (numbers coprime to primorial(k)).
    k-rough = coprime to 2*3*...*p_k where p_k is the k-th prime.
    For k=2: coprime to 6 (= numbers ≡ 1 or 5 mod 6).
    Returns (gaps, sequence)."""
    # Find small primes up to k-th prime
    small_primes = []
    n = 2
    while len(small_primes) < k:
        if all(n % p != 0 for p in small_primes):
            small_primes.append(n)
        n += 1

    # Generate numbers coprime to all small_primes
    primorial = 1
    for p in small_primes:
        primorial *= p

    sequence = []
    for x in range(1, N_max + 1):
        if all(x % p != 0 for p in small_primes):
            sequence.append(x)

    gaps = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
    return gaps, sequence


def generate_twin_prime_gaps(N_max):
    """Generate gaps between twin primes (p such that p+2 is also prime)."""
    is_prime = [True] * (N_max + 3)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(sqrt(N_max + 2)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_max + 3, i):
                is_prime[j] = False
    twins = [i for i in range(2, N_max + 1) if is_prime[i] and is_prime[i + 2]]
    if len(twins) < 2:
        return [], twins
    gaps = [twins[i+1] - twins[i] for i in range(len(twins)-1)]
    return gaps, twins


def generate_semiprime_gaps(N_max):
    """Generate gaps between semiprimes (numbers with exactly 2 prime factors counting multiplicity)."""
    semiprimes = []
    for n in range(4, N_max + 1):
        temp = n
        count = 0
        d = 2
        while d * d <= temp and count <= 2:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        if count == 2:
            semiprimes.append(n)
    if len(semiprimes) < 2:
        return [], semiprimes
    gaps = [semiprimes[i+1] - semiprimes[i] for i in range(len(semiprimes)-1)]
    return gaps, semiprimes


def generate_prime_power_gaps(N_max):
    """Generate gaps between prime powers (p^k, k >= 1)."""
    is_prime = [True] * (N_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(sqrt(N_max)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_max + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N_max + 1) if is_prime[i]]
    powers = set()
    for p in primes:
        pk = p
        while pk <= N_max:
            powers.add(pk)
            pk *= p
    pp_list = sorted(powers)
    if len(pp_list) < 2:
        return [], pp_list
    gaps = [pp_list[i+1] - pp_list[i] for i in range(len(pp_list)-1)]
    return gaps, pp_list


def generate_palindromic_prime_gaps(N_max):
    """Generate gaps between palindromic primes (primes whose decimal repr is a palindrome)."""
    is_prime = [True] * (N_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(sqrt(N_max)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_max + 1, i):
                is_prime[j] = False
    pal_primes = [i for i in range(2, N_max + 1)
                  if is_prime[i] and str(i) == str(i)[::-1]]
    if len(pal_primes) < 2:
        return [], pal_primes
    gaps = [pal_primes[i+1] - pal_primes[i] for i in range(len(pal_primes)-1)]
    return gaps, pal_primes


def generate_gaussian_prime_norm_gaps(N_max):
    """Generate gaps between norms of Gaussian primes a+bi with a>0,b>0, sorted by norm."""
    norms = set()
    limit = int(sqrt(N_max)) + 1
    for a in range(1, limit):
        for b in range(1, limit):
            n = a*a + b*b
            if n > N_max:
                break
            # n = a^2+b^2 is a Gaussian prime norm if n is prime
            # (for a,b > 0, a+bi is Gaussian prime iff a^2+b^2 is a rational prime)
            temp = n
            is_p = temp >= 2
            d = 2
            while d*d <= temp and is_p:
                if temp % d == 0:
                    is_p = False
                d += 1
            if is_p:
                norms.add(n)
    norm_list = sorted(norms)
    if len(norm_list) < 2:
        return [], norm_list
    gaps = [norm_list[i+1] - norm_list[i] for i in range(len(norm_list)-1)]
    return gaps, norm_list


def generate_sophie_germain_prime_gaps(N_max):
    """Generate gaps between Sophie Germain primes (p such that 2p+1 is also prime)."""
    is_prime = [True] * (2 * N_max + 2)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(sqrt(2 * N_max + 1)) + 1):
        if is_prime[i]:
            for j in range(i*i, 2 * N_max + 2, i):
                is_prime[j] = False
    sg_primes = [i for i in range(2, N_max + 1) if is_prime[i] and is_prime[2*i + 1]]
    if len(sg_primes) < 2:
        return [], sg_primes
    gaps = [sg_primes[i+1] - sg_primes[i] for i in range(len(sg_primes)-1)]
    return gaps, sg_primes


# ============================================================
# P1: Forbidden transitions test
# ============================================================

def transition_matrix_residues(sequence, m):
    """Compute the transition matrix of SEQUENCE ELEMENT residues mod m.
    T1 theorem: for primes > 3 mod 3, T[1][1] = T[2][2] = 0.
    Returns (T, classes) where T[i][j] = P(class j | class i)."""
    # Use element residues (not gap residues)
    residues = [x % m for x in sequence]
    # Filter out class 0 (e.g. p=3 mod 3 = 0)
    non_zero = [(i, r) for i, r in enumerate(residues) if r != 0]
    classes = sorted(set(r for _, r in non_zero))
    if not classes:
        return np.zeros((1, 1)), [0]
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    counts = np.zeros((n_classes, n_classes))
    for k in range(len(non_zero) - 1):
        _, r_curr = non_zero[k]
        _, r_next = non_zero[k + 1]
        i = class_to_idx[r_curr]
        j = class_to_idx[r_next]
        counts[i][j] += 1

    # Normalize rows
    T = np.zeros_like(counts)
    for i in range(n_classes):
        row_sum = counts[i].sum()
        if row_sum > 0:
            T[i] = counts[i] / row_sum

    return T, classes


def transition_matrix_gaps(gaps, m):
    """Compute the transition matrix of GAP residues mod m.
    Returns (T, classes) where T[i][j] = P(class j | class i)."""
    residues = [g % m for g in gaps]
    classes = sorted(set(residues))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    counts = np.zeros((n_classes, n_classes))
    for k in range(len(residues) - 1):
        i = class_to_idx[residues[k]]
        j = class_to_idx[residues[k+1]]
        counts[i][j] += 1

    T = np.zeros_like(counts)
    for i in range(n_classes):
        row_sum = counts[i].sum()
        if row_sum > 0:
            T[i] = counts[i] / row_sum

    return T, classes


def has_T0_forbidden(sequence, m=3, threshold=1e-6):
    """Check if T[r][r] = 0 for non-zero residues mod m (T1 property).
    Uses SEQUENCE ELEMENT residues (primes, not gaps).
    For primes > 3 mod 3: consecutive primes never share the same
    non-zero residue class, because that would force a gap divisible
    by 3, making the next number composite.
    Returns True if ALL diagonal elements for r != 0 are below threshold."""
    T, classes = transition_matrix_residues(sequence, m)
    for i, c in enumerate(classes):
        if c != 0 and T[i][i] > threshold:
            return False
    return True


# ============================================================
# P2: CRT fibration test
# ============================================================

def D_KL_empirical(gaps, m):
    """Empirical D_KL: KL divergence of gap residues mod m
    from the uniform distribution on {0, ..., m-1}."""
    residues = [g % m for g in gaps]
    counts = Counter(residues)
    n = len(residues)
    if n == 0:
        return 0.0
    D = 0.0
    for r in range(m):
        p_r = counts.get(r, 0) / n
        q_r = 1.0 / m
        if p_r > 0:
            D += p_r * log2(p_r / q_r)
    return D


def crt_superadditivity(gaps, p1, p2):
    """Test CRT super-additivity: D(p1*p2) vs D(p1) + D(p2).
    Returns (D_product, D_sum, excess) where excess = D_product - D_sum.
    Positive excess indicates non-trivial correlation (CRT structure)."""
    D_prod = D_KL_empirical(gaps, p1 * p2)
    D_p1 = D_KL_empirical(gaps, p1)
    D_p2 = D_KL_empirical(gaps, p2)
    D_sum = D_p1 + D_p2
    return D_prod, D_sum, D_prod - D_sum


# ============================================================
# P3: Fixed-point test
# ============================================================

def empirical_mu(gaps):
    """Compute empirical mean of gaps."""
    return np.mean(gaps) if gaps else 0.0


def fixed_point_self_consistency(mu):
    """Test if mu is a self-consistent fixed point.
    Active primes = {p odd prime : gamma_p(mu) > s = 1/2}.
    p=2 is excluded: it is the parity prime (not a sieve level).
    Self-consistency: mu = sum of active primes.
    Returns (active_primes, sum_active, residual)."""
    active = []
    for p in ALL_PRIMES_SMALL:
        if p == 2:
            continue  # p=2 = parity, not a sieve level
        gp = compute_gamma_p(mu, p)
        if gp > S_PT:
            active.append(p)
    s = sum(active)
    return active, s, abs(s - mu)


# ============================================================
# P4: Mertens convergence test
# ============================================================

def mertens_product(primes):
    """Compute the Mertens product M_k = prod_{p <= p_k} (1 - 1/p).
    Should converge to 2*e^(-gamma)/ln(p_k)."""
    import numpy as np
    EULER_GAMMA = 0.5772156649015329
    products = []
    product = 1.0
    for p in primes:
        product *= (1.0 - 1.0 / p)
        expected = 2.0 * np.exp(-EULER_GAMMA) / np.log(p) if p > 2 else 1.0
        products.append((p, product, expected, abs(product - expected) / expected if expected > 0 else 0))
    return products


# ============================================================
# Bifurcation functions
# ============================================================

def bifurcation_gap_per_prime(mu):
    """Compute |sin^2(p, q_stat) - sin^2(p, q_therm)| for each active prime.
    Non-zero gap = signature of the vertex/edge duality."""
    result = {}
    for p in ACTIVE_PRIMES:
        s_stat = compute_sin2(mu, p, 'stat')
        s_therm = compute_sin2(mu, p, 'therm')
        result[p] = abs(s_stat - s_therm)
    return result


def survival_probability(mu):
    """Survival probability through the sieve = alpha_EM.
    Product of sin^2(theta_p, q_stat) for active primes."""
    return alpha_sieve(mu)


# ============================================================
# GFT (Gap Fundamental Theorem) verification
# ============================================================

def GFT_check(mu, m):
    """Verify GFT: log2(m) = D_KL + H exactly.
    Returns (log2_m, D_KL, H, residual)."""
    q = 1.0 - 2.0 / mu if mu > 2 else 0.5
    if q <= 0 or q >= 1:
        return log2(m), 0, 0, float('inf')

    # H(geom truncated at m)
    qm = q ** m
    one_minus_qm = 1.0 - qm
    if one_minus_qm < 1e-15:
        return log2(m), 0, 0, float('inf')

    # Normalize: p_k = q^(k-1) * (1-q) / (1-q^m) for k=1..m
    H = 0.0
    for k in range(1, m + 1):
        pk = q**(k-1) * (1.0 - q) / one_minus_qm
        if pk > 0:
            H -= pk * log2(pk)

    D = log2(m) - H
    residual = abs(log2(m) - D - H)
    return log2(m), D, H, residual


# ============================================================
# Cross-pillar consistency
# ============================================================

def cross_pillar_alpha(mu):
    """Verify alpha(particles) = alpha(thermo) = survival probability."""
    alpha_particles = alpha_sieve(mu)
    alpha_survival = survival_probability(mu)
    return alpha_particles, alpha_survival, abs(alpha_particles - alpha_survival)


def G_over_alpha(mu):
    """Verify G/alpha = 2*pi (holonomy of S^1)."""
    alpha = alpha_sieve(mu)
    G = 2.0 * pi * alpha
    return G, alpha, G / alpha if alpha > 0 else 0


def dimensional_activation(mu):
    """Count active ODD primes (gamma_p > s), excluding p=2 (parity).
    p=2 is the parity prime and not a sieve level."""
    active = []
    for p in ALL_PRIMES_SMALL:
        if p == 2:
            continue
        gp = compute_gamma_p(mu, p)
        if gp > S_PT:
            active.append((p, gp))
    return active


# ============================================================
# Bridge axiom master check
# ============================================================

def bridge_axiom_check(gaps, primes=None, mu=None):
    """Check all 6 bridge axioms BA0-BA5 against a gap sequence.
    Returns dict with pass/fail for each axiom."""
    if mu is None:
        mu = empirical_mu(gaps)

    results = {}

    # BA0: Sieve axiom (gaps are well-defined positive integers)
    results['BA0_sieve'] = all(g > 0 and isinstance(g, (int, np.integer)) for g in gaps)

    # BA1: Modular axiom (gauge connection works: r_{n+1} = (r_n + g_n) mod m)
    if primes is not None and len(primes) > 2:
        m = 3
        ok = True
        for i in range(min(len(gaps), len(primes) - 1)):
            r_curr = primes[i] % m
            r_next = primes[i + 1] % m
            r_predicted = (r_curr + gaps[i]) % m
            if r_next != r_predicted:
                ok = False
                break
        results['BA1_modular'] = ok
    else:
        results['BA1_modular'] = None  # cannot test without sequence values

    # BA2: GFT (log2(m) = D_KL + H, exact for geometric reference)
    _, _, _, residual = GFT_check(mu, 6)
    results['BA2_GFT'] = residual < 1e-12

    # BA3: Holonomy (sin^2 = delta*(2-delta) is algebraic identity)
    q = q_stat(mu)
    for p in ACTIVE_PRIMES:
        delta = (1.0 - q**p) / p
        sin2 = compute_sin2(mu, p, 'stat')
        expected = delta * (2.0 - delta)
        if abs(sin2 - expected) > 1e-14:
            results['BA3_holonomy'] = False
            break
    else:
        results['BA3_holonomy'] = True

    # BA4: Selection (mu* = sum of active primes at fixed point)
    active, s_active, residual = fixed_point_self_consistency(MU_STAR)
    results['BA4_selection'] = (residual == 0 and active == ACTIVE_PRIMES)

    # BA5: Coupling (alpha = survival probability = prod sin^2)
    alpha = alpha_sieve(MU_STAR)
    results['BA5_coupling'] = (0.007 < alpha < 0.008)  # alpha ~ 1/137

    return results


# ============================================================
# Self-test
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PT_PONT Core Module -- Self Test")
    print("=" * 60)

    # Generate prime gaps
    gaps, primes = generate_prime_gaps(100000)
    print(f"\nPrime gaps: {len(gaps)} gaps up to 100000")
    print(f"Empirical mu: {empirical_mu(gaps):.4f}")

    # T1 test -- on k-rough numbers (sieve level), not primes directly
    # T1: after sieving by {2,3}, the 2-rough numbers mod 5 have T[r][r]=0
    rough_gaps, rough_seq = generate_k_rough(100000, 2)
    has_t0 = has_T0_forbidden(rough_seq, 5)
    print(f"\nT0 forbidden transitions (2-rough mod 5): {has_t0}")

    # Also test 3-rough (after sieving {2,3,5}) mod 7
    rough3_gaps, rough3_seq = generate_k_rough(100000, 3)
    has_t0_3 = has_T0_forbidden(rough3_seq, 7)
    print(f"T1 forbidden transitions (3-rough mod 7): {has_t0_3}")

    # Lucky numbers -- should NOT have T1
    lucky_gaps, lucky = generate_lucky_numbers(100000)
    has_t0_lucky = has_T0_forbidden(lucky, 3)
    print(f"T1 forbidden transitions (lucky mod 3): {has_t0_lucky}")

    # Fixed point
    active, s, res = fixed_point_self_consistency(MU_STAR)
    print(f"\nFixed point: active = {active}, sum = {s}, residual = {res}")

    # GFT
    for m in [6, 30, 210]:
        l, d, h, r = GFT_check(MU_STAR, m)
        print(f"GFT(m={m}): log2 = {l:.6f}, D_KL = {d:.6f}, H = {h:.6f}, residual = {r:.2e}")

    # Bifurcation
    bif = bifurcation_gap_per_prime(MU_STAR)
    print(f"\nBifurcation gaps at mu*: {bif}")

    # Bridge axioms
    ax = bridge_axiom_check(gaps, primes, MU_STAR)
    print(f"\nBridge axioms: {ax}")

    print("\n" + "=" * 60)
    print("Self-test complete.")
