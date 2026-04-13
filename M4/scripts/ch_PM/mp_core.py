"""
mp_core.py -- Shared utilities for Predictive Mathematics (PM) chapter scripts
==============================================================================
Standalone implementation of the PM observation matrix, probe construction,
and rank computation. No dependency on PT_New_Math/.

Status: [THM/ID]  |  Chapter: ch_PM  |  Tests: 6 scripts
"""
import math
import numpy as np
from itertools import combinations
from typing import List, Tuple, Optional, Sequence


# ── Primes ──────────────────────────────────────────────────────────
PRIMES_100 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def primes_up_to(limit: int) -> List[int]:
    return [p for p in range(2, limit + 1) if is_prime(p)]


def odd_primes(depth: int) -> Tuple[int, ...]:
    """First `depth` odd primes: (3, 5, 7, 11, ...)."""
    all_odd = [p for p in PRIMES_100 if p >= 3]
    return tuple(all_odd[:depth])


def prime_count(n: int) -> int:
    """pi(n): number of primes <= n."""
    return sum(1 for p in range(2, n + 1) if is_prime(p))


def sum_first_primes(k: int) -> int:
    """Sum of the first k primes: P(k) = 2 + 3 + 5 + ... + p_k."""
    ps = primes_up_to(300)
    return sum(ps[:k])


# ── Sieve survivors ────────────────────────────────────────────────
def survivors(active_primes: Sequence[int], limit: int = 0) -> List[int]:
    """Integers coprime to all primes in active_primes, up to product."""
    if limit == 0:
        limit = 1
        for p in active_primes:
            limit *= p
    result = []
    for n in range(1, limit + 1):
        if all(n % p != 0 for p in active_primes):
            result.append(n)
    return result


# ── Probes ──────────────────────────────────────────────────────────
# Probe = tuple: ("one",), ("chi3",), or ("eta", q, r)

Probe = Tuple


def probe_label(spec: Probe) -> str:
    if spec[0] == "one":
        return "1"
    if spec[0] == "chi3":
        return "chi3"
    return f"eta_{spec[1]}_{spec[2]}"


def probe_modulus(spec: Probe) -> Optional[int]:
    if spec[0] == "one":
        return None
    if spec[0] == "chi3":
        return 3
    return int(spec[1])


def chi3(n: int) -> float:
    r = n % 3
    if r == 1:
        return 1.0
    if r == 2:
        return -1.0
    return 0.0


def probe_value(spec: Probe, n: int) -> float:
    if spec[0] == "one":
        return 1.0
    if spec[0] == "chi3":
        return chi3(n)
    q, a = int(spec[1]), int(spec[2])
    return (1.0 if n % q == a else 0.0) - 1.0 / q


# ── Shell geometry ──────────────────────────────────────────────────
def shell_primes_before(shell: int) -> List[int]:
    """All odd primes strictly less than shell."""
    return [p for p in PRIMES_100 if 3 <= p < shell]


def max_depth_for_shell(shell: int) -> int:
    """Future depth = base_depth + 1 = (# odd primes < shell) + 1."""
    return len(shell_primes_before(shell)) + 1


def build_probe_families(shell: int):
    """
    Build base probes (all shells < target) and target shell probes.

    Returns (base_probes, target_probes).
    """
    # Base family: constant + chi3 + eta probes for p=5,7
    base_family = [("one",), ("chi3",)]
    eta5 = [("eta", 5, a) for a in (3, 4)]
    eta7 = [("eta", 7, a) for a in (5, 6)]

    base_shells = list(base_family) + eta5 + eta7

    # Add eta packets for primes 11, 13, ..., up to shell-1
    for p in shell_primes_before(shell):
        if p <= 7:
            continue
        packet = [("eta", p, r) for r in range(p - 4, p)]
        base_shells.extend(packet)

    base_shells = tuple(base_shells)

    # Target shell probes: 4 consecutive residues near shell
    eta_target = tuple(("eta", shell, r) for r in range(shell - 4, shell))

    return base_shells, eta_target


# ── Count tables (vectorized) ──────────────────────────────────────
def single_count_table_vec(active, cond_mod, cond_res, target_mod):
    """Count residues of survivors mod target_mod, conditioned on cond_mod=cond_res."""
    limit = 1
    for p in active:
        limit *= p
    surv = []
    for n in range(1, limit + 1):
        if all(n % p != 0 for p in active):
            if n % cond_mod == cond_res:
                surv.append(n)
    domain = list(range(target_mod))
    counts = np.zeros(target_mod, dtype=int)
    for n in surv:
        counts[n % target_mod] += 1
    return len(surv), domain, counts


def pair_count_table_vec(active, cond_mod, cond_res, mod1, mod2):
    """Joint count of (n mod mod1, n mod mod2) for conditioned survivors."""
    limit = 1
    for p in active:
        limit *= p
    surv = []
    for n in range(1, limit + 1):
        if all(n % p != 0 for p in active):
            if n % cond_mod == cond_res:
                surv.append(n)
    d1 = list(range(mod1))
    d2 = list(range(mod2))
    counts = np.zeros((mod1, mod2), dtype=int)
    for n in surv:
        counts[n % mod1, n % mod2] += 1
    return len(surv), d1, d2, counts


# ── Conditioned tables (object) ────────────────────────────────────
class ConditionedTables:
    """Pre-compute all single and joint distribution tables for one conditioning."""

    def __init__(self, active, cond_mod, cond_res, probe_moduli):
        self.active = active
        self.active_set = set(active)
        self.conditioning_modulus = cond_mod
        self.conditioning_residue = cond_res

        # Active primes other than cond_mod
        active_other = [p for p in active if p != cond_mod]
        # Inactive moduli from probes
        inactive_moduli = sorted(set(m for m in probe_moduli
                                      if m not in self.active_set and m != cond_mod))

        self.single = {}
        self.pair = {}

        # Single count tables for inactive moduli
        for modulus in inactive_moduli:
            denom, domain, counts = single_count_table_vec(
                active, cond_mod, cond_res, modulus)
            table = counts.astype(float) / float(denom) if denom else counts.astype(float)
            self.single[modulus] = (domain, table)

        # Joint tables
        for left, right in combinations(inactive_moduli, 2):
            denom, ld, rd, counts = pair_count_table_vec(
                active, cond_mod, cond_res, left, right)
            table = counts.astype(float) / float(denom) if denom else counts.astype(float)
            self.pair[(left, right)] = (ld, rd, table)

        for left in active_other:
            for right in inactive_moduli:
                denom, ld, rd, counts = pair_count_table_vec(
                    active, cond_mod, cond_res, left, right)
                table = counts.astype(float) / float(denom) if denom else counts.astype(float)
                self.pair[(left, right)] = (ld, rd, table)

    def prob(self, modulus, residue):
        if modulus in self.active_set:
            if modulus == self.conditioning_modulus:
                return 1.0 if residue == self.conditioning_residue else 0.0
            return 1.0 / (modulus - 1)
        domain, table = self.single[modulus]
        idx = domain.index(residue)
        return float(table[idx])

    def prob_joint(self, q1, a1, q2, a2):
        if q1 == q2:
            return self.prob(q1, a1) if a1 == a2 else 0.0
        if q1 == self.conditioning_modulus:
            return self.prob(q2, a2) if a1 == self.conditioning_residue else 0.0
        if q2 == self.conditioning_modulus:
            return self.prob(q1, a1) if a2 == self.conditioning_residue else 0.0

        q1_active = q1 in self.active_set
        q2_active = q2 in self.active_set
        if q1_active and q2_active:
            return 1.0 / ((q1 - 1) * (q2 - 1))

        key = (q1, q2)
        if key not in self.pair:
            key = (q2, q1)
            q1, a1, q2, a2 = q2, a2, q1, a1
        ld, rd, table = self.pair[key]
        li = ld.index(a1)
        ri = rd.index(a2)
        return float(table[li, ri])


# ── Probe expectations ─────────────────────────────────────────────
def probe_expectation(spec, cm, cr, prob_fn):
    if spec[0] == "one":
        return 1.0
    if spec[0] == "chi3":
        return prob_fn(3, 1) - prob_fn(3, 2)
    m = int(spec[1])
    t = int(spec[2])
    return prob_fn(m, t) - 1.0 / m


def probe_product_expectation(sl, sr, cm, cr, prob_fn, joint_fn):
    if sl[0] == "one":
        return probe_expectation(sr, cm, cr, prob_fn)
    if sr[0] == "one":
        return probe_expectation(sl, cm, cr, prob_fn)
    if sl[0] == "chi3" and sr[0] == "chi3":
        return 1.0
    if sl[0] == "chi3" and sr[0] == "eta":
        m, t = int(sr[1]), int(sr[2])
        return (joint_fn(3, 1, m, t) - joint_fn(3, 2, m, t)
                - (1.0 / m) * (prob_fn(3, 1) - prob_fn(3, 2)))
    if sl[0] == "eta" and sr[0] == "chi3":
        m, t = int(sl[1]), int(sl[2])
        return (joint_fn(3, 1, m, t) - joint_fn(3, 2, m, t)
                - (1.0 / m) * (prob_fn(3, 1) - prob_fn(3, 2)))
    q, a = int(sl[1]), int(sl[2])
    r, b = int(sr[1]), int(sr[2])
    if q == r:
        if a == b:
            return (1.0 - 2.0 / q) * prob_fn(q, a) + 1.0 / (q * q)
        return -(1.0 / q) * prob_fn(q, a) - (1.0 / q) * prob_fn(q, b) + 1.0 / (q * q)
    return (joint_fn(q, a, r, b)
            - (1.0 / r) * prob_fn(q, a)
            - (1.0 / q) * prob_fn(r, b)
            + 1.0 / (q * r))


# ── Observation matrix (exact, parallelizable) ─────────────────────
def _compute_block(args):
    """Worker: compute observation matrix block for one conditioning modulus."""
    active, conditioning_modulus, probes, probe_moduli_list = args
    n_probes = len(probes)
    rows = [np.zeros(n_probes * n_probes, dtype=float)]

    for cr in range(1, conditioning_modulus):
        tables = ConditionedTables(active, conditioning_modulus, cr, probe_moduli_list)
        gram = np.zeros((n_probes, n_probes), dtype=float)
        for i, sl in enumerate(probes):
            for j in range(i, n_probes):
                sr = probes[j]
                v = probe_product_expectation(sl, sr, conditioning_modulus, cr,
                                              tables.prob, tables.prob_joint)
                gram[i, j] = v
                gram[j, i] = v
            rows.append(gram.reshape(-1))

    block = np.vstack(rows)
    block -= np.mean(block, axis=0, keepdims=True)
    return block


def exact_observation_matrix(probes, depth):
    """Compute the exact observation matrix at given sieve depth."""
    active = odd_primes(depth)
    probe_moduli_list = [probe_modulus(s) for s in probes
                         if probe_modulus(s) is not None]
    blocks = []
    for cm in active:
        block = _compute_block((active, cm, probes, probe_moduli_list))
        blocks.append(block)
    return np.vstack(blocks)


# ── Historical matrix and rank computation ─────────────────────────
def build_historical_matrix(probes, max_d):
    """Build observation matrices for depths 1..max_d and return stacked history."""
    mats = []
    for d in range(1, max_d + 1):
        mat = exact_observation_matrix(probes, d)
        mats.append(mat)
    return mats


def cumulative_ranks(mats):
    """Compute cumulative ranks from list of observation matrices."""
    ranks = []
    hist = None
    for mat in mats:
        hist = mat if hist is None else np.vstack([hist, mat])
        r = int(np.linalg.matrix_rank(hist, tol=1e-10))
        ranks.append(r)
    return ranks


def layer_dimensions(ranks):
    """Layer dimension = rank increment at each depth."""
    dims = [ranks[0]]
    for i in range(1, len(ranks)):
        dims.append(ranks[i] - ranks[i - 1])
    return dims


# ── Shell data (13 verified shells) ────────────────────────────────
SHELL_DATA = {
    # shell: AT_observed, |B| = base_depth - 2 = (# odd primes < shell) - 2
    # First 6 shells fully computed; shells >= 41 from PM-PT conjecture
    11: {"AT": 1, "B": 1},
    13: {"AT": 1, "B": 2},
    17: {"AT": 2, "B": 3},
    19: {"AT": 2, "B": 4},
    23: {"AT": 2, "B": 5},
    29: {"AT": 3, "B": 6},
    31: {"AT": 3, "B": 7},
    37: {"AT": 3, "B": 8},
    41: {"AT": 1, "B": 9},
    43: {"AT": 1, "B": 10},
    47: {"AT": 1, "B": 11},
    53: {"AT": 1, "B": 12},
    59: {"AT": 1, "B": 13},
}


def at_formula(B: int) -> int:
    """
    Activation threshold formula (3 regimes):
    |B| = base_depth - 2 = (# odd primes < shell) - 2.
    - Ghost (|B| < 3): AT = 1
    - Construction (3 <= |B| <= 8): AT = ceil((|B|-2)/3) + 1
    - Asymptotic (|B| >= 9, i.e. p >= 41): AT = 1 (CRT perturbative reset)
    """
    if B < 3:
        return 1
    if B <= 8:
        return math.ceil((B - 2) / 3) + 1
    return 1


def dim_formula(d: int) -> int:
    """Layer dimension formula: dim(d) = P(d+1) - (2d+1)
    where P(k) = sum of first k primes."""
    return sum_first_primes(d + 1) - (2 * d + 1)


# ── Codes instantiation utilities ──────────────────────────────────
def enumerate_F2n(n: int) -> np.ndarray:
    """All vectors of F_2^n."""
    from itertools import product as iproduct
    return np.array(list(iproduct([0, 1], repeat=n)), dtype=np.int8)


def survivors_linear(all_words: np.ndarray, H_d: np.ndarray) -> np.ndarray:
    """Words satisfying H_d @ x = 0 mod 2."""
    if H_d.shape[0] == 0:
        return all_words
    syndrome = (all_words.astype(int) @ H_d.T) % 2
    mask = np.all(syndrome == 0, axis=1)
    return all_words[mask]


def code_observation_matrix(survivors_arr: np.ndarray, n: int) -> np.ndarray:
    """
    Observation matrix for error-correcting codes.
    Probes: psi_j(x) = x_j - mu_j (centered coordinates).
    Conditioning: x_i = v for i=0..n-1, v in {0,1}.
    """
    N = len(survivors_arr)
    if N < 2:
        return np.zeros((2 * n, n * n))
    surv = survivors_arr.astype(np.float64)
    mu = surv.mean(axis=0)
    centered = surv - mu
    rows = []
    for i in range(n):
        for v in [0, 1]:
            mask = survivors_arr[:, i] == v
            cnt = mask.sum()
            if cnt == 0:
                rows.append(np.zeros(n * n))
            else:
                sub = centered[mask]
                gram = (sub.T @ sub) / cnt
                rows.append(gram.flatten())
    return np.array(rows)
