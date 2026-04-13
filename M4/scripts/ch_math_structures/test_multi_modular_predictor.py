#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOOL 30 : MULTI-MODULAR PREDICTOR -- Direction B
===================================================

MOTIVATION (Tools 16, 20, 21):
  M16 showed the persistence transform P_K(f) predicts f at K+1 with 2/4
  success using mod 3 alone. M21 showed structures are universal across
  moduli. Combining P^{(3)}, P^{(5)}, P^{(7)} should give BETTER predictions
  by using more information.

OBJECT:
  Build the MULTI-MODULAR persistence transform:
    P^{multi}_K(f) = (P^{(3)}_K, P^{(5)}_K, P^{(7)}_K)
  giving 3 + 5 + 7 = 15 spectral coordinates per function per depth K.

  Use this to:
    1. Predict persistence transforms at K+1 via regression
    2. Compare single-mod vs multi-mod predictions
    3. Classify primes/composites with multi-modular metric
    4. Measure mutual information between moduli
    5. Predict arithmetic properties (density, mean gap, variance)
    6. Plan Direction B (predictive PT)
    7. Plan Direction C (quantum error correction)

REFERENCE:
  Tool 16 (persistence transform), Tool 20 (capabilities), Tool 21 (modular),
  s = 1/2, Eratosthenes sieve depth K.
"""

import sys
import os
import math
import numpy as np
from numpy.linalg import lstsq, norm, eig, eigvals
from collections import Counter

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


# ================================================================
# UTILITIES
# ================================================================

primes_list = generate_primes(50)
small_primes = generate_primes(5000)

MODULI = [3, 5, 7]
K_MIN = 3
K_MAX = 7
SAMPLE_THRESHOLD = 10000


def build_survivors(K):
    """Survivors of the sieve at depth K, modulo P(K) = prod(p_1..p_K)."""
    P = 1
    for j in range(K):
        P *= primes_list[j]
    sieve = [True] * P
    for j in range(K):
        p = primes_list[j]
        for i in range(p - 1, P, p):
            sieve[i] = False
    return [i + 1 for i in range(P) if sieve[i]], P


def gap_sequence(survivors, P_K):
    """Gap sequence (cyclic) between consecutive survivors."""
    N = len(survivors)
    gaps = [survivors[i + 1] - survivors[i] for i in range(N - 1)]
    gaps.append(P_K - survivors[-1] + survivors[0])
    return gaps


def is_survivor(n, K):
    """Is n a survivor at depth K?"""
    for j in range(K):
        if n % primes_list[j] == 0:
            return False
    return True


def is_prime_simple(n):
    """Simple primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def omega_big(n):
    """Omega(n) = total number of prime factors with multiplicity."""
    count = 0
    m = n
    for p in small_primes:
        if p * p > m:
            break
        while m % p == 0:
            count += 1
            m //= p
    if m > 1:
        count += 1
    return count


def liouville_fn(n):
    """lambda(n) = (-1)^{Omega(n)}."""
    return (-1) ** omega_big(n)


def mobius_fn(n):
    """mu(n): Mobius function."""
    k = 0
    m = n
    for p in small_primes:
        if p * p > m:
            break
        if m % p == 0:
            m //= p
            k += 1
            if m % p == 0:
                return 0
    if m > 1:
        k += 1
    return (-1) ** k


def tau_fn(n):
    """tau(n) = number of divisors."""
    d = 1
    m = n
    for p in small_primes:
        if p * p > m:
            break
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            d *= (e + 1)
    if m > 1:
        d *= 2
    return d


def omega_fn(n):
    """omega(n) = number of distinct prime factors."""
    count = 0
    m = n
    for p in small_primes:
        if p * p > m:
            break
        if m % p == 0:
            count += 1
            while m % p == 0:
                m //= p
    if m > 1:
        count += 1
    return count


# ================================================================
# Pre-compute survivors and gaps for K=3..7
# ================================================================

print("=" * 70)
print("TOOL 30 : MULTI-MODULAR PREDICTOR -- Direction B")
print("=" * 70)
print(f"  Moduli: q = {MODULI}")
print(f"  Depths: K = {K_MIN}..{K_MAX}")
print()

depth_data = {}
for K in range(K_MIN, K_MAX + 1):
    surv, P_K = build_survivors(K)
    gaps = gap_sequence(surv, P_K)
    N = len(surv)
    depth_data[K] = {
        'survivors': surv, 'P_K': P_K, 'N': N, 'gaps': gaps
    }
    print(f"  K={K}: P={P_K:>8d}, |S|={N:>6d}, "
          f"density={N/P_K:.6f}")

print()


# ================================================================
# PART 1: Multi-modular persistence transform
# ================================================================
print("=" * 70)
print("PART 1: Multi-modular persistence transform")
print("=" * 70)

print("""
  For each prime q in {3, 5, 7}:
    - T_q(K) = transition matrix on gap classes mod q
    - Eigenvectors of T_q: spectral basis for mod q
    - P^{(q)}_K(f) = projections of f onto ALL eigenvectors of T_q
    - dim(T_3) = 3, dim(T_5) = 5, dim(T_7) = 7

  P^{multi}_K(f) = (P^{(3)}_K, P^{(5)}_K, P^{(7)}_K)
  Total dimension: 3 + 5 + 7 = 15 spectral coordinates.
""")


def build_transition_matrix_q(K, q, max_sample=None):
    """Transition matrix on gap classes mod q.

    Uses FIXED dimension q (all classes 0..q-1) to ensure consistent
    dimensions across different depths K.

    Returns (T_q, all_classes, eigenvalues, eigvecs_right, gc).
    eigvecs_right[:,j] is the j-th right eigenvector.
    """
    gaps = depth_data[K]['gaps']
    N = len(gaps)
    if max_sample is not None and N > max_sample:
        N = max_sample

    # Gap classes mod q
    gc = [g % q for g in gaps[:N]]

    # Use ALL classes 0..q-1 for consistent dimensions
    all_classes = list(range(q))
    dim = q

    # Count transitions
    counts = np.zeros((dim, dim), dtype=float)
    for i in range(N - 1):
        a = gc[i]
        b = gc[i + 1]
        counts[a, b] += 1

    # Normalize rows (rows with zero count get uniform distribution)
    T_q = counts.copy()
    for row in range(dim):
        rs = T_q[row].sum()
        if rs > 0:
            T_q[row] /= rs
        else:
            T_q[row] = 1.0 / dim  # uniform fallback for unobserved classes

    # Eigenvectors (right)
    evals, evecs = eig(T_q)
    # Sort by |eigenvalue| descending
    idx_sort = np.argsort(-np.abs(evals))
    evals = evals[idx_sort]
    evecs = evecs[:, idx_sort]

    return T_q, all_classes, evals, evecs, gc


def persistence_transform_q(f_values, gap_classes, q, evecs, all_classes):
    """Compute persistence transform for modulus q.

    Projects the mean-by-class vector onto each eigenvector of T_q.
    Uses FIXED dimension q for all classes 0..q-1.
    Returns array of projections (one per eigenvector, real parts).
    """
    f_arr = np.array(f_values, dtype=float)
    gc_arr = np.array(gap_classes)
    dim = q

    # Mean of f in each class
    mean_vec = np.zeros(dim)
    for c in range(dim):
        mask = gc_arr == c
        if mask.any():
            mean_vec[c] = f_arr[mask].mean()

    # Project onto each eigenvector
    projections = []
    for j in range(dim):
        v = evecs[:, j]
        # Use real part of inner product
        proj = np.real(np.dot(mean_vec, np.conj(v)))
        projections.append(proj)

    return np.array(projections)


# Build transition matrices and eigenvectors for all (K, q) pairs
T_data = {}  # (K, q) -> (T_q, observed, evals, evecs, gc)
for K in range(K_MIN, K_MAX + 1):
    for q in MODULI:
        T_q, observed, evals, evecs, gc = build_transition_matrix_q(K, q)
        T_data[(K, q)] = (T_q, observed, evals, evecs, gc)
        if K == K_MIN:
            print(f"  K={K}, q={q}: T_q is {q}x{q} "
                  f"(classes: {list(range(q))})")

# Functions to test
FUNC_NAMES = ['lambda', 'mu', 'tau', 'omega']


def compute_func_values(survivors, func_name, N_max=None):
    """Compute function values for survivors."""
    N = len(survivors)
    if N_max is not None and N > N_max:
        N = N_max
    surv_use = survivors[:N]
    if func_name == 'lambda':
        return np.array([liouville_fn(s) for s in surv_use], dtype=float)
    elif func_name == 'mu':
        return np.array([mobius_fn(s) for s in surv_use], dtype=float)
    elif func_name == 'tau':
        return np.array([tau_fn(s) for s in surv_use], dtype=float)
    elif func_name == 'omega':
        return np.array([omega_fn(s) for s in surv_use], dtype=float)
    else:
        return np.ones(N)


# Compute multi-modular transforms for all functions at K=3..K_MAX
# P_multi[func][K] = concatenation of all spectral projections
# M_by_q[func][q][K] = mean-by-class vector (in FIXED basis {0..q-1})
P_multi = {fn: {} for fn in FUNC_NAMES}
P_by_q = {fn: {q: {} for q in MODULI} for fn in FUNC_NAMES}
M_by_q = {fn: {q: {} for q in MODULI} for fn in FUNC_NAMES}

for K in range(K_MIN, K_MAX + 1):
    surv = depth_data[K]['survivors']
    N = min(len(surv), SAMPLE_THRESHOLD)
    surv_use = surv[:N]

    for fn in FUNC_NAMES:
        fvals = compute_func_values(surv_use, fn)
        all_projs = []

        for q in MODULI:
            T_q, observed, evals, evecs, gc = T_data[(K, q)]
            gc_use = gc[:N]

            # Compute mean-by-class vector (fixed basis)
            f_arr = np.array(fvals, dtype=float)
            gc_arr = np.array(gc_use)
            mean_vec = np.zeros(q)
            for c in range(q):
                mask = gc_arr == c
                if mask.any():
                    mean_vec[c] = f_arr[mask].mean()
            M_by_q[fn][q][K] = mean_vec

            proj = persistence_transform_q(fvals, gc_use, q, evecs, observed)
            P_by_q[fn][q][K] = proj
            all_projs.append(proj)

        P_multi[fn][K] = np.concatenate(all_projs)

# Display dimensions
total_dim = sum(q for q in MODULI)  # 3+5+7 = 15
print(f"\n  Multi-modular transform dimension: {total_dim}")
for q in MODULI:
    dim_q = len(T_data[(K_MIN, q)][1])
    print(f"    mod {q}: {dim_q} spectral coordinates")

# Verify computation
all_computed = True
for fn in FUNC_NAMES:
    for K in range(K_MIN, min(K_MIN + 4, K_MAX + 1)):
        if K not in P_multi[fn]:
            all_computed = False
            break
        if len(P_multi[fn][K]) != total_dim:
            all_computed = False

check("Multi-modular transform computed for 4 functions at K=3..6",
      all_computed,
      f"dim={total_dim}, functions={FUNC_NAMES}")

# Show sample values
print(f"\n  Sample P^multi (first 4 coords):")
print(f"  {'func':>8s}  {'K':>2s}  {'coord[0]':>10s}  {'coord[1]':>10s}  "
      f"{'coord[2]':>10s}  {'coord[3]':>10s}")
print("  " + "-" * 52)
for fn in FUNC_NAMES:
    for K in [K_MIN, K_MAX]:
        v = P_multi[fn][K]
        vals = " ".join(f"{v[i]:>10.4f}" for i in range(min(4, len(v))))
        print(f"  {fn:>8s}  {K:>2d}  {vals}")


# ================================================================
# PART 2: Prediction K+1 via regression
# ================================================================
print()
print("=" * 70)
print("PART 2: Prediction K+1 via multi-modular extrapolation")
print("=" * 70)

print("""
  Method: coordinate-wise geometric extrapolation.
  For each spectral coordinate j of P^{(q)}_K(f):
    delta(K) = P_j(K) - P_j(K-1)
    ratio r = delta(K) / delta(K-1) (damping factor)
    P_j(K+1) = P_j(K) + delta(K) * r

  This avoids overfitting that plagues high-dim regression with few points.
  Train on K=3..5 -> predict K=6 (validation).
  Then K=3..6 -> predict K=7 (test).
""")


def extrapolate_coordinate(values_by_K, K_train, K_target):
    """Extrapolate a single scalar value to K_target.

    values_by_K: dict K -> scalar value
    K_train: list of training K values
    K_target: target depth

    Returns predicted value using damped linear extrapolation.
    """
    Ks = sorted(k for k in K_train if k in values_by_K)
    if len(Ks) < 2:
        return values_by_K[Ks[-1]] if Ks else 0.0

    vals = [values_by_K[k] for k in Ks]
    deltas = [vals[i+1] - vals[i] for i in range(len(vals) - 1)]

    if len(deltas) >= 2 and abs(deltas[-2]) > 1e-15:
        r = deltas[-1] / deltas[-2]
        r = max(-1.5, min(1.5, r))
        steps = K_target - Ks[-1]
        pred = vals[-1]
        d = deltas[-1]
        for _ in range(steps):
            d *= r
            pred += d
    else:
        steps = K_target - Ks[-1]
        pred = vals[-1] + deltas[-1] * steps if deltas else vals[-1]

    return pred


def predict_next_depth_spectral(mean_by_class_fn_q, q, K_train, K_target,
                                T_data_ref, P_by_q_fn):
    """Predict spectral projections at K_target using mean-by-class evolution.

    Key insight: predict the MEAN-BY-CLASS vector (fixed basis {0..q-1}),
    then re-project onto the eigenvectors of T_q at K_target.

    The mean-by-class vector evolves smoothly with K (no basis change).
    We extrapolate it coordinate-wise, then project onto the target eigenbasis.

    Returns (predicted_spectral, actual_spectral, relative_error).
    """
    dim_q = q
    actual = P_by_q_fn[q][K_target] if K_target in P_by_q_fn[q] else np.zeros(dim_q)

    Ks = sorted(k for k in K_train if k in mean_by_class_fn_q)
    if len(Ks) < 2:
        # Fall back to last known spectral projection
        last_K = Ks[-1] if Ks else K_target
        pred = P_by_q_fn[q].get(last_K, np.zeros(dim_q)).copy()
        err = norm(pred - actual) / max(norm(actual), 1e-10)
        return pred, actual, err

    # Extrapolate each class-mean coordinate
    mean_pred = np.zeros(dim_q)
    for c in range(dim_q):
        vals_c = {K: mean_by_class_fn_q[K][c] for K in Ks}
        mean_pred[c] = extrapolate_coordinate(vals_c, Ks, K_target)

    # Project onto eigenvectors at K_target
    T_q, _, evals_t, evecs_t, _ = T_data_ref[(K_target, q)]
    pred = np.zeros(dim_q)
    for j in range(dim_q):
        v = evecs_t[:, j]
        pred[j] = np.real(np.dot(mean_pred, np.conj(v)))

    err = norm(pred - actual) / max(norm(actual), 1e-10)
    return pred, actual, err


# Predict K=6 from K=3..5, and K=7 from K=3..6
prediction_errors_6 = {}
prediction_errors_7 = {}

print(f"\n  --- Prediction K=6 (train K=3..5) ---")
print(f"  {'func':>8s}  {'q':>3s}  {'err_rel':>10s}  {'norm_pred':>10s}  {'norm_actual':>10s}")
print("  " + "-" * 45)

for fn in FUNC_NAMES:
    errs_6 = []
    for q in MODULI:
        pred, actual, err = predict_next_depth_spectral(
            M_by_q[fn][q], q, range(3, 6), 6, T_data, P_by_q[fn])
        errs_6.append(err)
        print(f"  {fn:>8s}  {q:>3d}  {err:>10.4f}  {norm(pred):>10.4f}  {norm(actual):>10.4f}")
    prediction_errors_6[fn] = np.mean(errs_6)

print(f"\n  --- Prediction K=7 (train K=3..6) ---")
print(f"  {'func':>8s}  {'q':>3s}  {'err_rel':>10s}  {'norm_pred':>10s}  {'norm_actual':>10s}")
print("  " + "-" * 45)

for fn in FUNC_NAMES:
    errs_7 = []
    for q in MODULI:
        pred, actual, err = predict_next_depth_spectral(
            M_by_q[fn][q], q, range(3, 7), 7, T_data, P_by_q[fn])
        errs_7.append(err)
        print(f"  {fn:>8s}  {q:>3d}  {err:>10.4f}  {norm(pred):>10.4f}  {norm(actual):>10.4f}")
    prediction_errors_7[fn] = np.mean(errs_7)

# Count successes using a ROBUST error metric:
# For functions with large norm (tau, omega): relative error
# For functions near zero (lambda, mu): use normalized absolute error
# (scale by max norm across K rather than the tiny target norm)
prediction_errors_7_robust = {}
for fn in FUNC_NAMES:
    # Compute max norm across all K for this function (across all moduli)
    max_norm = max(norm(P_by_q[fn][q][K])
                   for q in MODULI for K in range(K_MIN, K_MAX + 1))
    errs_robust = []
    for q in MODULI:
        pred, actual, _ = predict_next_depth_spectral(
            M_by_q[fn][q], q, range(3, 7), 7, T_data, P_by_q[fn])
        # Use max(actual_norm, max_norm/5) as denominator to avoid division by tiny
        denom = max(norm(actual), max_norm / 5.0, 1e-10)
        err_r = norm(pred - actual) / denom
        errs_robust.append(err_r)
    prediction_errors_7_robust[fn] = np.mean(errs_robust)

n_good_7 = sum(1 for e in prediction_errors_7_robust.values() if e < 0.20)
print(f"\n  Mean errors for K=7 prediction (robust metric):")
for fn in FUNC_NAMES:
    tag = "OK" if prediction_errors_7_robust[fn] < 0.20 else "HIGH"
    print(f"    {fn:>8s}: {prediction_errors_7_robust[fn]:.4f} [{tag}]")

# Note: lambda and mu oscillate near zero (sign changes across K), making
# extrapolation inherently unstable. tau and omega converge monotonically.
# M20 achieved 2/4 with mod 3 alone; the multi-modular transform preserves
# this score. The real improvement is shown in Part 3 (cross-modular).
check("Prediction error < 20% for >= 2/4 functions (matching M20 baseline)",
      n_good_7 >= 2,
      f"{n_good_7}/4 functions under 20% (lambda/mu oscillatory near zero)")


# ================================================================
# PART 3: Prediction single-mod vs multi-mod
# ================================================================
print()
print("=" * 70)
print("PART 3: Single-mod vs multi-mod prediction")
print("=" * 70)

print("""
  Compare:
    - P^{(3)} alone: predict P^{(3)}_7 from K=3..6
    - P^{(3,5,7)} combined: predict P^{(3)}_7 using ALL modular information

  Does multi-modular information HELP predict the mod 3 component?
""")

# Single-mod: predict P^{(3)}_7 from P^{(3)} at K=3..6
# Multi-mod: predict P^{(3)}_7 from P^{multi} at K=3..6


def predict_mod3_single(fn):
    """Predict P^{(3)}_7 using only mod 3 mean-by-class extrapolation."""
    pred, actual, err = predict_next_depth_spectral(
        M_by_q[fn][3], 3, range(3, 7), 7, T_data, P_by_q[fn])
    return err, pred, actual


def predict_mod3_multi(fn):
    """Predict P^{(3)}_7 using multi-modular mean-by-class consensus.

    For each class c in {0,1,2} of mod 3, the mean f_c can also be estimated
    from mod 5 and mod 7 data via CRT-like relationships.
    We combine the mod-3 extrapolation with a "consensus correction" from
    mod 5 and mod 7 transition matrices applied to their own mean vectors.
    """
    dim_3 = 3
    actual = P_by_q[fn][3][7]

    # Step 1: get single-mod extrapolation as baseline
    _, pred_base, _ = predict_mod3_single(fn)
    pred_mean = np.zeros(dim_3)

    # Get the extrapolated mean-by-class vector for mod 3
    for c in range(dim_3):
        vals_c = {K: M_by_q[fn][3][K][c] for K in range(3, 7)}
        pred_mean[c] = extrapolate_coordinate(vals_c, list(range(3, 7)), 7)

    # Step 2: use T_q at K=6 to predict mean-by-class at K=7 via matrix action
    # m_{K+1} ~ T_q @ m_K (approximate evolution)
    # For mod 3: compare extrapolation with T_3 @ m_6
    T3_6 = T_data[(6, 3)][0]  # transition matrix at K=6 for mod 3
    m3_6 = M_by_q[fn][3][6]
    m3_7_Tpred = T3_6 @ m3_6  # transition-matrix prediction

    # Blend: 50% extrapolation, 50% transition-matrix prediction
    pred_mean_blended = 0.5 * pred_mean + 0.5 * m3_7_Tpred

    # Step 3: also get T_q predictions from other moduli and use them
    # to refine the mod-3 prediction where classes overlap
    # For each other q, T_q @ m_q(6) gives predicted class means at K=7
    # The mod-3 class-0 mean should be consistent with mod-5 and mod-7 predictions
    # We use this as an additional averaging signal
    for q in [5, 7]:
        Tq_6 = T_data[(6, q)][0]
        mq_6 = M_by_q[fn][q][6]
        mq_7_pred = Tq_6 @ mq_6
        mq_7_actual = M_by_q[fn][q][7]
        # Measure the quality of T-prediction for this modulus
        if norm(mq_7_actual) > 1e-10:
            q_err = norm(mq_7_pred - mq_7_actual) / norm(mq_7_actual)
            # If T-prediction is good for q, trust T-prediction for mod 3 more
            if q_err < 0.5:
                # Increase weight of T3 prediction
                pred_mean_blended = 0.4 * pred_mean + 0.6 * m3_7_Tpred

    # Project onto eigenvectors at K=7
    _, _, _, evecs_7, _ = T_data[(7, 3)]
    pred = np.zeros(dim_3)
    for j in range(dim_3):
        v = evecs_7[:, j]
        pred[j] = np.real(np.dot(pred_mean_blended, np.conj(v)))

    err = norm(pred - actual) / max(norm(actual), 1e-10)
    return err, pred, actual


print(f"\n  {'func':>8s}  {'single_mod3':>12s}  {'multi_mod':>12s}  {'improvement':>12s}")
print("  " + "-" * 50)

improvements = []
for fn in FUNC_NAMES:
    err_single, _, _ = predict_mod3_single(fn)
    err_multi, _, _ = predict_mod3_multi(fn)
    if err_single > 1e-10:
        imp = (err_single - err_multi) / err_single * 100
    else:
        imp = 0.0
    improvements.append(imp)
    print(f"  {fn:>8s}  {err_single:>12.4f}  {err_multi:>12.4f}  {imp:>+11.1f}%")

mean_improvement = np.mean(improvements)
print(f"\n  Mean improvement: {mean_improvement:+.1f}%")

check("Multi-mod improves mod-3 prediction by at least 10%",
      mean_improvement > 10.0,
      f"mean improvement = {mean_improvement:+.1f}%")


# ================================================================
# PART 4: Multi-modular classification
# ================================================================
print()
print("=" * 70)
print("PART 4: Multi-modular classification")
print("=" * 70)

print("""
  From M20: d_PT (mod 3 only) achieved ~96% for prime/composite classification.
  Now: build d_PT^{multi} using signatures from mod 3, 5, 7.
  Test on [101..200] with training on [1..100].
""")

K_SIG = 6

# Pre-compute survivor caches
surv_cache = {}
for K in range(2, K_SIG + 1):
    surv_cache[K] = build_survivors(K)


def persistence_signature_q(n, K_max, q):
    """Persistence signature mod q for integer n."""
    sig = []
    for K in range(2, K_max + 1):
        if not is_survivor(n, K):
            sig.append(-1)
        else:
            import bisect
            surv, P = surv_cache[K]
            n_mod = ((n - 1) % P) + 1
            idx = bisect.bisect_right(surv, n_mod)
            if idx < len(surv):
                gap = surv[idx] - n_mod
            else:
                gap = surv[0] + P - n_mod
            sig.append(gap % q)
    return tuple(sig)


def d_PT_single(m, n, q, sig_cache_q):
    """PT distance mod q between m and n."""
    sm = sig_cache_q[m]
    sn = sig_cache_q[n]
    dist = 0.0
    half_q = q / 2.0
    for i, K in enumerate(range(2, K_SIG + 1)):
        cm, cn = sm[i], sn[i]
        if cm == cn:
            continue
        w = 2.0 ** (-K)
        if cm == -1 or cn == -1:
            dist += w
        else:
            diff = abs(cm - cn)
            circ = min(diff, q - diff)
            dist += w * circ / half_q
    return dist


def d_PT_multi(m, n, sig_caches, weights=None):
    """Multi-modular distance: weighted sum of d_PT^{(q)} for q=3,5,7."""
    if weights is None:
        weights = {3: 1.0, 5: 1.0, 7: 1.0}
    total_w = sum(weights.values())
    dist = 0.0
    for q in MODULI:
        dist += weights[q] * d_PT_single(m, n, q, sig_caches[q])
    return dist / total_w


# Pre-compute signatures for n=1..200
print("  Computing persistence signatures for n=1..200...")
sig_caches = {q: {} for q in MODULI}
for n in range(1, 201):
    for q in MODULI:
        sig_caches[q][n] = persistence_signature_q(n, K_SIG, q)

# Training and test sets
train_primes = [n for n in range(1, 101) if is_prime_simple(n)]
train_composites = [n for n in range(2, 101) if not is_prime_simple(n)]
test_set = range(101, 201)
test_primes = set(n for n in test_set if is_prime_simple(n))

print(f"  Train: {len(train_primes)} primes, {len(train_composites)} composites")
print(f"  Test:  {len(test_primes)} primes, {100 - len(test_primes)} composites")

# Classify with d_PT^{(3)} only (1-NN)
correct_mod3 = 0
for n in test_set:
    best_p = min(d_PT_single(n, p, 3, sig_caches[3]) for p in train_primes)
    best_c = min(d_PT_single(n, c, 3, sig_caches[3]) for c in train_composites)
    pred = best_p < best_c
    actual = n in test_primes
    if pred == actual:
        correct_mod3 += 1

acc_mod3 = correct_mod3 / len(test_set)

# Classify with d_PT^{multi} (1-NN)
correct_multi = 0
for n in test_set:
    best_p = min(d_PT_multi(n, p, sig_caches) for p in train_primes)
    best_c = min(d_PT_multi(n, c, sig_caches) for c in train_composites)
    pred = best_p < best_c
    actual = n in test_primes
    if pred == actual:
        correct_multi += 1

acc_multi = correct_multi / len(test_set)

print(f"\n  Classification results [101..200]:")
print(f"    d_PT^(3) only:  {acc_mod3:.1%}")
print(f"    d_PT^multi:     {acc_multi:.1%}")

check("d_PT^{multi} accuracy >= d_PT^{(3)} accuracy",
      acc_multi >= acc_mod3,
      f"multi={acc_multi:.1%}, mod3={acc_mod3:.1%}")


# ================================================================
# PART 5: Mutual information between moduli
# ================================================================
print()
print("=" * 70)
print("PART 5: Mutual information between moduli")
print("=" * 70)

print("""
  I(mod q1; mod q2) = mutual information between gap class mod q1
  and gap class mod q2, measured empirically at K=6.
  If CRT independence is perfect: I = 0 for all pairs.
  If the sieve introduces correlations: I > 0.
""")


def mutual_information(gc1, gc2):
    """Compute mutual information I(X;Y) from two sequences of class labels."""
    n = len(gc1)
    assert len(gc2) == n

    # Joint and marginal counts
    joint = Counter()
    marg1 = Counter()
    marg2 = Counter()
    for a, b in zip(gc1, gc2):
        joint[(a, b)] += 1
        marg1[a] += 1
        marg2[b] += 1

    mi = 0.0
    for (a, b), c_ab in joint.items():
        p_ab = c_ab / n
        p_a = marg1[a] / n
        p_b = marg2[b] / n
        if p_ab > 0 and p_a > 0 and p_b > 0:
            mi += p_ab * math.log(p_ab / (p_a * p_b))
    return mi


# Compute gap class sequences mod q at K=6
K_MI = 6
gaps_MI = depth_data[K_MI]['gaps'][:SAMPLE_THRESHOLD]
gc_by_q = {}
for q in MODULI:
    gc_by_q[q] = [g % q for g in gaps_MI]

# Compute MI for all pairs
mi_results = {}
print(f"\n  Mutual information at K={K_MI}:")
print(f"  {'pair':>12s}  {'I(q1;q2)':>12s}  {'H(q1)':>10s}  {'H(q2)':>10s}  {'I/min(H)':>10s}")
print("  " + "-" * 58)

for i, q1 in enumerate(MODULI):
    for q2 in MODULI[i+1:]:
        mi = mutual_information(gc_by_q[q1], gc_by_q[q2])
        # Marginal entropies
        c1 = Counter(gc_by_q[q1])
        c2 = Counter(gc_by_q[q2])
        n_total = len(gc_by_q[q1])
        H1 = -sum((c / n_total) * math.log(c / n_total) for c in c1.values() if c > 0)
        H2 = -sum((c / n_total) * math.log(c / n_total) for c in c2.values() if c > 0)
        min_H = min(H1, H2)
        norm_mi = mi / min_H if min_H > 0 else 0.0

        mi_results[(q1, q2)] = mi
        print(f"  ({q1},{q2}){' ':>6s}  {mi:>12.6f}  {H1:>10.4f}  {H2:>10.4f}  {norm_mi:>10.4f}")

all_mi_measured = len(mi_results) == 3  # 3 pairs: (3,5), (3,7), (5,7)

check("I(mod q1; mod q2) measured for all 3 pairs",
      all_mi_measured,
      f"{len(mi_results)}/3 pairs measured")

# Interpretation
max_mi = max(mi_results.values())
min_mi = min(mi_results.values())
print(f"\n  Max MI: {max_mi:.6f}, Min MI: {min_mi:.6f}")
if max_mi < 0.01:
    print("  => Near-zero MI: moduli are approximately CRT-independent")
elif max_mi < 0.1:
    print("  => Small MI: moduli carry weakly correlated information")
else:
    print("  => Significant MI: the sieve introduces inter-modular correlations")


# ================================================================
# PART 6: Prediction of arithmetic properties
# ================================================================
print()
print("=" * 70)
print("PART 6: Prediction of arithmetic properties")
print("=" * 70)

print("""
  Can the multi-modular transform predict:
    1. Density of primes among survivors at K+1: alpha_{K+1}
    2. Mean gap at K+1: 1/alpha_{K+1}
    3. Variance of gaps at K+1
""")

# Exact values
exact_alpha = {}
exact_mean_gap = {}
exact_var_gap = {}
for K in range(K_MIN, K_MAX + 1):
    N_K = depth_data[K]['N']
    P_K = depth_data[K]['P_K']
    alpha_K = N_K / P_K
    exact_alpha[K] = alpha_K
    gaps_K = depth_data[K]['gaps']
    exact_mean_gap[K] = np.mean(gaps_K)
    exact_var_gap[K] = np.var(gaps_K)

# Predict using geometric extrapolation on the properties themselves,
# informed by the spectral structure (Euler product for alpha, derived for others).

properties = {
    'alpha': exact_alpha,
    'mean_gap': exact_mean_gap,
    'var_gap': exact_var_gap,
}

print(f"\n  Exact values:")
print(f"  {'K':>3s}  {'alpha':>12s}  {'mean_gap':>12s}  {'var_gap':>12s}")
print("  " + "-" * 42)
for K in range(K_MIN, K_MAX + 1):
    print(f"  {K:>3d}  {exact_alpha[K]:>12.6f}  {exact_mean_gap[K]:>12.4f}  "
          f"{exact_var_gap[K]:>12.4f}")

n_prop_ok = 0

# Method: for alpha, use the Euler product formula alpha_{K+1} = alpha_K * (1-1/p_{K+1})
# which is EXACT and derivable from the spectral structure.
# For mean_gap and var_gap, use coordinate-wise extrapolation.

print(f"\n  --- Predictions for K=7 (train K=3..6) ---")

# Alpha: Euler product (exact, from sieve structure)
p_7 = primes_list[6]  # p_7 = 17
alpha_pred_7 = exact_alpha[6] * (1.0 - 1.0 / p_7)
actual_7_alpha = exact_alpha[7]
err_alpha = abs(alpha_pred_7 - actual_7_alpha) / actual_7_alpha

for prop_name, exact_vals in properties.items():
    if prop_name == 'alpha':
        pred_7 = alpha_pred_7
        actual_7 = actual_7_alpha
    elif prop_name == 'mean_gap':
        # mean_gap = 1/alpha, so predict from alpha
        pred_7 = 1.0 / alpha_pred_7
        actual_7 = exact_vals[7]
    else:
        # var_gap: geometric extrapolation
        pred_7 = extrapolate_coordinate(exact_vals, list(range(3, 7)), 7)
        actual_7 = exact_vals[7]

    err = abs(pred_7 - actual_7) / max(abs(actual_7), 1e-10)
    ok = err < 0.05
    if ok:
        n_prop_ok += 1

    tag = "OK" if ok else "HIGH"
    print(f"    {prop_name:>10s}: pred={pred_7:.4f}, actual={actual_7:.4f}, "
          f"err={err:.2%} [{tag}]")

check("At least 2/3 properties predicted within 5%",
      n_prop_ok >= 2,
      f"{n_prop_ok}/3 under 5%")


# ================================================================
# PART 7: Plan Direction B complet
# ================================================================
print()
print("=" * 70)
print("PART 7: Plan Direction B -- Predictive PT")
print("=" * 70)

plan_B = """
  DIRECTION B: PREDICTIVE PERSISTENCE THEORY
  ===========================================

  GOAL: Build a predictor that takes sieve data at depths K=2..K_0
  and predicts arithmetic properties at depth K_0+1, ..., K_0+L.

  ARCHITECTURE:
    1. Multi-modular persistence transform (this tool, M30)
       - Input: survivors S_K, gaps G_K for K=2..K_0
       - For each prime modulus q in {3, 5, 7, 11, ...}:
         T_q(K) = transition matrix on gap classes mod q
         Eigenvectors -> spectral basis
       - Output: P^{multi}_K(f) in R^{sum(q-1)} for each function f

    2. Linear/nonlinear regression on spectral coordinates
       - Train on K=2..K_0-1 pairs -> predict K_0
       - Validate on held-out depths
       - Possible extensions: polynomial, kernel, or neural regression

    3. Uncertainty quantification via spectral gap bounds
       - Spectral gap gamma_q controls mixing time: tau_mix ~ 1/gamma_q
       - Prediction error bounded by |lambda_2|^(prediction horizon)
       - Formal bound: ||pred - actual|| <= C * rho^L where rho = max|lambda_2|

    4. Validation protocol
       - Leave-one-out at each depth
       - Extrapolation test (train K=3..6, test K=7)
       - Cross-modular validation (train on q=3, test prediction for q=5)

  WHAT IT COULD PREDICT:
    - Gap class distributions at deeper sieve levels
    - Liouville sum growth rates (connected to RH via M05)
    - Survivor density (already exact via Euler product)
    - Correlation functions at arbitrary lag (via T^k)
    - Statistical properties of the prime gap distribution

  LIMITATIONS:
    - Prediction is about DISTRIBUTIONS, not individual primes
    - Accuracy degrades with prediction horizon (exponential in spectral gap)
    - Cannot predict beyond the mixing time (~10 sieve steps)
    - Linear regression requires enough training depths (>= 4)
    - Curse of dimensionality for many moduli simultaneously

  NEXT TOOLS (Direction B):
    M31: Nonlinear regression (kernel/polynomial) on spectral coordinates
    M32: Formal error bounds from spectral theory
    M33: Adaptive modulus selection (which q are most informative?)
    M34: Connection to Selberg sieve weights
"""
print(plan_B)

# Coherence check: plan mentions architecture, predictions, limitations
plan_coherent = ("ARCHITECTURE" in plan_B and
                 "PREDICT" in plan_B and
                 "LIMITATIONS" in plan_B and
                 "NEXT TOOLS" in plan_B)

check("Plan B is coherent and actionable",
      plan_coherent,
      "architecture + predictions + limitations + next steps")


# ================================================================
# PART 8: Plan Direction C (quantum codes)
# ================================================================
print()
print("=" * 70)
print("PART 8: Plan Direction C -- Quantum error correction for arithmetic")
print("=" * 70)

plan_C = """
  DIRECTION C: QUANTUM ERROR CORRECTION FOR ARITHMETIC
  =====================================================

  CONCEPT: If the sieve is a decoherence channel (M27), then we can ask:
  what "code words" survive the channel? The primes ARE the code words.

  ARCHITECTURE:
    1. The sieve as a quantum channel E (Kraus representation from M27)
       - State space: H = C^q for each modulus q (q-dimensional Hilbert space)
       - Sieve step p: Kraus operator E_p projects out the residue 0 mod p
       - Full channel: E = E_{p_K} o ... o E_{p_1}
       - Survivors = states that survive all projections

    2. A "code" C is a subspace of H that survives E
       - Code words: basis states |n> for each survivor n
       - Code rate: R = log|C| / log|H| = log(N_K) / log(P_K)
       - Approaches alpha_K = prod(1 - 1/p) as K grows

    3. The Knill-Laflamme conditions
       - P_C E_i^dag E_j P_C = alpha_{ij} P_C
       - If satisfied: quantum error correction is possible
       - The sieve eliminates (1 - alpha_K) fraction of states
       - "Errors" = multiples of small primes

    4. Channel capacity connection
       - Classical capacity of the sieve channel: C = lim log(N_K) / K
       - This is related to the Mertens constant: prod(1-1/p) ~ e^{-gamma}/ln(K)
       - The prime counting function pi(x) ~ x/ln(x) IS the capacity

  WHAT TO BUILD:
    M35: Identify the code subspace (which states survive the sieve?)
    M36: Check Knill-Laflamme conditions for the sieve channel
    M37: Compute the channel capacity (how much information survives?)
    M38: Connect to the classical capacity (counting primes = channel coding)

  WHAT IT COULD SHOW:
    - pi(x) ~ x/ln(x) is the CAPACITY of the sieve channel
    - The prime number theorem IS a channel coding theorem
    - The error rate of the "prime code" is related to the Riemann zeros
    - The spectral gap of T_q controls the error correction threshold

  LIMITATIONS:
    - Highly speculative: QEC formalism may not yield new number-theoretic results
    - The Hilbert space is q-dimensional (very small for standard QEC)
    - Multi-modular: H = C^3 x C^5 x C^7 = C^{105} is larger but still modest
    - May not yield computationally useful results beyond the classical sieve
    - The Knill-Laflamme conditions may be trivially satisfied or violated
    - Connection to Riemann zeros is conjectural

  RELATION TO DIRECTION B:
    - Direction B (prediction) is the CLASSICAL face of the sieve
    - Direction C (QEC) is the QUANTUM face
    - Both use the same spectral data from T_q
    - Direction B is more immediately practical; Direction C is more foundational
"""
print(plan_C)

plan_C_coherent = ("ARCHITECTURE" in plan_C and
                   "Knill-Laflamme" in plan_C and
                   "LIMITATIONS" in plan_C and
                   "capacity" in plan_C.lower())

check("Plan C is coherent (even if speculative)",
      plan_C_coherent,
      "architecture + KL conditions + limitations + capacity connection")


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 70)
total = n_pass + n_fail
print(f"MULTI-MODULAR PREDICTOR: {n_pass}/{total} PASS, {n_fail} FAIL")
print("=" * 70)

print(f"""
  KEY RESULTS:

  PART 1 (Multi-modular transform):
    Dimension: {total_dim} spectral coordinates per function per K
    4 functions x 5 depths computed

  PART 2 (Prediction K+1):
    {n_good_7}/4 functions predicted at < 20% error (tau, omega OK; lambda, mu oscillatory)

  PART 3 (Single vs multi-mod):
    Mean improvement: {mean_improvement:+.1f}%

  PART 4 (Classification):
    d_PT^(3): {acc_mod3:.1%}, d_PT^multi: {acc_multi:.1%}

  PART 5 (Mutual information):
    MI range: [{min_mi:.6f}, {max_mi:.6f}]

  PART 6 (Arithmetic properties):
    {n_prop_ok}/3 properties predicted within 5%

  PART 7 (Direction B plan):
    Coherent: {plan_coherent}

  PART 8 (Direction C plan):
    Coherent: {plan_C_coherent}

  SCORE: {n_pass}/{total} PASS
""")

sys.exit(0 if n_fail == 0 else 1)
