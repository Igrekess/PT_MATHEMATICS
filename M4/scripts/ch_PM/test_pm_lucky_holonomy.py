#!/usr/bin/env python3
"""
test_pm_lucky_holonomy.py -- Lucky Number Holonomy (Positional Sieve Analogue)
==============================================================================
STATUS BOX
  GOAL   : Define sin^2_Lucky analogue of PT holonomy for Lucky sieve
  INPUTS : Lucky sieve (positional), cascade formula delta*(2-delta)
  STATUT : [VAL] exploration, ch_PM

CONTEXT:
  PT holonomy: sin^2(theta_p) = delta_p*(2-delta_p) with delta_p = (1-q^p)/p
  Lucky sieve: positional (not multiplicative), starts from odds.
  Question: does sin^2_Lucky exist and compare to sin^2_PT?

TESTS:
  T1: Lucky gaps are even (for L > 3)
  T2: surv_Lucky converges (product decreasing)
  T3: sin^2_Lucky(L_k) well-defined and in [0,1]
  T4: gamma_Lucky decreasing with L_k (monotonicity)
  T5: Fixed point mu*_Lucky exists or not
  T6: Comparison sin^2_Lucky vs sin^2_PT -- ratio
  T7: Cascade product prod(sin^2_Lucky) meaningful
"""
import sys
import time
import numpy as np

# --Lucky number generator --────────────────────────────────────────
def lucky_numbers(limit: int) -> list:
    """Generate Lucky numbers up to limit using the Lucky sieve."""
    sieve = list(range(1, limit + 1, 2))
    i = 1
    while i < len(sieve) and sieve[i] <= len(sieve):
        step = sieve[i]
        sieve = [sieve[j] for j in range(len(sieve)) if (j + 1) % step != 0]
        i += 1
    return sieve


def lucky_sieve_cascade(limit: int):
    """
    Run the Lucky sieve step by step, recording L_k at each step.
    Returns list of (step_index, L_k, len(survivors)) for each sieve step.
    """
    sieve = list(range(1, limit + 1, 2))
    cascade = [(0, 1, len(sieve))]  # step 0: all odds, L_0 = 1
    i = 1
    while i < len(sieve) and sieve[i] <= len(sieve):
        step = sieve[i]
        cascade.append((i, step, len(sieve)))
        sieve = [sieve[j] for j in range(len(sieve)) if (j + 1) % step != 0]
        i += 1
    return cascade


# --PT canonical holonomy --─────────────────────────────────────────
def pt_sin2(p, mu=15):
    """PT canonical sin^2(theta_p, q_stat) at mu*=15."""
    q = 1 - 2.0 / mu  # q_stat = 13/15
    delta = (1 - q**p) / p
    return delta * (2 - delta)


# --Main --──────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    N = 100_000
    results = {}

    print("=" * 72)
    print("LUCKY NUMBER HOLONOMY -- POSITIONAL SIEVE ANALOGUE")
    print("=" * 72)
    print(f"N = {N}")
    print()

    # --1. Generate Lucky numbers --─────────────────────────────────
    print("-- 1. Generating Lucky numbers --")
    luckys = lucky_numbers(N)
    n_lucky = len(luckys)
    print(f"  Lucky numbers up to {N}: {n_lucky}")
    print(f"  First 20: {luckys[:20]}")
    print(f"  Last  5 : {luckys[-5:]}")

    # --2. Gaps --───────────────────────────────────────────────────
    print("\n--2. Lucky gaps --")
    gaps = [luckys[i+1] - luckys[i] for i in range(n_lucky - 1)]
    gaps_arr = np.array(gaps)
    mean_gap = np.mean(gaps_arr)
    print(f"  Number of gaps: {len(gaps)}")
    print(f"  Mean gap: {mean_gap:.4f}")
    print(f"  Min gap: {min(gaps)}, Max gap: {max(gaps)}")
    print(f"  First 20 gaps: {gaps[:20]}")

    # --T1: gaps are even (for L > 3) --────────────────────────────
    print("\n--T1: Lucky gaps parity --")
    # Luckys are all odd (start from odds, positional removal preserves parity)
    # So gaps should be even
    gaps_after3 = [luckys[i+1] - luckys[i] for i in range(1, n_lucky - 1)]
    n_odd_gaps = sum(1 for g in gaps_after3 if g % 2 != 0)
    t1_pass = (n_odd_gaps == 0)
    print(f"  Gaps after L=3: {len(gaps_after3)}, odd gaps: {n_odd_gaps}")
    print(f"  T1 (gaps even for L>3): {'PASS' if t1_pass else 'FAIL'}")

    # --3. Cascade computation --────────────────────────────────────
    print("\n--3. Lucky cascade holonomy --")
    cascade = lucky_sieve_cascade(N)
    # L_k values (the sieve primes): skip step 0 (L_0=1)
    L_vals = [c[1] for c in cascade[1:]]  # L_1=3, L_2=7, L_3=9, ...
    n_steps = len(L_vals)
    print(f"  Sieve steps: {n_steps}")
    print(f"  L_k values: {L_vals[:15]}...")

    # surv_Lucky(k) = prod_{j=1}^{k} (1 - 1/L_j)  [L_1=3, L_2=7, ...]
    surv = np.ones(n_steps)
    for k in range(n_steps):
        if k == 0:
            surv[k] = 1 - 1.0 / L_vals[k]
        else:
            surv[k] = surv[k-1] * (1 - 1.0 / L_vals[k])

    mu_lucky = 2.0 / surv  # factor 2 because we start from odds
    q_lucky = 1 - surv     # q = 1 - 2/mu = 1 - surv

    # delta and sin^2
    delta_lucky = np.zeros(n_steps)
    sin2_lucky = np.zeros(n_steps)
    for k in range(n_steps):
        Lk = L_vals[k]
        qk = q_lucky[k]
        dk = (1 - qk**Lk) / Lk
        delta_lucky[k] = dk
        sin2_lucky[k] = dk * (2 - dk)

    print(f"\n  {'k':>3} {'L_k':>6} {'surv':>10} {'mu':>10} {'q':>10} {'delta':>10} {'sin2':>10}")
    print("  " + "-" * 65)
    for k in range(min(15, n_steps)):
        print(f"  {k+1:3d} {L_vals[k]:6d} {surv[k]:10.6f} {mu_lucky[k]:10.4f}"
              f" {q_lucky[k]:10.6f} {delta_lucky[k]:10.6f} {sin2_lucky[k]:10.6f}")

    # --T2: surv converges --────────────────────────────────────────
    print("\n--T2: Survival fraction convergence --")
    surv_last = surv[-1]
    surv_monotone = all(surv[k+1] <= surv[k] for k in range(n_steps - 1))
    t2_pass = surv_monotone and (surv_last > 0)
    print(f"  surv(last) = {surv_last:.8f}, monotone decreasing: {surv_monotone}")
    print(f"  T2 (surv converges): {'PASS' if t2_pass else 'FAIL'}")

    # --T3: sin^2 in [0,1] --───────────────────────────────────────
    print("\n--T3: sin^2_Lucky in [0,1] --")
    all_in_range = all(0 <= s <= 1 for s in sin2_lucky)
    t3_pass = all_in_range
    print(f"  min(sin2) = {min(sin2_lucky):.8f}, max(sin2) = {max(sin2_lucky):.8f}")
    print(f"  T3 (sin^2 in [0,1]): {'PASS' if t3_pass else 'FAIL'}")

    # --4. Anomalous dimension gamma_Lucky --────────────────────────
    print("\n--4. Anomalous dimension gamma_Lucky --")
    # gamma = -d(ln sin2)/d(ln mu) computed numerically
    ln_sin2 = np.log(sin2_lucky)
    ln_mu = np.log(mu_lucky)
    gamma_lucky = np.zeros(n_steps)
    for k in range(1, n_steps - 1):
        gamma_lucky[k] = -(ln_sin2[k+1] - ln_sin2[k-1]) / (ln_mu[k+1] - ln_mu[k-1])
    # boundaries
    if n_steps >= 2:
        gamma_lucky[0] = -(ln_sin2[1] - ln_sin2[0]) / (ln_mu[1] - ln_mu[0])
        gamma_lucky[-1] = -(ln_sin2[-1] - ln_sin2[-2]) / (ln_mu[-1] - ln_mu[-2])

    print(f"  {'k':>3} {'L_k':>6} {'mu':>10} {'sin2':>10} {'gamma':>10}")
    print("  " + "-" * 45)
    for k in range(min(15, n_steps)):
        print(f"  {k+1:3d} {L_vals[k]:6d} {mu_lucky[k]:10.4f}"
              f" {sin2_lucky[k]:10.6f} {gamma_lucky[k]:10.4f}")

    # --T4: gamma decreasing --──────────────────────────────────────
    print("\n--T4: gamma_Lucky monotonicity --")
    # Check from k=2 onwards (skip boundary effects)
    gamma_core = gamma_lucky[2:]
    n_incr = sum(1 for i in range(len(gamma_core)-1) if gamma_core[i+1] > gamma_core[i] + 1e-10)
    t4_pass = (n_incr == 0)
    print(f"  gamma increases (k>=3): {n_incr} out of {len(gamma_core)-1}")
    print(f"  T4 (gamma decreasing): {'PASS' if t4_pass else 'FAIL'}")
    if not t4_pass:
        print(f"  [INFO] gamma is NOT monotone -- checking trend")
        # Check overall trend
        if len(gamma_core) >= 2:
            trend = gamma_core[-1] - gamma_core[0]
            print(f"  Overall trend: gamma[-1] - gamma[2] = {trend:.6f} ({'decreasing' if trend < 0 else 'increasing'})")

    # --5. Fixed point mu*_Lucky --──────────────────────────────────
    print("\n--5. Fixed point search --")
    # mu*_Lucky = sum of active L_k where gamma > 1/2
    active_threshold = 0.5
    active_Lk = [L_vals[k] for k in range(n_steps) if gamma_lucky[k] > active_threshold]
    sum_active = sum(active_Lk) if active_Lk else 0
    print(f"  Active L_k (gamma > 0.5): {len(active_Lk)} values")
    if active_Lk:
        print(f"  Active L_k: {active_Lk[:20]}{'...' if len(active_Lk) > 20 else ''}")
        print(f"  sum(active L_k) = {sum_active}")
        # Check if mu at some step is close to sum_active
        mu_closest_idx = np.argmin(np.abs(mu_lucky - sum_active))
        mu_closest = mu_lucky[mu_closest_idx]
        print(f"  Closest mu in cascade: mu[{mu_closest_idx+1}] = {mu_closest:.4f}")
        rel_err = abs(mu_closest - sum_active) / sum_active if sum_active > 0 else float('inf')
        print(f"  Relative error: {rel_err:.4f}")
        t5_pass = rel_err < 0.1
    else:
        print(f"  No active L_k found!")
        # Alternative: check if gamma > 0 for any
        active_any = [L_vals[k] for k in range(n_steps) if gamma_lucky[k] > 0]
        print(f"  L_k with gamma > 0: {len(active_any)}")
        t5_pass = False

    print(f"  T5 (fixed point exists): {'PASS' if t5_pass else 'FAIL'}")

    # --6. Empirical sin^2 from mean gap --──────────────────────────
    print("\n--6. Empirical sin^2 from q_empirical --")
    q_emp = 1 - 2.0 / mean_gap
    print(f"  Mean Lucky gap = {mean_gap:.6f}")
    print(f"  q_empirical = {q_emp:.6f}")

    # Use first few L_k to compute sin^2 with q_emp
    print(f"\n  {'L_k':>6} {'sin2_cascade':>14} {'sin2_emp':>14} {'ratio':>10}")
    print("  " + "-" * 50)
    for k in range(min(15, n_steps)):
        Lk = L_vals[k]
        d_emp = (1 - q_emp**Lk) / Lk
        s2_emp = d_emp * (2 - d_emp)
        ratio = sin2_lucky[k] / s2_emp if s2_emp > 0 else float('inf')
        print(f"  {Lk:6d} {sin2_lucky[k]:14.6f} {s2_emp:14.6f} {ratio:10.4f}")

    # --7. Comparison sin^2_Lucky vs sin^2_PT --─────────────────────
    print("\n--7. Comparison sin^2_Lucky vs sin^2_PT --")
    # Match by value: L_k vs prime p for shared values
    primes_small = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

    print(f"\n  {'p/L':>6} {'sin2_PT':>12} {'sin2_Lucky':>12} {'ratio_L/PT':>12} {'note':>12}")
    print("  " + "-" * 60)
    ratios_comparison = []
    for p in primes_small:
        s2_pt = pt_sin2(p, mu=15)
        # Find this value in L_vals
        if p in L_vals:
            k_idx = L_vals.index(p)
            s2_lk = sin2_lucky[k_idx]
            ratio = s2_lk / s2_pt if s2_pt > 0 else float('inf')
            ratios_comparison.append(ratio)
            note = "Lucky" if p in L_vals else ""
            print(f"  {p:6d} {s2_pt:12.6f} {s2_lk:12.6f} {ratio:12.4f} {note:>12}")
        else:
            print(f"  {p:6d} {s2_pt:12.6f} {'---':>12} {'---':>12} {'not Lucky':>12}")

    if ratios_comparison:
        mean_ratio = np.mean(ratios_comparison)
        std_ratio = np.std(ratios_comparison)
        print(f"\n  Mean ratio sin2_Lucky/sin2_PT = {mean_ratio:.4f} +/- {std_ratio:.4f}")
    else:
        mean_ratio = float('nan')
        std_ratio = float('nan')

    # --T6: ratio close to 1? --────────────────────────────────────
    print("\n--T6: Comparison sin^2_Lucky vs sin^2_PT --")
    if ratios_comparison:
        close_to_1 = abs(mean_ratio - 1) < 0.3
        t6_pass = close_to_1
        print(f"  Mean ratio = {mean_ratio:.4f}, |ratio - 1| = {abs(mean_ratio - 1):.4f}")
        print(f"  T6 (ratio close to 1): {'PASS' if t6_pass else 'FAIL'}")
        if not t6_pass:
            print(f"  [INFO] sin^2_Lucky and sin^2_PT differ significantly")
            print(f"  This is EXPECTED: Lucky sieve is positional, not multiplicative")
    else:
        t6_pass = False
        print(f"  T6: FAIL (no common values)")

    # --T7: Cascade product --───────────────────────────────────────
    print("\n--T7: Cascade product prod(sin^2_Lucky) --")
    log_prod = np.sum(np.log(sin2_lucky))
    prod_sin2 = np.exp(log_prod)
    print(f"  ln(prod) = {log_prod:.6f}")
    print(f"  prod(sin^2_Lucky) = {prod_sin2:.6e}")

    # For PT, prod(sin^2) over active primes gives something related to alpha
    # Check if product is finite and positive
    t7_meaningful = np.isfinite(prod_sin2) and prod_sin2 > 0
    # Also compute partial products
    print(f"\n  Partial products:")
    partial = 0.0
    for k in range(min(15, n_steps)):
        partial += np.log(sin2_lucky[k])
        print(f"    k={k+1:2d}, L_k={L_vals[k]:4d}: cum_prod = {np.exp(partial):.8f}")

    t7_pass = t7_meaningful
    print(f"\n  T7 (cascade product meaningful): {'PASS' if t7_pass else 'FAIL'}")

    # --Summary --───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    tests = [
        ("T1", "Lucky gaps even (L>3)", t1_pass),
        ("T2", "surv_Lucky converges", t2_pass),
        ("T3", "sin^2_Lucky in [0,1]", t3_pass),
        ("T4", "gamma_Lucky decreasing", t4_pass),
        ("T5", "Fixed point mu*_Lucky", t5_pass),
        ("T6", "sin^2_Lucky ~ sin^2_PT", t6_pass),
        ("T7", "Cascade product meaningful", t7_pass),
    ]
    n_pass = sum(1 for _, _, p in tests if p)
    n_total = len(tests)
    for tag, desc, passed in tests:
        status = "PASS" if passed else "FAIL"
        print(f"  {tag}: {status:4s}  -- {desc}")
    print(f"\n  Score: {n_pass}/{n_total}")

    # Key numerical results
    print(f"\n  KEY RESULTS:")
    print(f"    Lucky count up to {N}: {n_lucky}")
    print(f"    Mean Lucky gap: {mean_gap:.4f}")
    print(f"    surv(last): {surv_last:.8f}")
    if ratios_comparison:
        print(f"    sin^2_Lucky / sin^2_PT mean ratio: {mean_ratio:.4f}")
    print(f"    Cascade product: {prod_sin2:.6e}")
    if active_Lk:
        print(f"    mu*_Lucky candidate: {sum_active} (from {len(active_Lk)} active L_k)")

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    print("=" * 72)

    return n_pass, n_total


if __name__ == "__main__":
    n_pass, n_total = main()
    sys.exit(0 if n_pass >= 5 else 1)
