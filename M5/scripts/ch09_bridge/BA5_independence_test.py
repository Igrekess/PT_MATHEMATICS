#!/usr/bin/env python3
"""
BA5 Independence Test -- Mutual Information on Prime Gap CRT Components
=======================================================================

Tests whether the mod-3, mod-5, mod-7 components of prime gaps are
statistically independent (as BA5 requires for the product form).

If MI(g mod 3, g mod 5) ~ 0, the components are independent and
the product form α_EM = ∏ sin²(θ_p) is structurally justified.

Date: 2026-03-06
"""

import numpy as np
from collections import defaultdict
import time

try:
    import primesieve
    import sys
    def get_primes(limit):
        return np.array(primesieve.primes(limit), dtype=np.int64)
    SIEVE_NAME = "primesieve"
except ImportError:
    SIEVE_NAME = "simple_sieve"
    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = b'\x00' * len(sieve[i*i::i])
        return np.array([i for i in range(2, limit + 1) if sieve[i]], dtype=np.int64)


def mutual_information(x, y, nx, ny):
    """Compute MI(X, Y) in bits from integer arrays x, y."""
    N = len(x)
    # Joint counts
    joint = np.zeros((nx, ny), dtype=np.int64)
    for i in range(N):
        joint[x[i], y[i]] += 1

    # Marginals
    px = joint.sum(axis=1) / N
    py = joint.sum(axis=0) / N
    pxy = joint / N

    mi = 0.0
    for a in range(nx):
        for b in range(ny):
            if pxy[a, b] > 0 and px[a] > 0 and py[b] > 0:
                mi += pxy[a, b] * np.log2(pxy[a, b] / (px[a] * py[b]))
    return mi


def conditional_test(x, y, nx, ny, label_x, label_y):
    """Test P(Y|X=a) ~ P(Y) for all a."""
    N = len(x)
    joint = np.zeros((nx, ny), dtype=np.int64)
    for i in range(N):
        joint[x[i], y[i]] += 1

    py = joint.sum(axis=0) / N  # marginal P(Y)

    print(f"\n  P({label_y}) marginal: {py}")
    max_dev = 0.0
    for a in range(nx):
        row_total = joint[a].sum()
        if row_total < 10:
            continue
        p_cond = joint[a] / row_total
        dev = np.max(np.abs(p_cond - py))
        if dev > max_dev:
            max_dev = dev
        print(f"  P({label_y}|{label_x}={a}): {p_cond}  "
              f"max|dev|={dev:.6f}  N={row_total:,}")
    return max_dev


def mi_vs_sample_size(gaps, mod_a, mod_b, n_points=8):
    """Check if MI decreases with sample size (convergence to 0)."""
    N = len(gaps)
    sizes = np.logspace(np.log10(1000), np.log10(N), n_points).astype(int)
    sizes = np.unique(sizes)

    print(f"\n  MI(g%{mod_a}, g%{mod_b}) vs sample size:")
    print(f"  {'N':>12}  {'MI (bits)':>12}  {'MI*N':>12}  {'1/sqrt(N)':>12}")
    print(f"  {'-'*52}")

    results = []
    for n in sizes:
        g = gaps[:n]
        x = (g % mod_a).astype(int)
        y = (g % mod_b).astype(int)
        mi = mutual_information(x, y, mod_a, mod_b)
        results.append((n, mi))
        print(f"  {n:>12,}  {mi:>12.8f}  {mi*n:>12.2f}  {1/np.sqrt(n):>12.8f}")
    return results


def chi2_independence(x, y, nx, ny):
    """Chi-squared test for independence."""
    N = len(x)
    joint = np.zeros((nx, ny), dtype=np.int64)
    for i in range(N):
        joint[x[i], y[i]] += 1

    row_sums = joint.sum(axis=1)
    col_sums = joint.sum(axis=0)

    chi2 = 0.0
    df = 0
    for a in range(nx):
        for b in range(ny):
            expected = row_sums[a] * col_sums[b] / N
            if expected > 0:
                chi2 += (joint[a, b] - expected) ** 2 / expected
                df += 1
    df -= (nx + ny - 1)  # degrees of freedom

    # For large N, chi2/df ~ 1 under independence
    return chi2, df, chi2 / df if df > 0 else 0


def main():
    LIMIT = 100_000_000
    print("=" * 70)
    print("   BA5 INDEPENDENCE TEST -- CRT COMPONENTS OF PRIME GAPS")
    print("   Test: MI(g mod p, g mod q) ~ 0 ?")
    print("=" * 70)

    t0 = time.time()
    print(f"\nGenerating primes up to {LIMIT:,} (engine: {SIEVE_NAME})...")
    primes = get_primes(LIMIT)
    print(f"  {len(primes):,} primes in {time.time() - t0:.1f}s")

    # Gaps (starting from p >= 5 to avoid trivial small gaps)
    mask = primes >= 5
    p = primes[mask]
    gaps = np.diff(p)
    N = len(gaps)
    print(f"  {N:,} gaps (primes >= 5)")

    # CRT components
    g3 = (gaps % 3).astype(int)
    g5 = (gaps % 5).astype(int)
    g7 = (gaps % 7).astype(int)

    # ===== TEST 1: Mutual Information =====
    print("\n" + "=" * 70)
    print("TEST 1: MUTUAL INFORMATION (MI)")
    print("=" * 70)
    print("If BA5 holds, MI should be ~ 0 (within statistical noise)")

    pairs = [(3, 5, g3, g5), (3, 7, g3, g7), (5, 7, g5, g7)]
    mi_values = []

    for mod_a, mod_b, xa, xb in pairs:
        mi = mutual_information(xa, xb, mod_a, mod_b)
        mi_values.append((mod_a, mod_b, mi))
        print(f"\n  MI(g%{mod_a}, g%{mod_b}) = {mi:.8f} bits")

    # Triple MI
    # MI(X;Y;Z) = MI(X;Y) + MI(X;Z|Y) - but for simplicity just report pairs

    # Baseline: MI for truly independent uniform would be 0
    # Statistical noise floor: ~ (nx-1)(ny-1) / (2*N*ln2) for uniform
    for mod_a, mod_b, mi in mi_values:
        noise_floor = (mod_a - 1) * (mod_b - 1) / (2 * N * np.log(2))
        ratio = mi / noise_floor if noise_floor > 0 else 0
        print(f"  MI(g%{mod_a}, g%{mod_b}): {mi:.8f} bits, "
              f"noise floor: {noise_floor:.8f}, ratio: {ratio:.2f}")

    # ===== TEST 2: MI vs sample size =====
    print("\n" + "=" * 70)
    print("TEST 2: MI vs SAMPLE SIZE (should -> 0 if independent)")
    print("=" * 70)

    for mod_a, mod_b in [(3, 5), (3, 7), (5, 7)]:
        mi_vs_sample_size(gaps, mod_a, mod_b)

    # ===== TEST 3: Conditional distributions =====
    print("\n" + "=" * 70)
    print("TEST 3: CONDITIONAL DISTRIBUTIONS P(Y|X) vs P(Y)")
    print("=" * 70)
    print("If independent: P(g%5 | g%3=a) = P(g%5) for all a")

    devs = []
    for mod_a, mod_b, xa, xb, la, lb in [
        (3, 5, g3, g5, "g%3", "g%5"),
        (3, 7, g3, g7, "g%3", "g%7"),
        (5, 7, g5, g7, "g%5", "g%7"),
    ]:
        print(f"\n  --- {la} vs {lb} ---")
        dev = conditional_test(xa, xb, mod_a, mod_b, la, lb)
        devs.append((la, lb, dev))

    # ===== TEST 4: Chi-squared =====
    print("\n" + "=" * 70)
    print("TEST 4: CHI-SQUARED INDEPENDENCE TEST")
    print("=" * 70)

    for mod_a, mod_b, xa, xb, label in [
        (3, 5, g3, g5, "g%3 vs g%5"),
        (3, 7, g3, g7, "g%3 vs g%7"),
        (5, 7, g5, g7, "g%5 vs g%7"),
    ]:
        chi2, df, ratio = chi2_independence(xa, xb, mod_a, mod_b)
        print(f"  {label}: chi2={chi2:.1f}, df={df}, chi2/df={ratio:.2f}")
        if ratio < 2:
            print(f"    -> COMPATIBLE with independence (chi2/df < 2)")
        elif ratio < 5:
            print(f"    -> WEAK dependence (2 < chi2/df < 5)")
        else:
            print(f"    -> SIGNIFICANT dependence (chi2/df > 5)")

    # ===== TEST 5: Joint vs product of marginals =====
    print("\n" + "=" * 70)
    print("TEST 5: JOINT vs PRODUCT OF MARGINALS (3-way)")
    print("=" * 70)

    # 3-way joint distribution P(g%3, g%5, g%7)
    joint3 = np.zeros((3, 5, 7), dtype=np.int64)
    for i in range(N):
        joint3[g3[i], g5[i], g7[i]] += 1

    p3 = np.bincount(g3, minlength=3) / N
    p5 = np.bincount(g5, minlength=5) / N
    p7 = np.bincount(g7, minlength=7) / N

    max_dev_3way = 0.0
    total_cells = 0
    total_dev = 0.0

    for a in range(3):
        for b in range(5):
            for c in range(7):
                p_joint = joint3[a, b, c] / N
                p_prod = p3[a] * p5[b] * p7[c]
                if p_prod > 1e-10:
                    dev = abs(p_joint - p_prod)
                    rel_dev = dev / p_prod
                    if dev > max_dev_3way:
                        max_dev_3way = dev
                        worst = (a, b, c, p_joint, p_prod, rel_dev)
                    total_dev += dev
                    total_cells += 1

    avg_dev = total_dev / total_cells if total_cells > 0 else 0
    print(f"  3-way joint P(g%3,g%5,g%7) vs product P(g%3)*P(g%5)*P(g%7):")
    print(f"  Max absolute deviation: {max_dev_3way:.8f}")
    print(f"  Mean absolute deviation: {avg_dev:.8f}")
    print(f"  Worst cell: ({worst[0]},{worst[1]},{worst[2]}), "
          f"P_joint={worst[3]:.6f}, P_prod={worst[4]:.6f}, "
          f"rel_dev={worst[5]:.4f}")
    print(f"  Statistical noise (1/sqrt(N)): {1/np.sqrt(N):.8f}")

    # ===== VERDICT =====
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    all_mi_small = all(mi < 0.001 for _, _, mi in mi_values)
    all_cond_small = all(d < 0.01 for _, _, d in devs)

    if all_mi_small and all_cond_small:
        print("  MI < 0.001 bits on all pairs")
        print("  Conditional deviations < 1%")
        print("  -> CRT components are EMPIRICALLY INDEPENDENT")
        print("  -> BA5 product form is STRUCTURALLY JUSTIFIED")
        print("  -> Status recommendation: BRIDGE -> DER")
    else:
        print("  Some MI or conditional deviations are significant")
        print("  -> CRT components show residual dependence")
        print("  -> BA5 product form is an APPROXIMATION")
        print("  -> Status remains: BRIDGE")

    for mod_a, mod_b, mi in mi_values:
        noise = (mod_a - 1) * (mod_b - 1) / (2 * N * np.log(2))
        status = "PASS (noise-level)" if mi < 10 * noise else "SIGNIFICANT"
        print(f"  MI(g%{mod_a}, g%{mod_b}) = {mi:.8f} [{status}]")

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    return results


if __name__ == "__main__":
    results = main()

    sys.exit(0 if all(results) else 1)
