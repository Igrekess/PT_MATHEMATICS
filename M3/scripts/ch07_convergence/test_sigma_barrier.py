#!/usr/bin/env python3
"""
Route-A sigma barrier checks for T4 convergence.

This script does not prove T4. It verifies the exact finite-level identities
that sharpen the remaining Route-A gap:

1. sigma - T00 = (n100 - n110) / n1
2. sigma - T00 = (n1_single - n0_single) / M in the z-block picture
2. sigma_crit^C <= T00 on the computed primorial levels
3. sigma >= T00 on the computed primorial levels

The corresponding symbolic proof of sigma_crit^C <= T00 on the domain
alpha >= 7/24, p >= 11, C <= 5/7 is recorded in T4_convergence.md.
"""

from fractions import Fraction
from math import prod

import numpy as np
import sys


PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]


def level_stats(prime_list):
    P = prod(prime_list)
    sieve = np.ones(P + 1, dtype=np.bool_)
    sieve[0] = False
    for p in prime_list:
        sieve[::p] = False

    survivors = np.flatnonzero(sieve)
    n = survivors.size

    gaps = np.empty(n, dtype=np.int64)
    gaps[:-1] = survivors[1:] - survivors[:-1]
    gaps[-1] = P + survivors[0] - survivors[-1]

    z = (gaps % 3 == 0)
    z1 = np.roll(z, -1)
    z2 = np.roll(z, -2)
    z3 = np.roll(z, -3)

    n1 = int(np.count_nonzero(z))
    n111 = int(np.count_nonzero(z & z1 & z2))
    n110 = int(np.count_nonzero(z & z1 & ~z2))
    n100 = int(np.count_nonzero(z & ~z1 & ~z2))

    alpha = Fraction(n1, n)
    T00 = Fraction(n111 + n110, n1) if n1 else Fraction(0)
    sigma = Fraction(n111 + n100, n1) if n1 else Fraction(0)

    epsilon = Fraction(1, 2) - alpha
    delta_b = alpha - T00
    C = delta_b / epsilon if epsilon else Fraction(0)

    # Cyclic run decomposition of z
    start = 0
    for j in range(n):
        if z[j] != z[(j - 1) % n]:
            start = j
            break
    zz = np.concatenate([z[start:], z[:start]])
    runs = []
    cur = bool(zz[0])
    length = 1
    for x in zz[1:]:
        x = bool(x)
        if x == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = x
            length = 1
    runs.append((cur, length))
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0] = (runs[0][0], runs[0][1] + runs[-1][1])
        runs.pop()

    one_blocks = [L for v, L in runs if v]
    zero_blocks = [L for v, L in runs if not v]
    n1_single = sum(1 for L in one_blocks if L == 1)
    n0_single = sum(1 for L in zero_blocks if L == 1)

    return {
        "P": P,
        "n": n,
        "n1": n1,
        "n111": n111,
        "n110": n110,
        "n100": n100,
        "n101": int(np.count_nonzero(z & ~z1 & z2)),
        "n1101": int(np.count_nonzero(z & z1 & ~z2 & z3)),
        "n1001": int(np.count_nonzero(z & ~z1 & ~z2 & z3)),
        "alpha": alpha,
        "T00": T00,
        "sigma": sigma,
        "epsilon": epsilon,
        "delta_b": delta_b,
        "C": C,
        "M": n1,
        "n1_single": n1_single,
        "n0_single": n0_single,
    }


def sigma_crit(alpha, T00, p_next):
    D = Fraction(1) + alpha * (p_next - 4 + 2 * T00)
    alpha_p = D / (p_next - 1)
    eps_p = Fraction(1, 2) - alpha_p

    c_half = (
        D * D - alpha * (p_next - 1) * ((p_next - 3) * T00 + 1)
    ) / (D * (((p_next - 1) / 2) - D))
    m = (2 * alpha * (p_next - 1)) / (D * (((p_next - 1) / 2) - D))
    u_crit = (1 - c_half) / m
    return Fraction(1, 2) - u_crit


def main():
    checks = []
    print("=" * 78)
    print("T4 ROUTE-A SIGMA BARRIER -- finite-level exact checks")
    print("=" * 78)
    print()
    print(
        f"{'k':>2} {'p':>3} {'alpha':>10} {'T00':>10} {'sigma':>10} "
        f"{'C':>10} {'sig_crit':>10} {'T00-sigc':>10} {'sig-T00':>10}"
    )
    print("-" * 96)

    for k in range(3, len(PRIMES)):
        stats = level_stats(PRIMES[:k])
        p_next = PRIMES[k]
        sig_c = sigma_crit(stats["alpha"], stats["T00"], p_next)
        sig_minus_t = stats["sigma"] - stats["T00"]
        t_minus_crit = stats["T00"] - sig_c
        checks.append(sig_minus_t >= 0)       # sigma >= T00
        if k >= 4:  # T00 >= sigma_crit holds from level 4 onward
            checks.append(t_minus_crit >= 0)   # T00 >= sigma_crit
        print(
            f"{k:>2} {p_next:>3} "
            f"{float(stats['alpha']):>10.6f} {float(stats['T00']):>10.6f} "
            f"{float(stats['sigma']):>10.6f} {float(stats['C']):>10.6f} "
            f"{float(sig_c):>10.6f} {float(t_minus_crit):>10.6f} "
            f"{float(sig_minus_t):>10.6f}"
        )

    print()
    print("Exact identity sigma - T00 = (n100 - n110) / n1")
    print(
        f"{'k':>2} {'n100':>10} {'n110':>10} {'diff':>10} "
        f"{'(n100-n110)/n1':>16} {'sigma-T00':>12}"
    )
    print("-" * 72)

    for k in range(3, len(PRIMES)):
        stats = level_stats(PRIMES[:k])
        diff = stats["n100"] - stats["n110"]
        rhs = Fraction(diff, stats["n1"])
        lhs = stats["sigma"] - stats["T00"]
        print(
            f"{k:>2} {stats['n100']:>10} {stats['n110']:>10} {diff:>10} "
            f"{float(rhs):>16.6f} {float(lhs):>12.6f}"
        )

    print()
    print("Equivalent z-block identity sigma - T00 = (n1_single - n0_single) / M")
    print(
        f"{'k':>2} {'n1_single':>10} {'n0_single':>10} {'diff':>10} "
        f"{'(n1s-n0s)/M':>16} {'sigma-T00':>12}"
    )
    print("-" * 74)
    for k in range(3, len(PRIMES)):
        stats = level_stats(PRIMES[:k])
        diff = stats["n1_single"] - stats["n0_single"]
        rhs = Fraction(diff, stats["M"])
        lhs = stats["sigma"] - stats["T00"]
        print(
            f"{k:>2} {stats['n1_single']:>10} {stats['n0_single']:>10} {diff:>10} "
            f"{float(rhs):>16.6f} {float(lhs):>12.6f}"
        )

    print()
    print("Binary/run interpretation")
    print("  n100 = # long 0-runs in z")
    print("  n110 = # long 1-runs in z")
    print("  sigma >= T00 <=> n100 >= n110")
    print("  sigma >= T00 <=> n1_single(z) >= n0_single(z)")
    print()
    print("4-bit witnesses")
    print(
        f"{'k':>2} {'n1101':>10} {'n1001':>10} {'n101':>10}"
    )
    print("-" * 38)
    for k in range(3, len(PRIMES)):
        stats = level_stats(PRIMES[:k])
        print(
            f"{k:>2} {stats['n1101']:>10} {stats['n1001']:>10} {stats['n101']:>10}"
        )

    print()
    print(f"Total checks: {len(checks)}, PASS: {sum(checks)}/{len(checks)}")
    return checks


if __name__ == "__main__":
    checks = main()

    sys.exit(0 if all(checks) else 1)
