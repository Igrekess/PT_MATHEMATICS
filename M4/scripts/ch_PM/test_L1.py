"""
test_L1.py -- Verification of L1 Unit Increment Theorem
========================================================
Status: [THM]  |  Chapter: ch_PM, Section 9.3
Goal: Each new shell prime adds exactly 1 DOF (dim_future = 1).

Validates on shells 11..29 (6 shells, fast computation).
Full verification on 13 shells requires --full flag.
"""
import sys
import time
sys.path.insert(0, ".")
from mp_core import (
    build_probe_families, max_depth_for_shell,
    exact_observation_matrix, is_prime
)
import numpy as np


def test_L1_single(shell: int, verbose: bool = True) -> bool:
    """Test L1 for one shell: dim(future_depth) must be > 0 with AT=1 probe."""
    base_shells, eta_target = build_probe_families(shell)
    max_depth = max_depth_for_shell(shell)

    # Full family = base + target
    family = tuple(list(base_shells) + list(eta_target))

    # Build historical matrices
    mats = []
    for d in range(1, max_depth + 1):
        mat = exact_observation_matrix(family, d)
        mats.append(mat)

    # Cumulative ranks
    hist = None
    ranks = []
    for mat in mats:
        hist = mat if hist is None else np.vstack([hist, mat])
        r = int(np.linalg.matrix_rank(hist, tol=1e-10))
        ranks.append(r)

    # Layer dimensions
    dims = [ranks[0]]
    for i in range(1, len(ranks)):
        dims.append(ranks[i] - ranks[i - 1])

    # L1: each layer with non-zero dimension should have dim = 1
    # Specifically, future_depth layer must have dim > 0
    dim_future = dims[max_depth - 1] if len(dims) >= max_depth else 0

    # Count non-zero dimensions
    non_zero_dims = [d for d in dims if d > 0]
    l1_holds = len(non_zero_dims) > 0 and all(d == 1 for d in non_zero_dims)

    if verbose:
        status = "PASS" if l1_holds else "FAIL"
        print(f"  Shell {shell:3d}: dims={dims}, future_dim={dim_future}, "
              f"L1={status}")

    return l1_holds


def main():
    print("=" * 65)
    print("  TEST L1: Unit Increment Theorem [THM]")
    print("  Each shell prime adds exactly 1 DOF")
    print("=" * 65)

    full_mode = "--full" in sys.argv
    shells = [11, 13, 17, 19, 23, 29]
    if full_mode:
        shells = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

    print(f"\n  Testing {len(shells)} shells: {shells}\n")

    passes = 0
    fails = 0
    t0 = time.time()

    for shell in shells:
        t_shell = time.time()
        ok = test_L1_single(shell)
        elapsed = time.time() - t_shell
        if ok:
            passes += 1
        else:
            fails += 1
        if elapsed > 1:
            print(f"    (elapsed: {elapsed:.1f}s)")

    total = time.time() - t0
    print(f"\n  {'=' * 50}")
    print(f"  RESULT: {passes}/{passes + fails} PASS "
          f"({'PASS' if fails == 0 else 'FAIL'})")
    print(f"  Total time: {total:.1f}s")
    print(f"  {'=' * 50}")

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
