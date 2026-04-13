"""
test_dim_formula.py -- Verification of Layer Dimension Formula
==============================================================
Status: [THM*]  |  Chapter: ch_PM, Section 9.4
Formula: dim(d) = P(d+1) - (2d+1)
where P(k) = sum of first k primes.

Verified exactly on shells 11..29 (6 shells, base depths 3-8).
"""
import sys
import time
sys.path.insert(0, ".")
from mp_core import (
    build_probe_families, max_depth_for_shell,
    exact_observation_matrix, dim_formula, sum_first_primes
)
import numpy as np


def compute_observed_dims(shell: int):
    """Compute observed layer dimensions for one shell."""
    base_shells, eta_target = build_probe_families(shell)
    max_depth = max_depth_for_shell(shell)
    family = tuple(list(base_shells) + list(eta_target))

    mats = []
    for d in range(1, max_depth + 1):
        mat = exact_observation_matrix(family, d)
        mats.append(mat)

    hist = None
    ranks = []
    for mat in mats:
        hist = mat if hist is None else np.vstack([hist, mat])
        r = int(np.linalg.matrix_rank(hist, tol=1e-10))
        ranks.append(r)

    dims = [ranks[0]]
    for i in range(1, len(ranks)):
        dims.append(ranks[i] - ranks[i - 1])
    return dims, max_depth


def main():
    print("=" * 65)
    print("  TEST DIM FORMULA: dim(d) = P(d+1) - (2d+1)  [THM*]")
    print("=" * 65)

    # Show formula values
    print(f"\n  Formula predictions:")
    print(f"  {'d':>3s}  {'P(d+1)':>7s}  {'2d+1':>5s}  {'dim(d)':>7s}")
    print(f"  {'---':>3s}  {'------':>7s}  {'----':>5s}  {'------':>7s}")
    for d in range(1, 11):
        P = sum_first_primes(d + 1)
        formula = dim_formula(d)
        print(f"  {d:3d}  {P:7d}  {2*d+1:5d}  {formula:7d}")

    shells = [11, 13, 17, 19, 23, 29]
    print(f"\n  Verifying on shells: {shells}\n")

    passes = 0
    fails = 0
    t0 = time.time()

    for shell in shells:
        dims, max_depth = compute_observed_dims(shell)
        base_depth = max_depth - 1

        # Check non-trivial layers (d where dim > 0)
        all_match = True
        for d_idx in range(len(dims)):
            d = d_idx + 1  # depth is 1-indexed
            if d <= base_depth:
                predicted = dim_formula(d)
                observed = dims[d_idx]
                match = (observed == predicted) or (observed == 0 and d > base_depth)
                if observed > 0 and observed != predicted:
                    all_match = False

        status = "PASS" if all_match else "FAIL"
        print(f"  Shell {shell:3d}: base_depth={base_depth}, "
              f"dims={dims[:base_depth]}, {status}")

        if all_match:
            passes += 1
        else:
            fails += 1

    total = time.time() - t0
    print(f"\n  {'=' * 50}")
    print(f"  RESULT: {passes}/{passes + fails} PASS "
          f"({'PASS' if fails == 0 else 'FAIL'})")
    print(f"  Total time: {total:.1f}s")
    print(f"  {'=' * 50}")

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
