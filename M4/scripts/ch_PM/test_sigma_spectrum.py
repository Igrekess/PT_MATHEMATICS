"""
test_sigma_spectrum.py -- Verification of Transversality Spectrum
================================================================
Status: [THM*]  |  Chapter: ch_PM, Section 9.5
Formula: Sigma_k = C(4, k) for k >= AT.

The transversality spectrum counts how many k-probe subsets of the
target shell detect dim_future > 0. For k >= AT, the count follows
binomial coefficients C(4, k).
"""
import sys
import time
from math import comb
sys.path.insert(0, ".")
from mp_core import (
    build_probe_families, max_depth_for_shell,
    exact_observation_matrix, at_formula, SHELL_DATA
)
import numpy as np
from itertools import combinations


def compute_sigma_spectrum(shell: int, verbose: bool = True):
    """Compute Sigma_k for all k = 1..4 for one shell."""
    base_shells, eta_target = build_probe_families(shell)
    max_depth = max_depth_for_shell(shell)
    base_depth = max_depth - 1

    n_target = len(eta_target)  # should be 4

    # Compute base historical matrix (without target probes)
    base_family = base_shells
    base_mats = []
    for d in range(1, max_depth + 1):
        mat = exact_observation_matrix(base_family, d)
        base_mats.append(mat)

    hist_base_d = np.vstack(base_mats[:base_depth])
    hist_base_d1 = np.vstack(base_mats[:max_depth])
    R_base_d = int(np.linalg.matrix_rank(hist_base_d, tol=1e-10))

    sigmas = {}
    for k in range(1, n_target + 1):
        count = 0
        for combo in combinations(range(n_target), k):
            # Build family with these k target probes
            family_k = list(base_family) + [eta_target[i] for i in combo]
            family_k = tuple(family_k)

            mats_k = []
            for d in range(1, max_depth + 1):
                mat = exact_observation_matrix(family_k, d)
                mats_k.append(mat)

            hist_k_d = np.vstack(mats_k[:base_depth])
            hist_k_d1 = np.vstack(mats_k[:max_depth])
            R_k_d = int(np.linalg.matrix_rank(hist_k_d, tol=1e-10))
            R_k_d1 = int(np.linalg.matrix_rank(hist_k_d1, tol=1e-10))

            dim_future = R_k_d1 - R_k_d
            if dim_future > 0:
                count += 1

        sigmas[k] = count
        predicted = comb(4, k) if k >= at_formula(SHELL_DATA[shell]["B"]) else None

        if verbose:
            tag = ""
            if predicted is not None:
                tag = f" (C(4,{k})={comb(4,k)}, " + \
                      ("MATCH" if count == predicted else f"MISMATCH: expected {predicted}") + ")"
            print(f"    Sigma_{k} = {count}{tag}")

    return sigmas


def main():
    print("=" * 65)
    print("  TEST SIGMA SPECTRUM: Sigma_k = C(4,k)  [THM*]")
    print("=" * 65)

    shells = [11, 13, 17, 19, 23, 29]
    print(f"\n  Testing {len(shells)} shells\n")

    passes = 0
    fails = 0
    t0 = time.time()

    for shell in shells:
        B = SHELL_DATA[shell]["B"]
        AT = at_formula(B)
        print(f"\n  Shell {shell}: |B|={B}, AT={AT}")

        t_shell = time.time()
        sigmas = compute_sigma_spectrum(shell, verbose=True)
        elapsed = time.time() - t_shell

        # Verify: for k >= AT, Sigma_k should equal C(4, k)
        shell_ok = True
        for k in range(AT, 5):
            if sigmas.get(k, 0) != comb(4, k):
                shell_ok = False

        status = "PASS" if shell_ok else "FAIL"
        print(f"  -> {status} ({elapsed:.1f}s)")

        if shell_ok:
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
