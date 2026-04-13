"""
test_phantom_rank.py -- Verification of Phantom Rank (C1, C2)
=============================================================
Status: [ID] + [THM*]  |  Chapter: ch_PM, Section 9.6.3
Identity C1: phantom(1) + struct(1) = n_base  [EXACT]
Theorem C2: For |B| >= 3, phantom(1) = n_base, struct(1) = 0

Verified on shells 11..29 (6 shells).
"""
import sys
import time
sys.path.insert(0, ".")
from mp_core import (
    build_probe_families, max_depth_for_shell,
    exact_observation_matrix, SHELL_DATA
)
import numpy as np


def phantom_diagnostic(shell: int, verbose: bool = True):
    """Compute phantom and structural ranks for one shell."""
    base_shells, eta_target = build_probe_families(shell)
    max_depth = max_depth_for_shell(shell)
    base_depth = max_depth - 1
    B = SHELL_DATA[shell]["B"]

    base_family = base_shells
    n_base = len(base_family)

    # Base rank at depth d and d+1
    base_mats = []
    for d in range(1, max_depth + 1):
        mat = exact_observation_matrix(base_family, d)
        base_mats.append(mat)

    hist_base_d = np.vstack(base_mats[:base_depth])
    hist_base_d1 = np.vstack(base_mats[:max_depth])
    R_base_d = int(np.linalg.matrix_rank(hist_base_d, tol=1e-10))
    R_base_d1 = int(np.linalg.matrix_rank(hist_base_d1, tol=1e-10))

    # Add probes one by one (up to 4)
    max_k = min(len(eta_target), 4)
    phantom_list = []
    struct_list = []

    for k in range(1, max_k + 1):
        family_k = tuple(list(base_family) + list(eta_target[:k]))
        mats_k = []
        for d in range(1, max_depth + 1):
            mat = exact_observation_matrix(family_k, d)
            mats_k.append(mat)

        hist_k_d = np.vstack(mats_k[:base_depth])
        hist_k_d1 = np.vstack(mats_k[:max_depth])
        R_k_d = int(np.linalg.matrix_rank(hist_k_d, tol=1e-10))
        R_k_d1 = int(np.linalg.matrix_rank(hist_k_d1, tol=1e-10))

        phantom_list.append(R_k_d - R_base_d)
        struct_list.append(R_k_d1 - R_k_d)

    if verbose:
        for k in range(len(phantom_list)):
            p = phantom_list[k]
            s = struct_list[k]
            print(f"    k={k+1}: phantom={p:3d}, struct={s:3d}, "
                  f"sum={p+s:3d}")

    return {
        "shell": shell,
        "B": B,
        "n_base": n_base,
        "phantom": phantom_list,
        "struct": struct_list,
    }


def main():
    print("=" * 65)
    print("  TEST PHANTOM RANK: C1 [ID] + C2 [THM*]")
    print("=" * 65)

    shells = [11, 13, 17, 19, 23, 29]
    print(f"\n  Testing {len(shells)} shells\n")

    c1_passes = 0
    c1_fails = 0
    c2_passes = 0
    c2_fails = 0
    t0 = time.time()

    for shell in shells:
        B = SHELL_DATA[shell]["B"]
        print(f"\n  Shell {shell}: |B|={B}")

        t_shell = time.time()
        result = phantom_diagnostic(shell, verbose=True)
        elapsed = time.time() - t_shell

        n_base = result["n_base"]
        p1 = result["phantom"][0]
        s1 = result["struct"][0]

        # C1: phantom(1) + struct(1) = n_base
        c1_ok = (p1 + s1 == n_base)
        c1_status = "PASS" if c1_ok else "FAIL"
        print(f"  C1: {p1} + {s1} = {p1+s1} {'==' if c1_ok else '!='} "
              f"n_base={n_base} -> {c1_status}")

        if c1_ok:
            c1_passes += 1
        else:
            c1_fails += 1

        # C2: for |B| >= 3, struct(1) = 0
        if B >= 3:
            c2_ok = (s1 == 0)
            c2_status = "PASS" if c2_ok else "FAIL"
            print(f"  C2: struct(1) = {s1} {'==' if c2_ok else '!='} 0 "
                  f"(|B|={B} >= 3) -> {c2_status}")
            if c2_ok:
                c2_passes += 1
            else:
                c2_fails += 1
        else:
            print(f"  C2: N/A (|B|={B} < 3, ghost shell)")

        print(f"    ({elapsed:.1f}s)")

    total = time.time() - t0
    print(f"\n  {'=' * 50}")
    print(f"  C1 [ID]:   {c1_passes}/{c1_passes + c1_fails} PASS "
          f"({'PASS' if c1_fails == 0 else 'FAIL'})")
    c2_total = c2_passes + c2_fails
    if c2_total > 0:
        print(f"  C2 [THM*]: {c2_passes}/{c2_total} PASS "
              f"({'PASS' if c2_fails == 0 else 'FAIL'})")
    total_ok = c1_fails == 0 and c2_fails == 0
    print(f"  OVERALL:   {'PASS' if total_ok else 'FAIL'}")
    print(f"  Total time: {total:.1f}s")
    print(f"  {'=' * 50}")

    return 0 if total_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
