#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test wrapper for all foundation proofs (D00--D08, L0).

Each D-script contains assert statements that raise on failure.
This wrapper runs them all and reports PASS/FAIL.
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "D00_forbidden_transitions.py",
    "D01_conservation_theorem.py",
    "D02_gft_ruelle.py",
    "D03_variational_sieve.py",
    "D04_master_formula_fp.py",
    "D05_mertens_law.py",
    "D06_proof_q_positive.py",
    "D07_sin2_identity.py",
    "D08_fixed_point_mu15.py",
    "L0_uniqueness_q_stat.py",
]


def main():
    n_pass = 0
    n_fail = 0
    failures = []

    for name in SCRIPTS:
        path = os.path.join(SCRIPT_DIR, name)
        if not os.path.exists(path):
            print(f"  SKIP  {name} (not found)")
            continue
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=60,
                cwd=SCRIPT_DIR,
                env={**os.environ, "PYTHONPATH": os.path.dirname(SCRIPT_DIR)},
            )
            if result.returncode == 0:
                n_pass += 1
                print(f"  PASS  {name}")
            else:
                n_fail += 1
                failures.append((name, result.stderr[-200:]))
                print(f"  FAIL  {name}")
        except subprocess.TimeoutExpired:
            n_fail += 1
            failures.append((name, "TIMEOUT"))
            print(f"  FAIL  {name} (timeout)")

    total = n_pass + n_fail
    print(f"\nFoundations: {n_pass}/{total} PASS")

    if failures:
        for name, err in failures:
            print(f"  {name}: {err.strip()[:100]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
