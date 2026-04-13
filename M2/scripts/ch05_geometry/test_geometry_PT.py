#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test wrapper for geometry proofs (Ch.05).

Runs the geometry uniqueness tests:
  test_T6_G2_uniqueness.py   7/7   D_KL uniqueness (Shore-Johnson)
  test_T6_G5_cencov.py       7/7   Fisher uniqueness (Cencov)

These scripts live in ch02_uniqueness/ (T6 programme) but are the
core geometry content of Ch.05.
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CH02_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "ch02_uniqueness")

SCRIPTS = [
    # (directory, script_name)
    (CH02_DIR, "test_T6_G2_uniqueness.py"),
    (CH02_DIR, "test_T6_G5_cencov.py"),
]


def main():
    n_pass = 0
    n_fail = 0
    failures = []

    for script_dir, name in SCRIPTS:
        path = os.path.join(script_dir, name)
        if not os.path.exists(path):
            print(f"  SKIP  {name} (not found)")
            continue
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=300,
                cwd=script_dir,
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
    print(f"\nGeometry: {n_pass}/{total} PASS")

    if failures:
        for name, err in failures:
            print(f"  {name}: {err.strip()[:100]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
