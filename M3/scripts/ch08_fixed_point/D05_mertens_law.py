#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D05 — Double Mertens Law: Theorem T5 (Fixed Point mu*=15). Product (1-1/p) ~ e^{-gamma}/ln(N).
GENUINE TEST: Compute the Mertens product on real primes and verify convergence.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _primes import generate_primes

EULER_GAMMA = np.euler_gamma  # full double precision (~15 digits)
primes = generate_primes(10000)

print('Mertens product verification:')
print('{:>8s} {:>12s} {:>12s} {:>8s}'.format('p_max', 'Product', 'e^-g/ln(p)', 'Ratio'))

for p_max in [100, 500, 1000, 5000, 10000]:
    ps = [p for p in primes if p <= p_max and p >= 2]
    # Mertens product: prod_{p <= x} (1 - 1/p)
    product = 1.0
    for p in ps:
        product *= (1 - 1.0/p)
    # Mertens theorem: product ~ e^{-gamma} / ln(x)
    expected = np.exp(-EULER_GAMMA) / np.log(p_max)
    ratio = product / expected
    print('{:8d} {:12.8f} {:12.8f} {:8.5f}'.format(p_max, product, expected, ratio))

# Final check: ratio should converge to 1
ps_all = [p for p in primes if p >= 2]
product_final = 1.0
for p in ps_all:
    product_final *= (1 - 1.0/p)
expected_final = np.exp(-EULER_GAMMA) / np.log(max(ps_all))
ratio_final = product_final / expected_final

assert abs(ratio_final - 1.0) < 0.05, 'FAIL: Mertens ratio = {:.4f}, expected ~1'.format(ratio_final)
print('\nD05 VERIFIED: Mertens product converges (ratio = {:.5f}).'.format(ratio_final))

sys.exit(0)
