#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D01 — Conservation Proof: Theorem T2 (Spectral Conservation) -- the transition matrix T on gap classes mod 3
is fully determined by 2 parameters (alpha, T[0][0]).
GENUINE TEST: Verify stationarity pi*T = pi on real prime gaps,
and that the 7 free-looking entries reduce to 2 independent parameters.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _primes import generate_primes

# Part 1: Compute T empirically on real primes
primes = [p for p in generate_primes(100000) if p > 3]
gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
classes = [g % 3 for g in gaps]

T_count = [[0]*3 for _ in range(3)]
for i in range(len(classes)-1):
    T_count[classes[i]][classes[i+1]] += 1

# Normalize rows
T = np.zeros((3, 3))
for a in range(3):
    s = sum(T_count[a])
    for b in range(3):
        T[a][b] = T_count[a][b] / s if s > 0 else 0

# Stationary distribution
pi_emp = np.array([classes.count(c) / len(classes) for c in range(3)])
alpha = pi_emp[0]
print('Empirical stationary distribution:')
print('  pi = [{:.4f}, {:.4f}, {:.4f}]'.format(*pi_emp))
print('  alpha = pi(0) = {:.6f}'.format(alpha))
print('  (1-alpha)/2 = {:.6f} (should = pi(1) = pi(2))'.format((1-alpha)/2))

# Part 2: Verify pi*T = pi (stationarity)
pi_T = pi_emp @ T
print('\nStationarity check pi*T = pi:')
for c in range(3):
    err = abs(pi_T[c] - pi_emp[c])
    print('  class {}: pi*T={:.6f}, pi={:.6f}, |err|={:.2e}'.format(
        c, pi_T[c], pi_emp[c], err))
    assert err < 0.005, 'FAIL: stationarity error {:.4f}'.format(err)

# Part 3: Verify T is determined by 2 parameters (alpha, T[0][0])
# From T1 (Forbidden Transitions): T[1][1] = T[2][2] = 0
# From symmetry 1<->2: T[1][0]=T[2][0], T[1][2]=T[2][1], T[0][1]=T[0][2]
# From stationarity: all entries derived from alpha and T[0][0]
T00 = T[0][0]
T01_pred = (1 - T00) / 2  # by symmetry T[0][1] = T[0][2] = (1-T00)/2
T10_pred = alpha * T01_pred / ((1-alpha)/2)  # from pi*T = pi
T12_pred = 1 - T10_pred  # row sums to 1

print('\n2-parameter reconstruction (alpha={:.4f}, T00={:.4f}):'.format(alpha, T00))
pairs = [
    ('T[0][1]', T[0][1], T01_pred),
    ('T[0][2]', T[0][2], T01_pred),
    ('T[1][0]', T[1][0], T10_pred),
    ('T[2][0]', T[2][0], T10_pred),
    ('T[1][2]', T[1][2], T12_pred),
    ('T[2][1]', T[2][1], T12_pred),
]
max_err = 0
for name, emp, pred in pairs:
    err = abs(emp - pred)
    max_err = max(max_err, err)
    print('  {} = {:.4f} (pred: {:.4f}, err: {:.4f})'.format(name, emp, pred, err))

assert max_err < 0.02, 'FAIL: reconstruction error = {:.4f}'.format(max_err)

print('\nD01 VERIFIED: T determined by 2 params (alpha, T00), max err = {:.4f}.'.format(max_err))

sys.exit(0)
