#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D00 — Forbidden Transitions Proof: Theorem T1 (Forbidden Transitions) mod 3 in prime gaps.
GENUINE TEST: Verify T[1->1] = T[2->2] = 0 exactly on 100K real primes,
then verify the MECHANISM (mod-6 alternation forces the constraint).
"""
from _primes import generate_primes
import sys

# Part 1: Count all 9 transition types on real primes > 3
N = 100000
primes = [p for p in generate_primes(N) if p > 3]
gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
classes = [g % 3 for g in gaps]

T = [[0]*3 for _ in range(3)]
for i in range(len(classes)-1):
    T[classes[i]][classes[i+1]] += 1

print('Transition matrix on {} consecutive gaps (primes > 3):'.format(len(gaps)))
print('       to_0    to_1    to_2    (n_row)')
for a in range(3):
    n = sum(T[a])
    f = [T[a][b]/n if n > 0 else 0 for b in range(3)]
    print('from_{}: {:7.4f} {:7.4f} {:7.4f}  ({})'.format(a, *f, n))

# EXACT zeros -- this is the theorem
assert T[1][1] == 0, 'FAIL: T[1][1] = {} (must be exactly 0)'.format(T[1][1])
assert T[2][2] == 0, 'FAIL: T[2][2] = {} (must be exactly 0)'.format(T[2][2])
# All other entries must be > 0
for a in range(3):
    for b in range(3):
        if (a, b) not in ((1, 1), (2, 2)):
            assert T[a][b] > 0, 'FAIL: T[{}][{}] = 0 (should be > 0)'.format(a, b)

# Part 2: VERIFY THE MECHANISM -- mod-6 alternation
# All primes > 3 are 1 or 5 mod 6
residues = [p % 6 for p in primes]
assert all(r in (1, 5) for r in residues), 'FAIL: prime not in {1,5} mod 6'

# Class-1 gap (gap = 1 mod 3 = 4 mod 6) can ONLY start from pos 1 mod 6
# Class-2 gap (gap = 2 mod 3 = 2 mod 6) can ONLY start from pos 5 mod 6
# This is WHY T[1][1] = T[2][2] = 0: after class-1 you're at pos 5 (can't do class-1),
# after class-2 you're at pos 1 (can't do class-2).
n_class1, n_class2 = 0, 0
for i in range(len(gaps)):
    c = classes[i]
    r = primes[i] % 6
    if c == 1:
        assert r == 1, 'FAIL: class-1 gap starting from {} mod 6'.format(r)
        n_class1 += 1
    elif c == 2:
        assert r == 5, 'FAIL: class-2 gap starting from {} mod 6'.format(r)
        n_class2 += 1

print('\nMechanism verified on all {} gaps:'.format(len(gaps)))
print('  Class-1 gaps: {} (all from pos 1 mod 6)'.format(n_class1))
print('  Class-2 gaps: {} (all from pos 5 mod 6)'.format(n_class2))
print('  => After class-1 (now at pos 5): class-1 impossible. QED.')
print('  => After class-2 (now at pos 1): class-2 impossible. QED.')

print('\nD00 VERIFIED: T[1][1]=T[2][2]=0 exact on {} primes, mechanism proven.'.format(N))

sys.exit(0)
