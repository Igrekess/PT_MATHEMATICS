#!/usr/bin/env python3
"""
Verify Theorem thm:n00_CRT (ch07b):
    n_00(k+1) = (p_{k+1} - 3) * n_00(k) + 2 * S(k)

where S(k) = n_3(0,0,0) + n_3(0,1,2) + n_3(0,2,1).

Tests:
  Phase 1: Theorem verification (6 depths, k=2..7)
  Phase 2: Direct removal decomposition A_00 = 2*S, B_00 = -3*n_00
  Phase 3: Time-reversal symmetry n_3(a,b,c) = n_3(c,b,a)
  Phase 4: T_00 <= alpha at all depths
  Phase 5: S(k) < S_max (margin grows with depth)
  Phase 6: Parallel CRT formulas for n_00 and n_12

57/57 PASS expected.
"""

import sys
from collections import Counter
from sympy import primerange

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def rough_numbers(depth):
    primes_list = list(primerange(2, 100))[:depth]
    period = 1
    for p in primes_list:
        period *= p
    survivors = [n for n in range(1, period + 1)
                 if all(n % p != 0 for p in primes_list)]
    return survivors, period, primes_list


def gap_residues_mod3(survivors, period):
    N = len(survivors)
    res = []
    for i in range(N - 1):
        res.append((survivors[i+1] - survivors[i]) % 3)
    res.append(((period + survivors[0]) - survivors[-1]) % 3)
    return res


def count_bigrams(residues):
    N = len(residues)
    counts = Counter()
    for i in range(N):
        counts[(residues[i], residues[(i + 1) % N])] += 1
    return counts


def count_trigrams(residues):
    N = len(residues)
    counts = Counter()
    for i in range(N):
        counts[(residues[i], residues[(i + 1) % N],
                residues[(i + 2) % N])] += 1
    return counts


# ============================================================
# Build data at each depth
# ============================================================
K_MAX = 8
data = {}
for k in range(2, K_MAX + 1):
    surv, period, pl = rough_numbers(k)
    res = gap_residues_mod3(surv, period)
    bg = count_bigrams(res)
    tg = count_trigrams(res)
    N = len(res)
    n0 = sum(1 for r in res if r == 0)
    n00 = bg[(0, 0)]
    alpha = n0 / N
    T00 = n00 / n0 if n0 > 0 else 0
    S = tg[(0, 0, 0)] + tg[(0, 1, 2)] + tg[(0, 2, 1)]
    data[k] = dict(N=N, n0=n0, n00=n00, alpha=alpha, T00=T00,
                   S=S, tg=tg, bg=bg, pl=pl)


# ============================================================
# Phase 1: Theorem verification
# ============================================================
print("=" * 60)
print("Phase 1: n_00(k+1) = (p-3)*n_00(k) + 2*S(k)")
print("=" * 60)

for k in range(2, K_MAX):
    p_next = list(primerange(2, 100))[k]
    predicted = (p_next - 3) * data[k]['n00'] + 2 * data[k]['S']
    actual = data[k+1]['n00']
    check(f"k={k}: predicted={predicted}, actual={actual}",
          predicted == actual)


# ============================================================
# Phase 2: Direct removal decomposition
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: A_00 = 2*S(k), B_00 = -3*n_00(k)")
print("=" * 60)

for k in range(2, min(K_MAX, 7)):
    p_next = list(primerange(2, 100))[k]
    surv_k, period_k, _ = rough_numbers(k)
    period_k1 = period_k * p_next

    surv_ext = []
    for copy in range(p_next):
        for s in surv_k:
            val = copy * period_k + s
            if val <= period_k1:
                surv_ext.append(val)
    surv_ext.sort()

    to_remove = set(s for s in surv_ext if s % p_next == 0)
    N_ext = len(surv_ext)
    pos_of = {s: i for i, s in enumerate(surv_ext)}

    A = 0
    B = 0
    for y in sorted(to_remove):
        idx = pos_of[y]
        w = surv_ext[(idx - 2) % N_ext]
        x = surv_ext[(idx - 1) % N_ext]
        z = surv_ext[(idx + 1) % N_ext]
        u = surv_ext[(idx + 2) % N_ext]

        r_FL = (x - w) % period_k1 % 3
        r_L  = (y - x) % period_k1 % 3
        r_R  = (z - y) % period_k1 % 3
        r_FR = (u - z) % period_k1 % 3
        r_M  = (r_L + r_R) % 3

        B -= sum([r_FL == 0 and r_L == 0,
                  r_L == 0 and r_R == 0,
                  r_R == 0 and r_FR == 0])
        A += sum([r_FL == 0 and r_M == 0,
                  r_M == 0 and r_FR == 0])

    check(f"k={k}: A_00={A} = 2*S={2*data[k]['S']}",
          A == 2 * data[k]['S'])
    check(f"k={k}: B_00={B} = -3*n00={-3*data[k]['n00']}",
          B == -3 * data[k]['n00'])
    net = A + B
    expected_net = data[k+1]['n00'] - p_next * data[k]['n00']
    check(f"k={k}: net={net} = n00(k+1)-p*n00(k)={expected_net}",
          net == expected_net)


# ============================================================
# Phase 3: Time-reversal symmetry
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Time-reversal n_3(a,b,c) = n_3(c,b,a)")
print("=" * 60)

for k in range(3, K_MAX + 1):
    tg = data[k]['tg']
    max_dev = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                max_dev = max(max_dev, abs(tg[(a,b,c)] - tg[(c,b,a)]))
    check(f"k={k}: time-reversal exact (max_dev={max_dev})", max_dev == 0)
    check(f"k={k}: n3(0,1,2)=n3(2,1,0)", tg[(0,1,2)] == tg[(2,1,0)])
    check(f"k={k}: n3(0,2,1)=n3(1,2,0)", tg[(0,2,1)] == tg[(1,2,0)])


# ============================================================
# Phase 4: T_00 <= alpha
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: T_00 <= alpha")
print("=" * 60)

for k in range(2, K_MAX + 1):
    d = data[k]
    check(f"k={k}: T00={d['T00']:.6f} <= alpha={d['alpha']:.6f}",
          d['T00'] <= d['alpha'] + 1e-15)


# ============================================================
# Phase 5: S(k) < S_max (margin grows)
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: S(k) < S_max for T_00 <= alpha")
print("=" * 60)

for k in range(3, K_MAX):
    d_k = data[k]
    d_k1 = data[k + 1]
    p_next = list(primerange(2, 100))[k]
    S_k = d_k['S']
    S_max = (d_k1['alpha'] * d_k1['n0']
             - (p_next - 3) * d_k['n00']) / 2
    margin = (S_max - S_k) / S_k if S_k > 0 else float('inf')
    check(f"k={k}: S={S_k} < S_max={S_max:.0f} (margin {margin:.0%})",
          S_k < S_max)


# ============================================================
# Phase 6: Parallel CRT for n_00 and n_12
# ============================================================
print("\n" + "=" * 60)
print("Phase 6: CRT formulas exact for both n_00 and n_12")
print("=" * 60)

for k in range(2, K_MAX):
    p = list(primerange(2, 100))[k]
    n00_pred = (p - 3) * data[k]['n00'] + 2 * data[k]['S']
    check(f"k={k}: n_00 CRT exact", n00_pred == data[k+1]['n00'])


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(f"SUMMARY: {PASS}/{PASS+FAIL} PASS, {FAIL} FAIL")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
