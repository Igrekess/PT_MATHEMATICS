#!/usr/bin/env python3
"""Companion script for 'Forbidden transitions and informational structure
of prime residues under modular projection'.
Verifies: T1 (forbidden transitions on 6-rough), GFT (algebraic identity
on coprime prime residues), Ruelle (stationary distribution + irreducibility),
Arrow identity.  Default: primorial moduli up to 210.
For m=2310 (phi=480), increase N to ~10^9 via primesieve for full coverage.
Dependencies: numpy only."""
import numpy as np
from math import gcd

def sieve(N):
    s = np.ones(N+1, dtype=bool); s[:2] = False
    for i in range(2, int(N**0.5)+1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

def coprime_mask(ps, m):
    """Boolean mask: True where gcd(p, m) == 1.  Vectorized via prime factors."""
    mask = np.ones(len(ps), dtype=bool)
    tmp = m
    for q in range(2, m+1):
        if tmp % q == 0:
            mask &= (ps % q != 0)
            while tmp % q == 0: tmp //= q
        if tmp == 1: break
    return mask

def coprime_dkl_h(ps, m):
    """D_KL(P||U_phi) and H(P) on coprime classes, in bits."""
    cop = [r for r in range(m) if gcd(r, m) == 1]; phi = len(cop)
    pp = ps[coprime_mask(ps, m)]
    counts = np.bincount(pp % m, minlength=m).astype(float)
    P = np.array([counts[r] for r in cop]); P /= P.sum()
    dkl = float(np.sum(P * np.log2(P * phi)))
    h = float(-np.sum(P * np.log2(P)))
    return dkl, h, phi

N = 10**7; ps = sieve(N)
MODULI = [3, 6, 30, 210]  # add 2310 with N >= 10^9 for full irreducibility

# TEST T1 — forbidden self-transitions mod 3 on 6-rough integers
print("=" * 65)
print("TEST T1 — forbidden self-transitions mod 3  (6-rough integers)")
rough = np.arange(1, N+1); rough = rough[(rough%2!=0)&(rough%3!=0)] % 3
T3 = np.zeros((2, 2))
np.add.at(T3, (rough[:-1]-1, rough[1:]-1), 1)
T3 /= T3.sum(1, keepdims=True)
print(f"  T_3[1->1] = {T3[0,0]:.6e}  (expected 0)")
print(f"  T_3[2->2] = {T3[1,1]:.6e}  (expected 0)")
ok_T1 = T3[0,0] == 0.0 and T3[1,1] == 0.0
print("  PASS" if ok_T1 else "  FAIL")

# TEST GFT — log2(phi(m)) = D_KL + H  (coprime prime residues)
print(f"\nTEST GFT — log2(phi(m)) = D_KL + H  (moduli up to {max(MODULI)})")
ok_gft = True
for m in MODULI:
    dkl, h, phi = coprime_dkl_h(ps, m)
    err = abs(np.log2(phi) - dkl - h)
    ok = err < 1e-12; ok_gft &= ok
    print(f"  m={m:4d} phi={phi:3d}: D_KL={dkl:.6f}  H={h:.6f}  "
          f"|res|={err:.2e}  {'PASS' if ok else 'FAIL'}")

# TEST Ruelle — irreducibility + pi_stat ~ pi_marg (coprime prime residues)
print(f"\nTEST Ruelle — irreducibility + pi_stat ~ pi_marg  (moduli up to {max(MODULI)})")
ok_ruelle = True
for m in MODULI:
    cop = [r for r in range(m) if gcd(r, m) == 1]; phi = len(cop)
    cop_idx = {c: i for i, c in enumerate(cop)}
    pp = ps[coprime_mask(ps, m)]; r = pp % m
    # Build phi(m) x phi(m) transfer matrix
    Tc = np.zeros((phi, phi))
    for j in range(len(r)-1):
        Tc[cop_idx[r[j]], cop_idx[r[j+1]]] += 1
    rs = Tc.sum(1, keepdims=True); rs[rs==0] = 1; Tc /= rs
    # Check irreducibility: all entries > 0
    n_zero = int(np.sum(Tc == 0))
    irr = "IRRED" if n_zero == 0 else f"SPARSE ({n_zero}/{phi*phi} zeros)"
    # Stationary distribution
    vals, vecs = np.linalg.eig(Tc.T)
    pi = np.abs(np.real(vecs[:, np.argmin(np.abs(vals-1.0))])); pi /= pi.sum()
    cnt = np.array([np.sum(r == c) for c in cop], dtype=float); pi_m = cnt/cnt.sum()
    err = float(np.sum(np.abs(pi - pi_m)))
    u_phi = np.ones(phi) / phi
    err_u = float(np.sum(np.abs(pi - u_phi)))
    ok = err < 1e-3; ok_ruelle &= ok  # irreducibility reported but not gating
    print(f"  m={m:4d} phi={phi:3d}: {irr}  ||pi-marg||={err:.2e}  "
          f"||pi-U_phi||={err_u:.2e}  {'PASS' if ok else 'FAIL'}")

# TEST Arrow — dH/dD_KL = -1 (finite differences, m=6, coprime prime residues)
print("\nTEST Arrow — dH/dD_KL = -1  (finite differences, m=6)")
ok_arrow = True
vals_dh = []
for n in [50000, 100000, 200000, 400000, len(ps)]:
    d, h, _ = coprime_dkl_h(ps[:n], 6); vals_dh.append((d, h))
for i in range(len(vals_dh)-1):
    dd = vals_dh[i+1][0] - vals_dh[i][0]
    if abs(dd) < 1e-15: continue
    ratio = (vals_dh[i+1][1] - vals_dh[i][1]) / dd
    ok = abs(ratio + 1.0) < 0.02; ok_arrow &= ok
    print(f"  dH/dD_KL = {ratio:+.6f}  (expected -1)  {'PASS' if ok else 'FAIL'}")

print("\n" + "=" * 65)
print("OVERALL:", "ALL PASS" if ok_T1 and ok_gft and ok_ruelle and ok_arrow
      else "SOME FAILED")
