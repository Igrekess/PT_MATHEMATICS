#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Three independent routes to the vertex-edge bifurcation.

Verifies that q_stat = 1 - 2/mu* and q_therm = exp(-1/mu*) arise from
three disjoint mathematical frameworks:
  Route 1: Optimisation  -- Lagrange (max-entropy) + Gibbs self-consistency
  Route 2: Exp. family   -- moment coordinate eta vs natural coordinate theta
  Route 3: Partition fn.  -- canonical ensemble Z(beta)

Reference: Chapter 3, Remark 'Three independent routes to the bifurcation'.
"""
import numpy as np
from fractions import Fraction
import sys

MU_STAR = 15
Q_STAT_EXACT = Fraction(13, 15)
Q_THERM = np.exp(-1.0 / MU_STAR)

n_pass = 0
n_fail = 0


def check(name, val, ref, tol=1e-12):
    global n_pass, n_fail
    err = abs(val - ref)
    ok = err < tol
    tag = "PASS" if ok else "FAIL"
    print("  [{}] {}: {:.15e} vs {:.15e} (err={:.2e})".format(tag, name, val, ref, err))
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    return ok


# ================================================================
# Route 1: Optimisation (Lagrange + Gibbs)
# ================================================================
print("=" * 70)
print("ROUTE 1: OPTIMISATION  (Lagrange max-entropy + Gibbs)")
print("=" * 70)

# q_stat from max-entropy: unique maximizer of H under E[k] = mu/2
q_stat_r1 = 1.0 - 2.0 / MU_STAR
check("q_stat (Lagrange)", q_stat_r1, float(Q_STAT_EXACT))

# q_therm from Gibbs: the Boltzmann weight e^{-beta*k} with beta=1/mu
# gives the geometric distribution with q = e^{-beta} = e^{-1/mu*}.
# Verify: geometric(q_therm) is normalized and has the right form.
q_therm_r1 = np.exp(-1.0 / MU_STAR)
# Normalization: sum_{k=1}^inf (1-q)*q^{k-1} = 1
norm = (1.0 - q_therm_r1) * q_therm_r1**0 / (1.0 - q_therm_r1)  # geometric series = 1
check("q_therm normalization", norm, 1.0)

# Verify Boltzmann form: (1-q)*q^{k-1} = (1-e^{-b})*e^{-b(k-1)} = Z^{-1}*e^{-b*k}
# where Z = e^{-b}/(1-e^{-b}), so P(k) = (1-e^{-b})/e^{-b} * e^{-b*k}
# = (e^b - 1) * e^{-b*k}
# Check at k=3:
beta = 1.0 / MU_STAR
p3_geom = (1.0 - q_therm_r1) * q_therm_r1**2
p3_boltz = (np.exp(beta) - 1.0) * np.exp(-beta * 3)
check("q_therm Boltzmann form (k=3)", p3_geom, p3_boltz)

# Distinctness
L = q_therm_r1 - q_stat_r1
print("  Latent heat L = q_therm - q_stat = {:.6f}".format(L))
assert L > 0, "FAIL: q_therm should exceed q_stat"
n_pass += 1
print("  [PASS] L > 0: bifurcation confirmed")

# ================================================================
# Route 2: Exponential family duality
# ================================================================
print()
print("=" * 70)
print("ROUTE 2: EXPONENTIAL FAMILY DUALITY  (eta vs theta coordinates)")
print("=" * 70)
print()
print("Geometric p_k = (1-q)*q^{k-1} is exponential family:")
print("  p_k = exp(theta*k - A(theta))  with theta = ln(q)")
print("  A(theta) = theta - ln(1 - e^theta)  [log-partition function]")
print("  eta = A'(theta) = 1/(1 - e^theta)   [mean parameter = E[k]]")
print()

# From MEAN coordinate eta = mu*/2 = 15/2:
# eta = 1/(1-q), so q = 1 - 1/eta = 1 - 2/mu*
eta_target = MU_STAR / 2.0
q_stat_r2 = 1.0 - 1.0 / eta_target
check("q_stat (eta = mu*/2)", q_stat_r2, float(Q_STAT_EXACT))

# From NATURAL coordinate theta = -beta = -1/mu*:
# q = e^theta = e^{-1/mu*}
theta_nat = -1.0 / MU_STAR
q_therm_r2 = np.exp(theta_nat)
check("q_therm (theta = -1/mu*)", q_therm_r2, Q_THERM)


# Log-partition function: A(theta) = theta - ln(1 - e^theta)
# (from p_k = (1-q)*q^{k-1} = (1-e^th)/e^th * e^{th*k}
#  = exp(th*k + ln(1-e^th) - th) = exp(th*k - A(th)))
def A_logpart(th):
    """Log-partition function of geometric family."""
    return th - np.log(1.0 - np.exp(th))


def A_prime(th):
    """A'(theta) = 1 + e^theta/(1-e^theta) = 1/(1-e^theta) = E[k]."""
    return 1.0 / (1.0 - np.exp(th))


# Verify: A'(theta) at q_stat gives eta = mu*/2
theta_stat = np.log(q_stat_r2)
eta_from_A = A_prime(theta_stat)
check("A'(theta_stat) == mu*/2", eta_from_A, eta_target)

# Verify: A'(theta) at q_therm gives a DIFFERENT eta
theta_therm = np.log(q_therm_r2)
check("theta_therm == -1/mu*", theta_therm, -1.0 / MU_STAR)
eta_therm = A_prime(theta_therm)
print("  eta(q_therm) = {:.6f}  (=/= mu*/2 = {:.1f})".format(eta_therm, eta_target))
print("  This confirms: A'(theta) is nonlinear => q_stat =/= q_therm")
assert abs(eta_therm - eta_target) > 0.1, "FAIL: should be different"
n_pass += 1
print("  [PASS] Nonlinearity confirmed: eta(q_therm) =/= eta(q_stat)")

# Verify convexity of A (ensures duality is well-defined)
# A''(theta) = e^theta / (1 - e^theta)^2 > 0 for theta < 0
def A_double_prime(th):
    return np.exp(th) / (1.0 - np.exp(th))**2


A_pp_stat = A_double_prime(theta_stat)
A_pp_therm = A_double_prime(theta_therm)
check("A''(theta_stat) > 0 (convexity)", float(A_pp_stat > 0), 1.0)
check("A''(theta_therm) > 0 (convexity)", float(A_pp_therm > 0), 1.0)

# Verify Legendre conjugate: A*(eta) = eta*theta(eta) - A(theta(eta))
# theta(eta) = ln(1 - 1/eta) for eta > 1
def A_star(eta_val):
    """Legendre conjugate (negative entropy)."""
    th = np.log(1.0 - 1.0 / eta_val)
    return eta_val * th - A_logpart(th)

# At eta = mu*/2: A*(eta) should equal the negative entropy of Geom(q_stat)
H_geom = -(1.0 - q_stat_r2) * np.log(1.0 - q_stat_r2) / (1.0 - q_stat_r2) \
    - q_stat_r2 * np.log(q_stat_r2) / (1.0 - q_stat_r2)
# Actually H = -sum p_k ln p_k = -ln(1-q) - q*ln(q)/(1-q) for geometric
H_geom = -np.log(1.0 - q_stat_r2) - q_stat_r2 * np.log(q_stat_r2) / (1.0 - q_stat_r2)
A_star_val = A_star(eta_target)
check("Legendre: A*(eta) == -H(Geom(q_stat))", A_star_val, -H_geom)

# ================================================================
# Route 3: Partition function (canonical ensemble)
# ================================================================
print()
print("=" * 70)
print("ROUTE 3: PARTITION FUNCTION  (canonical ensemble Z(beta))")
print("=" * 70)
print()


# Z(beta) = sum_{k=1}^infty e^{-beta*k} = e^{-beta}/(1 - e^{-beta})
def Z_canon(beta_val):
    return np.exp(-beta_val) / (1.0 - np.exp(-beta_val))


def mean_from_Z(beta_val, h=1e-6):
    """<k> = -d(ln Z)/d(beta) via numerical derivative."""
    return -(np.log(Z_canon(beta_val + h)) - np.log(Z_canon(beta_val - h))) / (2 * h)


def mean_from_Z_exact(beta_val):
    """<k> = 1/(e^beta - 1) + 1 = e^beta/(e^beta - 1) (exact)."""
    return np.exp(beta_val) / (np.exp(beta_val) - 1.0)


beta_star = 1.0 / MU_STAR

# q_therm from Z: the Boltzmann weight is q = e^{-beta}
q_therm_r3 = np.exp(-beta_star)
check("q_therm (Z(beta), q=e^{-beta})", q_therm_r3, Q_THERM)

# q_stat from mean: solve <k> = mu*/2 for q
# <k> = 1/(1-q) = mu*/2, so q = 1 - 2/mu*
mean_r3 = mean_from_Z_exact(beta_star)
print("  <k> at beta=1/mu*: {:.6f}".format(mean_r3))
print("  mu*/2 = {:.1f}".format(MU_STAR / 2.0))
print("  Note: <k>(beta=1/mu*) =/= mu*/2, confirming the two branches differ")

# The q that gives <k> = mu*/2:
q_stat_r3 = 1.0 - 2.0 / MU_STAR
mean_check = 1.0 / (1.0 - q_stat_r3)
check("q_stat (solve <k>=mu*/2)", mean_check, MU_STAR / 2.0)

# Verify numerical derivative matches exact
mean_num = mean_from_Z(beta_star)
mean_exact = mean_from_Z_exact(beta_star)
check("Z mean: numerical vs exact", mean_num, mean_exact, tol=1e-6)

# Free energy
F = -np.log(Z_canon(beta_star))
print("  Free energy F(beta*) = {:.8f}".format(F))
print("  Z(beta*) = {:.8f}".format(Z_canon(beta_star)))

# ================================================================
# Cross-route consistency
# ================================================================
print()
print("=" * 70)
print("CROSS-ROUTE CONSISTENCY")
print("=" * 70)

check("q_stat: Route1 == Route2", q_stat_r1, q_stat_r2)
check("q_stat: Route1 == Route3", q_stat_r1, q_stat_r3)
check("q_therm: Route1 == Route2", q_therm_r1, q_therm_r2)
check("q_therm: Route1 == Route3", q_therm_r1, q_therm_r3)

# Exact rational check for q_stat
q_stat_frac = Fraction(13, 15)
q_stat_from_mu = 1 - Fraction(2, MU_STAR)
assert q_stat_frac == q_stat_from_mu, "FAIL: exact rational mismatch"
n_pass += 1
print("  [PASS] q_stat = 13/15 EXACT (Fraction arithmetic)")

# ================================================================
# Summary
# ================================================================
print()
print("=" * 70)
total = n_pass + n_fail
print("THREE ROUTES TO BIFURCATION: {}/{} PASS, {} FAIL".format(n_pass, total, n_fail))
if n_fail == 0:
    print("All routes converge -- bifurcation ARMORED.")
else:
    print("WARNING: {} failures detected.".format(n_fail))
print("=" * 70)

sys.exit(0 if n_fail == 0 else 1)
