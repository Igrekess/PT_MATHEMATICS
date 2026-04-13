"""
Shared functions for complex PT scripts (M41--M49).

Complex variable: w_p = (1 - e^{2i*theta_p}) / 2
Circle theorem:   |w - 1/2|^2 = 1/4  (radius s = 1/2)
Key identity:     |w|^2 = Re(w) = sin^2(theta_p)

All derived from s = 1/2 via the modular sieve.
"""

import numpy as np
import sys
import os

# --- Add parent directory to path for pt_constants access ---
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from pt_constants import (
    s, q_stat, q_therm, mu_star,
    delta_p as _delta_p_const,
    sin2_theta as _sin2_const,
)

# ============================================================
# Constants
# ============================================================
MU_STAR = int(mu_star)
PRIMES_ACTIFS = [3, 5, 7]
PRIMES_GHOST = [11, 13]
PRIMES_ALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


# ============================================================
# Fundamental sieve functions (compatible with tool_4x signatures)
# ============================================================
def delta_p(p, q):
    """Algebraic deficit: delta_p = (1 - q^p) / p."""
    return (1.0 - q**p) / p


def sin2(p, q):
    """sin^2(theta_p) = delta*(2-delta)."""
    d = delta_p(p, q)
    return d * (2.0 - d)


def cos2(p, q):
    """cos^2(theta_p) = (1-delta)^2."""
    d = delta_p(p, q)
    return (1.0 - d)**2


def theta_p(p, q):
    """theta_p = arcsin(sqrt(sin^2))."""
    return np.arcsin(np.sqrt(sin2(p, q)))


# ============================================================
# Complex PT functions
# ============================================================
def w_p(p, q):
    """Complex PT variable: w_p = (1 - e^{2i*theta_p}) / 2.
    Re(w) = sin^2, Im(w) = -sin(2*theta)/2, |w|^2 = sin^2.
    """
    th = theta_p(p, q)
    z = np.exp(2j * th)
    return (1.0 - z) / 2.0


def W_complex(q, primes=None):
    """Complex product W = prod(w_p) for given primes.
    |W|^2 = alpha_bare.
    """
    if primes is None:
        primes = PRIMES_ACTIFS
    W = 1.0 + 0j
    for p in primes:
        W *= w_p(p, q)
    return W


def arg_W(q, primes=None):
    """Phase of the complex product: arg(W) = sum(arg(w_p))."""
    W = W_complex(q, primes)
    return np.angle(W)


def chi3(p):
    """Character mod 3: chi_3(p) = 0,+1,-1 according to p mod 3."""
    r = p % 3
    return 0 if r == 0 else (1 if r == 1 else -1)


def force_complex(p, q):
    """Holomorphic force F(w) = i/w - 2i. Pole at w=0, residue i."""
    w = w_p(p, q)
    return 1j / w - 2j


def momentum_complex(p, q):
    """Complex momentum p = dS/dtheta = -cot(theta) - i."""
    th = theta_p(p, q)
    return -np.cos(th) / np.sin(th) - 1j


def liouville_action(p, q):
    """Liouville action L_p = sin^2(theta_p) = |w_p|^2."""
    return sin2(p, q)
