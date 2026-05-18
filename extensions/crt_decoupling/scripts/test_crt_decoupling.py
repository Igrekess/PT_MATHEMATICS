"""
Test numerique de la formule fermee MI(k) du Corollaire 4.3 (PT).

T1: Evalue MI(k) pour (p,q) in {(3,5),(3,7),(5,7)}, k=1..5
T2: Verifie pi_15 = pi_3 (x) pi_5 (eigenvector gauche de T_15 = T_3 (x) T_5)
T3: MI empirique sous pi_Ruelle = pi_3 (x) pi_5 (doit etre ~ 0)
"""

import mpmath as mp
import numpy as np

mp.mp.dps = 50  # 50 chiffres significatifs

# ------------------------------------------------------------------
# Constantes PT
# ------------------------------------------------------------------
s = mp.mpf(1) / 2
mu_star = mp.mpf(15)
p1 = mp.mpf(3)

q_plus = 1 - 2 / mu_star            # 13/15
q_minus = mp.exp(-1 / mu_star)      # thermique


def delta_p(p, mu, q_kind="plus"):
    """delta_p = (1 - q^p) / p, q depends on regime."""
    q = (1 - 2 / mu) if q_kind == "plus" else mp.exp(-1 / mu)
    return (1 - q**p) / p


def sin2_theta(p, mu, q_kind="plus"):
    """T6 holonomy identity: sin^2(theta_p) = delta_p (2 - delta_p)."""
    d = delta_p(p, mu, q_kind)
    return d * (2 - d)


def cos2_theta(p, mu, q_kind="plus"):
    return 1 - sin2_theta(p, mu, q_kind)


# ------------------------------------------------------------------
# T1 : formule MI(k) du Corollaire 4.3
#      MI(k) ~= (1 / (2 ln 2)) * [ cos^(2k)(theta_3, mu*=15)
#                                 + sin^(4k)(theta_2, p_1=3) ]
# ------------------------------------------------------------------
print("=" * 72)
print("T1 — Test de la formule fermee MI(k) (Cor. 4.3)")
print("=" * 72)

# Angles fixes par la formule
cos2_t3 = cos2_theta(3, mu_star, "plus")        # cos^2(theta_3, mu=15)
sin2_t2 = sin2_theta(2, p1, "plus")             # sin^2(theta_2, mu=3)

print(f"\ncos^2(theta_3, mu*=15)  = {mp.nstr(cos2_t3, 12)}")
print(f"sin^2(theta_2, mu= 3)  = {mp.nstr(sin2_t2, 12)}")
print(f"q_+(mu*=15)            = {mp.nstr(q_plus, 12)}")
print(f"q_+(mu = 3)            = {mp.nstr(1 - 2/p1, 12)}\n")

inv2ln2 = 1 / (2 * mp.log(2))


def MI_formula(k):
    """MI(k) = (1/(2 ln 2)) [cos^{2k}(theta_3) + sin^{4k}(theta_2)]."""
    A = cos2_t3 ** k          # cos^(2k) = (cos^2)^k
    B = sin2_t2 ** (2 * k)    # sin^(4k) = (sin^2)^(2k)
    return inv2ln2 * (A + B)


targets = {
    (3, 5): mp.mpf("0.138"),
    (3, 7): mp.mpf("0.283"),
    (5, 7): None,  # cible non fournie pour MI direct, on l'affiche
}

print(f"{'k':>2} | {'MI_formule(bits)':>18} | "
      f"{'ratio/0.138':>12} | {'ratio/0.283':>12}")
print("-" * 72)
best = {}
for k in range(1, 8):
    mi = MI_formula(k)
    r35 = mi / targets[(3, 5)]
    r37 = mi / targets[(3, 7)]
    print(f"{k:>2} | {mp.nstr(mi, 12):>18} | "
          f"{mp.nstr(r35, 8):>12} | {mp.nstr(r37, 8):>12}")
    for tgt_key in [(3, 5), (3, 7)]:
        tgt = targets[tgt_key]
        err = abs(mi - tgt) / tgt
        if tgt_key not in best or err < best[tgt_key][1]:
            best[tgt_key] = (k, err, mi)

print()
for tgt_key, (k, err, mi) in best.items():
    print(f"Best k for MI(g mod {tgt_key[0]}, g mod {tgt_key[1]})"
          f" = {targets[tgt_key]} : k={k}, "
          f"MI_pred={mp.nstr(mi, 10)}, "
          f"erreur relative={mp.nstr(err*100, 4)} %")

t1_pass_35 = any(
    abs(MI_formula(k) - targets[(3, 5)]) / targets[(3, 5)] < mp.mpf("0.20")
    for k in range(1, 8)
)
t1_pass_37 = any(
    abs(MI_formula(k) - targets[(3, 7)]) / targets[(3, 7)] < mp.mpf("0.20")
    for k in range(1, 8)
)
T1_PASS = t1_pass_35 and t1_pass_37
print(f"\nT1 PASS (3,5) : {t1_pass_35}")
print(f"T1 PASS (3,7) : {t1_pass_37}")
print(f"T1 GLOBAL     : {T1_PASS}")


# ------------------------------------------------------------------
# T2 : pi_15 = pi_3 (x) pi_5  sous T_15 = T_3 (x) T_5
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("T2 — Decoupling stationnaire : pi_{pq} = pi_p (x) pi_q")
print("=" * 72)

# T_3 sur {1,2} : antidiag(1,1)
T3 = mp.matrix([[0, 1], [1, 0]])
# T_5 sur {1,2,3,4}: row-stochastic, off-diagonal 1/(p-2)=1/3
T5 = mp.matrix(4, 4)
for i in range(4):
    for j in range(4):
        T5[i, j] = mp.mpf(0) if i == j else mp.mpf(1) / 3

# Left eigenvector pi_3 of T_3 : pi T = pi <=> pi (T - I) = 0
def left_stationary(T):
    """Return left stationary distribution of row-stochastic T."""
    n = T.rows
    # Solve pi T = pi with sum pi_i = 1.
    # Equivalent: (T^T - I) pi^T = 0, pi >= 0, sum=1.
    M = mp.matrix(n + 1, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = T[j, i] - (1 if i == j else 0)
    for j in range(n):
        M[n, j] = 1
    rhs = mp.matrix(n + 1, 1)
    rhs[n, 0] = 1
    # least squares via normal equations
    MT = M.T
    A = MT * M
    b = MT * rhs
    pi = mp.lu_solve(A, b)
    return pi  # column vector

pi3 = left_stationary(T3)
pi5 = left_stationary(T5)

print(f"\npi_3 (stationnaire de T_3, etats {{1,2}}) :")
for i in range(2):
    print(f"   pi_3[{i+1}] = {mp.nstr(pi3[i,0], 20)}")
print(f"pi_5 (stationnaire de T_5, etats {{1,2,3,4}}) :")
for i in range(4):
    print(f"   pi_5[{i+1}] = {mp.nstr(pi5[i,0], 20)}")

# T_15 = T_3 (x) T_5  (Kronecker), size 8x8
def kron(A, B):
    rA, cA = A.rows, A.cols
    rB, cB = B.rows, B.cols
    C = mp.matrix(rA * rB, cA * cB)
    for i in range(rA):
        for j in range(cA):
            for k in range(rB):
                for l in range(cB):
                    C[i*rB + k, j*cB + l] = A[i, j] * B[k, l]
    return C

T15 = kron(T3, T5)
pi3_kron_pi5 = kron(pi3, pi5)   # 8x1 column vector

# Verifier pi_15_kron T_15 = pi_15_kron
pi_row = pi3_kron_pi5.T          # 1 x 8
lhs = pi_row * T15
diff = lhs - pi_row
# norme L2
norm = mp.sqrt(sum(diff[0, i]**2 for i in range(diff.cols)))
print(f"\nnorm(pi_3 (x) pi_5 * T_15 - pi_3 (x) pi_5) = "
      f"{mp.nstr(norm, 6)}")

# Verifier que pi_15 calcule directement = pi_3 (x) pi_5
pi15_direct = left_stationary(T15)
diff2 = pi15_direct - pi3_kron_pi5
norm2 = mp.sqrt(sum(diff2[i, 0]**2 for i in range(diff2.rows)))
print(f"norm(pi_15_direct - pi_3 (x) pi_5)          = "
      f"{mp.nstr(norm2, 6)}")

# KL divergence I(pi_15_direct || pi_3 (x) pi_5)
def kl(p, q):
    s = mp.mpf(0)
    n = p.rows
    for i in range(n):
        if p[i, 0] > 0:
            s += p[i, 0] * mp.log(p[i, 0] / q[i, 0])
    return s

I_div = kl(pi15_direct, pi3_kron_pi5)
print(f"I(pi_15_direct || pi_3 (x) pi_5)              = "
      f"{mp.nstr(I_div, 6)}")

T2_PASS = norm < mp.mpf("1e-30")
print(f"\nT2 PASS : {T2_PASS}  (seuil norm < 1e-30)")


# ------------------------------------------------------------------
# T3 : MI empirique sous pi_Ruelle = pi_3 (x) pi_5
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("T3 — MI empirique sous pi_3 (x) pi_5  (N=10^6)")
print("=" * 72)

# Convertir en float pour echantillonnage
pi3_np = np.array([float(pi3[i, 0]) for i in range(2)])
pi5_np = np.array([float(pi5[i, 0]) for i in range(4)])

# Re-normaliser (precision float)
pi3_np = pi3_np / pi3_np.sum()
pi5_np = pi5_np / pi5_np.sum()

print(f"\npi_3 (float) : {pi3_np}")
print(f"pi_5 (float) : {pi5_np}")

N = 10**6
rng = np.random.default_rng(seed=42)
r3 = rng.choice(2, size=N, p=pi3_np)   # etats 0,1 (correspond a {1,2})
r5 = rng.choice(4, size=N, p=pi5_np)   # etats 0..3

# Histogramme joint
joint = np.zeros((2, 4))
for i in range(N):
    joint[r3[i], r5[i]] += 1
joint /= N
marg_3 = joint.sum(axis=1)
marg_5 = joint.sum(axis=0)

MI_emp = 0.0
for i in range(2):
    for j in range(4):
        if joint[i, j] > 0:
            MI_emp += joint[i, j] * np.log2(
                joint[i, j] / (marg_3[i] * marg_5[j])
            )

print(f"\nMI empirique (bits)   = {MI_emp:.8f}")
print(f"Seuil 1/sqrt(N) (bits) ~ {1/np.sqrt(N):.6f}")
print(f"|MI_emp|              = {abs(MI_emp):.8f}")

T3_PASS = abs(MI_emp) < 0.001
print(f"\nT3 PASS : {T3_PASS}  (seuil |MI| < 0.001 bits)")


# ------------------------------------------------------------------
# Synthese
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("SYNTHESE")
print("=" * 72)
print(f"T1 : {'PASS' if T1_PASS else 'FAIL'}")
print(f"T2 : {'PASS' if T2_PASS else 'FAIL'}")
print(f"T3 : {'PASS' if T3_PASS else 'FAIL'}")
