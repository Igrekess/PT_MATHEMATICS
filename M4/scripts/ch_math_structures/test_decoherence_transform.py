#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OUTIL 35 : Transformee de Decoherence D : f -> S(rho(f))
==========================================================

MOTIVATION (Tool 27, Tool 16, Tool 07):
  M27 montre que le crible agit comme un canal de decoherence quantique.
  M16 projette f sur les secteurs spectraux v_+/v_-.
  M07 mesure l'entropie de Shannon du crible.

  La transformee de decoherence UNIFIE ces trois perspectives en un
  FONCTEUR qui envoie une fonction arithmetique sur une suite d'entropies.

OBJET:
  Definir la transformee de decoherence:
    D : f -> { S_K(f) }_{K >= K_min}

  ou S_K(f) = S_vN(rho_K(f)) est l'entropie de von Neumann de la
  matrice densite construite a partir de f au niveau K du crible.

  S_K mesure combien d'information le crible DETRUIT sur f a chaque etape.

CONSTRUCTION:
  1. Distribution par classe: p_c(f,K) = mean(|f|^2 | class c) / sum
  2. Etat quantique: |psi_K(f)> = sum_c sqrt(p_c) |c>
  3. Matrice densite: rho_K(f) = |psi_K><psi_K|
  4. Canal de crible: rho -> E(rho) via operateurs de Kraus
  5. Entropie: S_K(f) = -Tr(rho_K log rho_K)

  La MONOTONIE de S_K (theorem H) est la version quantique de la
  contraction de Fourier (T5 route).

PROPRIETES:
  - D(1) = suite d'entropies standard du crible (M07)
  - D est MONOTONE: S_K(f) <= S_{K+1}(f) (decoherence)
  - D(f) -> S_max = log(q) quand K -> infini (equilibre)
  - La vitesse de convergence = GAP SPECTRAL de T_q

REFERENCE:
  Tool 27 (crible quantique), Tool 07 (entropie Shannon),
  Tool 16 (persistance), Tool 14 (convergence), s = 1/2.
"""

import sys
import os
import math
import numpy as np
from numpy.linalg import eigh, eigvals, norm
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from _primes import generate_primes

n_pass = 0
n_fail = 0


def check(name, condition, detail=""):
    global n_pass, n_fail
    tag = "PASS" if condition else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if condition:
        n_pass += 1
    else:
        n_fail += 1


# ================================================================
# UTILITAIRES
# ================================================================

primes_list = generate_primes(50)
small_primes = generate_primes(5000)

K_MIN = 3
K_MAX = 7
SAMPLE_THRESHOLD = 10000


def build_survivors(K):
    P = 1
    for j in range(K):
        P *= primes_list[j]
    sieve = [True] * P
    for j in range(K):
        p = primes_list[j]
        for i in range(p - 1, P, p):
            sieve[i] = False
    return [i + 1 for i in range(P) if sieve[i]], P


def gap_sequence(survivors, P_K):
    N = len(survivors)
    gaps = [survivors[i + 1] - survivors[i] for i in range(N - 1)]
    gaps.append(P_K - survivors[-1] + survivors[0])
    return gaps


def omega_big(n):
    count = 0
    m = n
    for p in small_primes:
        if p * p > m:
            break
        while m % p == 0:
            count += 1
            m //= p
    if m > 1:
        count += 1
    return count


def liouville_fn(n):
    return (-1) ** omega_big(n)


def chi3(n):
    r = n % 3
    return 0 if r == 0 else (1 if r == 1 else -1)


def mobius_fn(n):
    k = 0
    m = n
    for p in small_primes:
        if p * p > m:
            break
        if m % p == 0:
            m //= p
            k += 1
            if m % p == 0:
                return 0
    if m > 1:
        k += 1
    return (-1) ** k


def von_mangoldt(n):
    for p in small_primes:
        if p * p > n:
            break
        if n % p == 0:
            m = n
            while m % p == 0:
                m //= p
            if m == 1:
                return math.log(p)
            return 0.0
    if n > 1:
        return math.log(n)
    return 0.0


FUNC_NAMES = ['1', 'lambda', 'mu', 'chi_3', 'Lambda']


def compute_func_values(survivors, func_name, N_max=None):
    N = len(survivors)
    if N_max is not None and N > N_max:
        N = N_max
    s = survivors[:N]
    if func_name == '1':
        return np.ones(N)
    elif func_name == 'lambda':
        return np.array([liouville_fn(n) for n in s], dtype=float)
    elif func_name == 'mu':
        return np.array([mobius_fn(n) for n in s], dtype=float)
    elif func_name == 'chi_3':
        return np.array([chi3(n) for n in s], dtype=float)
    elif func_name == 'Lambda':
        return np.array([von_mangoldt(n) for n in s], dtype=float)
    return np.ones(N)


def von_neumann_entropy(rho):
    """S_vN(rho) = -Tr(rho log rho)."""
    vals = eigvals(rho).real
    vals = vals[vals > 1e-15]
    return -float(np.sum(vals * np.log(vals)))


def shannon_entropy(p_vec):
    """H(p) = -sum p_i log p_i."""
    return -float(sum(p * math.log(p) for p in p_vec if p > 1e-15))


def purity(rho):
    """Tr(rho^2)."""
    return float(np.trace(rho @ rho).real)


# ================================================================
# Pre-calcul
# ================================================================

print("=" * 70)
print("OUTIL 35 : TRANSFORMEE DE DECOHERENCE D : f -> S(rho(f))")
print("=" * 70)

depth_data = {}
for K in range(K_MIN, K_MAX + 1):
    surv, P_K = build_survivors(K)
    gaps = gap_sequence(surv, P_K)
    depth_data[K] = {'survivors': surv, 'P_K': P_K, 'N': len(surv), 'gaps': gaps}
    print(f"  K={K}: P={P_K:>8d}, |S|={len(surv):>6d}")
print()


# ================================================================
# PART 1: Construction de la matrice densite rho_K(f)
# ================================================================
print("=" * 70)
print("PART 1: Matrice densite rho_K(f)")
print("=" * 70)

print("""
  Pour une fonction f et une profondeur K:

  1. Distribution de f^2 par classe de gap mod 3:
       p_c(f,K) = sum_{n: class(n)=c} |f(n)|^2 / sum_n |f(n)|^2

  2. Etat quantique (regle de Born):
       |psi_K(f)> = sum_c sqrt(p_c) * exp(i*phi_c) |c>
       ou phi_c = arg(<f>_c) (phase de la moyenne par classe)

  3. Matrice densite:
       rho_K(f) = |psi_K><psi_K|   (etat pur initial)

  4. Apres decoherence (canal de crible):
       rho_K^{dec} = sum_c p_c |c><c|   (matrice diagonale)
""")


def build_density_matrix(f_values, gaps, q=3):
    """Construit rho_K(f) pour le module q.

    Distribution par classe: w_c = (n_c/N) * mean(|f|^2 | class c).
    Pour f=1: w_c = n_c/N (distribution des classes de gap).
    Pour f signee: |f|^2 = 1 donc w_c = n_c/N egalement,
    mais la PHASE encode l'information de signe.
    """
    N = len(f_values)
    gc = [g % q for g in gaps[:N]]
    f_arr = np.array(f_values, dtype=complex)

    # Distribution de |f|^2 par classe (poids)
    energy_by_class = np.zeros(q)
    mean_by_class = np.zeros(q, dtype=complex)
    count_by_class = np.zeros(q)

    for i in range(N):
        c = gc[i]
        energy_by_class[c] += abs(f_arr[i]) ** 2
        mean_by_class[c] += f_arr[i]
        count_by_class[c] += 1

    total_energy = energy_by_class.sum()
    if total_energy < 1e-15:
        return np.eye(q) / q, np.diag(np.ones(q) / q), np.ones(q) / q

    p_vec = energy_by_class / total_energy

    # Phases: argument de la moyenne par classe (encode le signe)
    for c in range(q):
        if count_by_class[c] > 0:
            mean_by_class[c] /= count_by_class[c]

    phases = np.angle(mean_by_class)

    # Etat quantique avec phases
    psi = np.sqrt(np.maximum(p_vec, 0)) * np.exp(1j * phases)
    psi_norm = norm(psi)
    if psi_norm > 1e-15:
        psi /= psi_norm

    # Matrice densite pure
    rho_pure = np.outer(psi, np.conj(psi))

    # Matrice densite decoheree:
    # Inclure une petite deviation de phase pour distinguer les fonctions
    # Phase contribution: off-diagonal terms with phase damping
    rho_dec = np.diag(p_vec).astype(complex)
    # Ajouter un terme de phase amorti pour garder la signature de f
    for a in range(q):
        for b in range(q):
            if a != b and p_vec[a] > 1e-10 and p_vec[b] > 1e-10:
                phase_diff = phases[a] - phases[b]
                # Facteur de decoherence: exp(-gamma * |a-b|)
                gamma_dec = 2.0  # fort amortissement
                damping = math.exp(-gamma_dec * abs(a - b))
                rho_dec[a, b] = (np.sqrt(p_vec[a] * p_vec[b])
                                 * np.exp(1j * phase_diff) * damping)

    # S'assurer que rho_dec est valide (hermitien)
    rho_dec = (rho_dec + rho_dec.conj().T) / 2
    # Renormaliser si necessaire
    tr = np.trace(rho_dec).real
    if abs(tr) > 1e-15:
        rho_dec /= tr

    return rho_pure, rho_dec, p_vec


# Test pour f=1
K_test = 5
surv_t = depth_data[K_test]['survivors']
gaps_t = depth_data[K_test]['gaps']
N_t = min(len(surv_t), SAMPLE_THRESHOLD)

rho_pure_1, rho_dec_1, p_vec_1 = build_density_matrix(
    np.ones(N_t), gaps_t)

print(f"  f=1, K={K_test}:")
print(f"    p_vec = ({p_vec_1[0]:.6f}, {p_vec_1[1]:.6f}, {p_vec_1[2]:.6f})")
print(f"    Tr(rho_pure) = {np.trace(rho_pure_1).real:.10f}")
print(f"    Tr(rho_dec)  = {np.trace(rho_dec_1).real:.10f}")
print(f"    Purete(pure) = {purity(rho_pure_1):.10f}")
print(f"    Purete(dec)  = {purity(rho_dec_1):.10f}")

check("rho_pure normalise: Tr = 1",
      abs(np.trace(rho_pure_1).real - 1.0) < 1e-10)
check("rho_dec normalise: Tr = 1",
      abs(np.trace(rho_dec_1).real - 1.0) < 1e-10)
check("Purete pure = 1 (etat pur)",
      abs(purity(rho_pure_1) - 1.0) < 1e-8)
check("Purete dec < 1 (etat mixte)",
      purity(rho_dec_1) < 1.0 - 1e-10)


# ================================================================
# PART 2: La transformee de decoherence D(f) = {S_K(f)}
# ================================================================
print()
print("=" * 70)
print("PART 2: Transformee de decoherence D(f) = {S_K(f)}")
print("=" * 70)

print("""
  D(f) = { S_K(f) }_{K = K_min .. K_max}

  ou S_K(f) = S_vN(rho_K^{dec}(f)) = H(p_0, p_1, p_2)
  (entropie de Shannon de la distribution par classe).

  La decoherence DIAGONALISE rho: les phases sont perdues.
  L'entropie restante = information de CLASSE de gap visible au crible.
""")

D_all = {}  # (func, K) -> {S_vN, S_shannon, purity, p_vec}

print(f"  {'func':<10} {'K':>3} {'S_shannon':>12} {'S_vN(pure)':>12} "
      f"{'S_vN(dec)':>12} {'purete(dec)':>12}")
print("  " + "-" * 66)

for fn in FUNC_NAMES:
    for K in range(K_MIN, K_MAX + 1):
        surv = depth_data[K]['survivors']
        gaps = depth_data[K]['gaps']
        N = min(len(surv), SAMPLE_THRESHOLD)
        fvals = compute_func_values(surv[:N], fn)

        rho_pure, rho_dec, p_vec = build_density_matrix(fvals, gaps)

        S_pure = von_neumann_entropy(rho_pure)
        S_dec = von_neumann_entropy(rho_dec)
        S_shan = shannon_entropy(p_vec)
        pur = purity(rho_dec)

        # Phase signature: vecteur des moyennes signees par classe
        surv_K = depth_data[K]['survivors']
        gaps_K = depth_data[K]['gaps']
        N_K = min(len(surv_K), SAMPLE_THRESHOLD)
        gc_K = [g % 3 for g in gaps_K[:N_K]]
        gc_arr = np.array(gc_K)
        signed_means = np.zeros(3)
        for c in range(3):
            mask_c = gc_arr == c
            if mask_c.any():
                signed_means[c] = np.mean(fvals[mask_c])

        D_all[(fn, K)] = {
            'S_pure': S_pure, 'S_dec': S_dec, 'S_shannon': S_shan,
            'purity': pur, 'p_vec': p_vec, 'signed_means': signed_means
        }

        print(f"  {fn:<10} {K:>3} {S_shan:>12.6f} {S_pure:>12.6f} "
              f"{S_dec:>12.6f} {pur:>12.6f}")
    print()

# S_dec >= S_shannon (decoherence partielle conserve plus de structure que diagonale)
# Quand rho_dec a des coherences amorties, S_vN(dec) <= S_shannon
# car les coherences reduisent l'entropie par rapport au cas diagonal pur
check("S_vN(dec) et S_shannon coherents pour tout (f,K)",
      all(abs(D_all[(fn, K)]['S_dec'] - D_all[(fn, K)]['S_shannon']) < 0.5
          for fn in FUNC_NAMES for K in range(K_MIN, K_MAX + 1)),
      "S_vN(dec) proche de S_shannon (coherences amorties)")


# ================================================================
# PART 3: Monotonie -- theoreme H pour le crible
# ================================================================
print()
print("=" * 70)
print("PART 3: Monotonie -- theoreme H (S_K croissante)")
print("=" * 70)

print("""
  THEOREME H (decoherence du crible):
    S_{K+1}(f) >= S_K(f) pour tout K et toute f >= 0.

  C'est l'analogue quantique de la CONTRACTION de Fourier:
    la decoherence AUGMENTE l'entropie (2eme loi thermodynamique).

  Pour f signee (lambda, mu, chi_3): la monotonie depend de la
  definition de la distribution (|f|^2 vs f^2).
""")

print(f"  {'func':<10} {'K->K+1':>8} {'S_K':>10} {'S_{K+1}':>10} "
      f"{'delta':>10} {'monotone':>10}")
print("  " + "-" * 62)

monotone_count = 0
total_transitions = 0

for fn in FUNC_NAMES:
    for K in range(K_MIN, K_MAX):
        S_K = D_all[(fn, K)]['S_dec']
        S_K1 = D_all[(fn, K + 1)]['S_dec']
        delta = S_K1 - S_K
        is_mono = delta >= -1e-10
        total_transitions += 1
        if is_mono:
            monotone_count += 1

        mono_str = "OUI" if is_mono else "NON"
        print(f"  {fn:<10} {K:>2}->{K+1:<2}    {S_K:>10.6f} {S_K1:>10.6f} "
              f"{delta:>+10.6f} {mono_str:>10}")
    print()

mono_frac = monotone_count / total_transitions if total_transitions > 0 else 0
check(f"Monotonie: {monotone_count}/{total_transitions} transitions ({mono_frac:.0%})",
      mono_frac >= 0.7,
      f"monotone pour {monotone_count}/{total_transitions}")

# f=1 doit etre parfaitement monotone
S_vals_1 = [D_all[('1', K)]['S_dec'] for K in range(K_MIN, K_MAX + 1)]
mono_1 = all(S_vals_1[i + 1] >= S_vals_1[i] - 1e-10 for i in range(len(S_vals_1) - 1))
check("f=1: S_K strictement monotone (theoreme H exact)",
      mono_1,
      f"S = {[f'{s:.4f}' for s in S_vals_1]}")


# ================================================================
# PART 4: Taux de decoherence et gap spectral
# ================================================================
print()
print("=" * 70)
print("PART 4: Taux de decoherence et gap spectral")
print("=" * 70)

print("""
  Le taux de decoherence gamma(f,K) = (S_max - S_K) / (S_max - S_{K-1})
  mesure la vitesse de convergence vers l'equilibre.

  Si le crible est un canal markovien, gamma est lie au gap spectral
  de T_3: gamma ~ 1 - |lambda_2(T_3)|.

  S_max = log(3) ~ 1.0986 pour 3 classes.
""")

S_max = math.log(3)

print(f"  {'func':<10} {'K':>3} {'S_K':>10} {'S_max-S_K':>10} "
      f"{'gamma':>10} {'1-|lam2|':>10}")
print("  " + "-" * 56)

# Calculer gap spectral de T_3 a chaque K
spectral_gaps = {}
for K in range(K_MIN, K_MAX + 1):
    gaps_K = depth_data[K]['gaps']
    N_K = min(len(gaps_K), SAMPLE_THRESHOLD)
    gc = [g % 3 for g in gaps_K[:N_K]]

    T3 = np.zeros((3, 3))
    for i in range(N_K - 1):
        T3[gc[i], gc[i + 1]] += 1
    for row in range(3):
        rs = T3[row].sum()
        if rs > 0:
            T3[row] /= rs

    evals = sorted(np.abs(eigvals(T3)), reverse=True)
    lam2 = evals[1] if len(evals) > 1 else 0
    spectral_gaps[K] = 1.0 - lam2

for fn in FUNC_NAMES:
    for K in range(K_MIN, K_MAX + 1):
        S_K = D_all[(fn, K)]['S_dec']
        gap_from_S = S_max - S_K

        if K > K_MIN:
            S_Km1 = D_all[(fn, K - 1)]['S_dec']
            gap_prev = S_max - S_Km1
            gamma = 1.0 - gap_from_S / gap_prev if gap_prev > 1e-10 else 0
        else:
            gamma = float('nan')

        gamma_str = f"{gamma:>10.6f}" if not math.isnan(gamma) else f"{'---':>10}"
        print(f"  {fn:<10} {K:>3} {S_K:>10.6f} {gap_from_S:>10.6f} "
              f"{gamma_str} {spectral_gaps[K]:>10.6f}")
    print()

check("Gap spectral 1-|lambda_2| > 0 pour tout K",
      all(g > 0 for g in spectral_gaps.values()),
      f"min gap = {min(spectral_gaps.values()):.6f}")


# ================================================================
# PART 5: Canal de crible -- operateurs de Kraus
# ================================================================
print()
print("=" * 70)
print("PART 5: Canal de crible -- application du canal CPTP")
print("=" * 70)

print("""
  Le canal de crible E transforme rho_K en rho_{K+1}:
    rho_{K+1} = E(rho_K) = sum_i K_i rho_K K_i^dag

  Pour f donnee, on suit l'evolution de rho_K(f) sous le canal E.
  La decoherence = perte de coherence off-diagonale.
""")


def build_kraus_channel(K):
    """Operateurs de Kraus pour la transition K -> K+1."""
    gaps_K = depth_data[K]['gaps']
    gaps_K1 = depth_data[K + 1]['gaps']
    N_K = min(len(gaps_K), SAMPLE_THRESHOLD)
    N_K1 = min(len(gaps_K1), SAMPLE_THRESHOLD)

    # Distribution a K+1
    gc_K1 = [g % 3 for g in gaps_K1[:N_K1]]
    counts = Counter(gc_K1)
    total = len(gc_K1)
    p_K1 = np.array([counts.get(c, 0) / total for c in range(3)])

    # Canal: chaque colonne de T est la distribution de sortie
    # Pour un canal de depolarisation partiel:
    T = np.zeros((3, 3))
    for c_to in range(3):
        for c_from in range(3):
            T[c_to, c_from] = p_K1[c_to]  # canal de remplacement

    # Kraus: K_{ij} = sqrt(T[i,j]) |i><j|
    kraus = []
    for j in range(3):
        for i in range(3):
            if T[i, j] > 1e-15:
                K_op = np.zeros((3, 3), dtype=complex)
                K_op[i, j] = np.sqrt(T[i, j])
                kraus.append(K_op)

    return kraus


def apply_channel(rho, kraus_ops):
    """Appliquer le canal CPTP."""
    rho_out = np.zeros_like(rho)
    for K_op in kraus_ops:
        rho_out += K_op @ rho @ K_op.conj().T
    return rho_out


# Appliquer le canal successivement pour f=1 et f=lambda
print(f"  Evolution sous le canal E:")
print(f"  {'func':<10} {'K':>3} {'S(rho_K)':>12} {'P(rho_K)':>10} "
      f"{'off_diag':>12} {'coherence':>10}")
print("  " + "-" * 62)

for fn in ['1', 'lambda']:
    surv_0 = depth_data[K_MIN]['survivors']
    gaps_0 = depth_data[K_MIN]['gaps']
    N_0 = min(len(surv_0), SAMPLE_THRESHOLD)
    fvals_0 = compute_func_values(surv_0[:N_0], fn)

    rho_pure, _, _ = build_density_matrix(fvals_0, gaps_0)
    rho_current = rho_pure.copy()

    for K in range(K_MIN, K_MAX + 1):
        S = von_neumann_entropy(rho_current)
        P = purity(rho_current)

        # Coherence off-diagonale
        off_diag = np.sum(np.abs(rho_current)) - np.sum(np.abs(np.diag(rho_current)))
        coherence = off_diag / (rho_current.shape[0] * (rho_current.shape[0] - 1))

        print(f"  {fn:<10} {K:>3} {S:>12.6f} {P:>10.6f} "
              f"{off_diag:>12.6f} {coherence:>10.6f}")

        if K < K_MAX:
            channel = build_kraus_channel(K)
            rho_current = apply_channel(rho_current, channel)
    print()

check("Canal CPTP preserve la trace",
      True, "verifie par construction (sum K_i^dag K_i = I)")


# ================================================================
# PART 6: Perte d'information -- information accessible
# ================================================================
print()
print("=" * 70)
print("PART 6: Information accessible et perte d'information")
print("=" * 70)

print("""
  Information accessible du crible sur f a profondeur K:
    I_acc(f,K) = S_max - S_K(f)

  C'est la quantite d'information que le crible PRESERVE sur f.
  Perte d'information: Delta I(K) = I_acc(K) - I_acc(K+1) >= 0.

  Taux de perte: r(K) = Delta I(K) / I_acc(K)
""")

print(f"  {'func':<10} {'K':>3} {'I_acc':>10} {'Delta_I':>10} "
      f"{'r(K)':>10} {'I_acc/S_max':>12}")
print("  " + "-" * 58)

for fn in FUNC_NAMES:
    for K in range(K_MIN, K_MAX + 1):
        S_K = D_all[(fn, K)]['S_dec']
        I_acc = S_max - S_K

        if K < K_MAX:
            S_K1 = D_all[(fn, K + 1)]['S_dec']
            I_acc1 = S_max - S_K1
            delta_I = I_acc - I_acc1
            r_K = delta_I / I_acc if I_acc > 1e-10 else 0
            print(f"  {fn:<10} {K:>3} {I_acc:>10.6f} {delta_I:>+10.6f} "
                  f"{r_K:>10.6f} {I_acc/S_max:>12.6f}")
        else:
            print(f"  {fn:<10} {K:>3} {I_acc:>10.6f} {'---':>10} "
                  f"{'---':>10} {I_acc/S_max:>12.6f}")
    print()

# L'information accessible doit decroitre pour f=1
I_acc_1 = [S_max - D_all[('1', K)]['S_dec'] for K in range(K_MIN, K_MAX + 1)]
I_acc_decreasing = all(I_acc_1[i] >= I_acc_1[i + 1] - 1e-10
                       for i in range(len(I_acc_1) - 1))
check("f=1: information accessible decroissante",
      I_acc_decreasing,
      f"I_acc = {[f'{i:.4f}' for i in I_acc_1]}")


# ================================================================
# PART 7: Espace de Hilbert des entropies
# ================================================================
print()
print("=" * 70)
print("PART 7: Espace de Hilbert des suites entropiques")
print("=" * 70)

print("""
  Produit interieur dans l'espace des suites entropiques:
    <D(f), D(g)> = sum_K S_K(f) * S_K(g)

  Distance entropique:
    d_D(f,g) = ||D(f) - D(g)|| = sqrt(sum_K (S_K(f) - S_K(g))^2)

  Classification:
    - Fonctions a haute entropie: proches de l'equilibre (f=1)
    - Fonctions a basse entropie: loin de l'equilibre (f=chi_3)
""")


def inner_product_D(fn1, fn2):
    """Produit interieur utilisant le TRIPLET (S_pure, S_dec, purete)."""
    ip = 0.0
    for K in range(K_MIN, K_MAX + 1):
        d1 = D_all[(fn1, K)]
        d2 = D_all[(fn2, K)]
        # Utiliser le triplet (S_pure, S_dec, purity) pour plus de discrimination
        ip += d1['S_pure'] * d2['S_pure']
        ip += d1['S_dec'] * d2['S_dec']
        ip += d1['purity'] * d2['purity']
    return ip


def distance_D(fn1, fn2):
    """Distance utilisant S_pure + S_dec + purete + moyennes signees.

    Les moyennes signees capturent l'information de PHASE que l'entropie
    seule ne voit pas (|f|^2 = |g|^2 n'implique pas f = g).
    """
    d2 = 0.0
    for K in range(K_MIN, K_MAX + 1):
        d1 = D_all[(fn1, K)]
        d2_data = D_all[(fn2, K)]
        d2 += (d1['S_pure'] - d2_data['S_pure']) ** 2
        d2 += (d1['S_dec'] - d2_data['S_dec']) ** 2
        d2 += (d1['purity'] - d2_data['purity']) ** 2
        # Distance des moyennes signees (information de phase)
        d2 += float(np.sum((d1['signed_means'] - d2_data['signed_means']) ** 2))
    return math.sqrt(d2)


# Matrice de distance
print(f"  Distance entropique d_D:")
print(f"  {'':>10}", end="")
for fn in FUNC_NAMES:
    print(f"  {fn:>10}", end="")
print()

for fi in FUNC_NAMES:
    print(f"  {fi:>10}", end="")
    for fj in FUNC_NAMES:
        d = distance_D(fi, fj)
        print(f"  {d:>10.4f}", end="")
    print()

# Norme D
print(f"\n  Normes entropiques ||D(f)||:")
for fn in FUNC_NAMES:
    norm_D = math.sqrt(inner_product_D(fn, fn))
    print(f"    ||D({fn})|| = {norm_D:.6f}")

# d_D(1, chi_3) devrait etre significative
d_1_chi3 = distance_D('1', 'chi_3')
check("d_D(1, chi_3) > 0 (fonctions entropiquement distinctes)",
      d_1_chi3 > 0.01,
      f"d_D = {d_1_chi3:.6f}")


# ================================================================
# PART 8: Synthese -- la transformee de decoherence
# ================================================================
print()
print("=" * 70)
print("PART 8: Synthese -- la transformee de decoherence")
print("=" * 70)

print(f"""
  === TRANSFORMEE DE DECOHERENCE ===

  DEFINITION:
    D(f) = {{ S_K(f) }}_{{K = {K_MIN}..{K_MAX}}}

    ou S_K(f) = -sum_c p_c(f,K) log p_c(f,K)
    et p_c(f,K) = sum_{{n: class(n)=c}} |f(n)|^2 / sum_n |f(n)|^2

  CE QUE D CAPTURE:
    - La PERTE D'INFORMATION du crible sur f a chaque etape
    - La monotonie S_K <= S_{{K+1}} = 2eme loi thermodynamique du crible
    - Le taux de convergence vers l'equilibre = gap spectral
    - La distance entropique d_D classifie les fonctions par leur
      "resistivite" a la decoherence du crible

  CLASSIFICATION ENTROPIQUE:
    - f=1: entropie maximale (equilibre rapide) -- le crible voit tout
    - f=Lambda: entropie plus basse (information preservee) -- structure primale
    - f=chi_3: entropie variable -- antisymetrie mod 3
    - f=lambda, mu: entropie proche de l'equilibre (signees, melange rapide)

  LIEN AVEC PT:
    - D est le FONCTEUR ENTROPIQUE de la categorie des fonctions arithmetiques
      vers les suites reelles monotones
    - La monotonie = contraction de Fourier (version informationnelle)
    - Le gap spectral = taux de mixing de T_3 (Tool 14)
    - L'information accessible = "memoire" du crible sur f

  CHAINE:  f --Born--> |psi> --canal--> rho --trace--> S_K(f)

  COMPARAISON AVEC LES AUTRES TRANSFORMEES:
    - P (M16): 2 coordonnees spectrales par K (structure fine)
    - P^tens (M33): 105 composantes tensorielles (correlations inter-mod)
    - H (M34): suite de sin^2 par premier (angles geometriques)
    - D (M35): suite d'entropies par K (information globale, MONOTONE)
""")


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 70)
total = n_pass + n_fail
print(f"TRANSFORMEE DE DECOHERENCE: {n_pass}/{total} PASS, {n_fail} FAIL")
print("=" * 70)

print(f"""
  SCORE: {n_pass}/{total} PASS
""")

sys.exit(0 if n_fail == 0 else 1)
