"""
test_pm_sat.py -- PM Diagnostics on Random 3-SAT (Cross-Domain Test)
=====================================================================
Status: [VAL]  |  Chapter: ch_PM, Section 9.11 (candidate)
Tests PM invariants (observation matrix, rank, dim, AT) on random 3-SAT
instances as a function of clause/variable ratio.

SAT mapping:
  F = Boolean assignments {0,1}^n
  C = 3-SAT clauses added one by one (constraint cascade)
  Survivors = satisfying assignments of first d clauses
  Probes = centered bit indicators + pair products (analogue of CRT)
    Level 1: psi_j = x_j - 1/2
    Level 2: psi_j * psi_k (cross-variable correlations)

Key questions:
  - Does L1 (unit increment) hold? Under what conditions?
  - Does rank growth show structure near SAT phase transition?
  - How does dim(d) compare to the sieve formula?
"""
import sys
import time
import numpy as np
from itertools import combinations
import random


# ── SAT utilities ────────────────────────────────────────────────────

def random_3sat_clause(n_vars: int, rng: random.Random) -> tuple:
    vars_chosen = rng.sample(range(n_vars), 3)
    signs = [rng.choice([1, -1]) for _ in range(3)]
    return tuple(s * (v + 1) for s, v in zip(signs, vars_chosen))


def clause_satisfied(clause: tuple, assignment: tuple) -> bool:
    for lit in clause:
        var_idx = abs(lit) - 1
        val = assignment[var_idx]
        if (lit > 0 and val == 1) or (lit < 0 and val == 0):
            return True
    return False


def all_assignments(n_vars: int):
    for i in range(2 ** n_vars):
        yield tuple((i >> j) & 1 for j in range(n_vars))


# ── Probe system ─────────────────────────────────────────────────────

def build_probes(n_vars: int, level: int = 2):
    """
    Build probe functions.
    Level 1: psi_j = x_j - 1/2 (n probes)
    Level 2: psi_j * psi_k for j<k (n*(n-1)/2 probes)

    Returns list of (label, function) pairs.
    """
    probes = []
    # Level 1
    for j in range(n_vars):
        probes.append((f'x{j}', lambda x, j=j: x[j] - 0.5))
    # Level 2 (pair products)
    if level >= 2:
        for j in range(n_vars):
            for k in range(j + 1, n_vars):
                probes.append((f'x{j}x{k}',
                              lambda x, j=j, k=k: (x[j] - 0.5) * (x[k] - 0.5)))
    return probes


def evaluate_probes(assignment: tuple, probes: list) -> np.ndarray:
    return np.array([f(assignment) for _, f in probes])


# ── PM observation matrix ────────────────────────────────────────────

def sat_obs_matrix(survivors: list, probes: list,
                   cond_fn=None) -> np.ndarray:
    """
    Build observation matrix for SAT survivors.

    cond_fn: function mapping assignment -> conditioning key.
    Probes evaluated for each group, centered.
    """
    if len(survivors) == 0:
        return np.zeros((1, len(probes)))

    n_probes = len(probes)

    # Group by conditioning
    groups = {}
    for s in survivors:
        key = cond_fn(s) if cond_fn else 0
        groups.setdefault(key, []).append(s)

    rows = []
    for key in sorted(groups.keys()):
        grp = groups[key]
        vals = np.zeros(n_probes)
        for x in grp:
            vals += evaluate_probes(x, probes)
        vals /= len(grp)
        rows.append(vals)

    mat = np.array(rows)
    if mat.shape[0] > 1:
        mat -= mat.mean(axis=0)
    return mat


# ── PM cascade ───────────────────────────────────────────────────────

def pm_cascade(n_vars: int, clauses: list, probe_level: int = 2):
    """
    Run PM cascade on SAT instance.
    Returns (ranks, dims, survivors_counts).
    """
    probes = build_probes(n_vars, level=probe_level)

    all_assigns = list(all_assignments(n_vars))
    current = all_assigns[:]
    survivors_by_depth = [current[:]]

    for clause in clauses:
        current = [s for s in current if clause_satisfied(clause, s)]
        survivors_by_depth.append(current[:])

    hist = None
    ranks = []
    dims = []

    for d in range(len(clauses)):
        survs = survivors_by_depth[d + 1]
        if len(survs) < 2:
            ranks.append(ranks[-1] if ranks else 0)
            dims.append(0)
            continue

        # Conditioning: by values of variables in clause d
        clause_vars = sorted(set(abs(lit) - 1 for lit in clauses[d]))
        cond_fn = lambda x, cv=clause_vars: tuple(x[v] for v in cv)

        obs = sat_obs_matrix(survs, probes, cond_fn)

        if hist is None:
            hist = obs
        else:
            hist = np.vstack([hist, obs])

        r = int(np.linalg.matrix_rank(hist, tol=1e-8))
        prev_r = ranks[-1] if ranks else 0
        ranks.append(r)
        dims.append(r - prev_r)

    counts = [len(s) for s in survivors_by_depth]
    return ranks, dims, counts


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  PM CROSS-DOMAIN TEST: Random 3-SAT (with pair probes)")
    print("=" * 72)

    t0 = time.time()

    # ── Part 1: Detailed cascade ─────────────────────────────────────
    N = 10
    n_probes_1 = N
    n_probes_2 = N + N * (N - 1) // 2  # 10 + 45 = 55

    print(f"\n  Setup: n={N} variables, {n_probes_2} probes "
          f"({N} linear + {N*(N-1)//2} pairs)")
    print(f"  Max rank: {n_probes_2} (vs {N} with linear probes only)")

    for ratio_target in [2.0, 4.0, 4.27, 6.0]:
        n_clauses = max(1, int(ratio_target * N))
        rng = random.Random(42)
        clauses = [random_3sat_clause(N, rng) for _ in range(n_clauses)]

        ranks, dims, counts = pm_cascade(N, clauses, probe_level=2)

        # Find last non-zero dim
        active_dims = [(d + 1, dims[d]) for d in range(len(dims)) if dims[d] > 0]
        total_rank = ranks[-1] if ranks else 0

        l1_info = [d for d in dims if d > 0]
        n_unit = sum(1 for d in l1_info if d == 1)

        print(f"\n  Ratio {ratio_target:.2f} (m={n_clauses}):")
        print(f"    Final rank: {total_rank}/{n_probes_2}, "
              f"survivors: {counts[-1]}")
        print(f"    Dims (non-zero): {l1_info[:20]}")
        print(f"    L1 fraction: {n_unit}/{len(l1_info)} "
              f"({100*n_unit/len(l1_info) if l1_info else 0:.0f}%)")
        print(f"    Active range: clauses 1-{active_dims[-1][0] if active_dims else 0}")

    # ── Part 2: Full detailed output at transition ───────────────────
    print(f"\n\n  Part 2: Full cascade at ratio 4.27")
    print(f"  " + "-" * 60)

    n_clauses = int(4.27 * N)
    rng = random.Random(42)
    clauses = [random_3sat_clause(N, rng) for _ in range(n_clauses)]
    ranks, dims, counts = pm_cascade(N, clauses, probe_level=2)

    print(f"\n  {'d':>3s}  {'#surv':>6s}  {'ratio':>6s}  {'rank':>5s}  "
          f"{'dim':>4s}  {'surv_frac':>9s}")
    print(f"  {'---':>3s}  {'------':>6s}  {'-----':>6s}  {'-----':>5s}  "
          f"{'----':>4s}  {'---------':>9s}")

    for d in range(len(dims)):
        ns = counts[d + 1]
        ratio = (d + 1) / N
        r = ranks[d]
        dm = dims[d]
        frac = ns / 2**N
        marker = " <-- UNSAT" if ns == 0 else ""
        marker += " *" if dm > 0 else ""
        print(f"  {d+1:3d}  {ns:6d}  {ratio:6.2f}  {r:5d}  "
              f"{dm:4d}  {frac:9.4f}{marker}")
        if ns == 0:
            break

    # ── Part 3: Compare level 1 vs level 2 probes ───────────────────
    print(f"\n\n  Part 3: Linear probes vs pair probes")
    print(f"  " + "-" * 60)

    n_clauses = int(4.27 * N)
    rng = random.Random(42)
    clauses = [random_3sat_clause(N, rng) for _ in range(n_clauses)]

    ranks1, dims1, _ = pm_cascade(N, clauses, probe_level=1)
    ranks2, dims2, _ = pm_cascade(N, clauses, probe_level=2)

    nz1 = [d for d in dims1 if d > 0]
    nz2 = [d for d in dims2 if d > 0]

    print(f"  Linear (level 1): max_rank={max(ranks1)}/{N}, "
          f"dims={nz1[:15]}, L1={sum(1 for d in nz1 if d==1)}/{len(nz1)}")
    print(f"  Pairs  (level 2): max_rank={max(ranks2)}/{n_probes_2}, "
          f"dims={nz2[:15]}, L1={sum(1 for d in nz2 if d==1)}/{len(nz2)}")
    print(f"\n  Interpretation:")
    print(f"    Level 1 saturates at n={N} (each clause contributes ~1 "
          f"linear DOF)")
    print(f"    Level 2 reveals CROSS-VARIABLE structure "
          f"(analogue of CRT products)")

    # ── Part 4: Statistical sweep ────────────────────────────────────
    print(f"\n\n  Part 4: Statistical sweep (5 seeds x 11 ratios)")
    print(f"  " + "-" * 60)

    ratios = [1.0, 2.0, 3.0, 3.5, 4.0, 4.27, 4.5, 5.0, 6.0, 8.0, 10.0]
    N_SEEDS = 5

    print(f"\n  {'ratio':>6s}  {'max_rank':>9s}  {'n_active':>9s}  "
          f"{'L1%':>5s}  {'mean_dim':>9s}  {'UNSAT':>5s}")
    print(f"  {'------':>6s}  {'---------':>9s}  {'---------':>9s}  "
          f"{'-----':>5s}  {'---------':>9s}  {'-----':>5s}")

    for ratio in ratios:
        n_cl = max(1, int(ratio * N))
        all_max_r = []
        all_n_active = []
        all_l1 = []
        all_dims_nz = []
        n_unsat = 0

        for seed in range(N_SEEDS):
            rng = random.Random(seed + 300)
            cls = [random_3sat_clause(N, rng) for _ in range(n_cl)]
            rk, dm, cts = pm_cascade(N, cls, probe_level=2)

            if cts[-1] == 0:
                n_unsat += 1
            all_max_r.append(max(rk) if rk else 0)
            nz = [d for d in dm if d > 0]
            all_n_active.append(len(nz))
            all_l1.append(sum(1 for d in nz if d == 1) / len(nz) if nz else 0)
            all_dims_nz.extend(nz)

        print(f"  {ratio:6.2f}  {np.mean(all_max_r):9.1f}  "
              f"{np.mean(all_n_active):9.1f}  "
              f"{100*np.mean(all_l1):4.0f}%  "
              f"{np.mean(all_dims_nz) if all_dims_nz else 0:9.2f}  "
              f"{n_unsat}/{N_SEEDS}")

    # ── Part 5: Comparison table ─────────────────────────────────────
    print(f"\n\n  Part 5: Cross-domain comparison (4 instantiations)")
    print(f"  " + "-" * 60)

    # Compute SAT summary
    rng = random.Random(42)
    cls_rep = [random_3sat_clause(N, rng) for _ in range(int(4.27 * N))]
    rk_rep, dm_rep, ct_rep = pm_cascade(N, cls_rep, probe_level=2)
    nz_rep = [d for d in dm_rep if d > 0]
    l1_n = sum(1 for d in nz_rep if d == 1)
    l1_t = len(nz_rep)

    print(f"\n  {'Property':<28s} {'Eratosthene':<14s} {'Codes F2':<14s} "
          f"{'Lucky':<14s} {'3-SAT':<14s}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

    l1_str = f"{l1_n}/{l1_t}"
    act_str = f"{l1_t}/{len(cls_rep)}"
    sat_str = f"at {max(rk_rep)}"
    print(f"  {'L1 (unit dim.)':<28s} {'YES (13/13)':<14s} "
          f"{'COND':<14s} {'NO':<14s} "
          f"{l1_str:<14s}")
    print(f"  {'Rank saturation':<28s} {'~dim(d)':<14s} "
          f"{'fast':<14s} {'at 12':<14s} "
          f"{sat_str:<14s}")
    print(f"  {'Active clauses':<28s} {'all (L1)':<14s} "
          f"{'partial':<14s} {'partial':<14s} "
          f"{act_str:<14s}")
    print(f"  {'CRT/cross structure':<28s} {'YES':<14s} "
          f"{'NO':<14s} {'NO':<14s} "
          f"{'pair probes':<14s}")
    print(f"  {'Cascade type':<28s} {'Divisibility':<14s} "
          f"{'Parity':<14s} {'Positional':<14s} {'Boolean':<14s}")

    print(f"\n  Key insight: SAT clauses create structure ONLY in the")
    print(f"  first ~{l1_t} clauses. After that, clauses remove")
    print(f"  survivors without adding new observational directions.")
    print(f"  The 'active zone' (clauses contributing new DOFs) is")
    print(f"  ANALOGOUS to the sieve's cycle 1 transitoire.")

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  {'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
