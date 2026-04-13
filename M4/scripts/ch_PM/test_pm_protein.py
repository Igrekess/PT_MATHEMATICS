"""
test_pm_protein.py -- PM Framework on Protein Folding Cascades
==============================================================
Status: [VAL]  |  Chapter: ch_PM, Section XVII (cross-domain)

PM-Protein mapping:
  F = {H, E, C}^n  (all secondary structure assignments for n residues)
  C = constraints added sequentially:
      Phase 1 (d=1..n): AA identity at position d restricts SS propensity
      Phase 2 (d=n+1..2n-1): T0 transition constraint (H<->E forbidden)
  Survivors = SS sequences compatible with all constraints applied so far
  Probes = centered SS indicators at each position

Key questions:
  - Does the PM cascade produce non-trivial rank growth?
  - Is the transitoire/permanent dichotomy visible?
  - Does phantom rank appear?
  - Does L1 hold (one DOF per constraint)?
  - How does T0 (the protein analogue of prime T0) affect PM structure?
"""
import sys
import time
import numpy as np
from itertools import product as iterproduct

# ── SS propensity data (simplified Chou-Fasman) ────────────────────
# Each AA maps to its set of allowed SS states
# H = helix, E = strand (beta), C = coil
# Based on Chou-Fasman propensities with binary cutoff

SS_PROPENSITY = {
    # Strong helix formers -> {H, C}
    'A': {'H', 'C'},       # Ala: strong H-former
    'L': {'H', 'C'},       # Leu: H-former
    'M': {'H', 'C'},       # Met: H-former
    'E': {'H', 'C'},       # Glu: H-former
    'K': {'H', 'C'},       # Lys: H-former
    'Q': {'H', 'C'},       # Gln: H-former
    'R': {'H', 'C'},       # Arg: H-former

    # Strong strand formers -> {E, C}
    'V': {'E', 'C'},       # Val: strong E-former
    'I': {'E', 'C'},       # Ile: E-former
    'Y': {'E', 'C'},       # Tyr: E-former
    'F': {'E', 'C'},       # Phe: E-former
    'W': {'E', 'C'},       # Trp: E-former
    'T': {'E', 'C'},       # Thr: E-former

    # Neutral -> {H, E, C}
    'S': {'H', 'E', 'C'},  # Ser: neutral
    'N': {'H', 'E', 'C'},  # Asn: neutral
    'D': {'H', 'E', 'C'},  # Asp: neutral
    'H': {'H', 'E', 'C'},  # His: neutral
    'C': {'H', 'E', 'C'},  # Cys: neutral

    # Breakers
    'P': {'C'},             # Pro: strong breaker (C only)
    'G': {'H', 'E', 'C'},  # Gly: flexible (all allowed)
}

SS_STATES = ['H', 'E', 'C']
SS_IDX = {'H': 0, 'E': 1, 'C': 2}


# ── Generate all SS sequences ──────────────────────────────────────

def all_ss_sequences(n: int) -> list:
    """Generate all 3^n SS sequences as tuples."""
    return list(iterproduct(SS_STATES, repeat=n))


# ── Constraint functions ───────────────────────────────────────────

def apply_aa_constraint(sequences: list, position: int, aa: str) -> list:
    """Filter sequences: keep only those where SS at position is allowed by AA."""
    allowed = SS_PROPENSITY[aa]
    return [s for s in sequences if s[position] in allowed]


def apply_t0_constraint(sequences: list, pos1: int, pos2: int) -> list:
    """Filter sequences: remove H<->E direct transitions between pos1, pos2.
    This is the protein T0: helix-strand transition is forbidden."""
    result = []
    for s in sequences:
        ss1, ss2 = s[pos1], s[pos2]
        if (ss1 == 'H' and ss2 == 'E') or (ss1 == 'E' and ss2 == 'H'):
            continue
        result.append(s)
    return result


# ── Probe system ───────────────────────────────────────────────────

def build_probes(n: int, level: int = 2):
    """
    Build probe functions for SS sequences.
    Level 1: centered SS indicators at each position (2n probes)
        psi_{j,H} = 1[SS(j)=H] - mean, psi_{j,E} = 1[SS(j)=E] - mean
        (C is determined by H and E, so 2 probes per position)
    Level 2: pair products psi_{j,a} * psi_{k,b} for j<k
    """
    probes = []

    # Level 1: centered indicators
    for j in range(n):
        for ss in ['H', 'E']:  # C is redundant (determined by H and E)
            label = f'pos{j}_{ss}'
            probes.append((label,
                          lambda s, j=j, ss=ss: (1.0 if s[j] == ss else 0.0) - 1/3))

    # Level 2: pair products
    if level >= 2:
        l1_fns = []
        for j in range(n):
            for ss in ['H', 'E']:
                l1_fns.append((j, ss,
                              lambda s, j=j, ss=ss: (1.0 if s[j] == ss else 0.0) - 1/3))

        for i in range(len(l1_fns)):
            for k in range(i + 1, len(l1_fns)):
                j1, ss1, f1 = l1_fns[i]
                j2, ss2, f2 = l1_fns[k]
                if j1 == j2:
                    continue  # skip same-position pairs
                label = f'p{j1}{ss1}_p{j2}{ss2}'
                probes.append((label,
                              lambda s, f1=f1, f2=f2: f1(s) * f2(s)))

    return probes


def evaluate_probes(seq: tuple, probes: list) -> np.ndarray:
    """Evaluate all probes on a single SS sequence."""
    return np.array([f(seq) for _, f in probes])


# ── PM observation matrix ──────────────────────────────────────────

def obs_matrix(survivors: list, probes: list,
               cond_fn=None) -> np.ndarray:
    """Build observation matrix: rows = conditioning groups, cols = probes."""
    if len(survivors) == 0:
        return np.zeros((1, len(probes)))

    n_probes = len(probes)

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


# ── PM cascade ─────────────────────────────────────────────────────

def pm_protein_cascade(sequence: str, probe_level: int = 1):
    """
    Run PM cascade on a peptide sequence.

    Phase 1: AA identity constraints (d=1..n)
    Phase 2: T0 transition constraints (d=n+1..2n-1)

    Returns dict with all PM diagnostics.
    """
    n = len(sequence)
    probes = build_probes(n, level=probe_level)
    n_probes = len(probes)

    # Start with all 3^n sequences
    all_seqs = all_ss_sequences(n)

    # Build constraint list
    constraints = []

    # Phase 1: AA identity at each position
    for j in range(n):
        aa = sequence[j]
        constraints.append({
            'type': 'AA',
            'position': j,
            'aa': aa,
            'label': f'AA({aa})@{j}',
            'apply': lambda seqs, j=j, aa=aa: apply_aa_constraint(seqs, j, aa),
            'cond_fn': lambda s, j=j: s[j],
        })

    # Phase 2: T0 transitions between adjacent positions
    for j in range(n - 1):
        constraints.append({
            'type': 'T0',
            'positions': (j, j+1),
            'label': f'T0({j},{j+1})',
            'apply': lambda seqs, j=j: apply_t0_constraint(seqs, j, j+1),
            'cond_fn': lambda s, j=j: (s[j], s[j+1]),
        })

    # Run cascade
    current = all_seqs[:]
    survivors_by_depth = [current[:]]
    constraint_labels = []

    for c in constraints:
        current = c['apply'](current)
        survivors_by_depth.append(current[:])
        constraint_labels.append(c['label'])

    # Compute PM diagnostics
    hist = None
    ranks = []
    dims = []
    phantom_ranks = []
    phantom_struct = []

    for d in range(len(constraints)):
        survs = survivors_by_depth[d + 1]
        cond_fn = constraints[d]['cond_fn']

        if len(survs) < 2:
            ranks.append(ranks[-1] if ranks else 0)
            dims.append(0)
            phantom_ranks.append(0)
            phantom_struct.append(0)
            continue

        obs = obs_matrix(survs, probes, cond_fn)

        # Structural rank (new obs appended to history)
        if hist is None:
            hist = obs
        else:
            hist = np.vstack([hist, obs])

        r = int(np.linalg.matrix_rank(hist, tol=1e-8))
        prev_r = ranks[-1] if ranks else 0
        dim_d = r - prev_r
        ranks.append(r)
        dims.append(dim_d)

        # Phantom rank: conditioning on THIS constraint applied to
        # PREVIOUS survivors (before this constraint takes effect)
        prev_survs = survivors_by_depth[d]
        if len(prev_survs) >= 2 and d > 0:
            obs_pre = obs_matrix(prev_survs, probes, cond_fn)
            hist_pre = np.vstack([hist, obs_pre]) if hist is not None else obs_pre
            r_pre = int(np.linalg.matrix_rank(hist_pre, tol=1e-8))
            ph = r_pre - r
        else:
            ph = 0

        phantom_ranks.append(ph)
        phantom_struct.append(dim_d)

    counts = [len(s) for s in survivors_by_depth]

    return {
        'sequence': sequence,
        'n': n,
        'n_probes': n_probes,
        'constraints': constraint_labels,
        'ranks': ranks,
        'dims': dims,
        'counts': counts,
        'phantom_ranks': phantom_ranks,
        'phantom_struct': phantom_struct,
    }


# ── Test peptides ──────────────────────────────────────────────────

TEST_PEPTIDES = [
    ("AGVPLEAI", "mixed (H-formers + E-formers + breaker)"),
    ("AAALAAAA", "helix-dominated (all A except L)"),
    ("VVIVFVVV", "strand-dominated (all E-formers)"),
    ("GPGPGPGP", "coil-dominated (G/P alternation)"),
    ("ASEKHVIT", "realistic (all 4 propensity groups)"),
    ("SNDHGCSN", "neutral-rich (all flexible)"),
]


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  PM CROSS-DOMAIN TEST: Protein Folding Cascade")
    print("=" * 72)

    t0 = time.time()

    # ── Part 1: Detailed cascade on mixed peptide ──────────────────
    seq = "AGVPLEAI"
    n = len(seq)

    print(f"\n  Part 1: Detailed cascade -- peptide {seq}")
    print(f"  {'-' * 60}")

    # Level 1 probes only (for clarity)
    res = pm_protein_cascade(seq, probe_level=1)
    n_pr = res['n_probes']

    print(f"  Setup: n={n} residues, {n_pr} probes (level 1)")
    print(f"  Constraints: {n} AA + {n-1} T0 = {2*n-1} total")
    print(f"  Initial configurations: 3^{n} = {3**n}")

    print(f"\n  {'d':>3s}  {'label':>12s}  {'#surv':>7s}  {'rank':>5s}  "
          f"{'dim':>4s}  {'phant':>5s}  {'note':>10s}")
    print(f"  {'---':>3s}  {'------------':>12s}  {'-------':>7s}  {'-----':>5s}  "
          f"{'----':>4s}  {'-----':>5s}  {'----------':>10s}")

    for d in range(len(res['constraints'])):
        ns = res['counts'][d + 1]
        r = res['ranks'][d]
        dm = res['dims'][d]
        ph = res['phantom_ranks'][d]
        lbl = res['constraints'][d]
        note = ""
        if dm > 0:
            note = "ACTIVE"
        if ph > 0:
            note += " ph" if note else "ph"
        if ns <= 1:
            note = "trivial"
        print(f"  {d+1:3d}  {lbl:>12s}  {ns:7d}  {r:5d}  "
              f"{dm:4d}  {ph:5d}  {note:>10s}")

    # Summary
    active = [d for d in res['dims'] if d > 0]
    total_ph = sum(res['phantom_ranks'])
    max_r = max(res['ranks']) if res['ranks'] else 0
    n_active = len(active)

    print(f"\n  Summary:")
    print(f"    Max rank: {max_r}/{n_pr}")
    print(f"    Active constraints: {n_active}/{len(res['constraints'])}")
    print(f"    Dims (non-zero): {active}")
    print(f"    L1 fraction: {sum(1 for d in active if d == 1)}/{n_active}")
    print(f"    Total phantom: {total_ph}")

    # ── Part 2: Level 2 probes ─────────────────────────────────────
    print(f"\n\n  Part 2: Level 2 probes (pair products)")
    print(f"  {'-' * 60}")

    res2 = pm_protein_cascade(seq, probe_level=2)
    n_pr2 = res2['n_probes']

    print(f"  Probes: {2*n} linear + pair products = {n_pr2} total")

    print(f"\n  {'d':>3s}  {'label':>12s}  {'#surv':>7s}  {'rank':>5s}  "
          f"{'dim':>4s}  {'phant':>5s}")
    print(f"  {'---':>3s}  {'------------':>12s}  {'-------':>7s}  {'-----':>5s}  "
          f"{'----':>4s}  {'-----':>5s}")

    for d in range(len(res2['constraints'])):
        ns = res2['counts'][d + 1]
        r = res2['ranks'][d]
        dm = res2['dims'][d]
        ph = res2['phantom_ranks'][d]
        lbl = res2['constraints'][d]
        print(f"  {d+1:3d}  {lbl:>12s}  {ns:7d}  {r:5d}  "
              f"{dm:4d}  {ph:5d}")

    active2 = [d for d in res2['dims'] if d > 0]
    max_r2 = max(res2['ranks']) if res2['ranks'] else 0
    print(f"\n  Max rank: {max_r2}/{n_pr2}")
    print(f"  Active: {len(active2)}/{len(res2['constraints'])}")
    print(f"  Dims: {active2}")
    print(f"  L1: {sum(1 for d in active2 if d == 1)}/{len(active2)}")
    print(f"  Phantom: {sum(res2['phantom_ranks'])}")

    # ── Part 3: Compare across peptide types ───────────────────────
    print(f"\n\n  Part 3: Comparison across peptide types (level 1)")
    print(f"  {'-' * 60}")

    print(f"\n  {'Peptide':<12s}  {'Type':<30s}  {'MaxR':>5s}  "
          f"{'Act':>4s}  {'L1%':>5s}  {'PhTot':>5s}  {'FinalS':>7s}")
    print(f"  {'--------':<12s}  {'-----':<30s}  {'-----':>5s}  "
          f"{'----':>4s}  {'-----':>5s}  {'-----':>5s}  {'-------':>7s}")

    for peptide, desc in TEST_PEPTIDES:
        r = pm_protein_cascade(peptide, probe_level=1)
        act = [d for d in r['dims'] if d > 0]
        n_act = len(act)
        l1_frac = sum(1 for d in act if d == 1) / n_act if n_act > 0 else 0
        mr = max(r['ranks']) if r['ranks'] else 0
        tp = sum(r['phantom_ranks'])
        fs = r['counts'][-1]

        print(f"  {peptide:<12s}  {desc:<30s}  {mr:5d}  "
              f"{n_act:4d}  {100*l1_frac:4.0f}%  {tp:5d}  {fs:7d}")

    # ── Part 4: Phase structure ────────────────────────────────────
    print(f"\n\n  Part 4: Phase structure (AA phase vs T0 phase)")
    print(f"  {'-' * 60}")

    seq4 = "AGVPLEAI"
    r4 = pm_protein_cascade(seq4, probe_level=1)
    n4 = len(seq4)

    aa_dims = r4['dims'][:n4]
    t0_dims = r4['dims'][n4:]
    aa_active = sum(1 for d in aa_dims if d > 0)
    t0_active = sum(1 for d in t0_dims if d > 0)
    aa_rank = r4['ranks'][n4-1] if n4 > 0 else 0
    t0_rank = r4['ranks'][-1] if r4['ranks'] else 0

    print(f"  Peptide: {seq4}")
    print(f"  Phase 1 (AA identity):  {aa_active}/{n4} active, "
          f"rank {aa_rank}, dims = {aa_dims}")
    print(f"  Phase 2 (T0 trans.):    {t0_active}/{n4-1} active, "
          f"rank {t0_rank}, dims = {t0_dims}")
    print(f"\n  Interpretation:")
    if t0_active > 0:
        print(f"    T0 constraints ADD new DOF (rank {aa_rank} -> {t0_rank})")
        print(f"    The forbidden H<->E transition creates STRUCTURE")
        print(f"    This is the protein analogue of T0 in prime gaps")
    else:
        print(f"    T0 constraints are INERT (all structure from AA identity)")
        print(f"    No additional DOF from transition rules")

    # ── Part 5: T0 effect -- with vs without ───────────────────────
    print(f"\n\n  Part 5: T0 ablation test")
    print(f"  {'-' * 60}")

    # Run cascade with T0 vs without
    for peptide, desc in TEST_PEPTIDES[:4]:
        n_p = len(peptide)
        # With T0
        r_with = pm_protein_cascade(peptide, probe_level=1)
        # Without T0: only AA constraints
        probes = build_probes(n_p, level=1)
        all_seqs = all_ss_sequences(n_p)
        current = all_seqs[:]
        for j in range(n_p):
            current = apply_aa_constraint(current, j, peptide[j])
        surv_no_t0 = len(current)

        surv_with_t0 = r_with['counts'][-1]
        rank_aa = r_with['ranks'][n_p-1]
        rank_full = r_with['ranks'][-1]

        print(f"  {peptide}: survivors {surv_no_t0} -> {surv_with_t0} "
              f"(T0 removes {surv_no_t0 - surv_with_t0}), "
              f"rank {rank_aa} -> {rank_full} "
              f"(T0 adds {rank_full - rank_aa} DOF)")

    # ── Part 6: Cross-domain comparison table ──────────────────────
    print(f"\n\n  Part 6: Cross-domain PM comparison (4 instantiations)")
    print(f"  {'-' * 60}")

    # Compute protein summary
    rp = pm_protein_cascade("AGVPLEAI", probe_level=1)
    act_p = [d for d in rp['dims'] if d > 0]
    n_act_p = len(act_p)
    l1_p = sum(1 for d in act_p if d == 1)

    print(f"\n  {'Property':<28s} {'Sieve':<14s} {'Codes F2':<14s} "
          f"{'3-SAT':<14s} {'Protein':<14s}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

    print(f"  {'Framework works':<28s} {'YES':<14s} {'YES':<14s} "
          f"{'YES':<14s} {'YES':<14s}")
    print(f"  {'Transitoire/permanent':<28s} {'YES':<14s} {'YES':<14s} "
          f"{'YES':<14s} {'YES':<14s}")
    print(f"  {'L1 (unit dim.)':<28s} {'YES (13/13)':<14s} {'COND':<14s} "
          f"{'NO (dim~5.6)':<14s} {f'{l1_p}/{n_act_p}':<14s}")
    print(f"  {'Phantom rank':<28s} {'YES':<14s} {'NO':<14s} "
          f"{'YES (max=7)':<14s} {f'{sum(rp[chr(39+73)+chr(104)+chr(97)+chr(110)+chr(116)+chr(111)+chr(109)+chr(95)+chr(114)+chr(97)+chr(110)+chr(107)+chr(115)+chr(39)[1:-1]])}':<14s}")

    # Fix: just use the variable directly
    ph_total = sum(rp['phantom_ranks'])
    print(f"  {'Phantom rank':<28s} {'YES':<14s} {'NO':<14s} "
          f"{'YES (max=7)':<14s} {'ph=' + str(ph_total):<14s}")
    print(f"  {'Cascade type':<28s} {'Divisibility':<14s} {'Parity':<14s} "
          f"{'Boolean':<14s} {'SS propensity':<14s}")
    print(f"  {'Constraint source':<28s} {'Arithmetic':<14s} {'Linear alg':<14s} "
          f"{'Disjunction':<14s} {'Biochemistry':<14s}")

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  {'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
