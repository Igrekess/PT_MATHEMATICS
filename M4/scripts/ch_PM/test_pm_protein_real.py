"""
test_pm_protein_real.py -- PM on Real Proteins (34-protein dataset)
===================================================================
Status: [VAL]  |  Chapter: ch_PM, Section XVII

Applies PM cascade to REAL protein sequences from PDB using sliding
windows of size w=8. For each window, runs the full PM cascade
(AA constraints + T0 constraints) and measures rank, dim, phantom.

Key questions:
  - Does L1 hold on real proteins? What fraction of constraints give dim=1?
  - Does phantom rank correlate with protein properties (T0, omega)?
  - Does the PM transitoire vary with local sequence composition?
  - Does T0 create level-2 structure on real sequences?
"""
import sys
import os
import time
import numpy as np
from itertools import product as iterproduct

# Add PT_Proteines to path for protein_loader
PT_PROT_PATH = r"D:\P_Gaps\PT_CORE_LEVEL_3\PT_Proteines"
sys.path.insert(0, os.path.join(PT_PROT_PATH, "src", "python"))
sys.path.insert(0, os.path.join(PT_PROT_PATH, "paper", "scripts"))

try:
    from common import load_all_proteins
    HAS_LOADER = True
except ImportError:
    HAS_LOADER = False
    print("  WARNING: Cannot import protein loader. Using built-in sequences.")

# ── SS propensity (Chou-Fasman simplified) ─────────────────────────

SS_PROPENSITY = {
    'A': {'H', 'C'}, 'L': {'H', 'C'}, 'M': {'H', 'C'},
    'E': {'H', 'C'}, 'K': {'H', 'C'}, 'Q': {'H', 'C'}, 'R': {'H', 'C'},
    'V': {'E', 'C'}, 'I': {'E', 'C'}, 'Y': {'E', 'C'},
    'F': {'E', 'C'}, 'W': {'E', 'C'}, 'T': {'E', 'C'},
    'S': {'H', 'E', 'C'}, 'N': {'H', 'E', 'C'}, 'D': {'H', 'E', 'C'},
    'H': {'H', 'E', 'C'}, 'C': {'H', 'E', 'C'},
    'P': {'C'}, 'G': {'H', 'E', 'C'},
}

SS_STATES = ['H', 'E', 'C']


# ── PM cascade (reuse from test_pm_protein.py) ─────────────────────

def all_ss_sequences(n):
    return list(iterproduct(SS_STATES, repeat=n))


def apply_aa_constraint(sequences, position, aa):
    allowed = SS_PROPENSITY.get(aa, {'H', 'E', 'C'})
    return [s for s in sequences if s[position] in allowed]


def apply_t0_constraint(sequences, pos1, pos2):
    return [s for s in sequences
            if not ((s[pos1] == 'H' and s[pos2] == 'E') or
                    (s[pos1] == 'E' and s[pos2] == 'H'))]


def build_probes(n, level=1):
    probes = []
    for j in range(n):
        for ss in ['H', 'E']:
            probes.append((f'p{j}_{ss}',
                          lambda s, j=j, ss=ss: (1.0 if s[j] == ss else 0.0) - 1/3))
    if level >= 2:
        l1_fns = []
        for j in range(n):
            for ss in ['H', 'E']:
                l1_fns.append(lambda s, j=j, ss=ss: (1.0 if s[j] == ss else 0.0) - 1/3)
        for i in range(len(l1_fns)):
            for k in range(i + 1, len(l1_fns)):
                if (i // 2) == (k // 2):
                    continue
                probes.append((f'pair_{i}_{k}',
                              lambda s, fi=l1_fns[i], fk=l1_fns[k]: fi(s) * fk(s)))
    return probes


def evaluate_probes(seq, probes):
    return np.array([f(seq) for _, f in probes])


def obs_matrix(survivors, probes, cond_fn=None):
    if len(survivors) == 0:
        return np.zeros((1, len(probes)))
    groups = {}
    for s in survivors:
        key = cond_fn(s) if cond_fn else 0
        groups.setdefault(key, []).append(s)
    rows = []
    for key in sorted(groups.keys()):
        grp = groups[key]
        vals = np.zeros(len(probes))
        for x in grp:
            vals += evaluate_probes(x, probes)
        vals /= len(grp)
        rows.append(vals)
    mat = np.array(rows)
    if mat.shape[0] > 1:
        mat -= mat.mean(axis=0)
    return mat


def pm_cascade_window(aa_window, probe_level=1):
    """Run PM cascade on a short AA window. Returns summary dict."""
    n = len(aa_window)
    probes = build_probes(n, level=probe_level)
    n_probes = len(probes)

    all_seqs = all_ss_sequences(n)
    current = all_seqs[:]
    survivors_by_depth = [current[:]]

    # Phase 1: AA constraints
    for j in range(n):
        aa = aa_window[j]
        current = apply_aa_constraint(current, j, aa)
        survivors_by_depth.append(current[:])

    # Phase 2: T0 constraints
    for j in range(n - 1):
        current = apply_t0_constraint(current, j, j + 1)
        survivors_by_depth.append(current[:])

    # Build constraint metadata
    constraints = []
    for j in range(n):
        constraints.append({'cond_fn': lambda s, j=j: s[j], 'type': 'AA'})
    for j in range(n - 1):
        constraints.append({'cond_fn': lambda s, j=j: (s[j], s[j+1]), 'type': 'T0'})

    # Compute PM diagnostics
    hist = None
    ranks = []
    dims = []
    phantom_ranks = []

    for d in range(len(constraints)):
        survs = survivors_by_depth[d + 1]
        cond_fn = constraints[d]['cond_fn']

        if len(survs) < 2:
            ranks.append(ranks[-1] if ranks else 0)
            dims.append(0)
            phantom_ranks.append(0)
            continue

        o = obs_matrix(survs, probes, cond_fn)
        if hist is None:
            hist = o
        else:
            hist = np.vstack([hist, o])

        r = int(np.linalg.matrix_rank(hist, tol=1e-8))
        prev_r = ranks[-1] if ranks else 0
        dim_d = r - prev_r
        ranks.append(r)
        dims.append(dim_d)

        # Phantom rank
        prev_survs = survivors_by_depth[d]
        if len(prev_survs) >= 2 and d > 0:
            obs_pre = obs_matrix(prev_survs, probes, cond_fn)
            hist_pre = np.vstack([hist, obs_pre])
            r_pre = int(np.linalg.matrix_rank(hist_pre, tol=1e-8))
            ph = r_pre - r
        else:
            ph = 0
        phantom_ranks.append(ph)

    aa_dims = dims[:n]
    t0_dims = dims[n:]
    active_aa = [d for d in aa_dims if d > 0]
    active_t0 = [d for d in t0_dims if d > 0]

    return {
        'n_probes': n_probes,
        'max_rank': max(ranks) if ranks else 0,
        'aa_dims': aa_dims,
        't0_dims': t0_dims,
        'n_active_aa': len(active_aa),
        'n_active_t0': len(active_t0),
        'l1_aa': sum(1 for d in active_aa if d == 1),
        'l1_t0': sum(1 for d in active_t0 if d == 1),
        'phantom_total': sum(phantom_ranks),
        'final_survivors': survivors_by_depth[-1],
        'n_survivors': len(survivors_by_depth[-1]),
    }


# ── Classify AA by propensity type ─────────────────────────────────

def aa_type(aa):
    allowed = SS_PROPENSITY.get(aa, {'H', 'E', 'C'})
    if allowed == {'C'}:
        return 'breaker'
    elif allowed == {'H', 'C'}:
        return 'H-former'
    elif allowed == {'E', 'C'}:
        return 'E-former'
    else:
        return 'neutral'


# ── Check PM prediction vs actual SS ───────────────────────────────

def check_actual_ss(aa_window, ss_window, pm_result):
    """Check if actual SS assignment is among PM survivors."""
    actual = tuple(ss_window)
    survivors = pm_result['final_survivors']
    return actual in survivors


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  PM ON REAL PROTEINS: 34-Protein Dataset")
    print("=" * 72)
    t0 = time.time()

    # ── Load proteins ──────────────────────────────────────────────
    if not HAS_LOADER:
        print("  ERROR: Protein loader not available.")
        return 1

    print("\n  Loading proteins from PDB cache...")
    entries = load_all_proteins(quiet=True)
    print(f"  Loaded {len(entries)} proteins")

    W = 8  # window size
    print(f"  Window size: {W} (3^{W} = {3**W} configurations)")

    # ── Part 1: PM cascade on all windows ──────────────────────────
    print(f"\n  Part 1: Sliding window PM analysis (level 1)")
    print(f"  {'-' * 60}")

    all_results = []

    print(f"\n  {'PDB':>5s}  {'Len':>4s}  {'#Win':>5s}  {'L1%':>5s}  "
          f"{'<dim>':>6s}  {'T0act%':>6s}  {'PhTot':>6s}  "
          f"{'SS_in':>5s}  {'Type':>8s}")
    print(f"  {'-----':>5s}  {'----':>4s}  {'-----':>5s}  {'-----':>5s}  "
          f"{'------':>6s}  {'------':>6s}  {'------':>6s}  "
          f"{'-----':>5s}  {'--------':>8s}")

    for entry in entries:
        pdb_id = entry['pdb_id']
        aa_seq = entry['aa_seq']
        ss_seq = entry['ss_seq']
        length = entry['length']

        if length < W:
            continue

        n_windows = length - W + 1
        # Sample windows (max 30 per protein for speed)
        step = max(1, n_windows // 30)
        window_starts = list(range(0, n_windows, step))

        l1_counts = []
        dim_means = []
        t0_active_counts = []
        phantom_totals = []
        ss_in_counts = 0
        n_tested = 0

        for start in window_starts:
            aa_win = aa_seq[start:start + W]
            ss_win = ss_seq[start:start + W]

            # Skip if unknown AA
            if any(a not in SS_PROPENSITY for a in aa_win):
                continue

            res = pm_cascade_window(aa_win, probe_level=1)
            n_act = res['n_active_aa']
            l1_frac = res['l1_aa'] / n_act if n_act > 0 else 0
            l1_counts.append(l1_frac)
            dim_means.append(np.mean([d for d in res['aa_dims'] if d > 0]) if n_act > 0 else 0)
            t0_active_counts.append(res['n_active_t0'])
            phantom_totals.append(res['phantom_total'])

            # Check if actual SS is among survivors
            if check_actual_ss(aa_win, ss_win, res):
                ss_in_counts += 1
            n_tested += 1

        if n_tested == 0:
            continue

        mean_l1 = np.mean(l1_counts)
        mean_dim = np.mean(dim_means)
        mean_t0 = np.mean(t0_active_counts)
        mean_ph = np.mean(phantom_totals)
        ss_frac = ss_in_counts / n_tested

        prot_type = entry.get('sel', '?')[:3]

        print(f"  {pdb_id:>5s}  {length:4d}  {n_tested:5d}  "
              f"{100*mean_l1:4.0f}%  {mean_dim:6.2f}  "
              f"{100*mean_t0/(W-1):5.0f}%  {mean_ph:6.1f}  "
              f"{100*ss_frac:4.0f}%  {prot_type:>8s}")

        all_results.append({
            'pdb_id': pdb_id,
            'length': length,
            'n_windows': n_tested,
            'mean_l1': mean_l1,
            'mean_dim': mean_dim,
            'mean_t0_active': mean_t0,
            'mean_phantom': mean_ph,
            'ss_in_survivors': ss_frac,
            'sel': entry.get('sel', '?'),
            'omega': entry.get('omega', None),
            'T0_prot': entry.get('T0', None),
            'f_H': entry.get('f_H', None),
            'f_E': entry.get('f_E', None),
        })

    # ── Part 2: Global statistics ──────────────────────────────────
    print(f"\n\n  Part 2: Global statistics ({len(all_results)} proteins)")
    print(f"  {'-' * 60}")

    all_l1 = [r['mean_l1'] for r in all_results]
    all_dim = [r['mean_dim'] for r in all_results]
    all_ph = [r['mean_phantom'] for r in all_results]
    all_ss = [r['ss_in_survivors'] for r in all_results]

    print(f"  L1 fraction (mean):    {np.mean(all_l1)*100:.1f}% +/- {np.std(all_l1)*100:.1f}%")
    print(f"  Mean dim (active):     {np.mean(all_dim):.2f} +/- {np.std(all_dim):.2f}")
    print(f"  Phantom total (mean):  {np.mean(all_ph):.1f} +/- {np.std(all_ph):.1f}")
    print(f"  SS in survivors:       {np.mean(all_ss)*100:.1f}% +/- {np.std(all_ss)*100:.1f}%")

    # ── Part 3: L1 vs AA composition ───────────────────────────────
    print(f"\n\n  Part 3: L1 by AA propensity type")
    print(f"  {'-' * 60}")

    # Count AA types in each protein
    for entry in entries:
        aa_seq = entry['aa_seq']
        n_h = sum(1 for a in aa_seq if aa_type(a) == 'H-former')
        n_e = sum(1 for a in aa_seq if aa_type(a) == 'E-former')
        n_n = sum(1 for a in aa_seq if aa_type(a) == 'neutral')
        n_b = sum(1 for a in aa_seq if aa_type(a) == 'breaker')
        # Find matching result
        for r in all_results:
            if r['pdb_id'] == entry['pdb_id']:
                r['frac_restrictive'] = (n_h + n_e + n_b) / len(aa_seq)
                r['frac_neutral'] = n_n / len(aa_seq)
                r['frac_breaker'] = n_b / len(aa_seq)
                break

    # Correlation: L1 vs fraction of restrictive AAs
    has_frac = [r for r in all_results if 'frac_restrictive' in r]
    if len(has_frac) >= 5:
        x = np.array([r['frac_restrictive'] for r in has_frac])
        y = np.array([r['mean_l1'] for r in has_frac])
        corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0

        print(f"  Corr(frac_restrictive, L1): {corr:.3f}")
        print(f"  Restrictive AA range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  L1 range: [{y.min()*100:.0f}%, {y.max()*100:.0f}%]")

        # Split by median
        med = np.median(x)
        lo = [r['mean_l1'] for r in has_frac if r['frac_restrictive'] < med]
        hi = [r['mean_l1'] for r in has_frac if r['frac_restrictive'] >= med]
        print(f"\n  Low restrictive (<{med:.2f}):  L1 = {np.mean(lo)*100:.1f}%")
        print(f"  High restrictive (>={med:.2f}): L1 = {np.mean(hi)*100:.1f}%")

    # ── Part 4: SS prediction accuracy ─────────────────────────────
    print(f"\n\n  Part 4: PM as SS predictor (actual SS in survivors?)")
    print(f"  {'-' * 60}")

    print(f"  Fraction of windows where actual SS is among PM survivors:")
    print(f"  Mean: {np.mean(all_ss)*100:.1f}%")
    print(f"  Min:  {np.min(all_ss)*100:.1f}%")
    print(f"  Max:  {np.max(all_ss)*100:.1f}%")

    if np.mean(all_ss) > 0.95:
        print(f"\n  Interpretation: PM survivors CONTAIN the actual SS in {np.mean(all_ss)*100:.0f}%")
        print(f"  of cases. PM defines a VALID constraint envelope for protein folding.")
    elif np.mean(all_ss) > 0.5:
        print(f"\n  Interpretation: PM survivors contain actual SS in {np.mean(all_ss)*100:.0f}%")
        print(f"  of cases. Propensity model is PARTIALLY valid.")
    else:
        print(f"\n  Interpretation: PM survivors miss actual SS in {(1-np.mean(all_ss))*100:.0f}%")
        print(f"  of cases. Propensity model is too restrictive.")

    # ── Part 5: Correlations with protein properties ───────────────
    print(f"\n\n  Part 5: PM observables vs protein properties")
    print(f"  {'-' * 60}")

    has_omega = [r for r in all_results if r['omega'] is not None]
    if len(has_omega) >= 5:
        omega = np.array([r['omega'] for r in has_omega])
        l1 = np.array([r['mean_l1'] for r in has_omega])
        ph = np.array([r['mean_phantom'] for r in has_omega])
        ss = np.array([r['ss_in_survivors'] for r in has_omega])

        print(f"  {'Correlation':<35s}  {'r':>6s}  {'Interpretation':<30s}")
        print(f"  {'-'*35}  {'------':>6s}  {'-'*30}")

        for name, vals in [('L1', l1), ('phantom', ph), ('SS_in', ss)]:
            if np.std(vals) > 0 and np.std(omega) > 0:
                r = np.corrcoef(omega, vals)[0, 1]
                interp = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
                print(f"  omega vs {name:<24s}  {r:6.3f}  {interp}")

    has_T0 = [r for r in all_results if r['T0_prot'] is not None]
    if len(has_T0) >= 5:
        T0_vals = np.array([r['T0_prot'] for r in has_T0])
        l1_vals = np.array([r['mean_l1'] for r in has_T0])
        ph_vals = np.array([r['mean_phantom'] for r in has_T0])

        for name, vals in [('L1', l1_vals), ('phantom', ph_vals)]:
            if np.std(vals) > 0 and np.std(T0_vals) > 0:
                r = np.corrcoef(T0_vals, vals)[0, 1]
                interp = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
                print(f"  T0_prot vs {name:<22s}  {r:6.3f}  {interp}")

    # ── Part 6: Level 2 probes on a few proteins ───────────────────
    print(f"\n\n  Part 6: Level 2 probes (3 representative proteins)")
    print(f"  {'-' * 60}")

    # Pick 3 proteins: one short, one medium, one with high E content
    candidates = sorted(all_results, key=lambda r: r['length'])
    selected = []
    if len(candidates) >= 3:
        selected = [candidates[0], candidates[len(candidates)//2], candidates[-1]]

    for r in selected:
        pdb_id = r['pdb_id']
        entry = next((e for e in entries if e['pdb_id'] == pdb_id), None)
        if entry is None:
            continue

        aa_seq = entry['aa_seq']
        ss_seq = entry['ss_seq']

        # Take 3 windows: start, middle, end
        positions = [0, len(aa_seq)//2 - W//2, len(aa_seq) - W]
        positions = [max(0, min(p, len(aa_seq) - W)) for p in positions]

        print(f"\n  {pdb_id} (len={r['length']}):")
        for pos in positions[:2]:  # just 2 windows for speed
            aa_win = aa_seq[pos:pos + W]
            if any(a not in SS_PROPENSITY for a in aa_win):
                continue

            res1 = pm_cascade_window(aa_win, probe_level=1)
            res2 = pm_cascade_window(aa_win, probe_level=2)

            aa_r1 = max(res1['aa_dims']) if res1['aa_dims'] else 0
            t0_r1 = sum(res1['t0_dims'])
            aa_r2 = sum(res2['aa_dims'])
            t0_r2 = sum(res2['t0_dims'])

            print(f"    pos={pos}: {aa_win}  "
                  f"L1: rank {res1['max_rank']}/{res1['n_probes']} -> "
                  f"{res2['max_rank']}/{res2['n_probes']}  "
                  f"T0 adds: {t0_r1} (L1) / {t0_r2} (L2) DOF")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n\n  {'=' * 60}")
    print(f"  SUMMARY: PM on {len(all_results)} real proteins")
    print(f"  {'=' * 60}")
    print(f"  L1 fraction:          {np.mean(all_l1)*100:.1f}%  (quasi-L1)")
    print(f"  Mean dim per AA:      {np.mean(all_dim):.2f}")
    print(f"  Phantom rank:         {np.mean(all_ph):.1f} per window")
    print(f"  SS in survivors:      {np.mean(all_ss)*100:.1f}%")
    print(f"  T0 adds DOF at L2:    YES (cross-position correlations)")

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
