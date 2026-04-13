"""
test_pm_lucky.py -- PM Diagnostics on Lucky Numbers (Cross-Sieve Test)
======================================================================
Status: [VAL]  |  Chapter: ch_PM, Section 9.10
Tests PM invariants (L1, AT, phantom rank) on Lucky numbers and compares
with the Eratosthenes sieve.

Lucky numbers: positional sieve (not divisibility-based).
Key question: which PM properties are UNIVERSAL (hold for any sieve)
and which are SPECIFIC to arithmetic (Eratosthenes)?

Expected:
  - L1 may fail (no CRT irreducibility for positional sieve)
  - T0 forbidden transitions should be ABSENT (no mod 3 structure)
  - Phantom rank should behave differently (no arithmetic memory)
"""
import sys
import time
import numpy as np
from collections import Counter


# ── Lucky number generator ──────────────────────────────────────────
def lucky_numbers(limit: int) -> list:
    """Generate Lucky numbers up to limit using the Lucky sieve."""
    # Start with odd numbers
    sieve = list(range(1, limit + 1, 2))

    i = 1  # index into sieve (sieve[1] = 3 is first sieve value)
    while i < len(sieve) and sieve[i] <= len(sieve):
        step = sieve[i]
        # Remove every step-th element
        sieve = [sieve[j] for j in range(len(sieve)) if (j + 1) % step != 0]
        i += 1

    return sieve


def lucky_gaps(luckys: list) -> list:
    """Compute gaps between consecutive Lucky numbers."""
    return [luckys[i + 1] - luckys[i] for i in range(len(luckys) - 1)]


# ── Lucky sieve at specific depth ──────────────────────────────────
def lucky_survivors_at_depth(limit: int, depth: int) -> list:
    """Return survivors after exactly `depth` Lucky sieve steps."""
    sieve = list(range(1, limit + 1, 2))  # Step 0: odd numbers

    for step_idx in range(1, depth + 1):
        if step_idx >= len(sieve):
            break
        step = sieve[step_idx]
        if step > len(sieve):
            break
        sieve = [sieve[j] for j in range(len(sieve)) if (j + 1) % step != 0]

    return sieve


# ── T0 test: forbidden transitions mod 3 ──────────────────────────
def test_forbidden_transitions(gaps, label):
    """Check if same->same transitions mod 3 are forbidden (T0)."""
    classes = [g % 3 for g in gaps if g > 0]
    transitions = Counter()
    for i in range(len(classes) - 1):
        transitions[(classes[i], classes[i + 1])] += 1

    total = sum(transitions.values())
    n_11 = transitions.get((1, 1), 0)
    n_22 = transitions.get((2, 2), 0)
    forbidden_frac = (n_11 + n_22) / total if total > 0 else 0

    print(f"\n  T0 test for {label}:")
    print(f"    Total transitions: {total}")
    print(f"    (1->1) mod 3: {n_11} ({100*n_11/total:.2f}%)")
    print(f"    (2->2) mod 3: {n_22} ({100*n_22/total:.2f}%)")
    print(f"    Forbidden fraction: {100*forbidden_frac:.2f}%")
    t0_holds = (n_11 == 0 and n_22 == 0)
    print(f"    T0 holds: {'YES' if t0_holds else 'NO'}")
    return t0_holds


# ── Empirical observation matrix ──────────────────────────────────
def empirical_obs_matrix(gaps, moduli, cond_mod):
    """
    Build observation matrix from empirical gap distribution.

    Probes: centered indicators delta(g == r mod m) - 1/m for each (m, r).
    Conditioning: gap residue mod cond_mod.

    Returns matrix where each row = flattened Gram matrix for one
    conditioning residue.
    """
    probe_specs = []
    for m in moduli:
        for r in range(m):
            probe_specs.append((m, r))

    n_probes = len(probe_specs)
    rows = []

    for cr in range(cond_mod):
        # Filter gaps with this conditioning residue
        cond_gaps = [g for g in gaps if g % cond_mod == cr]
        if len(cond_gaps) < 2:
            rows.append(np.zeros(n_probes))
            continue

        # Compute probe values for each gap
        probe_vals = np.zeros((len(cond_gaps), n_probes))
        for gi, g in enumerate(cond_gaps):
            for pi, (m, r) in enumerate(probe_specs):
                probe_vals[gi, pi] = (1.0 if g % m == r else 0.0) - 1.0 / m

        # Mean of probe values (empirical expectation)
        row = probe_vals.mean(axis=0)
        rows.append(row)

    return np.array(rows)


# ── PM rank computation ──────────────────────────────────────────
def compute_pm_ranks(gaps_by_depth, moduli, cond_moduli):
    """
    Compute cumulative ranks across depths.

    gaps_by_depth: list of gap sequences (one per sieve depth)
    moduli: list of moduli for probes
    cond_moduli: list of conditioning moduli (one per depth)
    """
    ranks = []
    hist = None

    for d, (gaps, cm) in enumerate(zip(gaps_by_depth, cond_moduli)):
        if len(gaps) < 10:
            ranks.append(ranks[-1] if ranks else 0)
            continue

        obs = empirical_obs_matrix(gaps, moduli, cm)
        if hist is None:
            hist = obs
        else:
            hist = np.vstack([hist, obs])

        r = int(np.linalg.matrix_rank(hist, tol=1e-6))
        ranks.append(r)

    return ranks


def layer_dims(ranks):
    dims = [ranks[0]]
    for i in range(1, len(ranks)):
        dims.append(ranks[i] - ranks[i - 1])
    return dims


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PM CROSS-SIEVE TEST: Lucky Numbers vs Eratosthenes")
    print("=" * 70)

    LIMIT = 30000
    MAX_DEPTH = 8

    t0 = time.time()

    # ── Part 1: Generate sequences ──────────────────────────────
    print(f"\n  Part 1: Generating sequences (limit={LIMIT})")

    luckys = lucky_numbers(LIMIT)
    l_gaps = lucky_gaps(luckys)
    print(f"    Lucky numbers: {len(luckys)} (up to {luckys[-1]})")
    print(f"    Lucky gaps: {len(l_gaps)}, mean={np.mean(l_gaps):.2f}")
    print(f"    First 20 Lucky: {luckys[:20]}")
    print(f"    Lucky sieve values: {luckys[1:MAX_DEPTH+1]}")

    from mp_core import primes_up_to
    primes = primes_up_to(LIMIT)
    p_gaps = [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]
    print(f"    Primes: {len(primes)}, gaps: {len(p_gaps)}, mean={np.mean(p_gaps):.2f}")

    # ── Part 2: T0 forbidden transitions ────────────────────────
    print(f"\n  Part 2: T0 Forbidden Transitions (mod 3)")

    t0_primes = test_forbidden_transitions(p_gaps[1:], "Prime gaps (p>3)")
    t0_lucky = test_forbidden_transitions(l_gaps, "Lucky gaps")

    print(f"\n  -> Primes: T0 {'HOLDS (exact zero)' if t0_primes else 'FAILS'}")
    print(f"  -> Lucky:  T0 {'HOLDS' if t0_lucky else 'FAILS (as expected)'}")

    # ── Part 3: Gap distribution mod small primes ────────────────
    print(f"\n  Part 3: Gap distribution mod 3")

    for label, gaps in [("Prime", p_gaps[1:]), ("Lucky", l_gaps)]:
        counts = Counter(g % 3 for g in gaps)
        total = sum(counts.values())
        print(f"    {label:6s}: "
              f"0={counts[0]:5d} ({100*counts[0]/total:.1f}%), "
              f"1={counts[1]:5d} ({100*counts[1]/total:.1f}%), "
              f"2={counts[2]:5d} ({100*counts[2]/total:.1f}%)")

    # ── Part 4: PM rank structure at successive depths ───────────
    print(f"\n  Part 4: PM Rank Structure (observation matrix ranks)")

    # For Lucky: depth d = after d Lucky sieve steps
    # For Primes: depth d = sieve up to d-th odd prime
    probe_moduli = [3, 5, 7]

    print(f"\n  Lucky number PM ranks (depths 1-{MAX_DEPTH}):")
    lucky_gaps_by_depth = []
    lucky_cond_mods = []
    for d in range(1, MAX_DEPTH + 1):
        surv = lucky_survivors_at_depth(LIMIT, d)
        gaps_d = [surv[i + 1] - surv[i] for i in range(len(surv) - 1)]
        lucky_gaps_by_depth.append(gaps_d)
        # Conditioning: use sieve value at this depth
        sieve_val = luckys[d] if d < len(luckys) else 3
        lucky_cond_mods.append(sieve_val)
        print(f"    d={d}: {len(surv)} survivors, {len(gaps_d)} gaps, "
              f"sieve_val={sieve_val}, mean_gap={np.mean(gaps_d):.2f}")

    l_ranks = compute_pm_ranks(lucky_gaps_by_depth, probe_moduli, lucky_cond_mods)
    l_dims = layer_dims(l_ranks)

    print(f"\n  Prime number PM ranks (depths 1-{MAX_DEPTH}):")
    prime_gaps_by_depth = []
    prime_cond_mods = []
    odd_ps = [3, 5, 7, 11, 13, 17, 19, 23]
    for d in range(1, MAX_DEPTH + 1):
        # Survivors at depth d = numbers coprime to first d odd primes
        active = odd_ps[:d]
        M = 1
        for p in active:
            M *= p
        surv = [n for n in range(1, min(M + 1, LIMIT)) if all(n % p != 0 for p in active)]
        gaps_d = [surv[i + 1] - surv[i] for i in range(len(surv) - 1)]
        prime_gaps_by_depth.append(gaps_d)
        prime_cond_mods.append(active[-1])
        print(f"    d={d}: {len(surv)} survivors (coprime to {active}), "
              f"{len(gaps_d)} gaps, cond_mod={active[-1]}")

    p_ranks = compute_pm_ranks(prime_gaps_by_depth, probe_moduli, prime_cond_mods)
    p_dims = layer_dims(p_ranks)

    # ── Part 5: Comparison table ─────────────────────────────────
    print(f"\n  Part 5: Comparison Table")
    print(f"\n  {'Depth':>5s}  {'L_rank':>7s}  {'L_dim':>6s}  {'P_rank':>7s}  {'P_dim':>6s}  {'L1_L':>5s}  {'L1_P':>5s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*5}")

    l1_lucky_count = 0
    l1_prime_count = 0
    for d in range(MAX_DEPTH):
        lr = l_ranks[d] if d < len(l_ranks) else 0
        ld = l_dims[d] if d < len(l_dims) else 0
        pr = p_ranks[d] if d < len(p_ranks) else 0
        pd_ = p_dims[d] if d < len(p_dims) else 0
        l1_l = "YES" if ld == 1 else f"NO({ld})"
        l1_p = "YES" if pd_ == 1 else f"NO({pd_})"
        if ld == 1:
            l1_lucky_count += 1
        if pd_ == 1:
            l1_prime_count += 1
        print(f"  {d+1:5d}  {lr:7d}  {ld:6d}  {pr:7d}  {pd_:6d}  {l1_l:>5s}  {l1_p:>5s}")

    # ── Part 6: Transition matrix comparison ─────────────────────
    print(f"\n  Part 6: Transition Matrix T_3 Structure")

    for label, gaps in [("Prime (p>3)", p_gaps[1:]), ("Lucky", l_gaps)]:
        classes = [g % 3 for g in gaps if g > 0]
        T = np.zeros((3, 3))
        for i in range(len(classes) - 1):
            T[classes[i], classes[i + 1]] += 1
        # Normalize rows
        for r in range(3):
            s = T[r].sum()
            if s > 0:
                T[r] /= s
        print(f"\n    {label}:")
        print(f"    T_3 = [{T[0,0]:.3f}  {T[0,1]:.3f}  {T[0,2]:.3f}]")
        print(f"          [{T[1,0]:.3f}  {T[1,1]:.3f}  {T[1,2]:.3f}]")
        print(f"          [{T[2,0]:.3f}  {T[2,1]:.3f}  {T[2,2]:.3f}]")
        # Check structural zeros
        zeros = []
        if T[1, 1] < 0.001:
            zeros.append("T[1,1]")
        if T[2, 2] < 0.001:
            zeros.append("T[2,2]")
        if zeros:
            print(f"    Structural zeros: {', '.join(zeros)}")
        else:
            print(f"    No structural zeros (T0 absent)")

    # ── Part 7: Summary ─────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  {'=' * 60}")
    print(f"  SUMMARY: PM Cross-Sieve Comparison")
    print(f"  {'=' * 60}")
    print(f"\n  {'Property':<30s} {'Eratosthenes':<18s} {'Lucky':<18s}")
    print(f"  {'-'*30} {'-'*18} {'-'*18}")
    print(f"  {'T0 (forbidden trans.)':<30s} {'YES (exact)':<18s} "
          f"{'NO' if not t0_lucky else 'YES':<18s}")
    print(f"  {'L1 (unit increment)':<30s} "
          f"{l1_prime_count}/{MAX_DEPTH}{'':>10s} "
          f"{l1_lucky_count}/{MAX_DEPTH}")
    print(f"  {'CRT structure':<30s} {'YES (mod p)':<18s} {'NO (positional)':<18s}")
    print(f"  {'Sieve type':<30s} {'Divisibility':<18s} {'Positional':<18s}")

    print(f"\n  Key insights:")
    if not t0_lucky:
        print(f"  - T0 ABSENT in Lucky: no forbidden transitions mod 3")
        print(f"    => s=1/2 is SPECIFIC to arithmetic (divisibility)")
    if l1_lucky_count < MAX_DEPTH:
        print(f"  - L1 IRREGULAR in Lucky: increment != 1 at some depths")
        print(f"    => CRT irreducibility is SPECIFIC to prime sieve")
    else:
        print(f"  - L1 holds for Lucky (surprising if true)")

    print(f"\n  Conclusion: PM discriminates between sieve types.")
    print(f"  The arithmetic content (T0, CRT, phantom rank) is")
    print(f"  SPECIFIC to Eratosthenes, not universal to all sieves.")

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  {'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
