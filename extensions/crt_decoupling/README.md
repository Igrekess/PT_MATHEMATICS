# PT_CRT_Decoupling — Companion Article 11

**Title.** CRT Decoupling and the Geometric Invariance of Modular Mutual
Information.

**Author.** Yan Senez. **Date.** May 2026. **PDF.** `main.pdf` (32 pages).

## Build

```sh
make            # produces main.pdf via latexmk
make tests      # runs scripts/test_crt_decoupling.py
```

The article depends on `../shared/pt-common.sty` (symlinked).

## Formal verification (Lean 4)

The three algebraic theorems of the article are formally verified in
the Lean 4 project at
`/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_LEAN/` under the
namespace `PT.CrtDecoupling`.

| Paper statement | Lean module / theorem | Phase |
|---|---|---|
| **Theorem 3.1** (CRT Tensor Factorization) | `PT.CrtDecoupling.Tensor.crt_tensor_factor` | ✅ Phase 1 |
| **Lemma 3.2** (Ruelle Factorization) | `PT.CrtDecoupling.Tensor.ruelle_factor` | ✅ Phase 1 |
| **Theorem 6.2** (Geometric Decoupling, Ruelle form) | `PT.CrtDecoupling.Main.geometric_decoupling_ruelle` | ✅ Phase 1 |
| **Theorem 6.1** (Geometric Decoupling, empirical form) | `PT.CrtDecoupling.Empirical.empirical_invariance` | ✅ Phase 2 |
| Step 4 algebraic core (a.s.-constant conditioning) | `PT.CrtDecoupling.Empirical.mutualInformation_of_a_s_constant_factor` | ✅ Phase 2 |
| **Lemma 4.1** (Spectral Measurability of $G$, finite form) | `PT.CrtDecoupling.SpectralReduction.geometric_spectrally_measurable` | ✅ Phase 2.5 |
| **Theorem 6.1, closed form** (no hypothesis on $\pi$) | `PT.CrtDecoupling.SpectralReduction.empirical_invariance_closed` | ✅ Phase 2.5 |
| **Theorem 6.1, infinite-state form** (Birkhoff–Hopf via Mathlib `Ergodic`) | `PT.CrtDecoupling.Phase3.geometric_a_s_constant_ergodic`, `empirical_invariance_infinite` | ✅ Phase 3 |
| Path space + shift + `SieveErgodicSetup` | `PT.CrtDecoupling.Phase4.pathSpace`, `shift`, `shift_measurable`, `SieveErgodicSetup` | ✅ Phase 4 |
| **Theorem 6.1, sieve-specific closure** | `PT.CrtDecoupling.Phase4.sieve_empirical_invariance` | ✅ Phase 4 |
| Populated instance + closure application | `PT.CrtDecoupling.Phase4.trivialSetup`, `trivialSetup_empirical_invariance` | ✅ Phase 4.1 |

The Lean files build with `lake build PT.CrtDecoupling` (Phases 1 + 2 +
2.5 + 3 + 4 + 4.1 complete, May 2026). **0 sorry, 0 axiom** in all eight files.

**Phases 2.5–4.1 close the chain layer by layer**: spectral
measurability gives the finite-state version (Phase 2.5), Mathlib's
Birkhoff–Hopf gives the abstract infinite-state version (Phase 3),
Phase 4 packages the path-space machinery into a `SieveErgodicSetup`
structure, and **Phase 4.1 exhibits a concrete populated instance**
(`trivialSetup`, with state space `Unit` and Dirac measure) and
applies the closure theorem to it. The chain is *technically closed*:
all theorems are proved without `sorry` or `axiom`, and the closure
theorem produces non-vacuous statements when applied.

The *quantitative* content of Theorem 6.1 for the actual prime-sieve
Markov measure (rather than the trivial instance) requires populating
`SieveErgodicSetup` with a non-trivial instance from the corpus,
aggregating (i) `Mathlib.Probability.Kernel.IonescuTulcea` for the
Markov measure construction, (ii) `PT.Stochastic.T30FullSpectralAnalysis`
for primitivity of `T_p`, and (iii) the classical
"primitive Markov chain ⇒ ergodic" theorem (not yet in Mathlib for
arbitrary finite state spaces — the single remaining technical gap, a
natural Mathlib PR).

**Phase 3** (Corollary 7.1, Decoupling Closed Form) is **not**
formalisable as a theorem: $k_{p,q}^{\rm eff}$ is an empirically fitted
integer parameter, not a derived quantity. Only the prediction "the
formula at fixed $k$ yields value $X(k)$" can be made into a Lean
computation.

## Numerical tests

| Test | File | Status |
|---|---|---|
| T1 — closed form for $\mathcal{I}_{p,q}$ | `scripts/test_crt_decoupling.py` | PASS (4.5%, 8.1% rel. error) |
| T2 — Ruelle factorisation $\pi_{15} = \pi_3 \otimes \pi_5$ | id. | PASS ($5.88 \times 10^{-53}$ at 50 dps) |
| T3 — empirical MI under product measure | id. | PASS ($1.45 \times 10^{-6}$ bits, $N = 10^6$) |

T2 PASS is the numerical witness of Lemma 3.2 / `ruelle_factor`.
